"""Guarded B601-RS MIT driver for the independent Spectacles CAN bus.

The six arm joints use the vendor's factory coordinate system and URDF. The
recorded sleep pose is only a parking target, never a software zero. All motor
handles and CAN transactions stay on one control thread.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

# The vendor SDK (motorbridge, pinocchio, the URDF) in its own Python 3.11 environment, and the
# MacCAN library its CAN adapter needs. Session paths at the venue (docs/SPECTACLES.md).
B601_SDK = Path(os.environ.get("TRASHDROP_B601_SDK", "/private/tmp/trashdrop-rebot-sdk"))
B601_PCBUSB = Path(os.environ.get("TRASHDROP_B601_PCBUSB", "/private/tmp/trashdrop-pcbusb/PCBUSB"))


# The one motion limit for now (the user, 2026-09-27 14:40): no travel, turn or floor envelope. The
# old demo envelope (5 cm, 15 degrees, not below the start) refused 77% of the drag ticks of the
# 14:18 session, and a refused target left the arm standing until the next one got through.
JOINT_SPEED = math.radians(20.0)  # 15 until 2026-09-27 ~15:50
# The gripper motor turns about 340 degrees shut to open (b601_park.toml): at 15 degrees/s one
# opening took 23 s. 90 degrees/s of the motor opens it in about 4 s.
GRIP_SPEED = math.radians(90.0)
# Without measured gripper angles in b601_park.toml: this far from where it was at LIVE.
GRIP_ENVELOPE = math.radians(5.0)
# Closing on an item stops the jaw, and a far target would have the motor squeeze at full torque:
# the gripper is never commanded further than this from where it is (kp 50: about 4.4 N m).
GRIP_SQUEEZE = math.radians(5.0)
FOLLOW_ERROR = math.radians(5.0)  # the arm joints only: an item in the jaw blocks the gripper by design
FOLLOW_TICKS = 3  # ...and only when it lasts: one late reply is not a joint that failed to follow
# Each tick takes one damped least-squares step of the six arm joints towards the tool target.
IK_GAIN = 6.0  # 1/s: the step heads for the target at this rate, then JOINT_SPEED caps it
IK_DAMPING = 0.02  # m: keeps the step small and steady near a singular pose
ROTATION_WEIGHT = 0.1  # m per rad: a radian of tool orientation counts like 10 cm of position
LEAD_M = 0.05  # a tick aims at most this far ahead of the tool...
LEAD_RAD = math.radians(20.0)  # ...or turns towards at most this much of its orientation
LIMIT_MARGIN = math.radians(2.0)  # kept inside the URDF joint limits, the mechanical stops
# Gravity feed-forward, the vendor's own law (reBotArm_control_py GravityCompensation): g(q) from the
# URDF, times tau_scale from config/rebotarm_rs.yaml, joint directions +1. Without it the stiff MIT
# hold sagged about 2.3 degrees (14:18 session). It fades in over GRAVITY_RAMP_S after enabling.
TAU_SCALE = np.array([1.0, 0.98, 0.98, 1.0, 1.0, 1.0])
GRAVITY_RAMP_S = 1.0
TAU_LIMIT = np.array([18.0, 18.0, 18.0, 7.0, 7.0, 7.0])  # N m: half the URDF efforts, a sanity bound
HOST_ID = 0xFD
MECH_POS = 0x7019
MODELS = ("rs-06", "rs-06", "rs-06", "rs-00", "rs-00", "rs-00", "rs-00")
GAINS = ((50.0, 3.0), (150.0, 10.0), (150.0, 10.0),
         (50.0, 5.0), (50.0, 4.0), (50.0, 4.0), (50.0, 4.0))


def sleep_pose(path: Path) -> np.ndarray:
    """The human-selected parking pose, expressed in factory motor radians."""

    values = tomllib.loads(path.read_text())["joint_degrees"]
    pose = np.radians(np.asarray(values, dtype=float))
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("b601_park.toml needs seven finite joint angles")
    return pose


def gripper_range(path: Path, closed: float) -> tuple[float, float]:
    """The gripper motor's closed and open angles, radians.

    ``gripper_closed_degrees`` and ``gripper_open_degrees`` in b601_park.toml, once measured
    (``python -m trashdrop.b601_motor --read`` with the jaw shut, then open by hand). Until then:
    ``closed`` (where it was at LIVE) and GRIP_ENVELOPE further.
    """

    values = tomllib.loads(path.read_text())
    if "gripper_closed_degrees" in values and "gripper_open_degrees" in values:
        shut, wide = math.radians(float(values["gripper_closed_degrees"])), math.radians(
            float(values["gripper_open_degrees"]))
        if math.isfinite(shut) and math.isfinite(wide) and shut != wide:
            return shut, wide
        raise ValueError("b601_park.toml: gripper_closed_degrees and gripper_open_degrees must differ")
    return closed, closed + GRIP_ENVELOPE


def grip_step(previous: float, stepped: float, actual: float, elapsed: float) -> tuple[float, float]:
    """The gripper's set point, held within GRIP_SQUEEZE of where it is, and its velocity (rad/s).

    The velocity goes with the set point as a feed-forward: without it the damping held a moving
    jaw back (kd 4 at 90 degrees/s: 7 degrees). Held at the squeeze limit, the set point stops,
    and so does the feed-forward: what presses on an item is kp times GRIP_SQUEEZE, no more.
    """

    angle = min(max(stepped, actual - GRIP_SQUEEZE), actual + GRIP_SQUEEZE)
    velocity = (angle - previous) / elapsed if elapsed > 1e-4 else 0.0
    return angle, velocity


def velocity_step(current: np.ndarray, desired: np.ndarray, elapsed: float) -> np.ndarray:
    """Cap each joint's command velocity even after a delayed hand packet."""

    if not math.isfinite(elapsed) or elapsed < 0 or current.shape != (7,) or desired.shape != (7,):
        raise ValueError("invalid B601 joint step")
    if not np.isfinite(current).all() or not np.isfinite(desired).all():
        raise ValueError("non-finite B601 joint target")
    limit = np.array([JOINT_SPEED] * 6 + [GRIP_SPEED]) * min(elapsed, 0.05)
    return current + np.clip(desired - current, -limit, limit)


def dls_step(q: np.ndarray, error: np.ndarray, jacobian: np.ndarray, lower: np.ndarray, upper: np.ndarray,
             dt: float, speed: float = JOINT_SPEED) -> tuple[np.ndarray, list[int]]:
    """One damped least-squares step of the six arm joints towards a tool error; and the joints held at a limit.

    ``error``: the tool's position (m) and orientation (rotation vector, rad) short of the target,
    in the base frame, as ``jacobian`` (6 x 6, LOCAL_WORLD_ALIGNED). The step heads for the target
    at IK_GAIN; when any joint would pass ``speed``, the whole step shrinks together, so the tool
    keeps its direction (clipping joint by joint bends its path). A target out of reach leaves the
    arm as near as it gets, and it slides along a joint limit rather than stopping dead.
    """

    weights = np.array([1.0, 1.0, 1.0, ROTATION_WEIGHT, ROTATION_WEIGHT, ROTATION_WEIGHT])
    weighted = weights[:, None] * jacobian
    step = weighted.T @ np.linalg.solve(weighted @ weighted.T + IK_DAMPING ** 2 * np.eye(6), weights * error)
    velocity = IK_GAIN * step
    fastest = float(np.max(np.abs(velocity))) / speed
    if fastest > 1.0:
        velocity /= fastest
    wanted = q + velocity * max(dt, 0.0)
    # A joint that starts past its margin (the park pose sits at a factory stop) may come inward only.
    low = np.minimum(lower + LIMIT_MARGIN, q)
    high = np.maximum(upper - LIMIT_MARGIN, q)
    moved = np.clip(wanted, low, high)
    return moved, [joint for joint in range(len(q)) if moved[joint] != wanted[joint]]


class B601Motor:
    """One owner for seven motors; only enabled after an explicit LIVE command."""

    def __init__(self, *, channel: str, urdf: Path, park: Path) -> None:
        self.channel, self.urdf = channel, Path(urdf)
        self.park = sleep_pose(park)
        self.park_file = Path(park)
        self.grip_closed = self.grip_open = 0.0
        self._enabled_at = math.inf
        self.controller = None
        self.motors = []
        self.enabled = []
        self.model = self.data = None
        self.tip_id = None
        self.start = self.command = self.desired = None
        self.feedback = None
        self._lagging = 0
        self._last_at = 0.0
        self._park_settle_since = 0.0
        self.target: tuple[np.ndarray, np.ndarray] | None = None  # where the tool is headed, base frame
        self._sdk_root = self.urdf.parents[3]
        self.fault: str | None = None
        self.parking = False
        self.state = "off"

    def _read(self, motor) -> float:
        value = motor.robstride_get_param_f32_host_id(MECH_POS, HOST_ID, 500)
        if not math.isfinite(value):
            raise RuntimeError("non-finite B601 motor feedback")
        return value

    def connect(self) -> None:
        """Read, seed present positions, and enable without commanding zero."""

        from motorbridge import Controller, Mode
        import pinocchio as pin

        self.controller = Controller(self.channel)
        try:
            if str(self._sdk_root) not in sys.path:
                sys.path.insert(0, str(self._sdk_root))
            for index, model_name in enumerate(MODELS, 1):
                motor = self.controller.add_robstride_motor(index, HOST_ID, model_name)
                self.motors.append(motor)
                identity = motor.robstride_ping_host_id(HOST_ID, 500)
                if identity[0] != index:
                    raise RuntimeError(f"B601 J{index} replied as {identity!r}")
                faults, _ = motor.robstride_get_fault_report()
                if faults:
                    raise RuntimeError(f"B601 J{index} fault {faults:#x}")
            present = np.asarray([self._read(motor) for motor in self.motors])
            self.model = pin.buildModelFromUrdf(str(self.urdf))
            if self.model.nq < 6:
                raise RuntimeError("B601 URDF has fewer than six joints")
            self.data = self.model.createData()
            self.tip_id = self.model.getFrameId("gripper_end")
            if self.tip_id >= len(self.model.frames):
                raise RuntimeError("B601 URDF has no gripper_end frame")
            lower = np.asarray(self.model.lowerPositionLimit[:6])
            upper = np.asarray(self.model.upperPositionLimit[:6])
            # A small encoder offset at a factory limit is tolerated; targets
            # still have to move inward, never farther past that limit.
            if np.any(present[:6] < lower - math.radians(2)) or np.any(
                    present[:6] > upper + math.radians(2)):
                raise RuntimeError("B601 starting pose lies outside the URDF joint limits")
            self.start = present.copy()
            self.command = present.copy()
            self.desired = present.copy()
            self.feedback = present.copy()
            for motor in self.motors:
                motor.ensure_mode(Mode.MIT, 1000)
            # Send a present-position target while disabled. The first enabled
            # command must not be the SDK controller's default all-zero pose.
            for motor, angle, (kp, kd) in zip(self.motors, present, GAINS):
                motor.send_mit(float(angle), 0.0, kp, kd, 0.0)
            for index, (motor, angle, (kp, kd)) in enumerate(zip(self.motors, present, GAINS), 1):
                motor.enable()
                self.enabled.append(motor)
                motor.send_mit(float(angle), 0.0, kp, kd, 0.0)
                if abs(self._read(motor) - angle) > math.radians(1):
                    raise RuntimeError(f"B601 J{index} moved unexpectedly while enabling")
            self._last_at = time.monotonic()
            self._enabled_at = self._last_at
            self.grip_closed, self.grip_open = gripper_range(self.park_file, float(present[6]))
            self.state = "holding: pinch with either hand"
        except Exception:
            self.close()
            raise

    def tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        import pinocchio as pin

        if self.command is None:
            raise RuntimeError("B601 is not connected")
        q = np.zeros(self.model.nq)
        q[:6] = self.command[:6]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        frame = self.data.oMf[self.tip_id]
        return frame.translation.copy(), frame.rotation.copy()

    def target_tool(self, position: np.ndarray, rotation: np.ndarray) -> bool:
        """Head the tool for this pose; step() takes it there at no more than JOINT_SPEED."""

        if (position.shape != (3,) or rotation.shape != (3, 3) or
                not np.isfinite(position).all() or not np.isfinite(rotation).all()):
            raise ValueError("invalid B601 tool target")
        self.target = (position.copy(), rotation.copy())
        return True

    def _track(self, dt: float) -> None:
        """Set this tick's six joint set points one damped least-squares step towards the target."""

        import pinocchio as pin

        q = np.zeros(self.model.nq)
        q[:6] = self.command[:6]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        frame = self.data.oMf[self.tip_id]
        position, rotation = self.target
        short = position - frame.translation
        turn = pin.log3(rotation @ frame.rotation.T)
        distance, angle = float(np.linalg.norm(short)), float(np.linalg.norm(turn))
        if distance > LEAD_M:
            short *= LEAD_M / distance
        if angle > LEAD_RAD:
            turn *= LEAD_RAD / angle
        jacobian = pin.computeFrameJacobian(self.model, self.data, q, self.tip_id,
                                            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, :6]
        lower = np.asarray(self.model.lowerPositionLimit[:6])
        upper = np.asarray(self.model.upperPositionLimit[:6])
        moved, pinned = dls_step(self.command[:6], np.concatenate((short, turn)), jacobian, lower, upper, dt)
        self.desired[:6] = moved
        if pinned:
            self.state = "at a joint limit: " + ", ".join(f"J{joint + 1}" for joint in pinned)
        elif distance < 0.003 and angle < math.radians(1.0):
            self.state = "at the hand"
        else:
            self.state = f"following the hand · {math.degrees(JOINT_SPEED):.0f}°/s max"

    def _update_feedback(self) -> None:
        """Every motor's latest reply, as the vendor's SDK reads them (RebotArm.get_positions).

        A request frame each, the receive queue drained, the cached state read: nothing waits
        for an answer. The loop used to read a parameter and wait up to 500 ms for it; one that
        timed out (14:55) took the bridge down with all seven motors disabled, and the arm fell.
        A motor that has not answered keeps its last reading.
        """

        for motor in self.motors:
            try:
                motor.request_feedback()
            except Exception:
                pass
        try:
            self.controller.poll_feedback_once()
        except Exception:
            return
        for index, motor in enumerate(self.motors):
            try:
                state = motor.get_state()
            except Exception:
                continue
            if state is not None and math.isfinite(state.pos):
                self.feedback[index] = state.pos

    def _gravity(self, now: float) -> np.ndarray:
        """This tick's gravity torques for the six arm joints, N m, faded in after enabling."""

        import pinocchio as pin

        q = np.zeros(self.model.nq)
        q[:6] = self.command[:6]
        torque = TAU_SCALE * np.asarray(pin.computeGeneralizedGravity(self.model, self.data, q))[:6]
        fade = min(max((now - self._enabled_at) / GRAVITY_RAMP_S, 0.0), 1.0)
        return np.clip(torque * fade, -TAU_LIMIT, TAU_LIMIT)

    def set_grip(self, open_grip: bool) -> None:
        self.set_grip_fraction(1.0 if open_grip else 0.0)

    def set_grip_fraction(self, fraction: float) -> None:
        """Open the jaw this share of the way from closed (0) to open (1): gripper_range."""

        share = min(max(float(fraction), 0.0), 1.0)
        self.desired[6] = self.grip_closed + share * (self.grip_open - self.grip_closed)

    def grip_fraction(self, *, now: bool = False) -> float:
        """How far open the jaw is to be, 0 to 1; ``now``: where it is being driven this moment."""

        angle = self.command[6] if now else self.desired[6]
        share = float(angle - self.grip_closed) / (self.grip_open - self.grip_closed)
        return min(max(share, 0.0), 1.0)

    def hold(self) -> None:
        self.target = None
        self.desired = self.command.copy()
        self.parking = False
        self.state = "holding: release and re-pinch to continue"

    def start_park(self) -> bool:
        if self.command is None:
            return False
        # From anywhere now that nothing keeps the arm near it (it refused past 15 degrees): the
        # joints go straight to the sleep pose at JOINT_SPEED, so watch a long way home.
        self.target = None
        self.desired = self.park.copy()
        self.desired[6] = self.command[6]
        self.parking = True
        self._park_settle_since = 0.0
        self.state = "parking slowly to saved sleep pose"
        return True

    def step(self) -> bool:
        """Send one speed-limited set point; return true once parking finishes."""

        if self.command is None:
            return False
        now = time.monotonic()
        elapsed = max(0.0, now - self._last_at)
        if self.target is not None and not self.parking:
            self._track(min(elapsed, 0.05))
        previous = float(self.command[6])
        self.command = velocity_step(self.command, self.desired, elapsed)
        self._last_at = now
        self._update_feedback()
        # The gripper's set point stays within GRIP_SQUEEZE of where it is.
        self.command[6], grip_velocity = grip_step(previous, float(self.command[6]), float(self.feedback[6]),
                                                   min(elapsed, 0.05))
        torque = np.append(self._gravity(now), 0.0)  # the gripper holds by position alone
        velocity = [0.0] * 6 + [grip_velocity]
        for motor, angle, (kp, kd), speed, feed in zip(self.motors, self.command, GAINS, velocity, torque):
            motor.send_mit(float(angle), float(speed), kp, kd, float(feed))
        behind = np.abs(self.feedback[:6] - self.command[:6]) > FOLLOW_ERROR
        self._lagging = self._lagging + 1 if behind.any() else 0
        if self._lagging >= FOLLOW_TICKS:
            self._lagging = 0
            self.hold()
            self.fault = f"J{int(np.argmax(behind)) + 1} did not follow: holding"
            self.state = self.fault
        if self.parking:  # the gripper stays as it is: it may hold an item
            close_command = np.max(np.abs(self.command[:6] - self.park[:6])) < math.radians(0.2)
            close_feedback = np.max(np.abs(self.feedback[:6] - self.park[:6])) < math.radians(1)
            if close_command and close_feedback:
                if self._park_settle_since == 0:
                    self._park_settle_since = now
                elif now - self._park_settle_since >= 0.5:
                    self.close()
                    self.state = "parked: motors disabled"
                    return True
            else:
                self._park_settle_since = 0.0
        return False

    def close(self) -> None:
        for motor in reversed(self.enabled):
            try:
                motor.disable()
            except Exception:
                pass
        self.enabled.clear()
        for motor in self.motors:
            motor.close()
        self.motors.clear()
        if self.controller is not None:
            try:
                self.controller.close_bus()
            finally:
                self.controller.close()
            self.controller = None
        self.command = self.desired = None
        self.parking = False


def read_angles(channel: str = "can0@1000000") -> np.ndarray:
    """All seven motors' angles, degrees, enabling none: to measure the gripper, or a pose."""

    from motorbridge import Controller

    controller = Controller(channel)
    motors = []
    try:
        for index, model_name in enumerate(MODELS, 1):
            motors.append(controller.add_robstride_motor(index, HOST_ID, model_name))
        return np.degrees([motor.robstride_get_param_f32_host_id(MECH_POS, HOST_ID, 500) for motor in motors])
    finally:
        for motor in motors:
            motor.close()
        try:
            controller.close_bus()
        finally:
            controller.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trashdrop.b601_motor",
                                     description="B601-RS readings; stop `trashdrop web --b601` first (one CAN owner)")
    parser.add_argument("--read", action="store_true", help="print all seven angles; enables no motor")
    parser.add_argument("--channel", default="can0@1000000")
    args = parser.parse_args(argv)
    if not args.read:
        parser.print_help()
        return 1
    angles = read_angles(args.channel)
    print("B601 angles, degrees, J1..J6 then the gripper J7: " + ", ".join(f"{angle:.3f}" for angle in angles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
