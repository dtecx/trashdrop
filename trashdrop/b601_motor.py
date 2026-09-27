"""Guarded B601-RS MIT driver for the independent Spectacles CAN bus.

The six arm joints use the vendor's factory coordinate system and URDF. The
recorded sleep pose is only a parking target, never a software zero. All motor
handles and CAN transactions stay on one control thread.
"""

from __future__ import annotations

import math
import sys
import time
import tomllib
from pathlib import Path

import numpy as np


JOINT_SPEED = math.radians(4.0)  # below the first J1 test's five-degree-per-second ramp
GRIP_SPEED = math.radians(2.0)
JOINT_ENVELOPE = math.radians(8.0)  # small first live workspace, relative to session start
GRIP_ENVELOPE = math.radians(5.0)
MAX_TOOL_TRAVEL = 0.03  # metres from the session's starting tool position
MAX_TOOL_ROTATION = math.radians(10.0)
FOLLOW_ERROR = math.radians(5.0)
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


def velocity_step(current: np.ndarray, desired: np.ndarray, elapsed: float) -> np.ndarray:
    """Cap each joint's command velocity even after a delayed hand packet."""

    if not math.isfinite(elapsed) or elapsed < 0 or current.shape != (7,) or desired.shape != (7,):
        raise ValueError("invalid B601 joint step")
    if not np.isfinite(current).all() or not np.isfinite(desired).all():
        raise ValueError("non-finite B601 joint target")
    limit = np.array([JOINT_SPEED] * 6 + [GRIP_SPEED]) * min(elapsed, 0.05)
    return current + np.clip(desired - current, -limit, limit)


class B601Motor:
    """One owner for seven motors; only enabled after an explicit LIVE command."""

    def __init__(self, *, channel: str, urdf: Path, park: Path) -> None:
        self.channel, self.urdf = channel, Path(urdf)
        self.park = sleep_pose(park)
        self.controller = None
        self.motors = []
        self.enabled = []
        self.model = self.data = None
        self.tip_id = None
        self.start = self.command = self.desired = None
        self.feedback = None
        self._feedback_index = 0
        self._last_at = 0.0
        self._park_settle_since = 0.0
        self._start_position = self._start_rotation = None
        self._floor_z = 0.0
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
            self._start_position, self._start_rotation = self.tool_pose()
            self._floor_z = float(self._start_position[2])
            self.state = "holding: pinch with the right hand"
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
        """Solve a small 6-DoF tool step, refusing limits or downward motion."""

        import pinocchio as pin
        from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik

        if (position.shape != (3,) or rotation.shape != (3, 3) or
                not np.isfinite(position).all() or not np.isfinite(rotation).all()):
            raise ValueError("invalid B601 tool target")
        q = np.zeros(self.model.nq)
        q[:6] = self.command[:6]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        if position[2] < self._floor_z - 1e-4:
            self.state = "at the table: downward motion blocked"
            return False
        if np.linalg.norm(position - self._start_position) > MAX_TOOL_TRAVEL + 1e-4:
            self.state = "at the short demo reach limit"
            return False
        if np.linalg.norm(pin.log3(rotation @ self._start_rotation.T)) > MAX_TOOL_ROTATION + 1e-4:
            self.state = "at the short demo turn limit"
            return False
        result = solve_ik(self.model, self.data, self.tip_id, pin.SE3(rotation, position), q,
                          IKParams(max_iter=60, tolerance=0.003, step_size=0.5, damping=1e-4),
                          controlled_joints=6)
        if not result.success or not np.isfinite(result.q).all():
            self.state = "at the IK limit: holding"
            return False
        candidate = np.asarray(result.q[:6])
        lower = np.asarray(self.model.lowerPositionLimit[:6])
        upper = np.asarray(self.model.upperPositionLimit[:6])
        if (np.any(candidate < lower - math.radians(2)) or
                np.any(candidate > upper + math.radians(2)) or
                np.any((self.start[:6] < lower) & (candidate < self.start[:6] - 1e-4)) or
                np.any((self.start[:6] > upper) & (candidate > self.start[:6] + 1e-4)) or
                np.any(np.abs(candidate - self.start[:6]) > JOINT_ENVELOPE)):
            self.state = "at a joint limit: holding"
            return False
        self.desired[:6] = candidate
        self.state = "following right hand · 4°/s max"
        return True

    def set_grip(self, open_grip: bool) -> None:
        self.desired[6] = self.start[6] + (GRIP_ENVELOPE if open_grip else 0.0)

    def hold(self) -> None:
        self.desired = self.command.copy()
        self.parking = False
        self.state = "holding: release and re-pinch to continue"

    def start_park(self) -> bool:
        if self.command is None:
            return False
        if np.max(np.abs(self.command - self.park)) > math.radians(15):
            self.state = "too far from saved sleep pose: holding"
            return False
        self.desired = self.park.copy()
        self.parking = True
        self._park_settle_since = 0.0
        self.state = "parking slowly to saved sleep pose"
        return True

    def step(self) -> bool:
        """Send one speed-limited set point; return true once parking finishes."""

        if self.command is None:
            return False
        now = time.monotonic()
        self.command = velocity_step(self.command, self.desired, max(0.0, now - self._last_at))
        self._last_at = now
        for motor, angle, (kp, kd) in zip(self.motors, self.command, GAINS):
            motor.send_mit(float(angle), 0.0, kp, kd, 0.0)
        motor = self.motors[self._feedback_index]
        actual = self._read(motor)
        self.feedback[self._feedback_index] = actual
        if abs(actual - self.command[self._feedback_index]) > FOLLOW_ERROR:
            self.hold()
            self.fault = f"J{self._feedback_index + 1} did not follow: holding"
            self.state = self.fault
        self._feedback_index = (self._feedback_index + 1) % len(self.motors)
        if self.parking:
            close_command = np.max(np.abs(self.command - self.park)) < math.radians(0.2)
            close_feedback = np.max(np.abs(self.feedback - self.park)) < math.radians(1)
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
