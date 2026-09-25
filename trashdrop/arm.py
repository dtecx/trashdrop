"""Driving one real SO-101: slow, smooth, interruptible moves.

The servos count in ticks, 4096 to a turn. The arms were calibrated with
LeRobot, which leaves two things in each motor's EEPROM: a homing offset that
makes the calibration pose read 2047 on every joint, and the joint's limits.
People read and write degrees from that calibration pose instead -- and the
gripper as percent open -- and this module converts.

Three rules keep a real arm, and the people next to it, safe:

* Torque goes on only after every goal register holds the joint's present
  position. The servos power up with goal 0: enabling torque as they are
  would throw every joint to its end stop at full speed.
* A move is a trajectory, never a jump. Goals are streamed at 50 Hz along a
  smooth path no faster than ``max_speed``, and each servo's own speed limit
  is set to twice that, so that even a wrong goal arrives slowly.
* Every target is clamped inside the joint's EEPROM limits, with a margin.

Ctrl+C during a move stops the arm where it is and keeps it holding: letting
go of the torque mid-air would drop it.
"""

from __future__ import annotations

import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .servo import MOTORS
from .station import repository_root

TICKS_PER_TURN = 4096
CALIBRATION_TICK = 2047  # where LeRobot's homing puts the calibration pose
DEG_PER_TICK = 360.0 / TICKS_PER_TURN
LIMIT_MARGIN = 20  # ticks kept clear of each EEPROM limit, ~1.8 degrees
RATE_HZ = 50
MIN_JERK_PEAK = 1.875  # peak speed of a minimum-jerk move, relative to its average
SETTLED_TICKS = 25  # how close a joint must end up to call a move done
GRIPPER = "gripper"
POSES_FILE = repository_root() / "poses.toml"


@dataclass(frozen=True)
class JointLimits:
    low: int
    high: int


class Arm:
    """One arm on its servo bus. Degrees in, degrees out; ticks stay inside."""

    def __init__(self, name: str, bus, max_speed: float, *, clock=time.monotonic, sleep=time.sleep) -> None:
        self.name = name
        self.bus = bus
        self.max_speed = max_speed
        self._clock, self._sleep = clock, sleep
        self.limits = {
            joint: JointLimits(bus.read(motor, "min_limit"), bus.read(motor, "max_limit"))
            for joint, motor in MOTORS.items()
        }

    # --- units ---------------------------------------------------------------

    def to_ticks(self, joint: str, value: float) -> int:
        """Degrees from the calibration pose (gripper: percent open), clamped."""

        limits = self.limits[joint]
        if joint == GRIPPER:
            ticks = limits.low + value / 100.0 * (limits.high - limits.low)
        else:
            ticks = CALIBRATION_TICK + value / DEG_PER_TICK
        return int(round(min(max(ticks, limits.low + LIMIT_MARGIN), limits.high - LIMIT_MARGIN)))

    def from_ticks(self, joint: str, ticks: int) -> float:
        limits = self.limits[joint]
        if joint == GRIPPER:
            return 100.0 * (ticks - limits.low) / (limits.high - limits.low)
        return (ticks - CALIBRATION_TICK) * DEG_PER_TICK

    def pose(self) -> dict[str, float]:
        """Where every joint is now, in degrees (gripper: percent open)."""

        return {joint: round(self.from_ticks(joint, ticks), 1) for joint, ticks in self.bus.positions().items()}

    # --- torque --------------------------------------------------------------

    def torque_is_on(self) -> bool:
        return all(self.bus.read(motor, "torque_enable") for motor in MOTORS.values())

    def torque_on(self) -> None:
        """Hold the arm where it is. Rule one: goals first, then torque."""

        present = self.bus.positions()
        speed_cap = int(round(2 * self.max_speed / DEG_PER_TICK))
        for joint, motor in MOTORS.items():
            self.bus.write(motor, "goal_position", present[joint])
            if abs(self.bus.read(motor, "goal_position") - present[joint]) > 2:
                raise RuntimeError(f"{self.name} {joint}: goal did not take; torque left off")
            self.bus.write(motor, "goal_speed", speed_cap)
        for motor in MOTORS.values():
            self.bus.write(motor, "torque_enable", 1)

    def torque_off(self) -> None:
        """Go limp. Mid-air, the arm falls: someone must be holding it."""

        for motor in MOTORS.values():
            self.bus.write(motor, "torque_enable", 0)

    def hold(self) -> None:
        """Stop wherever the joints are now, and keep holding there."""

        present = self.bus.positions()
        self.bus.write_goals({MOTORS[joint]: ticks for joint, ticks in present.items()})

    # --- moving --------------------------------------------------------------

    def move(self, targets: dict[str, float], *, speed: float | None = None) -> dict[str, float]:
        """Move the named joints to ``targets``; every other joint stays put.

        Returns the pose reached. Joints not named keep their current goal, not
        their present position, so a joint sagging under gravity does not creep
        further down with every move.
        """

        unknown = set(targets) - set(MOTORS)
        if unknown:
            raise ValueError(f"no such joint: {', '.join(sorted(unknown))}")
        if not self.torque_is_on():
            raise RuntimeError(f"{self.name} arm is limp: turn torque on first")
        speed = min(speed or self.max_speed, self.max_speed)

        start = {joint: self.bus.read(motor, "goal_position") for joint, motor in MOTORS.items()}
        goal = dict(start)
        goal.update({joint: self.to_ticks(joint, value) for joint, value in targets.items()})
        travel = max(abs(goal[joint] - start[joint]) for joint in MOTORS) * DEG_PER_TICK
        duration = max(MIN_JERK_PEAK * travel / speed, 0.3)

        began = self._clock()
        try:
            while True:
                s = min((self._clock() - began) / duration, 1.0)
                blend = s**3 * (10 - 15 * s + 6 * s**2)  # minimum jerk
                self.bus.write_goals(
                    {MOTORS[joint]: round(start[joint] + (goal[joint] - start[joint]) * blend) for joint in MOTORS}
                )
                if s >= 1.0:
                    break
                self._sleep(1.0 / RATE_HZ)
            deadline = self._clock() + 1.5
            while self._clock() < deadline:
                present = self.bus.positions()
                if all(abs(present[joint] - goal[joint]) <= SETTLED_TICKS for joint in MOTORS):
                    break
                self._sleep(1.0 / RATE_HZ)
        except KeyboardInterrupt:
            self.hold()
            raise
        return self.pose()


def connect(name: str, rig=None) -> Arm:
    """The arm called ``name`` in rig.toml -- "left", "right", or its label."""

    from .rig import load_rig
    from .servo import ServoBus

    rig = rig or load_rig()
    key = resolve_arm(name, rig)
    devices = rig.arms[key]
    if not devices.bus:
        raise RuntimeError(f"the {key} arm has no bus in rig.toml: run `uv run trashdrop rig identify`")
    return Arm(key, ServoBus.by_serial(devices.bus), devices.max_speed)


def resolve_arm(name: str, rig) -> str:
    wanted = name.strip().lower()
    for key, devices in rig.arms.items():
        if wanted in (key, devices.label.lower()):
            return key
    raise ValueError(f"no arm called {name!r}; rig.toml has: " + ", ".join(
        f"{key} ({devices.label})" for key, devices in rig.arms.items()))


# --- named poses ----------------------------------------------------------------


def load_poses(path: Path = POSES_FILE) -> dict[str, dict[str, dict[str, float]]]:
    """arm -> pose name -> joint -> degrees (gripper: percent open)."""

    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def save_pose(arm: str, name: str, values: dict[str, float], path: Path = POSES_FILE) -> Path:
    poses = load_poses(path)
    poses.setdefault(arm, {})[name] = values
    lines = [
        "# Named poses, per arm: degrees from the pose the arm was calibrated in,",
        "# the gripper in percent open. Edit freely. To record one, pose the limp",
        "# arm by hand and run:   uv run trashdrop arm save left rest",
        "# and to play it back:   uv run trashdrop arm go left rest",
    ]
    for arm_name in sorted(poses):
        for pose_name in sorted(poses[arm_name]):
            lines += ["", f"[{arm_name}.{pose_name}]"]
            for joint, value in poses[arm_name][pose_name].items():
                lines.append(f"{joint} = {float(value):.1f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
