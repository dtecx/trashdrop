"""Driving one real SO-101: slow, smooth, interruptible moves.

The servos count in ticks, 4096 to a turn. The arms were calibrated with
LeRobot, which leaves each joint's homing offset and range in the motor's
EEPROM. People read and write LeRobot's own units instead, so that a pose here
means the same as in ``lerobot-teleoperate`` or a recorded dataset: degrees
from the middle of each joint's calibrated range (LeRobot's ``use_degrees``),
and the gripper 0..100 across its range, 0 closed.

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

import math
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .servo import MOTORS
from .station import repository_root

# LeRobot divides by the largest tick value, not by ticks per turn.
DEG_PER_TICK = 360.0 / 4095
LIMIT_MARGIN = 20  # ticks kept clear of each EEPROM limit, ~1.8 degrees
RATE_HZ = 50
MIN_JERK_PEAK = 1.875  # peak speed of a minimum-jerk move, relative to its average
SETTLED_TICKS = 25  # how close a joint must end up to call a move done
SERVO_ACCELERATION = 254  # what LeRobot sets; it resets when a servo loses power
STILL_TICKS = 1  # ... or how little it may move between two readings, 20 ms apart,
STILL_READINGS = 5  # this many times in a row, to have stopped where it is
GRIPPER = "gripper"
FIRST_POSE = "organizers_first"
POSES_FILE = repository_root() / "poses.toml"


class Stopped(KeyboardInterrupt):
    """A stop asked for from elsewhere -- the web page's button -- and handled like Ctrl+C: hold."""


@dataclass(frozen=True)
class JointLimits:
    low: int
    high: int


class Arm:
    """One arm on its servo bus. Degrees in, degrees out; ticks stay inside."""

    def __init__(self, name: str, bus, max_speed: float, *, clock=time.monotonic, sleep=time.sleep,
                 stop: threading.Event | None = None) -> None:
        self.name = name
        self.bus = bus
        self.max_speed = max_speed
        self._clock, self._sleep = clock, sleep
        self.stop = stop  # set from another thread, it stops a move where it is
        self.limits = {
            joint: JointLimits(bus.read(motor, "min_limit"), bus.read(motor, "max_limit"))
            for joint, motor in MOTORS.items()
        }

    # --- units ---------------------------------------------------------------

    def to_ticks(self, joint: str, value: float) -> int:
        """LeRobot degrees (gripper: 0..100), clamped inside the joint's range."""

        limits = self.limits[joint]
        if joint == GRIPPER:
            ticks = limits.low + value / 100.0 * (limits.high - limits.low)
        else:
            ticks = (limits.low + limits.high) / 2 + value / DEG_PER_TICK
        return int(round(min(max(ticks, limits.low + LIMIT_MARGIN), limits.high - LIMIT_MARGIN)))

    def from_ticks(self, joint: str, ticks: int) -> float:
        limits = self.limits[joint]
        if joint == GRIPPER:
            return 100.0 * (ticks - limits.low) / (limits.high - limits.low)
        return (ticks - (limits.low + limits.high) / 2) * DEG_PER_TICK

    def limits_degrees(self) -> dict[str, tuple[float, float]]:
        """Each joint's usable range in degrees, margin included (gripper left out)."""

        return {
            joint: (self.from_ticks(joint, limits.low + LIMIT_MARGIN), self.from_ticks(joint, limits.high - LIMIT_MARGIN))
            for joint, limits in self.limits.items()
            if joint != GRIPPER
        }

    def pose(self) -> dict[str, float]:
        """Where every joint is now, in degrees (gripper: percent open)."""

        return {joint: round(self.from_ticks(joint, ticks), 1) for joint, ticks in self.bus.positions().items()}

    # --- torque --------------------------------------------------------------

    def torque_is_on(self) -> bool:
        return all(self.bus.read(motor, "torque_enable") for motor in MOTORS.values())

    def torque_on(self) -> None:
        """Hold the arm where it is. Rule one: goals first, then torque."""

        present = self.bus.positions()
        for joint, motor in MOTORS.items():
            self.bus.write(motor, "goal_position", present[joint])
            if abs(self.bus.read(motor, "goal_position") - present[joint]) > 2:
                raise RuntimeError(f"{self.name} {joint}: goal did not take; torque left off")
        self.limit_speed()
        for motor in MOTORS.values():
            self.bus.write(motor, "torque_enable", 1)

    def limit_speed(self) -> None:
        """The servos' own speed limit, twice max_speed, behind the streamed path; LeRobot's acceleration.

        Written on connecting too, not only with torque: an arm left holding
        from an earlier run kept the limit of the max_speed it had then, and a
        faster max_speed in rig.toml changed next to nothing.
        """

        speed_cap = int(round(2 * self.max_speed / DEG_PER_TICK))
        for motor in MOTORS.values():
            self.bus.write(motor, "goal_speed", speed_cap)
            self.bus.write(motor, "acceleration", SERVO_ACCELERATION)

    def torque_off(self) -> None:
        """Go limp. Mid-air, the arm falls: someone must be holding it."""

        for motor in MOTORS.values():
            self.bus.write(motor, "torque_enable", 0)

    def relax_joints_hold_gripper(self) -> None:
        """Make the five arm joints limp while the gripper holds its opening."""

        motor = MOTORS[GRIPPER]
        present = self.bus.read(motor, "position")
        # A gripper that was limp may have an old goal. Point it at its present
        # position before torque goes on so preparing a hand-taught pose cannot
        # unexpectedly open or close the jaw.
        self.bus.write(motor, "goal_position", present)
        if abs(self.bus.read(motor, "goal_position") - present) > 2:
            raise RuntimeError(f"{self.name} gripper: goal did not take; arm left unchanged")
        self.bus.write(motor, "goal_speed", int(round(2 * self.max_speed / DEG_PER_TICK)))
        self.bus.write(motor, "acceleration", SERVO_ACCELERATION)
        self.bus.write(motor, "torque_enable", 1)
        for joint, joint_motor in MOTORS.items():
            if joint != GRIPPER:
                self.bus.write(joint_motor, "torque_enable", 0)

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

        return move_together([(self, targets)], speed=speed)[0]

    def _plan(self, targets: dict[str, float], speed: float | None) -> tuple[dict, dict, float]:
        """(start ticks, goal ticks, seconds) of one move."""

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
        return start, goal, max(MIN_JERK_PEAK * travel / speed, 0.3)


def move_together(moves: list[tuple[Arm, dict[str, float]]], *, speed: float | None = None) -> list[dict[str, float]]:
    """Move several arms at once, each to its own targets along its own smooth path.

    Returns each arm's pose reached. Ctrl+C stops every one of them where it
    is, and they all keep holding.

    A move ends when every joint is within SETTLED_TICKS of its goal, or has
    stopped: a jaw closed on an item never reaches its goal, and used to wait
    out the whole 1.5 s for it on every grasp.
    """

    plans = [arm._plan(targets, speed) for arm, targets in moves]
    clock, sleep = moves[0][0]._clock, moves[0][0]._sleep

    def check() -> None:
        if any(arm.stop is not None and arm.stop.is_set() for arm, _ in moves):
            raise Stopped

    began = clock()
    try:
        while True:
            check()
            elapsed = clock() - began
            for (arm, _), (start, goal, duration) in zip(moves, plans):
                s = min(elapsed / duration, 1.0)
                blend = s**3 * (10 - 15 * s + 6 * s**2)  # minimum jerk
                arm.bus.write_goals(
                    {MOTORS[joint]: round(start[joint] + (goal[joint] - start[joint]) * blend) for joint in MOTORS}
                )
            if all(elapsed >= duration for _, _, duration in plans):
                break
            sleep(1.0 / RATE_HZ)
        deadline = clock() + 1.5
        pending = {index: (None, 0) for index in range(len(moves))}  # index -> (last reading, still readings)
        while pending and clock() < deadline:
            check()
            for index, (last, still) in list(pending.items()):
                present = moves[index][0].bus.positions()
                goal = plans[index][1]
                if all(abs(present[joint] - goal[joint]) <= SETTLED_TICKS for joint in MOTORS):
                    del pending[index]
                    continue
                moved = last is None or any(abs(present[joint] - last[joint]) > STILL_TICKS for joint in MOTORS)
                still = 0 if moved else still + 1
                if still >= STILL_READINGS:
                    del pending[index]
                else:
                    pending[index] = (present, still)
            if pending:
                sleep(1.0 / RATE_HZ)
    except KeyboardInterrupt:
        for arm, _ in moves:
            arm.hold()
        raise
    return [arm.pose() for arm, _ in moves]


def move_named_pose_together(
    arms: dict[str, Arm], poses: dict[str, dict[str, dict[str, float]]], pose_name: str, speed: float
) -> list[dict[str, float]]:
    """Move every arm straight from its current pose to one saved pose, together.

    The saved gripper value is deliberately ignored. Each jaw holds wherever
    it is while the five arm joints follow a simultaneous minimum-jerk path.
    """

    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    moves = []
    arm_joints = tuple(joint for joint in MOTORS if joint != GRIPPER)
    for name, arm in arms.items():
        saved = poses.get(name, {}).get(pose_name)
        if saved is None:
            raise ValueError(f"no pose {pose_name!r} for the {name} arm in poses.toml")
        missing = [joint for joint in arm_joints if joint not in saved]
        if missing:
            raise ValueError(f"pose {name}.{pose_name} has no {', '.join(missing)}")
        moves.append((arm, {joint: saved[joint] for joint in arm_joints}))

    # The user may have posed a limp arm by hand, or moved a powered arm away
    # from its old goal. Start from the measured position in either case.
    for arm, _ in moves:
        arm.hold()
        if not arm.torque_is_on():
            arm.torque_on()
    return move_together(moves, speed=speed)


def move_both_to_pose(arms: dict[str, Arm], poses: dict[str, dict[str, dict[str, float]]], pose_name: str,
                      speed: float, *, clamp: bool = True) -> list[dict[str, float]]:
    """Both arms to one saved pose together, with whatever they hold between them.

    Torque goes on where it is off (goals first), then, with ``clamp``, the
    grippers close on what they hold and keep squeezing; then the five arm
    joints move together. The saved gripper value is not used.
    """

    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    arm_joints = tuple(joint for joint in MOTORS if joint != GRIPPER)
    moves = []
    for name, arm in arms.items():
        saved = poses.get(name, {}).get(pose_name)
        if saved is None:
            raise ValueError(f"no pose {pose_name!r} for the {name} arm in poses.toml")
        moves.append((arm, {joint: saved[joint] for joint in arm_joints if joint in saved}))
    for arm, _ in moves:
        if not arm.torque_is_on():
            arm.torque_on()
    if clamp:
        move_together([(arm, {GRIPPER: 0.0}) for arm, _ in moves], speed=speed)
    return move_together(moves, speed=speed)


# A first test on a real arm: from the joint that can do least harm to the one
# that swings the whole arm.
TEST_ORDER = ("gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan")
GRIPPER_TEST = 15.0  # percent the gripper opens in the test


def nudge_joints(arm: Arm, degrees: float, speed: float, *, ask=input, log=print) -> list[str]:
    """Nudge each joint +degrees and -degrees around where it is, back in between.

    ``ask(prompt)`` answers "" to go ahead, "s" to skip that joint, "q" to
    stop. Every joint ends where it started. Returns the joints moved.
    """

    # Unrounded, so that "back to the start" is the very tick it started at.
    start = {joint: arm.from_ticks(joint, ticks) for joint, ticks in arm.bus.positions().items()}
    moved = []
    for joint in TEST_ORDER:
        amount = GRIPPER_TEST if joint == GRIPPER else degrees
        unit = "%" if joint == GRIPPER else " deg"
        answer = ask(f"  {joint:13s} Enter = move it +-{amount:g}{unit}, s = skip, q = stop: ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            continue
        # The gripper only opens a little and closes again: it may already be shut.
        offsets = (amount, 0.0) if joint == GRIPPER else (amount, 0.0, -amount, 0.0)
        report = []
        for offset in offsets:
            reached = arm.move({joint: start[joint] + offset}, speed=speed)
            report.append(f"{start[joint] + offset:+.1f}->{reached[joint]:+.1f}")
        log(f"    wanted->reached: {', '.join(report)}")
        moved.append(joint)
    return moved


# --- pick and drop by taught poses ---------------------------------------------

# The poses a pick is played from, taught by hand and saved with `arm save`.
PICK_POSES = ("above", "grab", "drop", "neutral")
# A gripper told to close that stops below this is closed on nothing.
MISS_BELOW = 4.0
# How far the gripper opens to let go, percent.
RELEASE_OPEN = 60.0


def pick_and_drop(arm: Arm, poses: dict[str, dict[str, float]], *, speed: float | None = None,
                  check: bool = True, log=print) -> bool:
    """Play one pick from taught poses; True if something was carried and dropped.

    From "above" the arm comes down to "grab" with the jaws as they were
    taught (open), closes, and looks at where the jaws stopped: closed all
    the way means nothing is between them, and the arm goes back without
    pretending. While carrying, the gripper is left out of every move, so
    nothing on the way can open it.
    """

    missing = [name for name in PICK_POSES if name not in poses]
    if missing:
        raise ValueError(f"the {arm.name} arm has no pose {', '.join(missing)} in poses.toml")

    def carry(pose: dict[str, float]) -> dict[str, float]:
        return {joint: value for joint, value in pose.items() if joint != GRIPPER}

    log(f"{arm.name}: above the item")
    arm.move(poses["above"], speed=speed)
    log(f"{arm.name}: down to grab")
    arm.move(poses["grab"], speed=speed)
    log(f"{arm.name}: closing")
    held = arm.move({GRIPPER: 0.0}, speed=speed)[GRIPPER]
    if check and held < MISS_BELOW:
        log(f"{arm.name}: the jaws closed to {held:.0f} %: nothing between them, a miss")
        arm.move(poses["above"], speed=speed)
        arm.move(poses["neutral"], speed=speed)
        return False
    log(f"{arm.name}: holding it, jaws at {held:.0f} % -- lifting")
    arm.move(carry(poses["above"]), speed=speed)
    log(f"{arm.name}: over the drop zone")
    arm.move(carry(poses["drop"]), speed=speed)
    arm.move({GRIPPER: RELEASE_OPEN}, speed=speed)
    log(f"{arm.name}: dropped; back to neutral")
    arm.move(poses["neutral"], speed=speed)
    return True


def connect(name: str, rig=None, *, stop: threading.Event | None = None) -> Arm:
    """The arm called ``name`` in rig.toml -- "left", "right", or its label."""

    from .rig import load_rig
    from .servo import ServoBus

    rig = rig or load_rig()
    key = resolve_arm(name, rig)
    devices = rig.arms[key]
    if not devices.bus:
        raise RuntimeError(f"the {key} arm has no bus in rig.toml: run `uv run trashdrop rig identify`")
    arm = Arm(key, ServoBus.by_serial(devices.bus), devices.max_speed, stop=stop)
    arm.limit_speed()
    return arm


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
        "# Named poses, per arm, in LeRobot's units: degrees from the middle of each",
        "# joint's calibrated range, the gripper 0..100 (0 = closed). Edit freely.",
        "# To record one, pose the limp arm by hand and run:",
        "#     uv run trashdrop arm save left rest",
        "# and to play it back:",
        "#     uv run trashdrop arm go left rest",
    ]
    for arm_name in sorted(poses):
        for pose_name in sorted(poses[arm_name]):
            lines += ["", f"[{arm_name}.{pose_name}]"]
            for joint, value in poses[arm_name][pose_name].items():
                lines.append(f"{joint} = {float(value):.1f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
