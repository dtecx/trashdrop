"""The organisers' merch task: a dance of hand gestures, both arms as a person's two arms.

Kept apart from the sorting code on purpose: it reads what the rest provides
-- the arms, their neutral poses, the model, where the arms stand -- and
changes none of it.

    uv run python -m trashdrop.groove --preview   # no arms: the checks, and out/groove.png
    uv run python -m trashdrop.groove             # both arms dance; Ctrl+C holds them
    uv run python -m trashdrop.groove --speed 0.5 # half as fast; 1.5 half again as fast

The team acted the four moves out with their own hands (photos, 2026-09-26),
and then said how the arms should do them. Each arm stands in for a forearm
held up, its gripper for the hand:

* jaws -- a mouth: the gripper level, pointing ahead, turned so that its jaw
  opens up and down, opening wide and snapping shut.
* point -- the hand flat and still, the base turning: the gripper level,
  pointing ahead, and the bases sweeping it to one side and to the other.
* raise -- one arm straight up with the hand open, the other lowered in
  front, then the other way round.
* twist -- hands spread like claws, turning at the wrists: the grippers open
  and the wrist rolls turn back and forth, mirrored.

Every pose is an offset from each arm's neutral (straight up) -- the gripper
alone is absolute, percent open -- so each arm's own zeros do not matter. A
pose is reached on the beat, each move is eased into over two beats, and the
dance ends where it began. Between moves both arms first stand up straight:
at the venue an arm hit the wall on the way into raise, its base turning
back while it went down, and one move now never sweeps into the next. The
arms face about 16 degrees apart at the venue; for the dance each base turns
half of that, so that both stand, lean and point the same way.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

BPM = 60.0  # slow, to begin with
REPS = 4  # times through each move
LEAD_BEATS = 2.0  # into each move, and back to neutral at the end
BETWEEN_BEATS = 1.5  # from the end of one move to standing straight up, before the next
MAX_JOINT_SPEED = 220.0  # deg/s at the fastest moment of any step; an STS3215 manages about 270
DANCE_SPEED = 150.0  # the servos' own limit while dancing is twice this, the same for both arms
MIN_APART_CM = 8.0  # between the two arms' centre lines, anywhere along them
MIN_ABOVE_TABLE_CM = 10.0  # the lowest point of either arm past its shoulder
MAX_BEHIND_CM = 2.0  # no part of either arm behind where its base stands: out of sight, and near the wall
GRIPPER_DEG_PER_PERCENT = 1.3  # the jaw swings about 128 degrees over 0..100 %
ARMS = ("left", "right")
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MIRRORED = ("shoulder_pan", "wrist_roll")  # turned the other way on the right arm, for a mirror image


def _both(pose: dict[str, float]) -> dict[str, dict[str, float]]:
    return {"left": dict(pose), "right": dict(pose)}


def _mirrored(pose: dict[str, float]) -> dict[str, dict[str, float]]:
    return {"left": dict(pose), "right": {joint: -value if joint in MIRRORED else value for joint, value in pose.items()}}


OPEN = 70.0  # percent: a hand spread wide, a mouth open wide
# Level and pointing ahead; turned a quarter so the jaw opens up and down.
# The other quarter, lower jaw moving, is past the right wrist's limit.
MOUTH = {"wrist_flex": 90.0, "wrist_roll": 90.0}
FLAT = {"wrist_flex": 90.0}  # level and pointing ahead, the jaw closing sideways: a flat hand
SWEEP = 75.0  # the bases' turn for point, each way; the left base stops at 84 with its trim on
UP = {"gripper": OPEN}  # straight up, the hand open
DOWN = {"shoulder_lift": 70.0, "wrist_flex": 20.0, "gripper": 20.0}  # lowered in front, the hand loose
# Each move: its poses in order, as (arm -> offsets from neutral, beats to reach it).
MOVES: dict[str, tuple[tuple[dict[str, dict[str, float]], float], ...]] = {
    "jaws": ((_both({**MOUTH, "gripper": OPEN}), 1.0), (_both({**MOUTH, "gripper": 0.0}), 1.0)),
    "point": ((_both({**FLAT, "shoulder_pan": -SWEEP}), 2.0), (_both({**FLAT, "shoulder_pan": SWEEP}), 2.0)),
    "raise": (({"left": DOWN, "right": UP}, 2.0), ({"left": UP, "right": DOWN}, 2.0)),
    "twist": ((_mirrored({"gripper": 60.0, "wrist_roll": 40.0}), 1.0),
              (_mirrored({"gripper": 60.0, "wrist_roll": -40.0}), 1.0)),
}
ORDER = tuple(MOVES)


# --- the dance as keyframes -----------------------------------------------------------


@dataclass
class Routine:
    """Both arms' keyframes and the seconds to each: where every joint is at any moment."""

    keyframes: list[dict[str, dict[str, float]]]  # arm -> joint -> degrees (gripper: percent open)
    seconds: list[float]  # from keyframes[i] to keyframes[i + 1]
    labels: list[str]  # the move each keyframe belongs to

    @property
    def total(self) -> float:
        return float(sum(self.seconds))

    def _step(self, t: float) -> tuple[int, float]:
        """(the step under way at ``t``, how far through it, 0..1)."""

        t = min(max(t, 0.0), self.total)
        for index, span in enumerate(self.seconds):
            if t < span or index == len(self.seconds) - 1:
                return index, min(t / span, 1.0) if span > 0 else 1.0
            t -= span
        return len(self.seconds) - 1, 1.0

    def at(self, t: float) -> dict[str, dict[str, float]]:
        index, u = self._step(t)
        eased = (1.0 - math.cos(math.pi * u)) / 2.0  # still at both ends of every step: no jerk on the beat
        a, b = self.keyframes[index], self.keyframes[index + 1]
        return {arm: {joint: a[arm][joint] + (b[arm][joint] - a[arm][joint]) * eased for joint in a[arm]}
                for arm in a}

    def label(self, t: float) -> str:
        return self.labels[self._step(t)[0] + 1]


def build(start: dict[str, dict[str, float]], neutral: dict[str, dict[str, float]], *, moves=ORDER,
          reps: int = REPS, bpm: float = BPM, trims: dict[str, float] | None = None) -> Routine:
    """The dance from ``start`` back to ``neutral``: each of ``moves`` ``reps`` times, at ``bpm``.

    ``trims`` turn each base by that many degrees throughout, so that both
    arms face the same way (see facing()).
    """

    unknown = [name for name in moves if name not in MOVES]
    if unknown or not moves:
        raise ValueError(f"the moves are {', '.join(MOVES)}; not {', '.join(unknown) or 'none'}")
    if not (0 < bpm <= 200 and 1 <= reps <= 32):
        raise ValueError("bpm 1-200, reps 1-32")
    beat, trims = 60.0 / bpm, trims or {}
    keyframes = [{arm: dict(start[arm]) for arm in ARMS}]
    seconds: list[float] = []
    labels = ["start"]
    for number, name in enumerate(moves):
        if number:  # straight up between moves: nothing sweeps from one move's last pose into the next's first
            keyframes.append({arm: _absolute(neutral[arm], {}, trims.get(arm, 0.0)) for arm in ARMS})
            seconds.append(BETWEEN_BEATS * beat)
            labels.append("upright")
        for rep in range(reps):
            for index, (pose, beats) in enumerate(MOVES[name]):
                keyframes.append({arm: _absolute(neutral[arm], pose[arm], trims.get(arm, 0.0)) for arm in ARMS})
                seconds.append((LEAD_BEATS if rep == 0 and index == 0 else beats) * beat)
                labels.append(name)
    keyframes.append({arm: dict(neutral[arm]) for arm in ARMS})
    seconds.append(LEAD_BEATS * beat)
    labels.append("neutral")
    return Routine(keyframes, seconds, labels)


def _absolute(neutral: dict[str, float], offsets: dict[str, float], trim: float) -> dict[str, float]:
    pose = {joint: neutral[joint] + offsets.get(joint, 0.0) for joint in ARM_JOINTS}
    pose["shoulder_pan"] += trim
    pose["gripper"] = offsets.get("gripper", neutral["gripper"])
    return pose


def fastest(routine: Routine) -> tuple[float, str]:
    """The fastest any joint turns, deg/s, and where. Eased, a step peaks at pi/2 its average speed."""

    top, where = 0.0, ""
    for index, span in enumerate(routine.seconds):
        a, b = routine.keyframes[index], routine.keyframes[index + 1]
        for arm in a:
            for joint in a[arm]:
                turn = abs(b[arm][joint] - a[arm][joint]) * (GRIPPER_DEG_PER_PERCENT if joint == "gripper" else 1.0)
                speed = math.pi / 2 * turn / span
                if speed > top:
                    top, where = speed, f"the {arm} {joint} in {routine.labels[index + 1]}"
    return top, where


LIMIT_SLACK_DEG = 2.0  # past Arm.limits_degrees, which keeps its own margin: the left elbow's neutral is there


def outside(routine: Routine, limits: dict[str, dict[str, tuple[float, float]]]) -> list[str]:
    """Targets past an arm's joint limits (degrees, as Arm.limits_degrees gives them), said in words.

    A target past a limit would be clamped, and the move no longer the one
    planned: the arms would stop pointing the same way, say.
    """

    found, seen = [], set()
    for keyframe, label in zip(routine.keyframes[1:], routine.labels[1:]):
        for arm, pose in keyframe.items():
            for joint, (low, high) in limits[arm].items():
                if not low - LIMIT_SLACK_DEG <= pose[joint] <= high + LIMIT_SLACK_DEG and (arm, joint) not in seen:
                    seen.add((arm, joint))
                    found.append(f"{arm} {joint} {pose[joint]:+.0f} in {label} (limits {low:+.0f}..{high:+.0f})")
    return found


# --- the model: where the arms are ----------------------------------------------------

LINKS = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper")
# Along each arm: its links to the fixed fingertip, and the moving jaw off the gripper.
SEGMENTS = [(i, i + 1) for i in range(6)] + [(5, 7), (7, 8)]


class Body:
    """An arm on the model, standing where it stands on the table: its joints and fingertips, cm."""

    def __init__(self, kinematics, placement) -> None:
        import mujoco

        self.kinematics, self.placement, self._mujoco = kinematics, placement, mujoco
        model = kinematics.model
        self._links = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in LINKS]
        self._jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        self._gripper = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]
        # The moving fingertip in its jaw's own frame: where the fixed one is with the jaw shut.
        upright = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -90.0, "wrist_flex": 0.0,
                   "wrist_roll": 0.0}
        tip = kinematics.fingertip(upright)
        self._jaw_tip = kinematics.data.xmat[self._jaw].reshape(3, 3).T @ (tip - kinematics.data.xpos[self._jaw])

    def to_table(self, arm_xy_cm) -> np.ndarray:
        """A point in the arm's frame, as a point on the table's (the marker sheet's), cm."""

        angle = np.radians(self.placement.yaw)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return rotation.T @ (np.asarray(arm_xy_cm, dtype=float) - [self.placement.x, self.placement.y])

    def points(self, pose: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """(the arm's points in the table's frame, cm; each one's height above the table, cm).

        Base, shoulder, upper arm, lower arm, wrist and gripper joints, the
        fixed fingertip, then the moving jaw's hinge and fingertip.
        """

        kinematics, data = self.kinematics, self.kinematics.data
        fixed_tip = kinematics.fingertip(pose)  # leaves the model at ``pose``, jaw shut
        data.qpos[self._gripper] = np.radians(max(pose.get("gripper", 0.0), 0.0))  # about a degree a percent
        self._mujoco.mj_kinematics(kinematics.model, data)
        jaw = data.xpos[self._jaw].copy()
        moving_tip = jaw + data.xmat[self._jaw].reshape(3, 3) @ self._jaw_tip
        arm = np.array([*(data.xpos[i] for i in self._links), fixed_tip, jaw, moving_tip]) * 100
        heights = arm[:, 2] - np.array([self.placement.table_height(x, y) for x, y in arm[:, :2]])
        table = np.column_stack([np.array([self.to_table(xy) for xy in arm[:, :2]]), arm[:, 2]])
        return table, heights

    def heading(self, pose: dict[str, float]) -> float:
        """Which way the arm reaches on the table at ``pose``, degrees in the table's frame."""

        tcp = self.kinematics.tcp(pose)[:2] * 100
        way = self.to_table(tcp) - self.to_table(self.kinematics.pan_axis * 100)
        return math.degrees(math.atan2(way[1], way[0]))


def _wrap(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def facing(bodies: dict[str, Body], neutral: dict[str, dict[str, float]]) -> tuple[dict[str, float], float]:
    """(per arm, the base turn that makes both face the same way; the way they then face), degrees.

    Each turns half the difference, found on the model: which way the arm
    reaches leaning forward, and which way a turn of its base takes that.
    """

    headings, turns = {}, {}
    for arm, body in bodies.items():
        leaning = dict(neutral[arm], shoulder_lift=45.0)
        headings[arm] = body.heading(leaning)
        turns[arm] = _wrap(body.heading(dict(leaning, shoulder_pan=leaning["shoulder_pan"] + 10.0)) - headings[arm]) / 10.0
    mean = headings["left"] + _wrap(headings["right"] - headings["left"]) / 2.0
    return {arm: _wrap(mean - headings[arm]) / turns[arm] for arm in bodies}, mean


def _segment_gap(p1, q1, p2, q2) -> float:
    """The closest two segments come (Ericson, Real-Time Collision Detection, 5.1.9)."""

    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = d1 @ r
        if e < 1e-12:
            s, t = float(np.clip(-c / a, 0.0, 1.0)), 0.0
        else:
            b = d1 @ d2
            denominator = a * e - b * b
            s = float(np.clip((b * f - c * e) / denominator, 0.0, 1.0)) if denominator > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = float(np.clip(-c / a, 0.0, 1.0)), 0.0
            elif t > 1.0:
                s, t = float(np.clip((b - c) / a, 0.0, 1.0)), 1.0
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


class Clearance(NamedTuple):
    """Over the whole dance, on the model, cm."""

    apart: float  # the closest the two arms' centre lines come
    above: float  # the lowest either comes above the table, past its shoulder
    behind: float  # the furthest any part goes behind where its base stands
    out: float  # the furthest either reaches out to its own side
    ahead: float  # the furthest either reaches ahead of where its base stands


def check(routine: Routine, bodies: dict[str, Body], heading: float, step_s: float = 0.05) -> Clearance:
    """How close the arms come to each other and the table, and how far they reach; ``heading`` is ahead."""

    forward = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading))])
    outwards = {"left": np.array([-forward[1], forward[0]]), "right": np.array([forward[1], -forward[0]])}
    bases = {arm: body.to_table(np.zeros(2)) for arm, body in bodies.items()}
    apart = above = math.inf
    behind = out = ahead = -math.inf
    for t in np.arange(0.0, routine.total + step_s, step_s):
        now = routine.at(t)
        shapes = {}
        for arm, body in bodies.items():
            shapes[arm], heights = body.points(now[arm])
            above = min(above, float(heights[2:].min()))  # base and shoulder stand on the table
            flat = shapes[arm][:, :2] - bases[arm]
            behind = max(behind, float(-(flat @ forward).min()))
            ahead = max(ahead, float((flat @ forward).max()))
            out = max(out, float((flat @ outwards[arm]).max()))
        left, right = shapes["left"], shapes["right"]
        apart = min(apart, min(_segment_gap(left[i], left[j], right[k], right[m])
                               for i, j in SEGMENTS for k, m in SEGMENTS))
    return Clearance(apart, above, behind, out, ahead)


def storyboard(bodies: dict[str, Body], neutral: dict[str, dict[str, float]], trims: dict[str, float],
               heading: float, moves, path: Path) -> Path:
    """Each move's poses as the model has them, from behind the arms and from their right side."""

    import cv2

    scale, width, height, floor = 3.2, 330, 250, 225  # px per cm, one panel, the table's line in it
    forward = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading))])
    right = np.array([forward[1], -forward[0]])
    middle = np.mean([body.to_table(body.kinematics.pan_axis * 100) for body in bodies.values()], axis=0)
    colours = {"left": (215, 130, 40), "right": (40, 140, 235)}
    rows = []
    for name in moves:
        panels = []
        for view, across in (("from behind", right), ("from the right", forward)):
            for number, (pose, _) in enumerate(MOVES[name], start=1):
                panel = np.full((height, width, 3), 250, np.uint8)
                cv2.line(panel, (0, floor), (width, floor), (170, 170, 170), 2)
                for arm, body in bodies.items():
                    points, _ = body.points(_absolute(neutral[arm], pose[arm], trims.get(arm, 0.0)))
                    screen = [(int(width / 2 + ((p[:2] - middle) @ across) * scale), int(floor - p[2] * scale))
                              for p in points]
                    for i, j in SEGMENTS:
                        cv2.line(panel, screen[i], screen[j], colours[arm], 6 if j <= 6 else 3, cv2.LINE_AA)
                    for point in screen[:6]:
                        cv2.circle(panel, point, 4, (60, 60, 60), -1, cv2.LINE_AA)
                cv2.putText(panel, f"{name} {number}/{len(MOVES[name])}, {view}", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
                panels.append(panel)
        rows.append(np.hstack(panels))
    sheet = np.vstack(rows)
    legend = np.full((30, sheet.shape[1], 3), 250, np.uint8)
    cv2.putText(legend, "blue: left arm   orange: right arm   (the model, at this rig's placements)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([sheet, legend]))
    return path


# --- the real arms -----------------------------------------------------------------------


def perform(arms: dict, routine: Routine, *, log=print, clock=time.monotonic, sleep=time.sleep) -> None:
    """Stream the routine to both arms at once, one tick for both; Ctrl+C or STOP holds them."""

    from .arm import RATE_HZ, Stopped
    from .servo import MOTORS

    for arm in arms.values():
        if not arm.torque_is_on():
            arm.torque_on()
    speeds = {name: arm.max_speed for name, arm in arms.items()}
    try:
        for arm in arms.values():  # the same servo speed limit for both: neither lags the other
            arm.max_speed = DANCE_SPEED
            arm.limit_speed()
        began, said = clock(), None
        while True:
            t = min(clock() - began, routine.total)
            if any(arm.stop is not None and arm.stop.is_set() for arm in arms.values()):
                raise Stopped
            now = routine.at(t)
            for name, arm in arms.items():
                arm.bus.write_goals({MOTORS[joint]: arm.to_ticks(joint, value) for joint, value in now[name].items()})
            if routine.label(t) != said:
                said = routine.label(t)
                log(f"{t:5.1f} s  {said}")
            if t >= routine.total:
                break
            sleep(1.0 / RATE_HZ)
    except KeyboardInterrupt:
        for arm in arms.values():
            arm.hold()
        raise
    finally:
        for name, arm in arms.items():
            arm.max_speed = speeds[name]
            try:
                arm.limit_speed()
            except Exception:  # a bus that went away must not hide what happened first
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trashdrop.groove", description=__doc__.split("\n\n")[0])
    parser.add_argument("--speed", type=float, default=1.0,
                        help="how fast: 1 the usual tempo, 0.5 half as fast, 1.5 half again as fast")
    parser.add_argument("--bpm", type=float, default=BPM, help=f"the usual tempo, beats a minute (default {BPM:g})")
    parser.add_argument("--reps", type=int, default=REPS, help=f"times through each move (default {REPS})")
    parser.add_argument("--moves", default=",".join(ORDER), help=f"which, in order (default {','.join(ORDER)})")
    parser.add_argument("--preview", action="store_true", help="no arms: check the dance and draw out/groove.png")
    args = parser.parse_args(argv)

    from .arm import connect, load_poses, move_together
    from .kinematics import Kinematics
    from .placement import Placement
    from .rig import load_rig
    from .station import repository_root

    moves = [name.strip() for name in args.moves.split(",") if name.strip()]
    if args.speed <= 0:
        print("--speed must be above 0")
        return 1
    bpm = args.bpm * args.speed
    rig, poses = load_rig(), load_poses()
    neutral = {arm: dict(poses[arm]["neutral"]) for arm in ARMS}
    bodies, trims, heading = {}, {}, 0.0
    if all(rig.arms[arm].sheet for arm in ARMS):
        bodies = {arm: Body(Kinematics(rig.arms[arm].wrist_roll_offset), Placement(*rig.arms[arm].sheet))
                  for arm in ARMS}
        trims, heading = facing(bodies, neutral)
    try:
        routine = build(neutral, neutral, moves=moves, reps=args.reps, bpm=bpm, trims=trims)
    except ValueError as error:
        print(error)
        return 1
    speed, where = fastest(routine)
    print(f"{' -> '.join(moves)}, {args.reps} times each at speed {args.speed:g} ({bpm:g} bpm): "
          f"{routine.total:.0f} s. Fastest: {where}, {speed:.0f} deg/s at its peak.")
    if speed > MAX_JOINT_SPEED:
        print(f"too fast for the servos (at most {MAX_JOINT_SPEED:g} deg/s): --speed "
              f"{math.floor(args.speed * MAX_JOINT_SPEED / speed * 100) / 100:g} at most")
        return 1
    if bodies:
        print("the bases turn " + ", ".join(f"{arm} {trim:+.1f} deg" for arm, trim in trims.items())
              + " so that both face the same way")
        room = check(routine, bodies, heading)
        print(f"on the model the arms come {room.apart:.0f} cm apart at the closest, {room.above:.0f} cm above "
              f"the table at the lowest; they reach {room.out:.0f} cm out to the sides, {room.ahead:.0f} cm ahead "
              f"and {max(0.0, room.behind):.0f} cm behind their bases")
        if room.apart < MIN_APART_CM or room.above < MIN_ABOVE_TABLE_CM or room.behind > MAX_BEHIND_CM:
            print(f"too close: at least {MIN_APART_CM:g} cm apart and {MIN_ABOVE_TABLE_CM:g} cm above the table, "
                  f"at most {MAX_BEHIND_CM:g} cm behind the bases")
            return 1
        print(f"poses drawn: {storyboard(bodies, neutral, trims, heading, moves, repository_root() / 'out' / 'groove.png')}")
    else:
        print("no arm placements in rig.toml: the bases are not turned to match, and nothing is checked on the model")
    if args.preview:
        return 0

    arms = {}
    try:
        arms = {name: connect(name, rig) for name in ARMS}
        problems = outside(routine, {name: arm.limits_degrees() for name, arm in arms.items()})
        if problems:
            print("past the joints' limits: " + "; ".join(problems))
            return 1
        input(f"both arms go to neutral, then dance {routine.total:.0f} s. The lowered arm reaches over the "
              "pick zone: clear it. Ctrl+C holds both where they are. Enter starts...")
        for arm in arms.values():
            if not arm.torque_is_on():
                arm.torque_on()
        move_together([(arm, neutral[name]) for name, arm in arms.items()])
        start = {name: arm.pose() for name, arm in arms.items()}
        perform(arms, build(start, neutral, moves=moves, reps=args.reps, bpm=bpm, trims=trims))
    except (ValueError, RuntimeError) as error:
        print(error)
        return 1
    except KeyboardInterrupt:
        print("\nstopped; both arms hold where they are")
        return 130
    finally:
        for arm in arms.values():
            arm.bus.close()
    print("done: both arms back in neutral")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
