"""Crab rave: both arms rock what they hold between them to a beat, as one.

The organisers' mini task. The arms stand straight up -- crab_start in
poses.toml: neutral, wrists turned, grippers clamped -- holding a giant can
between their grippers. Holding one thing, they have to move as one: one
clock, the same offsets for both at every tick, the goals for both written in
the same 50 Hz loop, and both servos' own speed limits alike, so neither lags
behind the other.

Straight up, an SO-101 cannot lower its gripper straight down: that bends
the elbow further back, and the left one is at its end stop there. So the
can goes down and up the way the arms can take it: rocking forwards and back
from the shoulders -- the wrists turning against them, so the can stays
level -- and swaying left and right from the bases. Everything is an offset
from where the arms stand when the dance starts, eased in and out, so it
starts and ends where it began; Ctrl+C holds both where they are.

The two arms do not face quite the same way (11 degrees apart at the venue),
so the same lean would carry their grippers apart -- 3.3 cm at 30 degrees,
tearing at the can. Given the kinematics and the arms' placements, each arm
also turns its base a little with the lean (parallel_pans), worked out on
the model so both grippers move square to the line between them and stay
the same distance apart.
"""

from __future__ import annotations

import math
import time

import numpy as np

BPM = 60.0  # slow, to begin with; Crab Rave itself is 125
BEATS_PER_ROCK = 2  # forwards and back again
BEATS_PER_SWAY = 4  # left and right again
ROCK_DEG = 5.0
SWAY_DEG = 3.0
MAX_ROCK_DEG = 35.0
MAX_SWAY_DEG = 30.0
STYLES = ("rock", "bang")  # rock: forwards and back; bang: a nod forwards on every beat
EASE_S = 2.0
# The servos' own speed limit while dancing, the same for both arms: with the
# right arm's usual 45 deg/s it would lag the left one and twist the can.
DANCE_SPEED = 150.0


def offsets(t: float, total: float, *, bpm: float = BPM, rock: float = ROCK_DEG,
            sway: float = SWAY_DEG, style: str = "rock") -> dict[str, float]:
    """Joint offsets in degrees at ``t`` seconds into a ``total``-second dance, the same for both arms."""

    ease = _ease(t, total)
    beat = 60.0 / bpm
    if style == "bang":
        lean = ease * rock * math.sin(math.pi * t / beat) ** 2  # forwards and back up, every beat
    else:
        lean = ease * rock * math.sin(2 * math.pi * t / (beat * BEATS_PER_ROCK))
    return {
        "shoulder_lift": lean,
        "wrist_flex": -lean,  # the gripper keeps pointing the way it did: the can stays level
        "shoulder_pan": ease * sway * math.sin(2 * math.pi * t / (beat * BEATS_PER_SWAY)),
    }


def parallel_pans(kinematics: dict, placements: dict, start: dict[str, dict[str, float]], max_lean: float,
                  samples: int = 41) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per arm, (leans, pan turns): the base turn that keeps each gripper moving square to the line
    between the two grippers as it leans, so they stay the same distance apart. Degrees throughout."""

    names = list(start)
    rest = {name: _tcp_on_table(kinematics[name], placements[name], start[name]) for name in names}
    gap = rest[names[-1]] - rest[names[0]]
    along = gap / max(float(np.linalg.norm(gap)), 1e-9)
    leans = np.linspace(-max_lean, max_lean, samples)
    tables = {}
    for name in names:

        def sideways(turn: float, lean: float) -> float:
            pose = dict(start[name])
            pose["shoulder_lift"] += lean
            pose["wrist_flex"] -= lean
            pose["shoulder_pan"] += turn
            return float((_tcp_on_table(kinematics[name], placements[name], pose) - rest[name]) @ along)

        turns = []
        for lean in leans:
            a, b = 0.0, 1.0
            fa, fb = sideways(a, lean), sideways(b, lean)
            for _ in range(8):  # secant: the sideways drift is smooth in the turn
                if abs(fb - fa) < 1e-12:
                    break
                a, b = b, b - fb * (b - a) / (fb - fa)
                fa, fb = fb, sideways(b, lean)
            turns.append(b)
        tables[name] = (leans, np.array(turns))
    return tables


def _tcp_on_table(kinematics, placement, pose: dict[str, float]) -> np.ndarray:
    """The TCP in the table's frame, cm."""

    tcp = kinematics.tcp(pose)[:2] * 100
    angle = np.radians(placement.yaw)
    back = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    return back @ (tcp - [placement.x, placement.y])


def _ease(t: float, total: float) -> float:
    """0 at both ends of the dance, 1 in between, smooth: no jump starting or stopping."""

    edge = min(t, total - t, EASE_S) / EASE_S
    edge = max(0.0, min(1.0, edge))
    return edge * edge * (3 - 2 * edge)


def dance(arms: dict, *, seconds: float, bpm: float = BPM, rock: float = ROCK_DEG, sway: float = SWAY_DEG,
          style: str = "rock", kinematics: dict | None = None, placements: dict | None = None,
          log=print, clock=time.monotonic, sleep=time.sleep) -> None:
    """Dance ``seconds`` with every arm in ``arms`` at once, from where they stand now.

    With ``kinematics`` and ``placements`` for two arms, their bases turn with
    the lean to keep the grippers the same distance apart (parallel_pans).
    """

    from .arm import DEG_PER_TICK, LIMIT_MARGIN, RATE_HZ, Stopped
    from .servo import MOTORS

    if not (0 < bpm <= 200 and 0 <= rock <= MAX_ROCK_DEG and 0 <= sway <= MAX_SWAY_DEG and seconds > 0
            and style in STYLES):
        raise ValueError(f"bpm 1-200, rock 0-{MAX_ROCK_DEG:g} deg, sway 0-{MAX_SWAY_DEG:g} deg, seconds > 0, "
                         f"style {' or '.join(STYLES)}")
    for arm in arms.values():
        if not arm.torque_is_on():
            arm.torque_on()
    start = {name: {joint: arm.bus.read(motor, "goal_position") for joint, motor in MOTORS.items()}
             for name, arm in arms.items()}
    speeds = {name: arm.max_speed for name, arm in arms.items()}
    pans = {}
    if kinematics and placements and len(arms) == 2 and rock > 0:
        start_degrees = {name: {joint: arms[name].from_ticks(joint, ticks) for joint, ticks in start[name].items()}
                         for name in arms}
        pans = parallel_pans(kinematics, placements, start_degrees, rock)
        widest = max(float(np.max(np.abs(turns))) for _, turns in pans.values())
        log(f"keeping the grippers the same distance apart: the bases turn up to {widest:.1f} deg with the lean")

    def goals_at(t: float, name: str) -> dict[int, int]:
        moved = offsets(t, seconds, bpm=bpm, rock=rock, sway=sway, style=style)
        if name in pans:
            leans, turns = pans[name]
            moved["shoulder_pan"] += float(np.interp(moved["shoulder_lift"], leans, turns))
        limits = arms[name].limits
        return {
            motor: min(max(start[name][joint] + round(moved.get(joint, 0.0) / DEG_PER_TICK),
                           limits[joint].low + LIMIT_MARGIN), limits[joint].high - LIMIT_MARGIN)
            for joint, motor in MOTORS.items()
        }

    try:
        for arm in arms.values():
            arm.max_speed = DANCE_SPEED
            arm.limit_speed()
        log(f"dancing {seconds:g} s at {bpm:g} bpm, {style}: leaning {rock:g} deg, swaying {sway:g} deg")
        began = clock()
        while True:
            t = min(clock() - began, seconds)
            if any(arm.stop is not None and arm.stop.is_set() for arm in arms.values()):
                raise Stopped
            for name, arm in arms.items():  # one tick, every arm: they stay together
                arm.bus.write_goals(goals_at(t, name))
            if t >= seconds:
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
    log("done: back where they started")
