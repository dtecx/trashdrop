"""Where each arm's wrist roll really has its zero.

LeRobot calibrates the wrist roll as a full turn around wherever a hand held
it during the "middle of the range" step, so its zero is that arm's own and
may be a quarter turn from the model's. Then the jaw closes along an item
instead of across it: every position is right and every grasp misses.

`trashdrop rig roll` finds the difference. The arm holds its fingers down
over the middle of the pick zone with the moving jaw sent towards the far
tape edge; a person says where it really opened. That fixes the offset to a
quarter turn; a few nudges by eye, until the fingers line up square to the
tape, fix the rest. rig.toml keeps it per arm:

    model roll = LeRobot roll + wrist_roll_offset

The fixed fingertip is 8 mm off the roll axis, so an offset that changes
moves where every touched corner was, by up to 1.6 cm.
"""

from __future__ import annotations

import numpy as np

from .kinematics import LEANS_DEG
from .placement import Placement

# Table directions, as the overhead camera shows them: far is the top of its picture.
SIDES = {"far": (0.0, 1.0), "near": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}
KEYS = {"f": "far", "n": "near", "l": "left", "r": "right"}
HOVER_CM = 10.0  # TCP above the table while it is looked at
NUDGE_DEG = 3.0


def _opening(kinematics, pose: dict[str, float], placement: Placement) -> np.ndarray:
    """Which way the moving jaw opens in ``pose``, in the table's frame."""

    _, across = kinematics.pointing(pose)
    angle = np.radians(placement.yaw)
    # The arm's frame back into the table's: the placement's rotation, undone.
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) @ across[:2]


def jaw_side(kinematics, pose: dict[str, float], placement: Placement) -> str:
    """Which tape edge the moving jaw opens towards, in ``pose``."""

    opening = _opening(kinematics, pose, placement)
    return max(SIDES, key=lambda side: float(np.dot(SIDES[side], opening)))


def degrees_from_far(kinematics, pose: dict[str, float], placement: Placement) -> float:
    """How far the moving jaw's opening direction is turned from the far edge, seen from above."""

    opening = _opening(kinematics, pose, placement)
    return float(np.degrees(np.arctan2(opening[0], opening[1])))


def turned_by(kinematics, pose: dict[str, float], placement: Placement, seen: str) -> float:
    """Degrees the real wrist is turned from where ``kinematics`` thinks, if its jaw opened towards ``seen``."""

    for delta in (0.0, 90.0, 180.0, -90.0):
        if jaw_side(kinematics, dict(pose, wrist_roll=pose["wrist_roll"] + delta), placement) == seen:
            return delta
    raise ValueError(f"no quarter turn opens the jaw towards {seen!r}")


def hover_pose(kinematics, placement: Placement, limits=None) -> dict[str, float] | None:
    """Fingers down over the pick zone's middle, the moving jaw towards the far edge; None if out of reach."""

    x, y = placement.to_arm((0.0, 0.0))
    target = np.array([x, y, placement.table_height(x, y) + HOVER_CM]) / 100
    yaw = placement.direction_to_arm(90.0)  # the table's +y: far
    for lean in LEANS_DEG:
        solution = kinematics.solve(target, yaw_deg=yaw, limits=limits, lean_deg=lean)
        if solution.reachable:
            return solution.degrees
    return None


def settled_offset(kinematics, pose: dict[str, float], placement: Placement, nudged: float) -> float:
    """The arm's offset, once a person nudged its roll ``nudged`` degrees from ``pose`` until the jaw looked square.

    The hover itself is only solved to a degree or two; what ``kinematics``
    says was left over is taken out too.
    """

    left_over = degrees_from_far(kinematics, pose, placement)
    per_degree = (degrees_from_far(kinematics, dict(pose, wrist_roll=pose["wrist_roll"] + 1.0), placement)
                  - degrees_from_far(kinematics, dict(pose, wrist_roll=pose["wrist_roll"] - 1.0), placement)) / 2.0
    return wrap(kinematics.wrist_roll_offset - nudged - left_over / per_degree)


def wrap(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0
