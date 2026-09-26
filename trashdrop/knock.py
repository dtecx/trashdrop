"""Standing items: pushed over from the side, then picked up lying down.

Nothing an SO-101 does from above takes a standing bottle. Its cap is about
20 cm up, and fingers pointing down reach 9 cm high at the zone's distances,
17 cm leaning 30 degrees (kinematics); the jaw would have to come down past
the cap as well. Seen from above it is a round outline 6-8 cm across, wider
than the jaw closes around. So an item that looks like that is pushed over:
the closed gripper comes down beside it at two thirds of a bottle's height
and moves through where it stands, towards the middle of the zone, so that
it falls inside. Then it is found again and picked up like everything else.

Either arm may push, towards or away from its base: whichever lets the item
fall nearest the middle of the zone. The pick that follows is still made by
the arm on its material's side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import LEANS_DEG
from .placement import Placement
from .sorter import base_on_sheet, sheet_points

UPRIGHT_ASPECT = 1.4  # longest side of its outline over the shortest, at most
UPRIGHT_SIZE_CM = (4.5, 12.0)  # shortest side at least (no pinch fits), longest at most
PUSH_HEIGHT_CM = 8.0  # the TCP above the table while pushing: above a can's middle
CLEAR_CM = 4.0  # the gripper comes down this far from the item's edge
THROUGH_CM = 4.0  # and pushes on to this far past where its middle was
ABOVE_CM = 4.0  # it comes down to the push height from this much higher (fingers down reach ~12-17 cm)
PUSH_SPEED = 30.0  # deg/s: firm, not a swipe


@dataclass(frozen=True)
class KnockPlan:
    arm: str
    above: dict[str, float]  # over the start, clear of the item
    start: dict[str, float]  # beside the item, at the push height
    middle: dict[str, float]  # where it stood: the push goes through here, so it stays level
    end: dict[str, float]  # past where it stood
    after: dict[str, float]  # lifted clear again
    direction: tuple[float, float]  # which way it is pushed, a unit vector on the table
    pulls: bool  # the gripper moves towards the arm's base, not away from it


def outline_cm(item, scale: float, homography) -> np.ndarray:
    """The item's outline on the table, cm."""

    import cv2

    contours, _ = cv2.findContours(item.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = np.vstack([contour.reshape(-1, 2) for contour in contours]).astype(float)
    return sheet_points(homography, (points[:, 0] + 0.5) * scale, (points[:, 1] + 0.5) * scale)


def looks_upright(item, scale: float, homography) -> tuple[bool, float, float]:
    """(round and too wide to pinch, shortest side cm, longest side cm) of the item's outline."""

    import cv2

    (_, _), (a, b), _ = cv2.minAreaRect(outline_cm(item, scale, homography).astype(np.float32))
    short, long = sorted((float(a), float(b)))
    upright = long <= UPRIGHT_ASPECT * short and UPRIGHT_SIZE_CM[0] <= short and long <= UPRIGHT_SIZE_CM[1]
    return upright, short, long


def plan_knock(item, scale: float, homography, placements: dict[str, Placement], kinematics,
               limits: dict[str, dict[str, tuple[float, float]]], zone_centre_cm) -> tuple[KnockPlan | None, str]:
    """How to push the item over so it falls towards the middle of the zone; or (None, why not)."""

    outline = outline_cm(item, scale, homography)
    centre = outline.mean(axis=0)
    radius = float(np.max(np.linalg.norm(outline - centre, axis=1)))
    inward = np.asarray(zone_centre_cm, float) - centre
    options = []
    for name, placement in placements.items():
        arm_kinematics = kinematics[name] if isinstance(kinematics, dict) else kinematics
        base = base_on_sheet(placement)
        away = (centre - base) / max(np.linalg.norm(centre - base), 1e-9)
        for pulls, direction in ((False, away), (True, -away)):
            # Right in the middle any way in is as good; otherwise, towards it.
            score = 1.0 if np.linalg.norm(inward) < 2.0 else float(direction @ inward / np.linalg.norm(inward))
            options.append((score - 0.1 * pulls, name, placement, arm_kinematics, pulls, direction))
    reasons = []
    for _, name, placement, arm_kinematics, pulls, direction in sorted(options, key=lambda option: -option[0]):
        start_xy = centre - direction * (radius + CLEAR_CM)
        end_xy = centre + direction * THROUGH_CM
        targets = []
        for xy, lift in ((start_xy, ABOVE_CM), (start_xy, 0.0), (centre, 0.0), (end_xy, 0.0), (end_xy, ABOVE_CM)):
            x, y = placement.to_arm(xy)
            targets.append(np.array([x, y, placement.table_height(x, y) + PUSH_HEIGHT_CM + lift]) / 100)
        poses = _fingers_down(arm_kinematics, targets, limits.get(name))
        if poses is not None:
            return KnockPlan(name, *poses, (float(direction[0]), float(direction[1])), pulls), ""
        reasons.append(f"{name} {'pulling' if pulls else 'pushing'}: out of reach")
    return None, "; ".join(reasons) or "no calibrated arm"


def execute_knock(arm, plan: KnockPlan, neutral: dict[str, float], *, log=print) -> None:
    """Jaw shut, down beside the item, through it, up and back to neutral."""

    log(f"{arm.name}: closing the jaw to push with")
    arm.move({"gripper": 0.0})
    arm.move(plan.above)
    arm.move(plan.start)
    log(f"{arm.name}: {'pulling' if plan.pulls else 'pushing'} it over")
    arm.move(plan.middle, speed=PUSH_SPEED)
    arm.move(plan.end, speed=PUSH_SPEED)
    arm.move(plan.after)
    arm.move(neutral)


def _fingers_down(kinematics, targets, limits) -> list[dict[str, float]] | None:
    """Joint angles for every target, fingers down with one lean for all, so the push stays level."""

    for lean in LEANS_DEG:
        poses = []
        for target in targets:
            solution = kinematics.solve(target, start=poses[-1] if poses else None, limits=limits, lean_deg=lean)
            if not solution.reachable:
                break
            poses.append(solution.degrees)
        if len(poses) == len(targets):
            return poses
    return None
