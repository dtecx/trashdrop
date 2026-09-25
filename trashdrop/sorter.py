"""One pick, end to end: the overhead camera finds an item, an arm takes it.

    camera frame -> item mask, against a picture of the empty table
    -> grasp plan in pixels (perception/grasp.py)
    -> through the camera's homography to cm on the calibration sheet
    -> through each arm's placement into that arm's frame
    -> inverse kinematics, fingers down, the jaw turned to the plan
    -> above, down, close, check it holds, lift, turn to the arm's side,
       open, back to neutral.

The sheet need not stay on the table: the homography and the placements
describe the table plane, and hold for as long as the camera and the arm
bases stay where they were when they were calibrated.

Items are looked for only where some arm can pick with its fingers down --
a ring around each base -- so a hand or a laptop at the edge of the picture
is never mistaken for rubbish. Both arms should be in their neutral pose
(straight up) when the camera looks: they then stand over their own bases,
outside every ring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import LEANS_DEG
from .perception.grasp import GraspPlan, plan_grasp
from .placement import Placement

ANALYSIS_WIDTH = 320
# Where the TCP goes above the table to grasp, and how far back along the
# fingers the approach starts -- as far as the reach allows at that spot.
GRASP_HEIGHT_CM = 2.5
BACK_OFF_CM = (7.0, 6.0, 5.0, 4.0)
DESCENT_SPEED = 15.0  # deg/s for the last few centimetres
# The ring around a base where items are looked for, cm: fingers straight down
# reach from about 10 cm, fingers leaning 45 degrees to about 42 cm.
RING_CM = (9.0, 42.0)
# Tip opening of the SO-101 jaw against the gripper joint angle (CAD, see
# station.py), and the joint's travel over the LeRobot 0..100 range.
_OPENING_M = (0.032, 0.046, 0.060, 0.074, 0.087)
_JAW_RAD = (0.2, 0.4, 0.6, 0.8, 1.0)
_JAW_RANGE_RAD = (-0.17, 1.75)


def gripper_percent_for(opening_m: float) -> float:
    """LeRobot gripper percent that opens the fingertips about ``opening_m``."""

    angle = float(np.interp(opening_m, _OPENING_M, _JAW_RAD, right=_JAW_RANGE_RAD[1]))
    low, high = _JAW_RANGE_RAD
    return float(np.clip(100.0 * (angle - low) / (high - low), 0.0, 100.0))


@dataclass(frozen=True)
class PickPlan:
    arm: str
    above: dict[str, float]
    grasp: dict[str, float]
    open_percent: float
    grasp_plan: GraspPlan
    target_cm: tuple[float, float]  # TCP in the arm's frame
    pixel: tuple[float, float]  # where the fixed finger goes, full-resolution pixels
    lean_deg: float = 0.0  # how far the fingers lean away from the base


def sheet_points(homography, us, vs) -> np.ndarray:
    """Pixels -> sheet cm, vectorised."""

    matrix = homography.matrix
    ones = np.ones_like(us, dtype=float)
    projected = matrix @ np.vstack([us, vs, ones])
    return (projected[:2] / projected[2]).T * 100.0


def reach_mask(shape: tuple[int, int], scale: float, homography, placements: dict[str, Placement]) -> np.ndarray:
    """Analysis-resolution mask of table pixels inside some arm's ring."""

    height, width = shape
    vs, us = np.mgrid[0:height, 0:width]
    points = sheet_points(homography, (us.ravel() + 0.5) * scale, (vs.ravel() + 0.5) * scale)
    inside = np.zeros(len(points), bool)
    for placement in placements.values():
        base = base_on_sheet(placement)
        distance = np.linalg.norm(points - base, axis=1)
        inside |= (distance >= RING_CM[0]) & (distance <= RING_CM[1])
    return (inside.reshape(height, width) * 255).astype(np.uint8)


def base_on_sheet(placement: Placement) -> np.ndarray:
    angle = np.radians(placement.yaw)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotation.T @ (-np.array([placement.x, placement.y]))


def find_item(frame, background, valid_small):
    """(item mask at analysis resolution, scale to full resolution) or (None, reason)."""

    import cv2

    from .dataset.autolabel import EDGE_MARGIN_PX, _foreground
    from .perception.regions import find_item_region

    height, width = frame.shape[:2]
    scale = width / ANALYSIS_WIDTH
    size = (ANALYSIS_WIDTH, round(height / scale))
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    reference = cv2.resize(background, size, interpolation=cv2.INTER_AREA)
    mask, fit = _foreground(small, reference, 28, valid_small)
    if not fit.trusted:
        return None, "the picture changed too much to compare with the empty table (light? camera moved?)"
    region, reason = find_item_region(mask, edge_margin_px=EDGE_MARGIN_PX, valid_mask=valid_small)
    if region is None:
        return None, {"nothing_changed": "no item found where an arm can reach",
                      "touches_frame_edge": "the item sticks out of the reachable area"}.get(reason, reason)
    hull = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(hull, [region.hull.reshape(-1, 2)], 255)
    item = ((mask > 0) & (hull > 0)).astype(np.uint8) * 255
    item = cv2.morphologyEx(item, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(item, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(item)
    cv2.drawContours(filled, contours, -1, 255, -1)
    # A clear item shows up as scattered fragments; its hull is then the honest outline.
    if (filled > 0).sum() < 0.6 * max((hull > 0).sum(), 1):
        filled = hull
    return filled > 0, scale


def plan_pick(item_small, scale: float, homography, placements: dict[str, Placement], kinematics,
              limits: dict[str, dict[str, tuple[float, float]]], prefer: str | None = None):
    """A PickPlan for the item, or (None, reason)."""

    centre = np.argwhere(item_small).mean(axis=0)[::-1] * scale  # (u, v) full resolution
    u0, v0 = centre
    here, right = sheet_points(homography, np.array([u0, u0 + 10.0]), np.array([v0, v0]))
    m_per_px_small = np.linalg.norm(right - here) / 100.0 / 10.0 * scale

    reasons = []
    order = sorted(placements, key=lambda name: name != prefer)
    for name in order:
        placement = placements[name]
        base = base_on_sheet(placement)
        # The fixed finger on the side of the item facing this arm's base.
        side_px = sheet_to_pixel_direction(homography, centre, base - sheet_points(homography, np.array([u0]), np.array([v0]))[0])
        grasp = plan_grasp(item_small, m_per_px_small, fixed_side=tuple(side_px))
        if grasp.mode != "pinch":
            reasons.append(f"{name}: {grasp.reason}")
            continue
        fixed_px = np.array(grasp.fixed_finger(m_per_px_small)) * scale
        across_px = np.array(grasp.across)
        ends = sheet_points(homography, np.array([fixed_px[0], fixed_px[0] + across_px[0] * 20]),
                            np.array([fixed_px[1], fixed_px[1] + across_px[1] * 20]))
        target_sheet, direction_sheet = ends[0], ends[1] - ends[0]
        yaw = placement.direction_to_arm(float(np.degrees(np.arctan2(direction_sheet[1], direction_sheet[0]))))
        x, y = placement.to_arm(target_sheet)
        target = np.array([x, y, placement.table_z + GRASP_HEIGHT_CM]) / 100
        found = None
        for lean in LEANS_DEG:
            down = kinematics.solve(target, yaw_deg=yaw, limits=limits.get(name), lean_deg=lean)
            if not down.reachable:
                continue
            # Start the approach back along the fingers, so the last move is along them.
            approach = kinematics.approach_for(target, lean)
            for back in BACK_OFF_CM:
                above = kinematics.solve(target - approach * back / 100, yaw_deg=yaw, start=down.degrees,
                                         limits=limits.get(name), lean_deg=lean)
                if above.reachable:
                    found = (above, down, lean)
                    break
            if found:
                break
        if found is None:
            reasons.append(f"{name}: out of reach, {np.hypot(x, y):.0f} cm from its base (it reaches 10-42 cm)")
            continue
        above, down, lean = found
        return PickPlan(name, above.degrees, down.degrees, gripper_percent_for(grasp.opening_m), grasp,
                        (float(x), float(y)), (float(fixed_px[0]), float(fixed_px[1])), lean), None
    return None, "; ".join(reasons) or "no calibrated arm"


def sheet_to_pixel_direction(homography, pixel, sheet_direction) -> np.ndarray:
    """A direction on the sheet, as a direction in the picture at ``pixel``."""

    here = sheet_points(homography, np.array([pixel[0]]), np.array([pixel[1]]))[0]
    target = here + 5.0 * np.asarray(sheet_direction) / max(np.linalg.norm(sheet_direction), 1e-9)
    u, v = homography.world_to_pixel(target[0] / 100.0, target[1] / 100.0)
    return np.array([u - pixel[0], v - pixel[1]])


def drop_pose(arm: str) -> dict[str, float]:
    """Turned to the arm's own side -- left for the left arm, right for the right -- and lifted."""

    pan = 80.0 if arm == "left" else -80.0
    return {"shoulder_pan": pan, "shoulder_lift": 0.0, "elbow_flex": -60.0, "wrist_flex": 60.0, "wrist_roll": 0.0}


MISS_BELOW = 4.0  # percent: jaws told to close that stop below this hold nothing
RELEASE_OPEN = 60.0


def execute_pick(arm, plan: PickPlan, neutral: dict[str, float], *, dry_run: bool = False,
                 sleep=None, log=print) -> bool:
    """Carry out a PickPlan with a real (or fake) Arm; True if something was dropped."""

    import time

    sleep = sleep or time.sleep
    gripper = "gripper"
    log(f"{arm.name}: opening the jaws to {plan.open_percent:.0f} %")
    arm.move({gripper: plan.open_percent})
    log(f"{arm.name}: above the item")
    arm.move(plan.above)
    if dry_run:
        log(f"{arm.name}: dry run -- hovering over the item for 3 s, then back")
        sleep(3.0)
        arm.move(neutral)
        return False
    log(f"{arm.name}: down")
    arm.move(plan.grasp, speed=DESCENT_SPEED)
    held = arm.move({gripper: 0.0})[gripper]
    if held < MISS_BELOW:
        log(f"{arm.name}: the jaws closed to {held:.0f} %: nothing between them, a miss")
        arm.move(plan.above)
        arm.move(neutral)
        return False
    log(f"{arm.name}: holding it (jaws at {held:.0f} %) -- lifting and turning to its side")
    carry = {joint: value for joint, value in plan.above.items() if joint != gripper}
    arm.move(carry)
    arm.move(drop_pose(arm.name))
    arm.move({gripper: RELEASE_OPEN})
    log(f"{arm.name}: dropped; back to neutral")
    arm.move(neutral)
    return True
