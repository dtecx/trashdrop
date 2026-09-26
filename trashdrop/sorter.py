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

Items are looked for only inside the pick zone (rig.toml), and there only
where some arm can reach, so laptops, hands and pens elsewhere on the table
are never mistaken for rubbish. Only the zone has to be empty when the empty
table is photographed. Both arms should be in their neutral pose (straight
up) when the camera looks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import FINGERTIP_BEYOND_TCP, LEANS_DEG
from .perception.grasp import GraspPlan, plan_grasp
from .placement import Placement

ANALYSIS_WIDTH = 320
# How far above the table the fixed fingertip comes down to grasp, cm (`pick
# --fingertips` changes it). At 1.3 cm the pads held only the top edge of a
# 2.2 cm pack at the venue; 0.5 cm covers its side, and the pads still reach
# up past the middle of a lying bottle. The TCP goes this much higher, and the
# approach starts back along the fingers -- as far as the reach allows there.
FINGERTIPS_CM = 0.5
LOWEST_FINGERTIPS_CM = 0.2  # any lower and a millimetre of calibration puts them in the table
GRASP_HEIGHT_CM = FINGERTIPS_CM + FINGERTIP_BEYOND_TCP * 100  # the TCP, above the table
BACK_OFF_CM = (7.0, 6.0, 5.0, 4.0)
DESCENT_SPEED = 15.0  # deg/s for the last few centimetres
# The ring around a base where items are looked for, cm: fingers straight down
# reach from about 10 cm, fingers leaning 45 degrees to about 42 cm.
RING_CM = (9.0, 42.0)
# Tip opening of the SO-101 jaw against the gripper joint angle (CAD, see
# station.py), and the joint's travel over the LeRobot 0..100 range. Past the
# last entry the jaw mostly swings up, not out: it is never opened further,
# where closing would bring it down from high above onto the item.
_OPENING_M = (0.032, 0.046, 0.060, 0.074, 0.087)
_JAW_RAD = (0.2, 0.4, 0.6, 0.8, 1.0)
_JAW_RANGE_RAD = (-0.17, 1.75)


def gripper_percent_for(opening_m: float) -> float:
    """LeRobot gripper percent that opens the fingertips about ``opening_m``, at most the table's widest."""

    angle = float(np.interp(opening_m, _OPENING_M, _JAW_RAD))
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
    moving_pixel: tuple[float, float] | None = None  # where the moving finger comes down
    table_cm: float | None = None  # the table's height under the target, in the arm's frame
    tcp_above_table_cm: float = GRASP_HEIGHT_CM
    distances_cm: dict[str, float] = field(default_factory=dict)  # item to each arm's base

    @property
    def fingertip_above_table_cm(self) -> float:
        """Where the plan puts the fixed fingertip, above the table."""

        return self.tcp_above_table_cm - FINGERTIP_BEYOND_TCP * 100 * float(np.cos(np.radians(self.lean_deg)))


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


@dataclass
class Detection:
    """What the camera found: the item's mask, or why there is none."""

    item: np.ndarray | None  # analysis-resolution mask of the item
    scale: float  # full-resolution pixels per analysis pixel
    reason: str
    changed: np.ndarray  # everything that differs from the empty table, for a look


# Items are looked for this far inside the zone's edge, clear of tape marking it.
ZONE_INSET_CM = 1.0


def zone_mask(shape: tuple[int, int], scale: float, homography, zone) -> np.ndarray:
    """Analysis-resolution mask of the pick zone, ZONE_INSET_CM inside its edge."""

    import cv2

    corners_cm = np.array(zone.corners(), float)
    polygon = np.array([homography.world_to_pixel(x / 100, y / 100) for x, y in corners_cm]) / scale
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    # Pixels per cm along the zone's edge, to turn the inset into an erosion.
    edge_px = np.linalg.norm(np.diff(np.vstack([polygon, polygon[:1]]), axis=0), axis=1).sum()
    edge_cm = np.linalg.norm(np.diff(np.vstack([corners_cm, corners_cm[:1]]), axis=0), axis=1).sum()
    inset_px = int(round(ZONE_INSET_CM * edge_px / edge_cm))
    if inset_px > 0:
        mask = cv2.erode(mask, np.ones((2 * inset_px + 1, 2 * inset_px + 1), np.uint8))
    return mask


def find_item(frame, background, valid_small) -> Detection:
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
        return Detection(None, scale, "the picture changed too much to compare with the empty table "
                         "(light changed? camera moved?) -- press b to photograph the empty table again", mask)
    region, reason = find_item_region(mask, edge_margin_px=EDGE_MARGIN_PX, valid_mask=valid_small)
    if region is None:
        changed = int((mask > 0).sum())
        explanation = {
            "nothing_changed": f"no item found where an arm can reach ({changed} changed pixels there). Was it "
                               "on the table when the empty photo was taken? Press b to take it again",
            "touches_frame_edge": "the item runs out of the reachable area: move it closer to an arm",
            "more_than_one_object": "two separate things changed: one item at a time, and nothing moving nearby",
        }
        return Detection(None, scale, explanation.get(reason, reason), mask)
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
    return Detection(filled > 0, scale, "ok", mask)


# The detector works at ANALYSIS_WIDTH: about 4 mm per pixel over the zone.
# That finds an item, but a ragged pixel on a box's edge is then a 4 mm step,
# and at the venue one read to the grasp planner as a bottle cap sticking out
# of a cigarette pack. Around the item its outline is found again at this
# width, about 1.4 mm per pixel, before a grasp is planned on it.
REFINE_WIDTH = 960
REFINE_MARGIN_PX = 8  # analysis pixels of table kept around the item, for the exposure fit


def refine_item(frame, background, detection: Detection, valid_small) -> tuple[np.ndarray, float]:
    """(the item's mask at REFINE_WIDTH, full-resolution pixels per pixel of it).

    Falls back to the detector's own mask, scaled up, when the finer look
    cannot be trusted: the light changed, or it finds a very different item.
    """

    import cv2

    from .dataset.autolabel import _foreground

    height, width = frame.shape[:2]
    scale = width / REFINE_WIDTH
    size = (REFINE_WIDTH, round(height / scale))
    coarse = cv2.resize(detection.item.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero(detection.item)
    ratio = detection.scale / scale  # fine pixels per analysis pixel
    x0 = max(int((xs.min() - REFINE_MARGIN_PX) * ratio), 0)
    x1 = min(int((xs.max() + 1 + REFINE_MARGIN_PX) * ratio), size[0])
    y0 = max(int((ys.min() - REFINE_MARGIN_PX) * ratio), 0)
    y1 = min(int((ys.max() + 1 + REFINE_MARGIN_PX) * ratio), size[1])

    live = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)[y0:y1, x0:x1]
    reference = cv2.resize(background, size, interpolation=cv2.INTER_AREA)[y0:y1, x0:x1]
    valid = cv2.resize(valid_small, size, interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1]
    mask, fit = _foreground(live, reference, 28, valid)
    if not fit.trusted:
        return coarse, scale
    # Only what belongs to the item the detector found, a little past its outline.
    near = cv2.dilate(coarse[y0:y1, x0:x1].astype(np.uint8), np.ones((2 * int(ratio) + 1,) * 2, np.uint8)) > 0
    item = ((mask > 0) & near).astype(np.uint8) * 255
    item = cv2.morphologyEx(item, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(item, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return coarse, scale
    filled = np.zeros_like(item)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    found = (filled > 0).sum()
    expected = coarse[y0:y1, x0:x1].sum()
    if not 0.6 * expected <= found <= 1.4 * expected:
        return coarse, scale  # fragments of clear plastic, or something else: the detector knows better
    fine = np.zeros(coarse.shape, bool)
    fine[y0:y1, x0:x1] = filled > 0
    return fine, scale


# Which side each material goes to, and so which arm takes it: each arm drops
# only on its own side. Metal goes with plastic, as in Poland's yellow bin
# ("metale i tworzywa sztuczne"). Anything else is left for a person.
SIDE_OF = {"plastic": "left", "metal": "left", "paper": "right"}
# The classifier's certainty an item needs before it is sorted. On 17 of our
# objects it had never seen (training/evaluate.py, one object left out at a
# time), this rule sent nothing to the wrong side from 0.7 up; at 0.8 it
# sorted 380 of 407 frames and left the other 27.
MIN_CONFIDENCE = 0.8
CROP_PADDING_PX = 10  # table kept around the item, as around the training crops


def side_for(probabilities: dict[str, float], min_confidence: float = MIN_CONFIDENCE) -> tuple[str | None, float]:
    """(which side the item goes to, how sure of that side); the side is None when not sure enough."""

    sure: dict[str, float] = {}
    for material, probability in probabilities.items():
        side = SIDE_OF.get(material)
        if side is not None:
            sure[side] = sure.get(side, 0.0) + probability
    if not sure:
        return None, 0.0
    side = max(sure, key=sure.get)
    return (side if sure[side] >= min_confidence else None), sure[side]


def item_crop(frame, item, scale: float) -> np.ndarray:
    """The item cut out of the full-resolution frame, a little table around it."""

    height, width = frame.shape[:2]
    ys, xs = np.nonzero(item)
    x0 = max(int(xs.min() * scale) - CROP_PADDING_PX, 0)
    x1 = min(int(np.ceil((xs.max() + 1) * scale)) + CROP_PADDING_PX, width)
    y0 = max(int(ys.min() * scale) - CROP_PADDING_PX, 0)
    y1 = min(int(np.ceil((ys.max() + 1) * scale)) + CROP_PADDING_PX, height)
    return frame[y0:y1, x0:x1].copy()


def draw_zone(frame, zone_small, searched_small) -> np.ndarray:
    """The pick zone outlined in green; its part no arm can reach shaded."""

    import cv2

    height, width = frame.shape[:2]
    view = frame.copy()
    zone = cv2.resize(zone_small, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    searched = cv2.resize(searched_small, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    out_of_reach = zone & ~searched
    view[out_of_reach] = (0.5 * view[out_of_reach]).astype(np.uint8)
    contours, _ = cv2.findContours(zone.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (60, 220, 60), 4)
    return view


def draw_detection(frame, detection: Detection, valid_small) -> np.ndarray:
    """What the detector saw: changed pixels in red, the searched zone outlined."""

    import cv2

    height, width = frame.shape[:2]
    view = frame.copy()
    changed = cv2.resize(detection.changed, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    view[changed] = (0.4 * view[changed] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    zone = cv2.resize(valid_small, (width, height), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (60, 220, 60), 3)
    return view


def plan_pick(item_small, scale: float, homography, placements: dict[str, Placement], kinematics,
              limits: dict[str, dict[str, tuple[float, float]]], prefer: str | None = None,
              fingertips_cm: float = FINGERTIPS_CM):
    """(PickPlan, what the arms passed over said), or (None, why no arm can).

    The arm whose base is nearest the item plans first, the other only if it
    cannot; ``prefer`` names an arm to try first regardless. ``kinematics`` is
    one Kinematics for every arm, or one per arm by name: each arm's wrist
    roll has its own zero. ``fingertips_cm`` is how far above the table the
    fixed fingertip comes down.
    """

    tcp_cm = max(fingertips_cm, LOWEST_FINGERTIPS_CM) + FINGERTIP_BEYOND_TCP * 100

    centre = np.argwhere(item_small).mean(axis=0)[::-1] * scale  # (u, v) full resolution
    u0, v0 = centre
    here, right = sheet_points(homography, np.array([u0, u0 + 10.0]), np.array([v0, v0]))
    m_per_px_small = np.linalg.norm(right - here) / 100.0 / 10.0 * scale

    here = sheet_points(homography, np.array([u0]), np.array([v0]))[0]
    distances = {name: float(np.linalg.norm(base_on_sheet(placement) - here)) for name, placement in placements.items()}
    reasons = []
    order = sorted(placements, key=lambda name: (name != prefer, distances[name]))
    for name in order:
        placement = placements[name]
        arm_kinematics = kinematics[name] if isinstance(kinematics, dict) else kinematics
        base = base_on_sheet(placement)
        # The fixed finger on the side of the item facing this arm's base; the
        # other side when the wrist cannot turn that far.
        towards_base = sheet_to_pixel_direction(homography, centre, base - sheet_points(homography, np.array([u0]), np.array([v0]))[0])
        grasp = None
        for side_px in (towards_base, -towards_base):
            grasp = plan_grasp(item_small, m_per_px_small, fixed_side=tuple(side_px))
            if grasp.mode != "pinch":
                break
            fixed_px = np.array(grasp.fixed_finger(m_per_px_small)) * scale
            moving_px = np.array(grasp.moving_finger(m_per_px_small)) * scale
            across_px = np.array(grasp.across)
            ends = sheet_points(homography, np.array([fixed_px[0], fixed_px[0] + across_px[0] * 20]),
                                np.array([fixed_px[1], fixed_px[1] + across_px[1] * 20]))
            target_sheet, direction_sheet = ends[0], ends[1] - ends[0]
            yaw = placement.direction_to_arm(float(np.degrees(np.arctan2(direction_sheet[1], direction_sheet[0]))))
            x, y = placement.to_arm(target_sheet)
            table_cm = placement.table_height(x, y)
            target = np.array([x, y, table_cm + tcp_cm]) / 100
            found = _approach(arm_kinematics, target, yaw, limits.get(name))
            if found is not None:
                above, down, lean = found
                return PickPlan(name, above.degrees, down.degrees, gripper_percent_for(grasp.opening_m), grasp,
                                (float(x), float(y)), (float(fixed_px[0]), float(fixed_px[1])), lean,
                                (float(moving_px[0]), float(moving_px[1])), table_cm, tcp_cm,
                                distances), "; ".join(reasons) or None
        if grasp.mode != "pinch":
            reasons.append(f"{name}: {grasp.reason}")
        else:
            reasons.append(f"{name}: out of reach, {np.hypot(x, y):.0f} cm from its base (it reaches 10-42 cm)")
    return None, "; ".join(reasons) or "no calibrated arm"


def _approach(kinematics, target, yaw: float, limits):
    """(above, down, lean) solutions for a grasp at ``target``, or None."""

    for lean in LEANS_DEG:
        down = kinematics.solve(target, yaw_deg=yaw, limits=limits, lean_deg=lean)
        if not down.reachable:
            continue
        # Start the approach back along the fingers, so the last move is along them.
        approach = kinematics.approach_for(target, lean)
        for back in BACK_OFF_CM:
            above = kinematics.solve(target - approach * back / 100, yaw_deg=yaw, start=down.degrees,
                                     limits=limits, lean_deg=lean)
            if above.reachable:
                return above, down, lean
    return None


def sheet_to_pixel_direction(homography, pixel, sheet_direction) -> np.ndarray:
    """A direction on the sheet, as a direction in the picture at ``pixel``."""

    here = sheet_points(homography, np.array([pixel[0]]), np.array([pixel[1]]))[0]
    target = here + 5.0 * np.asarray(sheet_direction) / max(np.linalg.norm(sheet_direction), 1e-9)
    u, v = homography.world_to_pixel(target[0] / 100.0, target[1] / 100.0)
    return np.array([u - pixel[0], v - pixel[1]])


def draw_plan(frame, item_small, scale: float, plan: PickPlan | None) -> np.ndarray:
    """The camera frame with the item's outline and, if there is a plan, where each finger goes."""

    import cv2

    preview = frame.copy()
    contours, _ = cv2.findContours(item_small.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV draws integer contours only; the item was found at a smaller scale.
    cv2.drawContours(preview, [np.rint(c * scale).astype(np.int32) for c in contours], -1, (255, 255, 0), 3)
    if plan is not None:
        fixed = tuple(int(round(v)) for v in plan.pixel)
        if plan.moving_pixel is not None:
            moving = tuple(int(round(v)) for v in plan.moving_pixel)
            cv2.line(preview, fixed, moving, (0, 200, 255), 3)
            cv2.circle(preview, moving, 12, (255, 120, 0), 3)
        cv2.circle(preview, fixed, 12, (0, 0, 255), -1)
        cv2.putText(preview, f"{plan.arm} arm, lean {plan.lean_deg:.0f} deg", (fixed[0] + 18, fixed[1] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return preview


def drop_pose(arm: str) -> dict[str, float]:
    """Turned to the arm's own side -- left for the left arm, right for the right -- and lifted."""

    pan = 80.0 if arm == "left" else -80.0
    return {"shoulder_pan": pan, "shoulder_lift": 0.0, "elbow_flex": -60.0, "wrist_flex": 60.0, "wrist_roll": 0.0}


MISS_BELOW = 4.0  # percent: jaws told to close that stop below this hold nothing
RELEASE_OPEN = 60.0

# The descent goes where the plan says, uncorrected. Raising it by the sag the
# hover showed was tried at the venue: the hover came up 2-7 mm short, but the
# grasp itself barely sags, and the fingertips ended 5 mm higher than planned.
# Where they really are is reported instead, from the joints' own readings.
SETTLE_S = 0.5


def execute_pick(arm, plan: PickPlan, neutral: dict[str, float], *, kinematics=None, dry_run: bool = False,
                 sleep=None, log=print) -> bool:
    """Carry out a PickPlan with a real (or fake) Arm; True if something was dropped.

    With ``kinematics`` the fingertips' real height above the table is reported.
    """

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
    if kinematics is not None and plan.table_cm is not None:
        sleep(SETTLE_S)
        tip_cm = kinematics.fingertip(arm.pose())[2] * 100 - plan.table_cm
        log(f"{arm.name}: fingertips {tip_cm:.1f} cm above the table (planned {plan.fingertip_above_table_cm:.1f})")
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
