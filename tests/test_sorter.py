"""One pick end to end, on a synthetic camera and a fake arm.

The camera sees the table at 1 mm per pixel; the left arm stands where the
real one was calibrated at the venue. What is pinned down: an item in reach
gets a plan whose fixed finger lands where the grasp planner put it, fingers
down, and executing a plan never opens the jaws before the drop.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
pytest.importorskip("mujoco", reason="needs the simulation extra")

from trashdrop.kinematics import Kinematics  # noqa: E402
from trashdrop.perception.calibration import HomographyCalibration  # noqa: E402
from trashdrop.placement import Placement  # noqa: E402
from trashdrop.rig import PickZone  # noqa: E402
from trashdrop.sorter import (  # noqa: E402
    GRASP_HEIGHT_CM,
    PICK_EDGE_MARGIN_PX,
    RELEASE_OPEN,
    draw_detection,
    draw_zone,
    find_item,
    refine_item,
    zone_mask,
    base_on_sheet,
    draw_plan,
    drop_pose,
    execute_pick,
    gripper_percent_for,
    plan_pick,
    reach_mask,
    sheet_points,
)

LEFT = Placement(x=24.65, y=-19.64, yaw=-92.31, table_z=-2.42)  # the venue's left arm
RIGHT = Placement(x=17.97, y=20.73, yaw=-84.12, table_z=-0.47)  # and its right arm, facing the same way
SCALE = 6.0  # full-resolution pixels per analysis pixel


def camera() -> HomographyCalibration:
    """1 mm per full-resolution pixel; sheet centre at pixel (960, 540), sheet y up the picture."""

    sheet_cm = [(-10, 7.5), (10, 7.5), (10, -7.5), (-10, -7.5)]
    pixels = [(960 + x * 10, 540 - y * 10) for x, y in sheet_cm]
    return HomographyCalibration(pixels, [(x / 100, y / 100) for x, y in sheet_cm])


def item_at(homography, sheet_cm, size_cm) -> np.ndarray:
    """An analysis-resolution mask of a rectangular item centred at a sheet point."""

    mask = np.zeros((180, 320), bool)
    u, v = homography.world_to_pixel(sheet_cm[0] / 100, sheet_cm[1] / 100)
    half_w, half_h = size_cm[0] * 10 / 2 / SCALE, size_cm[1] * 10 / 2 / SCALE
    cu, cv = u / SCALE, v / SCALE
    mask[int(cv - half_h) : int(cv + half_h), int(cu - half_w) : int(cu + half_w)] = True
    return mask


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = Kinematics()
        cls.homography = camera()

    def test_the_base_is_the_arms_origin(self) -> None:
        self.assertTrue(np.allclose(LEFT.to_arm(base_on_sheet(LEFT)), [0.0, 0.0], atol=1e-9))

    def test_an_item_in_reach_is_planned_fixed_finger_down_on_its_edge(self) -> None:
        item = item_at(self.homography, (-10.0, -8.0), (3.0, 12.0))  # a 3 cm wide bottle lying along y
        plan, reason = plan_pick(item, SCALE, self.homography, {"left": LEFT}, self.kinematics, {})
        self.assertIsNotNone(plan, reason)
        self.assertEqual(plan.arm, "left")
        self.assertAlmostEqual(plan.grasp_plan.width_m, 0.03, delta=0.01)
        reached = self.kinematics.tcp(plan.grasp) * 100
        self.assertLess(np.hypot(reached[0] - plan.target_cm[0], reached[1] - plan.target_cm[1]), 0.3)
        self.assertAlmostEqual(reached[2], LEFT.table_z + GRASP_HEIGHT_CM, delta=0.3)
        fingers, _ = self.kinematics.pointing(plan.grasp)
        self.assertLess(fingers[2], -0.99)
        # The fixed finger is just outside the item, not on top of it.
        sheet_target = sheet_points(self.homography, np.array([plan.pixel[0]]), np.array([plan.pixel[1]]))[0]
        self.assertGreater(abs(sheet_target[0] - (-10.0)), 1.5 - 0.2)

    def test_the_grasp_height_follows_a_table_read_lower_further_out(self) -> None:
        tilted = Placement(LEFT.x, LEFT.y, LEFT.yaw, 0.88, -0.095, 0.104)  # the venue's left arm, touched
        item = item_at(self.homography, (-10.0, -8.0), (3.0, 12.0))
        plan, reason = plan_pick(item, SCALE, self.homography, {"left": tilted}, self.kinematics, {})
        self.assertIsNotNone(plan, reason)
        self.assertAlmostEqual(plan.table_cm, tilted.table_height(*plan.target_cm), places=6)
        reached = self.kinematics.tcp(plan.grasp) * 100
        self.assertAlmostEqual(reached[2], plan.table_cm + GRASP_HEIGHT_CM, delta=0.3)
        tip = self.kinematics.fingertip(plan.grasp)[2] * 100 - plan.table_cm
        self.assertAlmostEqual(tip, plan.fingertip_above_table_cm, delta=0.3)
        self.assertGreater(tip, 0.2, "the fingertip clears the table")

    def test_how_low_the_fingertips_come_is_asked_for_and_never_into_the_table(self) -> None:
        item = item_at(self.homography, (-10.0, -8.0), (3.0, 12.0))
        for asked, expected in ((1.3, 1.3), (0.5, 0.5), (-1.0, 0.2)):
            with self.subTest(asked=asked):
                plan, reason = plan_pick(item, SCALE, self.homography, {"left": LEFT}, self.kinematics, {},
                                         fingertips_cm=asked)
                self.assertIsNotNone(plan, reason)
                tip = self.kinematics.fingertip(plan.grasp)[2] * 100 - plan.table_cm
                self.assertAlmostEqual(tip, expected, delta=0.3)

    def test_an_arm_with_its_own_wrist_zero_still_closes_its_jaw_across_the_item(self) -> None:
        # The venue's miss: a screwdriver lying along the table's y, the jaw closing along it.
        turned = Kinematics(wrist_roll_offset=90.0)
        item = item_at(self.homography, (-10.0, -8.0), (3.0, 12.0))
        plan, reason = plan_pick(item, SCALE, self.homography, {"left": LEFT}, {"left": turned}, {})
        self.assertIsNotNone(plan, reason)
        _, across = turned.pointing(plan.grasp)
        angle = np.radians(LEFT.yaw)
        on_sheet = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) @ across[:2]
        self.assertGreater(abs(on_sheet[0]), 0.99, "square to the item, which lies along the sheet's y")

    def test_an_item_in_the_middle_of_the_table_is_taken_with_leaning_fingers(self) -> None:
        # Where the bottle lay at the venue: the sheet's centre, 31.5 cm from the left base.
        item = item_at(self.homography, (0.0, 0.0), (6.5, 20.0))
        plan, reason = plan_pick(item, SCALE, self.homography, {"left": LEFT}, self.kinematics, {})
        self.assertIsNotNone(plan, reason)
        self.assertGreater(plan.lean_deg, 0.0)
        reached = self.kinematics.tcp(plan.grasp) * 100
        self.assertLess(np.hypot(reached[0] - plan.target_cm[0], reached[1] - plan.target_cm[1]), 0.3)

    def test_the_plan_is_drawn_on_the_full_size_frame(self) -> None:
        # The venue crash: the item's outline, found at analysis size, scaled
        # by a float into coordinates OpenCV refuses to draw.
        frame = np.full((1080, 1920, 3), 200, np.uint8)
        item = item_at(self.homography, (-10.0, -8.0), (3.0, 12.0))
        self.assertEqual(draw_plan(frame, item, SCALE, None).shape, frame.shape)
        plan, reason = plan_pick(item, SCALE, self.homography, {"left": LEFT}, self.kinematics, {})
        self.assertIsNotNone(plan, reason)
        drawn = draw_plan(frame, item, SCALE, plan)
        u, v = (int(round(value)) for value in plan.pixel)
        self.assertTrue((drawn[v, u] == (0, 0, 255)).all(), "the fixed finger's dot is where the plan says")

    def test_the_arm_nearer_the_item_takes_it(self) -> None:
        both = {"left": LEFT, "right": RIGHT}
        for sheet_cm, nearer in (((10.0, -8.0), "right"), ((-10.0, -8.0), "left")):
            with self.subTest(item=sheet_cm):
                item = item_at(self.homography, sheet_cm, (3.0, 12.0))
                plan, reason = plan_pick(item, SCALE, self.homography, both, self.kinematics, {})
                self.assertIsNotNone(plan, reason)
                self.assertEqual(plan.arm, nearer)
                self.assertLess(plan.distances_cm[nearer], min(plan.distances_cm.values()) + 1e-9)
        item = item_at(self.homography, (10.0, -8.0), (3.0, 12.0))
        plan, _ = plan_pick(item, SCALE, self.homography, both, self.kinematics, {}, prefer="left")
        self.assertEqual(plan.arm, "left", "an arm asked for goes first however far it is")

    def test_the_further_arm_takes_what_the_nearer_cannot_reach(self) -> None:
        # Right beside the right arm's base: too close for fingers down.
        item = item_at(self.homography, (17.0, -18.0), (3.0, 6.0))
        plan, passed_over = plan_pick(item, SCALE, self.homography, {"left": LEFT, "right": RIGHT},
                                      self.kinematics, {})
        self.assertIsNotNone(plan, passed_over)
        self.assertEqual(plan.arm, "left")
        self.assertIn("right", passed_over)

    def test_an_item_out_of_reach_is_refused_with_a_reason(self) -> None:
        item = item_at(self.homography, (30.0, 20.0), (3.0, 8.0))
        plan, reason = plan_pick(item, SCALE, self.homography, {"left": LEFT}, self.kinematics, {})
        self.assertIsNone(plan)
        self.assertIn("reach", reason)

    def test_the_camera_only_looks_inside_the_arms_ring(self) -> None:
        mask = reach_mask((180, 320), SCALE, self.homography, {"left": LEFT})
        base_u, base_v = self.homography.world_to_pixel(*(base_on_sheet(LEFT) / 100))
        self.assertEqual(mask[int(base_v / SCALE), int(base_u / SCALE)], 0, "not at the base itself")
        self.assertGreater(mask.sum(), 0)

    def test_shared_reach_excludes_places_only_one_arm_can_take(self) -> None:
        placements = {"left": LEFT, "right": RIGHT}
        either = reach_mask((180, 320), SCALE, self.homography, placements)
        both = reach_mask((180, 320), SCALE, self.homography, placements, require_all=True)
        self.assertTrue(np.all((both == 0) | (either > 0)))
        self.assertGreater((either > 0).sum(), (both > 0).sum())


class DetectionTests(unittest.TestCase):
    """A busy table: things stay on it, people reach across it. Only the zone counts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.homography = camera()
        cls.zone = zone_mask((180, 320), SCALE, cls.homography, PickZone())  # 30 x 21 cm, the sheet's footprint

    @staticmethod
    def table() -> np.ndarray:
        frame = np.full((1080, 1920, 3), (215, 220, 218), np.uint8)
        noise = np.random.default_rng(0).normal(0, 2, frame.shape)
        return np.clip(frame + noise, 0, 255).astype(np.uint8)

    def test_the_zone_is_the_sheets_footprint(self) -> None:
        self.assertEqual(self.zone[540 // 6, 960 // 6], 255, "the sheet's centre")
        self.assertEqual(self.zone[540 // 6, 1300 // 6], 0, "20 cm right of the sheet's edge")

    def test_a_hand_outside_the_zone_does_not_hide_the_item_in_it(self) -> None:
        empty = self.table()
        frame = empty.copy()
        cv2.rectangle(frame, (900, 500), (1000, 560), (60, 90, 170), -1)  # the item, inside
        cv2.rectangle(frame, (1500, 100), (1900, 400), (90, 120, 160), -1)  # a hand, far outside
        detection = find_item(frame, empty, self.zone)
        self.assertIsNotNone(detection.item, detection.reason)
        self.assertEqual(draw_detection(frame, detection, self.zone).shape, frame.shape)
        self.assertEqual(draw_zone(frame, self.zone, self.zone).shape, frame.shape)

    def test_the_item_is_outlined_again_finer_for_the_grasp(self) -> None:
        # The venue's cigarette pack: 5.5 x 8.5 cm lying 10 degrees askew.
        from trashdrop.perception.grasp import plan_grasp

        empty = self.table()
        frame = empty.copy()
        cv2.fillPoly(frame, [cv2.boxPoints(((960.0, 540.0), (55.0, 85.0), 10.0)).astype(np.int32)], (70, 90, 60))
        detection = find_item(frame, empty, self.zone)
        self.assertIsNotNone(detection.item, detection.reason)
        item, scale = refine_item(frame, empty, detection, self.zone)
        self.assertEqual(scale, 2.0)
        plan = plan_grasp(item, scale * 0.001, fixed_side=(0.0, 1.0))
        coarse = plan_grasp(detection.item, detection.scale * 0.001, fixed_side=(0.0, 1.0))
        self.assertEqual(plan.mode, "pinch")
        # The difference image blurs a dark item a little wider than it is:
        # safe, the jaw opens wider. Never narrower, and much closer than
        # the detector's own 4 mm pixels (73 mm here).
        self.assertGreaterEqual(plan.width_m, 0.053)
        self.assertLess(plan.width_m, 0.066)
        self.assertLess(plan.width_m, coarse.width_m - 0.005)
        self.assertGreater(abs(np.dot(plan.across, (np.cos(np.radians(10)), np.sin(np.radians(10))))), 0.97)
        self.assertLess(np.hypot(*(np.array(plan.center) * scale - (960, 540))), 15.0, "through the middle")

    def test_live_pick_accepts_an_item_near_the_zone_edge_but_not_on_it(self) -> None:
        self.assertEqual(PICK_EDGE_MARGIN_PX, 1)
        left = int(np.flatnonzero(self.zone[90] > 0)[0])
        empty = self.table()
        for offset, expected in ((0, "touches_frame_edge"), (2, "ok")):
            with self.subTest(offset=offset):
                frame = empty.copy()
                x = int((left + offset) * SCALE)
                cv2.rectangle(frame, (x, 500), (x + 72, 560), (40, 40, 40), -1)
                self.assertEqual(find_item(frame, empty, self.zone).code, expected)

    def test_an_empty_zone_says_so(self) -> None:
        # What the pick checks after every drop: nothing left means it was caught.
        empty = self.table()
        self.assertEqual(find_item(empty.copy(), empty, self.zone).code, "nothing_changed")
        frame = empty.copy()
        cv2.rectangle(frame, (900, 500), (1000, 560), (60, 90, 170), -1)
        self.assertEqual(find_item(frame, empty, self.zone).code, "ok")

    def test_an_item_already_there_in_the_empty_photo_is_explained(self) -> None:
        with_item = self.table()
        cv2.rectangle(with_item, (900, 500), (1000, 560), (60, 90, 170), -1)
        detection = find_item(with_item, with_item, self.zone)
        self.assertIsNone(detection.item)
        self.assertIn("press b", detection.reason.lower())


class GripperTests(unittest.TestCase):
    def test_wider_items_open_the_jaw_further(self) -> None:
        openings = [gripper_percent_for(w) for w in (0.03, 0.05, 0.07, 0.09)]
        self.assertEqual(openings, sorted(openings))

    def test_the_jaw_is_never_swung_up_past_its_widest(self) -> None:
        # The venue's cigarette pack asked for 9 cm and got 100 %: the jaw swung
        # up and came down on top of it.
        self.assertEqual(gripper_percent_for(0.20), gripper_percent_for(0.087))
        self.assertLess(gripper_percent_for(0.20), 65.0)


class FakeArm:
    """Records moves; the jaws stop at ``stops_at`` percent when told to close."""

    def __init__(self, name: str = "left", stops_at: float = 30.0) -> None:
        self.name, self.stops_at, self.moves = name, stops_at, []

    def move(self, targets, speed=None):
        self.moves.append(dict(targets))
        gripper = targets.get("gripper")
        reached = dict(targets)
        if gripper == 0.0:
            reached["gripper"] = self.stops_at
        return reached


class ReportingArm(FakeArm):
    """Also says where its joints are: where they were last sent."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.goal = {}

    def move(self, targets, speed=None):
        self.goal.update(targets)
        return super().move(targets, speed)

    def pose(self):
        return dict(self.goal)


class SpeedArm(FakeArm):
    def __init__(self) -> None:
        super().__init__()
        self.speeds = []

    def move(self, targets, speed=None):
        self.speeds.append(speed)
        return super().move(targets, speed)


class ExecuteTests(unittest.TestCase):
    PLAN = type("P", (), {
        "above": {"shoulder_pan": -60.0, "shoulder_lift": 10.0, "elbow_flex": 5.0, "wrist_flex": 80.0, "wrist_roll": 20.0},
        "grasp": {"shoulder_pan": -60.0, "shoulder_lift": 30.0, "elbow_flex": 0.0, "wrist_flex": 70.0, "wrist_roll": 20.0},
        "open_percent": 55.0,
        "table_cm": -2.0,
        "fingertip_above_table_cm": 1.3,
    })()
    NEUTRAL = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -90.0, "wrist_flex": 0.0, "wrist_roll": 0.0}

    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = Kinematics()

    def pick(self, arm, **kwargs) -> tuple[bool, list[str]]:
        lines: list[str] = []
        done = execute_pick(arm, self.PLAN, self.NEUTRAL, kinematics=self.kinematics, sleep=lambda _: None,
                            log=lines.append, **kwargs)
        return done, lines

    def test_the_descent_goes_where_planned_and_says_where_the_fingertips_are(self) -> None:
        arm = ReportingArm()
        done, lines = self.pick(arm)
        self.assertTrue(done)
        self.assertIn(self.PLAN.grasp, arm.moves)
        self.assertTrue(any("above the table" in line for line in lines), lines)

    def test_a_dry_run_with_kinematics_never_goes_down(self) -> None:
        arm = ReportingArm()
        done, _ = self.pick(arm, dry_run=True)
        self.assertFalse(done)
        self.assertNotIn(self.PLAN.grasp, arm.moves)

    def test_the_final_descent_uses_its_configured_speed(self) -> None:
        arm = SpeedArm()
        execute_pick(arm, self.PLAN, self.NEUTRAL, descent_speed=12.0, sleep=lambda _: None, log=lambda *_: None)
        self.assertEqual(arm.speeds[arm.moves.index(self.PLAN.grasp)], 12.0)

    def test_a_dry_run_never_goes_down(self) -> None:
        arm = FakeArm()
        self.assertFalse(execute_pick(arm, self.PLAN, self.NEUTRAL, dry_run=True, sleep=lambda _: None, log=lambda *_: None))
        self.assertNotIn(self.PLAN.grasp, arm.moves)
        self.assertEqual(arm.moves[-1], self.NEUTRAL)

    def test_a_jaw_closed_almost_shut_still_carries_its_item(self) -> None:
        # A crumpled receipt closes the jaw as far as nothing does. At the venue
        # such items were taken for misses and never let go.
        arm = FakeArm(stops_at=1.0)
        self.assertTrue(execute_pick(arm, self.PLAN, self.NEUTRAL, log=lambda *_: None))
        released = arm.moves.index(drop_pose("left")) + 1
        self.assertEqual(arm.moves[released], {"gripper": RELEASE_OPEN})

    def test_the_jaws_open_only_over_the_arms_own_side(self) -> None:
        arm = FakeArm("right")
        self.assertTrue(execute_pick(arm, self.PLAN, self.NEUTRAL, log=lambda *_: None))
        closed = next(i for i, move in enumerate(arm.moves) if move.get("gripper") == 0.0)
        opened = next(i for i, move in enumerate(arm.moves) if i > closed and "gripper" in move)
        self.assertEqual(arm.moves[opened - 1], drop_pose("right"))
        self.assertIn("wrist_roll", arm.moves[opened + 1], "a shake of the wrist once the jaw is open")
        self.assertEqual(arm.moves[-1], self.NEUTRAL)

    def test_each_arm_drops_on_its_own_outer_side(self) -> None:
        # In an arm's own frame +y is its left. The venue's left arm once
        # dropped a bottle to its right, between the two arms.
        left = self.kinematics.tcp(drop_pose("left"))
        right = self.kinematics.tcp(drop_pose("right"))
        self.assertGreater(left[1], 0.2, "the left arm drops to its left")
        self.assertLess(right[1], -0.2, "the right arm drops to its right")
        for name, tcp in (("left", left), ("right", right)):
            self.assertGreater(tcp[2], 0.12, "and high enough to drop, not place")
            fingers, _ = self.kinematics.pointing(drop_pose(name))
            self.assertLess(fingers[2], -0.8, "the jaw opens downward, so the item falls out")


if __name__ == "__main__":
    unittest.main()
