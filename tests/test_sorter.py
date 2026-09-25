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
from trashdrop.sorter import (  # noqa: E402
    GRASP_HEIGHT_CM,
    base_on_sheet,
    drop_pose,
    execute_pick,
    gripper_percent_for,
    plan_pick,
    reach_mask,
    sheet_points,
)

LEFT = Placement(x=24.65, y=-19.64, yaw=-92.31, table_z=-2.42)  # the venue's left arm
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


class GripperTests(unittest.TestCase):
    def test_wider_items_open_the_jaw_further(self) -> None:
        openings = [gripper_percent_for(w) for w in (0.03, 0.05, 0.07, 0.09)]
        self.assertEqual(openings, sorted(openings))
        self.assertEqual(gripper_percent_for(0.20), 100.0)


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


class ExecuteTests(unittest.TestCase):
    PLAN = type("P", (), {
        "above": {"shoulder_pan": -60.0, "shoulder_lift": 10.0, "elbow_flex": 5.0, "wrist_flex": 80.0, "wrist_roll": 20.0},
        "grasp": {"shoulder_pan": -60.0, "shoulder_lift": 30.0, "elbow_flex": 0.0, "wrist_flex": 70.0, "wrist_roll": 20.0},
        "open_percent": 55.0,
    })()
    NEUTRAL = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -90.0, "wrist_flex": 0.0, "wrist_roll": 0.0}

    def test_a_dry_run_never_goes_down(self) -> None:
        arm = FakeArm()
        self.assertFalse(execute_pick(arm, self.PLAN, self.NEUTRAL, dry_run=True, sleep=lambda _: None, log=lambda *_: None))
        self.assertNotIn(self.PLAN.grasp, arm.moves)
        self.assertEqual(arm.moves[-1], self.NEUTRAL)

    def test_a_miss_is_not_carried(self) -> None:
        arm = FakeArm(stops_at=1.0)
        self.assertFalse(execute_pick(arm, self.PLAN, self.NEUTRAL, log=lambda *_: None))
        self.assertNotIn(drop_pose("left"), arm.moves)

    def test_the_jaws_open_only_over_the_arms_own_side(self) -> None:
        arm = FakeArm("right")
        self.assertTrue(execute_pick(arm, self.PLAN, self.NEUTRAL, log=lambda *_: None))
        closed = next(i for i, move in enumerate(arm.moves) if move.get("gripper") == 0.0)
        opened = next(i for i, move in enumerate(arm.moves) if i > closed and "gripper" in move)
        self.assertEqual(arm.moves[opened - 1], drop_pose("right"))
        self.assertLess(drop_pose("right")["shoulder_pan"], 0, "the right arm turns right")
        self.assertEqual(arm.moves[-1], self.NEUTRAL)


if __name__ == "__main__":
    unittest.main()
