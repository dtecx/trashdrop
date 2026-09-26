"""Where the gripper takes an item, planned on synthetic top-down masks.

Masks are rasterised from an item's width profile, so the test needs numpy
only. What it pins down are the decisions the real jaw depends on: a bottle
is taken by the neck, a can across the middle, a sheet only from an edge, and
the fixed finger never comes down on top of the item.
"""

from __future__ import annotations

import unittest

import numpy as np

from trashdrop.perception.grasp import plan_grasp
from trashdrop.station import FIXED_JAW_CLEARANCE, JAW_SPAN, MAX_GRASP_WIDTH

M_PER_PX = 0.001
SIZE = 400

# (distance from one end, width), metres. A 0.5 L PET bottle lying down.
BOTTLE = ((0.0, 0.065), (0.16, 0.065), (0.19, 0.027), (0.205, 0.027), (0.2051, 0.030), (0.22, 0.030))
CAN = ((0.0, 0.066), (0.168, 0.066))
SHEET = ((0.0, 0.10), (0.15, 0.10))


def rasterise(profile, angle_deg: float = 0.0, gap: tuple[float, float] | None = None):
    """A top-down mask of an item centred in the image; returns (mask, to_item)."""

    xs, ws = np.array(profile).T
    length = xs[-1]
    angle = np.radians(angle_deg)
    centre = SIZE / 2

    def to_item(x, y):
        dx, dy = (np.asarray(x) - centre) * M_PER_PX, (np.asarray(y) - centre) * M_PER_PX
        s = dx * np.cos(angle) + dy * np.sin(angle) + length / 2
        t = -dx * np.sin(angle) + dy * np.cos(angle)
        return s, t

    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    s, t = to_item(xx, yy)
    mask = (s >= 0) & (s <= length) & (np.abs(t) <= np.interp(s, xs, ws) / 2)
    if gap is not None:
        mask &= ~((s >= gap[0]) & (s <= gap[1]))
    return mask, to_item


def boxes(parts, angle_deg: float = 0.0):
    """A mask made of rectangles (centre x, centre y, width, height) in metres."""

    angle = np.radians(angle_deg)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    dx, dy = (xx - SIZE / 2) * M_PER_PX, (yy - SIZE / 2) * M_PER_PX
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    mask = np.zeros((SIZE, SIZE), bool)
    for cx, cy, w, h in parts:
        mask |= (np.abs(u - cx) <= w / 2) & (np.abs(v - cy) <= h / 2)

    def to_item(x, y):
        px, py = (np.asarray(x) - SIZE / 2) * M_PER_PX, (np.asarray(y) - SIZE / 2) * M_PER_PX
        return px * np.cos(angle) + py * np.sin(angle), -px * np.sin(angle) + py * np.cos(angle)

    return mask, to_item


def distance_to_item(mask, point) -> float:
    ys, xs = np.nonzero(mask)
    return float(np.min(np.hypot(xs - point[0], ys - point[1]))) * M_PER_PX


class GraspPlanTests(unittest.TestCase):
    def test_a_bottle_is_taken_by_the_neck(self) -> None:
        for angle in (0.0, 30.0, 115.0):
            with self.subTest(angle=angle):
                mask, to_item = rasterise(BOTTLE, angle)
                plan = plan_grasp(mask, M_PER_PX)
                self.assertEqual(plan.mode, "pinch")
                s, _ = to_item(*plan.center)
                self.assertGreater(s, 0.185, "the jaw should sit on the neck and cap, not the body")
                self.assertLess(plan.width_m, 0.035)

    def test_a_can_is_taken_across_the_middle(self) -> None:
        for angle in (0.0, 60.0):
            with self.subTest(angle=angle):
                mask, to_item = rasterise(CAN, angle)
                plan = plan_grasp(mask, M_PER_PX)
                self.assertEqual(plan.mode, "pinch")
                s, t = to_item(*plan.center)
                self.assertAlmostEqual(s, 0.084, delta=0.02)
                self.assertAlmostEqual(t, 0.0, delta=0.003)
                self.assertAlmostEqual(plan.width_m, 0.066, delta=0.003)

    def test_a_sheet_wider_than_the_jaw_only_gets_an_edge_plan(self) -> None:
        mask, _ = rasterise(SHEET, 20.0)
        plan = plan_grasp(mask, M_PER_PX)
        self.assertEqual(plan.mode, "edge")
        fixed = plan.fixed_finger(M_PER_PX)
        moving = plan.moving_finger(M_PER_PX)
        self.assertFalse(mask[round(fixed[1]), round(fixed[0])])
        self.assertTrue(mask[round(moving[1]), round(moving[0])], "the moving finger lands on the sheet")

    def test_the_fixed_finger_never_comes_down_on_the_item(self) -> None:
        for profile in (BOTTLE, CAN):
            for angle in (0.0, 45.0, 170.0):
                with self.subTest(profile=len(profile), angle=angle):
                    mask, _ = rasterise(profile, angle)
                    plan = plan_grasp(mask, M_PER_PX)
                    fixed = plan.fixed_finger(M_PER_PX)
                    moving = plan.moving_finger(M_PER_PX)
                    self.assertGreater(distance_to_item(mask, fixed), FIXED_JAW_CLEARANCE / 2)
                    self.assertGreater(distance_to_item(mask, moving), FIXED_JAW_CLEARANCE / 2)
                    gap = np.hypot(moving[0] - fixed[0], moving[1] - fixed[1]) * M_PER_PX
                    self.assertAlmostEqual(gap, plan.opening_m, places=6)

    def test_the_fixed_finger_goes_on_the_requested_side(self) -> None:
        mask, _ = rasterise(CAN, 90.0)  # lying along the image's y axis
        for side, sign in (((1.0, 0.0), 1), ((-1.0, 0.0), -1)):
            with self.subTest(side=side):
                plan = plan_grasp(mask, M_PER_PX, fixed_side=side)
                fixed = plan.fixed_finger(M_PER_PX)
                self.assertGreater(sign * (fixed[0] - plan.center[0]), 0)

    def test_an_invisible_stretch_is_never_under_the_jaw(self) -> None:
        # Clear plastic can vanish against the table in the middle of an item;
        # its width there is unknown, so the jaw must not rely on it.
        gap = (0.07, 0.08)
        mask, to_item = rasterise(((0.0, 0.03), (0.12, 0.03)), 25.0, gap=gap)
        plan = plan_grasp(mask, M_PER_PX)
        self.assertEqual(plan.mode, "pinch")
        s, _ = to_item(*plan.center)
        self.assertTrue(s + JAW_SPAN / 2 < gap[0] or s - JAW_SPAN / 2 > gap[1])

    def test_a_crushed_bottle_is_taken_by_the_cap_sticking_out_of_its_side(self) -> None:
        # Body flattened to 12 x 9 cm -- too wide anywhere -- with a 3 cm cap
        # standing 2.5 cm proud of one long side.
        for angle in (0.0, 35.0):
            with self.subTest(angle=angle):
                mask, to_item = boxes([(0.0, 0.0, 0.12, 0.09), (0.01, 0.0575, 0.03, 0.025)], angle)
                plan = plan_grasp(mask, M_PER_PX)
                self.assertEqual(plan.mode, "pinch")
                u, v = to_item(*plan.center)
                self.assertGreater(v, 0.045, "the jaw should close on the cap")
                self.assertAlmostEqual(u, 0.01, delta=0.006)
                self.assertLess(plan.width_m, 0.035)

    def test_a_thin_shadow_tail_is_not_a_handle(self) -> None:
        # A standing can seen from above: its lid and side read ~85 mm, too
        # wide, and the detector left a 10 mm tail of shadow attached to it.
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        can = np.hypot(xx - 200, yy - 200) <= 42
        tail, _ = boxes([(0.06, 0.0, 0.05, 0.01)], 40.0)
        self.assertEqual(plan_grasp(can | tail, M_PER_PX).mode, "edge")

        # The same tail on a can lying down must not stop the can being taken,
        # nor end up under a finger.
        lying, _ = rasterise(CAN, 0.0)
        tail, _ = boxes([(0.0, 0.045, 0.01, 0.03)])
        mask = lying | tail
        plan = plan_grasp(mask, M_PER_PX)
        self.assertEqual(plan.mode, "pinch")
        self.assertLessEqual(plan.width_m, MAX_GRASP_WIDTH)
        for finger in (plan.fixed_finger(M_PER_PX), plan.moving_finger(M_PER_PX)):
            self.assertGreater(distance_to_item(mask, finger), FIXED_JAW_CLEARANCE / 2)

    def test_a_ragged_corner_of_a_box_is_not_a_cap(self) -> None:
        # The venue's cigarette pack, 10 degrees askew, as the detector's 4 mm
        # pixels saw it. A two-pixel step on its left edge once read as a cap
        # to hang the jaw on, and the jaw closed along the pack's edge.
        rows = [
            ".........#######...", ".################..", *[".#################."] * 7,
            *["...###############."] * 5, *["..################."] * 7, "...##############..",
        ]
        mask = np.array([[c == "#" for c in row] for row in rows])
        plan = plan_grasp(mask, 0.0039, fixed_side=(-0.61, 0.79))
        self.assertEqual(plan.mode, "pinch")
        self.assertGreater(abs(plan.across[0]), 0.95, "across the pack's short side")
        ys, xs = np.nonzero(mask)
        self.assertLess(abs(plan.center[1] - ys.mean()), 3.0, "through its middle, not at an end")

    def test_a_narrower_limit_turns_a_can_into_an_edge_plan(self) -> None:
        mask, _ = rasterise(CAN)
        self.assertEqual(plan_grasp(mask, M_PER_PX).mode, "pinch")
        self.assertEqual(plan_grasp(mask, M_PER_PX, max_width_m=0.05).mode, "edge")
        self.assertGreaterEqual(MAX_GRASP_WIDTH, 0.066, "a 0.5 L can must stay graspable")

    def test_an_empty_mask_gives_no_plan(self) -> None:
        self.assertEqual(plan_grasp(np.zeros((50, 50), bool), M_PER_PX).mode, "none")


if __name__ == "__main__":
    unittest.main()
