"""Standing items: told apart from lying ones from above, and pushed over.

The camera is synthetic, 1 mm per full-resolution pixel with the table's
frame in its middle; the arms stand where the venue's did. What is pinned
down: a round outline too wide to pinch is taken for a standing item and a
lying bottle or can is not; the push comes down beside the item, fingers
down, and goes level through where it stood, falling it towards the middle
of the zone where the arms allow.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
pytest.importorskip("mujoco", reason="needs the simulation extra")

from trashdrop.kinematics import Kinematics  # noqa: E402
from trashdrop.knock import PUSH_HEIGHT_CM, execute_knock, looks_upright, plan_knock  # noqa: E402
from trashdrop.perception.calibration import HomographyCalibration  # noqa: E402
from trashdrop.placement import Placement  # noqa: E402

LEFT = Placement(x=24.65, y=-19.64, yaw=-92.31, table_z=-2.42)
RIGHT = Placement(x=17.97, y=20.73, yaw=-84.12, table_z=-0.47)
SCALE = 2.0  # full-resolution pixels per mask pixel


def camera() -> HomographyCalibration:
    corners = [(-10, 7.5), (10, 7.5), (10, -7.5), (-10, -7.5)]
    return HomographyCalibration([(960 + x * 10, 540 - y * 10) for x, y in corners],
                                 [(x / 100, y / 100) for x, y in corners])


def outline(centre_cm, size_cm, angle_deg: float = 0.0, round_: bool = False) -> np.ndarray:
    """A mask of an item on the table, at half resolution."""

    mask = np.zeros((540, 960), np.uint8)
    u, v = 960 + centre_cm[0] * 10, 540 - centre_cm[1] * 10
    if round_:
        cv2.circle(mask, (int(u / SCALE), int(v / SCALE)), int(size_cm[0] * 10 / 2 / SCALE), 1, -1)
    else:
        box = cv2.boxPoints(((u / SCALE, v / SCALE), (size_cm[0] * 10 / SCALE, size_cm[1] * 10 / SCALE), angle_deg))
        cv2.fillPoly(mask, [np.rint(box).astype(np.int32)], 1)
    return mask > 0


class UprightTests(unittest.TestCase):
    def test_a_standing_bottle_seen_from_above_is_upright(self) -> None:
        upright, short, long = looks_upright(outline((0, 0), (7.0,), round_=True), SCALE, camera())
        self.assertTrue(upright)
        self.assertAlmostEqual(short, 7.0, delta=0.4)

    def test_lying_items_and_pinchable_ones_are_not(self) -> None:
        for name, mask in (
            ("a bottle lying down", outline((0, 0), (6.5, 20.0), 30.0)),
            ("a can lying down", outline((0, 0), (6.6, 12.0), -20.0)),
            ("a small round thing the jaw closes around", outline((0, 0), (3.5,), round_=True)),
        ):
            with self.subTest(name):
                self.assertFalse(looks_upright(mask, SCALE, camera())[0])


class KnockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = Kinematics()

    def plan(self, centre_cm):
        return plan_knock(outline(centre_cm, (7.0,), round_=True), SCALE, camera(),
                          {"left": LEFT, "right": RIGHT}, self.kinematics, {}, (0.0, 0.0))

    def test_the_push_comes_down_beside_it_and_goes_level_through_it(self) -> None:
        plan, why = self.plan((4.0, -3.0))
        self.assertIsNotNone(plan, why)
        placement = LEFT if plan.arm == "left" else RIGHT
        start, end = (self.kinematics.tcp(plan.start) * 100, self.kinematics.tcp(plan.end) * 100)
        for tcp in (start, end):
            self.assertAlmostEqual(tcp[2] - placement.table_height(*tcp[:2]), PUSH_HEIGHT_CM, delta=0.3)
        for a, b in ((plan.start, plan.middle), (plan.middle, plan.end)):
            for s in np.linspace(0, 1, 21):
                pose = {joint: a[joint] + (b[joint] - a[joint]) * s for joint in a}
                tcp = self.kinematics.tcp(pose) * 100
                self.assertAlmostEqual(tcp[2] - placement.table_height(*tcp[:2]), PUSH_HEIGHT_CM, delta=1.0,
                                       msg="level all the way through")
        fingers, _ = self.kinematics.pointing(plan.start)
        self.assertLess(fingers[2], -0.7, "fingers down")

    def test_it_falls_towards_the_middle_when_an_arm_can_push_that_way(self) -> None:
        # Near the zone's left edge, left of both arms' reach towards it: a push
        # from the right arm or a pull by the left sends it rightwards, inwards.
        plan, why = self.plan((-8.0, 0.0))
        self.assertIsNotNone(plan, why)
        self.assertGreater(plan.direction[0], 0.3, "inwards")

    def test_the_jaw_shuts_before_it_comes_down_and_it_ends_in_neutral(self) -> None:
        plan, _ = self.plan((0.0, 0.0))
        moves: list[dict] = []
        arm = type("A", (), {"name": plan.arm, "move": lambda self, targets, speed=None: moves.append(dict(targets))})()
        neutral = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -90.0, "wrist_flex": 0.0, "wrist_roll": 0.0}
        execute_knock(arm, plan, neutral, log=lambda *_: None)
        self.assertEqual(moves[0], {"gripper": 0.0})
        self.assertEqual(moves[1:], [plan.above, plan.start, plan.middle, plan.end, plan.after, neutral])


if __name__ == "__main__":
    unittest.main()
