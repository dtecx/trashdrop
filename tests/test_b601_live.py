"""The B601 LIVE gate, hand clutch, park pose and per-joint speed cap."""

import json
import math
import unittest
from pathlib import Path

import numpy as np

from trashdrop.b601 import B601HandMotion
from trashdrop.b601_glasses import B601GlassesBridge
from trashdrop.b601_motor import GRIP_SPEED, JOINT_SPEED, sleep_pose, velocity_step


def packet(*, pinch: bool, x: float = 0.0) -> dict:
    return {"head": {"p": [0, 0, 0], "look": [0, 0, 1]},
            "right": {"tracked": True, "pinch": pinch,
                      "thumb": [x, 0, 20], "index": [x + 1, 0, 20],
                      "wrist": [0, -1, 20], "indexKnuckle": [1, 0, 20],
                      "middleKnuckle": [0, 0, 20], "pinkyKnuckle": [-1, 0, 20]}}


class B601LiveTests(unittest.TestCase):
    def test_clutch_requires_release_and_loss_requires_another_release(self) -> None:
        motion = B601HandMotion(scale=0.5)
        self.assertIsNone(motion.update(packet(pinch=True), 0))
        self.assertIsNone(motion.update(packet(pinch=False), 0))
        first = motion.update(packet(pinch=True), 0)
        np.testing.assert_allclose(first[0], [0, 0, 0])
        moved = motion.update(packet(pinch=True, x=2), 0)
        np.testing.assert_allclose(np.linalg.norm(moved[0]), 0.01)
        np.testing.assert_allclose(moved[1], np.eye(3))
        self.assertIsNone(motion.update(packet(pinch=True, x=3), 0.5))
        self.assertIsNone(motion.update(packet(pinch=True, x=4), 0))
        self.assertIsNone(motion.update(packet(pinch=False), 0))
        self.assertIsNotNone(motion.update(packet(pinch=True), 0))

    def test_rate_limiter_caps_all_seven_motors_after_a_delayed_packet(self) -> None:
        start = np.zeros(7)
        target = np.ones(7)
        result = velocity_step(start, target, 0.5)
        np.testing.assert_allclose(result[:6], JOINT_SPEED * 0.05)
        self.assertAlmostEqual(result[6], GRIP_SPEED * 0.05)
        with self.assertRaises(ValueError):
            velocity_step(start, np.full(7, math.nan), 0.03)

    def test_sleep_pose_is_not_rewritten_as_a_motor_zero(self) -> None:
        pose = sleep_pose(Path(__file__).resolve().parents[1] / "b601_park.toml")
        self.assertEqual(pose.shape, (7,))
        self.assertAlmostEqual(math.degrees(pose[3]), -51.374, places=3)

    def test_live_button_is_locked_until_bridge_explicitly_allows_it(self) -> None:
        bridge = B601GlassesBridge(live_allowed=False, sdk_root=Path("/unused"))
        bridge._command({"command": "b601_live", "enabled": True})
        self.assertIsNone(bridge.driver)
        self.assertIn("--live", bridge.error)

    def test_hold_can_resume_and_neutral_parks_without_hiding_the_menu(self) -> None:
        class FakeMotor:
            state = "holding"
            feedback = np.zeros(7)

            def hold(self):
                self.state = "holding"

            def start_park(self):
                return True

        bridge = B601GlassesBridge(live_allowed=True, sdk_root=Path("/unused"))
        bridge.driver = FakeMotor()
        bridge.mode = "live"
        bridge._command({"command": "b601_live", "enabled": False})
        self.assertEqual(bridge.mode, "hold")
        bridge._report()
        self.assertFalse(json.loads(bridge.hands.report)["manual"])
        bridge._command({"command": "b601_live", "enabled": True})
        self.assertEqual(bridge.mode, "live")
        bridge._command({"command": "presentation", "enabled": False})
        self.assertTrue(bridge.hands.presentation)
        self.assertEqual(bridge.mode, "hold")
        bridge._command({"command": "neutral"})
        self.assertEqual(bridge.mode, "parking")


if __name__ == "__main__":
    unittest.main()
