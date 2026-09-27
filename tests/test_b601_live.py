"""The B601 LIVE gate, hand clutch, park pose and per-joint speed cap."""

import json
import math
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from trashdrop.b601 import B601HandMotion
from trashdrop.b601_glasses import B601GlassesBridge
from trashdrop.b601_motor import GRIP_SPEED, JOINT_SPEED, sleep_pose, velocity_step
from trashdrop.web.b601 import B601WebControl


def packet(*, pinch: bool, x: float = 0.0, y: float = 0.0,
           middle_touch: bool = False, twist_deg: float = 0.0) -> dict:
    angle = math.radians(twist_deg)
    across_x, across_z = math.cos(angle), math.sin(angle)
    return {"head": {"p": [0, 0, 0], "look": [0, 0, 1]},
            "right": {"tracked": True, "pinch": pinch,
                      "thumb": [x, y, 20], "index": [x + (1 if pinch else 6), y, 20],
                      "middleTip": [x + (1 if middle_touch else 5), y, 20],
                      "wrist": [x, y - 1, 20],
                      "indexKnuckle": [x + across_x, y, 20 + across_z],
                      "middleKnuckle": [x, y, 20],
                      "pinkyKnuckle": [x - across_x, y, 20 - across_z]}}


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

    def test_lifting_is_palm_translation_without_orientation_noise(self) -> None:
        motion = B601HandMotion(scale=1.0)
        motion.update(packet(pinch=False), 0)
        motion.update(packet(pinch=True), 0)
        lift = motion.update(packet(pinch=True, y=3, twist_deg=30), 0)
        self.assertAlmostEqual(lift[0][2], 0.03)
        np.testing.assert_allclose(lift[1], np.eye(3), atol=1e-8)

    def test_middle_touch_rotates_without_translating(self) -> None:
        motion = B601HandMotion(scale=1.0)
        motion.update(packet(pinch=False), 0, now=0.0)
        self.assertIsNone(motion.update(packet(pinch=False, middle_touch=True), 0, now=0.1))
        motion.update(packet(pinch=False, middle_touch=True), 0, now=0.3)
        turn = motion.update(packet(pinch=False, x=4, middle_touch=True, twist_deg=10), 0, now=0.4)
        np.testing.assert_allclose(turn[0], np.zeros(3))
        self.assertFalse(np.allclose(turn[1], np.eye(3)))

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
        self.assertEqual(bridge.mode, "preview")
        self.assertIn("motor enabling is locked", bridge.error)

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

    def test_web_b601_mode_keeps_so101_disconnected_and_requires_parking_to_exit(self) -> None:
        class FakeCell:
            version = 0

        web = B601WebControl(FakeCell(), live_allowed=True, sdk_root=Path("/unused"))
        web.bridge._report()
        self.assertFalse(web.state()["presentation"])
        self.assertEqual(web.configure(target="so101"), "SO-101 arms are unplugged; select B601")
        self.assertIn("glasses' B601 button", web.command({"command": "b601_live"}))
        self.assertEqual(web.bridge.hands.take_commands(), [])
        web.bridge.hands.connected = True
        self.assertIsNone(web.set_presentation(True))
        web.bridge.driver = object()
        self.assertIn("Park B601", web.set_presentation(False))
        self.assertTrue(web.state()["presentation"])

    def test_dry_run_b601_button_previews_without_enabling_motors(self) -> None:
        bridge = B601GlassesBridge(live_allowed=False, sdk_root=Path("/unused"))
        bridge._command({"command": "b601_live", "enabled": True})
        bridge._report()
        self.assertEqual(bridge.mode, "preview")
        self.assertIsNone(bridge.driver)
        self.assertIn("preview only", bridge.hands.status)

    def test_b601_cannot_enable_while_glasses_ui_is_hidden(self) -> None:
        bridge = B601GlassesBridge(live_allowed=True, sdk_root=Path("/unused"))
        bridge.hands.presentation = False
        bridge._command({"command": "b601_live", "enabled": True})
        self.assertIsNone(bridge.driver)
        self.assertIn("Enter Spectacles UI", bridge.error)

    def test_web_parser_has_explicit_camera_only_b601_mode(self) -> None:
        from trashdrop.__main__ import build_parser

        args = build_parser().parse_args(["web", "--b601", "--open"])
        self.assertTrue(args.b601)
        self.assertFalse(args.dry_run)

    def test_camera_only_cell_refuses_sorting_without_so101(self) -> None:
        from trashdrop.cell import Cell

        cell = Cell.__new__(Cell)
        cell.b601_only = True
        self.assertIn("SO-101 arms are unplugged", cell.begin("auto"))

    def test_camera_only_cell_starts_without_opening_so101_adapters(self) -> None:
        from trashdrop.cell import Cell

        class FakeCapture:
            def read(self):
                time.sleep(0.01)
                return True, np.zeros((480, 640, 3), dtype=np.uint8)

            def release(self):
                pass

        with patch("trashdrop.__main__._resolve_camera", return_value=0), \
                patch("trashdrop.dataset.capture.open_camera", return_value=FakeCapture()):
            cell = Cell.open_camera_only()
            try:
                cell.start()
                self.assertTrue(cell.b601_only)
                self.assertEqual(cell.arms, {})
                self.assertEqual(cell.state()["scene"]["size"], [640, 480])
            finally:
                cell.close()


if __name__ == "__main__":
    unittest.main()
