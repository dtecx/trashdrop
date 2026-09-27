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
        self.assertAlmostEqual(math.degrees(pose[3]), -0.159, places=3)

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

            def grip_fraction(self):
                return 0.0

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

            def log(self, text):
                pass

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


def touching(*, x: float = 0.0, pinky: bool = False) -> dict:
    """The right hand, index well apart, thumb touching the middle tip or (``pinky``) the pinky tip.

    Looking along +z, as ``packet`` does, the wearer's right is -x.
    """

    message = packet(pinch=False, x=x, middle_touch=not pinky)
    hand = message["right"]
    hand["middleTip"] = [x + (1 if not pinky else 5), 0, 20]
    hand["pinkyTip"] = [x + (1 if pinky else 6), 0, 20]
    return message


class B601ControlTests(unittest.TestCase):
    """The pinch control made for the B601 after the 14:18 session: no range envelope, one speed cap."""

    def test_a_step_keeps_its_direction_when_the_speed_cap_shrinks_it(self) -> None:
        from trashdrop.b601_motor import dls_step

        q = np.zeros(6)
        jacobian = np.diag([1.0, 0.5, 0.25, 1.0, 1.0, 1.0])  # one metre a radian on x, half on y...
        error = np.array([0.05, 0.05, 0.0, 0, 0, 0])
        moved, pinned = dls_step(q, error, jacobian, np.full(6, -3.0), np.full(6, 3.0), 0.05)
        self.assertEqual(pinned, [])
        self.assertAlmostEqual(np.max(np.abs(moved)), JOINT_SPEED * 0.05)  # the cap, on the fastest joint
        self.assertAlmostEqual(moved[1] / moved[0], 2.0, delta=0.01)  # joint by joint it would be 1; damping: 1.998

    def test_a_target_out_of_reach_slides_along_a_limit_instead_of_stopping(self) -> None:
        from trashdrop.b601_motor import LIMIT_MARGIN, dls_step

        q = np.array([0.5 - LIMIT_MARGIN, 0, 0, 0, 0, 0])
        error = np.array([0.05, 0.05, 0, 0, 0, 0])
        moved, pinned = dls_step(q, error, np.eye(6), np.full(6, -0.5), np.full(6, 0.5), 0.05)
        self.assertEqual(pinned, [0])
        self.assertAlmostEqual(moved[0], q[0])  # held at the limit...
        self.assertGreater(moved[1], 0.0)  # ...while the other joint still goes

    def test_a_joint_starting_past_its_stop_may_only_come_back(self) -> None:
        from trashdrop.b601_motor import dls_step

        q = np.array([-0.012, 0, 0, 0, 0, 0])  # the park pose: J2 at -0.69 degrees, its limit 0
        down, pinned = dls_step(q, np.array([-0.05, 0, 0, 0, 0, 0]), np.eye(6), np.zeros(6), np.ones(6), 0.05)
        self.assertEqual((down[0], pinned), (q[0], [0]))
        up, _ = dls_step(q, np.array([0.05, 0, 0, 0, 0, 0]), np.eye(6), np.zeros(6), np.ones(6), 0.05)
        self.assertGreater(up[0], q[0])

    def test_thumb_middle_and_a_move_right_turn_the_tool_clockwise_from_above(self) -> None:
        motion = B601HandMotion(scale=1.0)
        motion.update(packet(pinch=False), 0, now=0.0)
        motion.update(touching(), 0, now=0.1)
        motion.update(touching(), 0, now=0.3)
        delta, rotation = motion.update(touching(x=-4.0), 0, now=0.4)  # 4 cm to the wearer's right
        np.testing.assert_allclose(delta, np.zeros(3))
        self.assertAlmostEqual(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])), -20.0)
        np.testing.assert_allclose(rotation[:, 2], [0, 0, 1], atol=1e-12)  # about the vertical only

    def test_thumb_pinky_and_a_sideways_move_set_the_jaw_while_the_arm_holds(self) -> None:
        from trashdrop.b601 import JAW_PER_CM

        motion = B601HandMotion(scale=1.0)
        motion.update(packet(pinch=False), 0, now=0.0)
        self.assertIsNone(motion.update(touching(pinky=True), 0, now=0.1))  # not held long enough yet
        motion.update(touching(pinky=True), 0, now=0.3)
        delta, rotation = motion.update(touching(pinky=True, x=3.0), 0, now=0.4)  # 3 cm left: opens
        np.testing.assert_allclose(delta, np.zeros(3))
        np.testing.assert_allclose(rotation, np.eye(3))
        self.assertAlmostEqual(motion.jaw, 3 * JAW_PER_CM)
        motion.update(packet(pinch=False), 0, now=0.5)  # let go: it stays
        self.assertAlmostEqual(motion.jaw, 3 * JAW_PER_CM)
        motion.update(touching(pinky=True, x=10.0), 0, now=0.6)
        motion.update(touching(pinky=True, x=10.0), 0, now=0.8)
        motion.update(touching(pinky=True, x=-20.0), 0, now=0.9)  # far right: shut, and no further
        self.assertEqual(motion.jaw, 0.0)

    def test_either_hand_drives_whichever_starts_a_gesture(self) -> None:
        def left(message: dict) -> dict:
            return {"head": message["head"], "left": message["right"]}

        motion = B601HandMotion(scale=1.0)
        self.assertIsNone(motion.update(left(packet(pinch=False)), 0))
        self.assertEqual(motion.side, "left")
        motion.update(left(packet(pinch=True)), 0)
        moved = motion.update(left(packet(pinch=True, y=2)), 0)
        self.assertAlmostEqual(moved[0][2], 0.02)
        both = {**packet(pinch=True, y=5), "left": left(packet(pinch=True, y=4))["left"]}
        self.assertAlmostEqual(motion.update(both, 0)[0][2], 0.04, msg="mid-gesture the other hand cannot take over")

    def test_the_gripper_range_comes_from_the_park_file_once_measured(self) -> None:
        import tempfile

        from trashdrop.b601_motor import GRIP_ENVELOPE, gripper_range

        with tempfile.TemporaryDirectory() as folder:
            park = Path(folder) / "park.toml"
            park.write_text("joint_degrees = [0, 0, 0, 0, 0, 0, 0]\n")
            self.assertEqual(gripper_range(park, 0.25), (0.25, 0.25 + GRIP_ENVELOPE))
            park.write_text("joint_degrees = [0, 0, 0, 0, 0, 0, 0]\ngripper_closed_degrees = 10\n"
                            "gripper_open_degrees = -80\n")
            closed, opened = gripper_range(park, 0.25)
            self.assertAlmostEqual(math.degrees(closed), 10)
            self.assertAlmostEqual(math.degrees(opened), -80)

    def test_the_measured_gripper_travel_is_about_one_motor_turn(self) -> None:
        from trashdrop.b601_motor import gripper_range

        closed, opened = gripper_range(Path(__file__).resolve().parents[1] / "b601_park.toml", 0.0)
        self.assertAlmostEqual(math.degrees(closed), 1.707)
        self.assertGreater(math.degrees(opened - closed), 300)

    def test_closing_on_an_item_squeezes_no_harder_than_the_limit(self) -> None:
        from trashdrop.b601_motor import GRIP_SQUEEZE, grip_step

        free, velocity = grip_step(1.0, 0.9, 0.95, 0.05)  # moving freely: as stepped, with its velocity
        self.assertAlmostEqual(free, 0.9)
        self.assertAlmostEqual(velocity, -2.0)
        blocked, velocity = grip_step(1.0 - GRIP_SQUEEZE, 0.2, 1.0, 0.05)  # the jaw stopped on an item at 1.0
        self.assertAlmostEqual(blocked, 1.0 - GRIP_SQUEEZE)
        self.assertAlmostEqual(velocity, 0.0)  # held there: no feed-forward pushing on

    def test_a_can_fault_holds_the_arm_and_never_disables_the_motors(self) -> None:
        # 14:55: one position read timed out, the bridge closed the driver, all seven motors went
        # limp and the arm fell.
        class FlakyMotor:
            state = "following"
            feedback = np.zeros(7)
            closed = held = 0

            def step(self):
                raise RuntimeError("robstride_get_param_f32_host_id failed")

            def hold(self):
                self.held += 1

            def close(self):
                self.closed += 1

            def grip_fraction(self):
                return 0.4

        bridge = B601GlassesBridge(live_allowed=True, sdk_root=Path("/unused"))
        motor = FlakyMotor()
        bridge.driver, bridge.mode = motor, "live"
        bridge.tick()
        self.assertIs(bridge.driver, motor)
        self.assertEqual(motor.closed, 0)
        self.assertGreater(motor.held, 0)
        self.assertIn("holding", bridge.error)
        self.assertEqual(bridge.motion.jaw, 0.4)  # the jaw is not dropped either

    def test_the_jaw_frame_levels_its_fingers_and_reads_back(self) -> None:
        from trashdrop.b601 import heading_and_pitch, jaw_frame

        np.testing.assert_allclose(jaw_frame(0, 0), np.eye(3), atol=1e-12)  # the URDF zero pose's tool
        down = jaw_frame(30, -90)
        np.testing.assert_allclose(down[:, 0], [0, 0, -1], atol=1e-12)
        for heading, pitch in ((30, -90), (-45, -20), (120, 0)):
            frame = jaw_frame(heading, pitch)
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(frame[2, 1], 0.0)  # the fingers' axis stays level
            np.testing.assert_allclose(heading_and_pitch(frame), (heading, pitch), atol=1e-9)

    def test_while_dragging_the_jaw_points_as_far_down_as_the_hand(self) -> None:
        from trashdrop.b601 import hand_pitch

        level = {"wrist": [0, 0, 20], "middleKnuckle": [0, 0, 28]}
        bent = {"wrist": [0, 0, 20], "middleKnuckle": [0, -8, 20.0]}  # the Lens world is y up
        self.assertAlmostEqual(hand_pitch(level), 0.0)
        self.assertAlmostEqual(hand_pitch(bent), -90.0)
        motion = B601HandMotion(scale=1.0)
        motion.update(packet(pinch=False), 0)
        motion.update(packet(pinch=True), 0)
        start = motion.pitch
        tilted = packet(pinch=True)
        tilted["right"]["middleKnuckle"] = [0, -8, 20]
        tilted["right"]["wrist"] = [0, 0, 20]
        for _ in range(30):
            motion.update(tilted, 0)
        self.assertNotEqual(start, motion.pitch)
        self.assertAlmostEqual(motion.pitch, -90.0, delta=1.0)

    def test_letting_go_of_thumb_and_pinky_stops_the_jaw_where_it_is(self) -> None:
        class Motion:
            gesture, jaw, anchor_generation, state = "free", 0.9, 0, "free"

            def update(self, packet, age):
                return None

        class Motor:
            state = "holding"
            feedback = np.zeros(7)
            set_to = None

            def hold(self):
                pass

            def grip_fraction(self, *, now=False):
                return 0.35 if now else 0.9

            def set_grip_fraction(self, fraction):
                self.set_to = fraction

            def step(self):
                return False

        bridge = B601GlassesBridge(live_allowed=True, sdk_root=Path("/unused"))
        bridge.driver, bridge.mode, bridge.motion, bridge.gripping = Motor(), "live", Motion(), True
        bridge.hands.connected = True
        bridge.tick()
        self.assertEqual(bridge.motion.jaw, 0.35)  # where it was being driven, not on to 0.9
        self.assertEqual(bridge.driver.set_to, 0.35)

    def test_web_b601_carries_on_in_the_sdk_environment_without_motorbridge(self) -> None:
        import argparse
        import os
        import tempfile

        from trashdrop import __main__ as cli

        with tempfile.TemporaryDirectory() as folder:
            python = Path(folder) / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            args = argparse.Namespace(camera="auto")
            with patch("trashdrop.b601_motor.B601_SDK", Path(folder)), \
                    patch.object(cli, "_resolve_camera", return_value=0), \
                    patch.object(cli.sys, "argv", ["trashdrop", "web", "--b601", "--open"]), \
                    patch.dict(os.environ, {}, clear=False), \
                    patch("os.execve") as execve:
                os.environ.pop("TRASHDROP_IN_B601_SDK", None)
                cli._web_in_b601_sdk(args)
        program, argv, env = execve.call_args.args
        self.assertEqual(program, str(python))
        self.assertEqual(argv[1:], ["-m", "trashdrop", "web", "--b601", "--open", "--camera", "0"])
        self.assertIn("PCBUSB", env["DYLD_LIBRARY_PATH"])
        self.assertEqual(env["TRASHDROP_IN_B601_SDK"], "1")  # and so it cannot loop


if __name__ == "__main__":
    unittest.main()
