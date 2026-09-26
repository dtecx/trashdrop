"""Telling the two arms and their wrist cameras apart, and naming them in rig.toml.

No hardware: servo buses are fakes that report joint positions, cameras are
peaks of picture change. What is pinned down is the procedure a person runs at
the venue -- move the front arm by hand, and the right bus and camera get the
right name.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trashdrop.camera.uvc import UvcCamera
from trashdrop.rig import MOVE_TICKS, ArmDevices, PickZone, Rig, first_moved, load_rig, save_rig, wait_until_still
from trashdrop.servo import decode_offset


class FakeBus:
    """Joint positions; ``moves`` = {poll number: (joint, ticks)}."""

    def __init__(self, moves=None) -> None:
        self.polls = 0
        self.moves = moves or {}
        self.current = {"shoulder_pan": 2048, "elbow_flex": 2048}

    def positions(self) -> dict[str, int]:
        self.polls += 1
        if self.polls in self.moves:
            joint, ticks = self.moves[self.polls]
            self.current[joint] += ticks
        return dict(self.current)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class RigFileTests(unittest.TestCase):
    def test_rig_toml_round_trips(self) -> None:
        rig = Rig(overhead="046d:08e5", pick_zone=PickZone(x=2.0, y=-3.0, width=40.0, height=25.0))
        rig.arms["left"] = ArmDevices(bus="5AAF220303", camera="2993:0858", label="F01",
                                      max_speed=20.0, descent_speed=12.0)
        rig.arms["right"] = ArmDevices(bus="5AAF219965", camera=None, label="F02")
        with tempfile.TemporaryDirectory() as directory:
            path = save_rig(rig, Path(directory) / "rig.toml")
            loaded = load_rig(path)
        self.assertEqual(loaded, rig)
        self.assertEqual(loaded.cameras(), {"overhead": "046d:08e5", "left wrist": "2993:0858"})

    def test_older_rig_file_uses_default_descent_speed(self) -> None:
        rig = Rig()
        with tempfile.TemporaryDirectory() as directory:
            path = save_rig(rig, Path(directory) / "rig.toml")
            path.write_text("\n".join(line for line in path.read_text().splitlines()
                                      if not line.startswith("descent_speed =")) + "\n")
            loaded = load_rig(path)
        self.assertEqual(loaded.arms["left"].descent_speed, rig.arms["left"].descent_speed)

    def test_zone_corners_run_far_left_far_right_near_right_near_left(self) -> None:
        zone = PickZone(x=0.0, y=0.0, width=30.0, height=28.0)
        self.assertEqual(zone.corners(), [(-15.0, 14.0), (15.0, 14.0), (15.0, -14.0), (-15.0, -14.0)])
        self.assertEqual(zone.corners(inset=1.0)[0], (-14.0, 13.0), "inset keeps tape on the edge out")

    def test_tape_calibration_survives_the_file(self) -> None:
        rig = Rig()
        rig.arms["left"] = ArmDevices(bus="5AAF219965", label="F01",
                                      touches={"far_left": (32.9, -10.3, -2.5), "near_left": (17.4, -9.8, -1.3)})
        rig.tape_pixels = {"far_left": (705.5, 420.0), "near_right": (1180.2, 805.9)}
        rig.pick_zone = PickZone(polygon=((-16.0, 15.5), (15.0, 13.0), (14.0, -14.5), (-15.5, -12.0)))
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_rig(save_rig(rig, Path(directory) / "rig.toml"))
        self.assertEqual(loaded.arms["left"].touches, rig.arms["left"].touches)
        self.assertEqual(loaded.tape_pixels, rig.tape_pixels)
        self.assertEqual(loaded.pick_zone.corners(), [(-16.0, 15.5), (15.0, 13.0), (14.0, -14.5), (-15.5, -12.0)])

    def test_the_table_plane_survives_the_file_and_older_files_still_load(self) -> None:
        rig = Rig()
        rig.arms["left"] = ArmDevices(bus="5AAF219965", sheet=(19.12, -21.7, -95.36, 0.88, -0.095, 0.1037))
        with tempfile.TemporaryDirectory() as directory:
            path = save_rig(rig, Path(directory) / "rig.toml")
            self.assertEqual(load_rig(path).arms["left"].sheet, rig.arms["left"].sheet)
            path.write_text(path.read_text().replace(", dz_dx = -0.0950, dz_dy = 0.1037", ""))
            older = load_rig(path)
        self.assertEqual(older.arms["left"].sheet, (19.12, -21.7, -95.36, 0.88, 0.0, 0.0), "a level table")

    def test_the_wrist_zero_and_the_touch_poses_survive_the_file(self) -> None:
        rig = Rig()
        pose = {"shoulder_pan": 20.0, "shoulder_lift": 45.5, "elbow_flex": -35.0, "wrist_flex": 80.0, "wrist_roll": 5.2}
        rig.arms["left"] = ArmDevices(bus="5AAF219965", touches={"far_left": (33.84, -9.56, -3.33)},
                                      touch_poses={"far_left": pose}, wrist_roll_offset=90.0)
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_rig(save_rig(rig, Path(directory) / "rig.toml"))
        self.assertEqual(loaded.arms["left"].touch_poses, {"far_left": pose})
        self.assertEqual(loaded.arms["left"].wrist_roll_offset, 90.0)
        self.assertEqual(loaded.arms["right"].wrist_roll_offset, 0.0)

    def test_no_rig_toml_means_nothing_identified(self) -> None:
        rig = load_rig(Path("/nonexistent/rig.toml"))
        self.assertTrue(all(arm.bus is None for arm in rig.arms.values()))


class IdentifyTests(unittest.TestCase):
    def test_the_bus_whose_joint_is_pushed_is_found(self) -> None:
        clock = FakeClock()
        buses = {"still": FakeBus(), "pushed": FakeBus({4: ("elbow_flex", MOVE_TICKS + 20)})}
        found = first_moved(buses, 10.0, clock=clock, sleep=clock.sleep)
        self.assertEqual(found, ("pushed", "elbow_flex"))

    def test_servo_jitter_is_not_a_push(self) -> None:
        clock = FakeClock()
        buses = {"a": FakeBus({3: ("shoulder_pan", 6)}), "b": FakeBus({5: ("elbow_flex", -9)})}
        self.assertIsNone(first_moved(buses, 2.0, clock=clock, sleep=clock.sleep))

    def test_the_joint_that_moved_most_is_reported_not_a_nudged_neighbour(self) -> None:
        # Bending an elbow by hand drags the shoulder along a little.
        clock = FakeClock()
        bus = FakeBus({3: ("elbow_flex", MOVE_TICKS + 200)})
        bus.moves_extra = {3: ("shoulder_pan", MOVE_TICKS + 10)}
        original = bus.positions

        def positions():
            result = original()
            if bus.polls == 3:
                joint, ticks = bus.moves_extra[3]
                bus.current[joint] += ticks
                result[joint] = bus.current[joint]
            return result

        bus.positions = positions
        self.assertEqual(first_moved({"arm": bus}, 5.0, clock=clock, sleep=clock.sleep), ("arm", "elbow_flex"))

    def test_a_joint_already_identified_does_not_answer_the_next_question(self) -> None:
        # The person is still turning the base when asked for the shoulder.
        clock = FakeClock()
        bus = FakeBus({2: ("shoulder_pan", MOVE_TICKS + 300), 6: ("elbow_flex", MOVE_TICKS + 50)})
        found = first_moved({"arm": bus}, 5.0, ignore={("arm", "shoulder_pan")}, clock=clock, sleep=clock.sleep)
        self.assertEqual(found, ("arm", "elbow_flex"))

    def test_the_next_question_waits_for_the_arm_to_settle(self) -> None:
        clock = FakeClock()
        wobbling = FakeBus({n: ("elbow_flex", 40 if n % 2 else -40) for n in range(2, 12)})
        self.assertTrue(wait_until_still({"arm": wobbling}, calm=0.5, limit=5.0, clock=clock, sleep=clock.sleep))
        self.assertGreater(clock.now, 1.0, "it waited out the wobble before calling the arm still")


class CameraChoiceTests(unittest.TestCase):
    def test_a_lone_camera_needs_no_id(self) -> None:
        self.assertEqual(UvcCamera._only([(0x046D, 0x08E5)]), (0x046D, 0x08E5))

    def test_several_cameras_are_refused_rather_than_guessed(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            UvcCamera._only([(0x10BB, 0x2B08), (0x046D, 0x08E5), (0x2993, 0x0858)])
        self.assertIn("046d:08e5", str(caught.exception))
        self.assertIn("rig.toml", str(caught.exception))

    def test_no_camera_is_a_clear_error(self) -> None:
        with self.assertRaises(RuntimeError):
            UvcCamera._only([])


class ServoTests(unittest.TestCase):
    def test_homing_offset_is_sign_magnitude(self) -> None:
        self.assertEqual(decode_offset(1374), 1374)
        self.assertEqual(decode_offset(0x800 | 1277), -1277)


if __name__ == "__main__":
    unittest.main()
