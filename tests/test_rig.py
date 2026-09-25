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
from trashdrop.rig import MOVE_TICKS, ArmDevices, MotionMeter, Rig, first_moved, load_rig, save_rig
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
        rig = Rig(overhead="046d:08e5")
        rig.arms["left"] = ArmDevices(bus="5AAF220303", camera="2993:0858", label="F01", max_speed=20.0)
        rig.arms["right"] = ArmDevices(bus="5AAF219965", camera=None, label="F02")
        with tempfile.TemporaryDirectory() as directory:
            path = save_rig(rig, Path(directory) / "rig.toml")
            loaded = load_rig(path)
        self.assertEqual(loaded, rig)
        self.assertEqual(loaded.cameras(), {"overhead": "046d:08e5", "left wrist": "2993:0858"})

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

    def test_the_camera_that_swung_with_the_arm_is_picked(self) -> None:
        meter = MotionMeter({"2993:0858": None, "10bb:2b08": None})
        meter.peak = {"2993:0858": 41.0, "10bb:2b08": 7.5}
        self.assertEqual(meter.mover(), "2993:0858")

    def test_two_cameras_that_changed_alike_are_not_guessed_between(self) -> None:
        meter = MotionMeter({"2993:0858": None, "10bb:2b08": None})
        meter.peak = {"2993:0858": 20.0, "10bb:2b08": 17.0}
        self.assertIsNone(meter.mover())


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
