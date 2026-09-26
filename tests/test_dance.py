"""Crab rave: two arms rocking one can between them, as one.

What is pinned down: the dance starts and ends exactly where the arms stood,
never jumping; every tick writes both arms, with the same offsets; the
grippers keep squeezing what they hold; the can stays level (on the model);
the servos' usual speed limits come back afterwards; and a stop holds both.
"""

from __future__ import annotations

import threading
import unittest

import numpy as np
import pytest

from tests.test_arm import FakeBus, FakeClock
from trashdrop.arm import Arm, Stopped
from trashdrop.dance import DANCE_SPEED, dance, offsets
from trashdrop.servo import MOTORS


class ChoreographyTests(unittest.TestCase):
    def test_it_starts_and_ends_where_the_arms_stood_without_a_jump(self) -> None:
        self.assertEqual(offsets(0.0, 20.0), {"shoulder_lift": 0.0, "wrist_flex": -0.0, "shoulder_pan": 0.0})
        self.assertTrue(all(abs(v) < 1e-9 for v in offsets(20.0, 20.0).values()))
        ticks = [offsets(t, 20.0, bpm=125, rock=8, sway=6) for t in np.arange(0, 20, 0.02)]
        steps = [max(abs(b[j] - a[j]) for j in a) for a, b in zip(ticks, ticks[1:])]
        self.assertLess(max(steps), 1.5, "no more than 1.5 degrees between two ticks, even at 125 bpm")

    def test_the_wrists_turn_against_the_shoulders(self) -> None:
        for t in np.arange(0, 20, 0.37):
            moved = offsets(t, 20.0)
            self.assertAlmostEqual(moved["wrist_flex"], -moved["shoulder_lift"])
            self.assertLessEqual(abs(moved["shoulder_lift"]), 5.0 + 1e-9)

    def test_the_can_stays_level_on_the_model(self) -> None:
        pytest.importorskip("mujoco", reason="needs the simulation extra")
        from trashdrop.kinematics import NEUTRAL, Kinematics

        kinematics = Kinematics(-80.0)
        start = dict(NEUTRAL, wrist_roll=120.0)
        for lean in (-8.0, 8.0):
            pose = dict(start, shoulder_lift=lean, wrist_flex=-lean)
            fingers, _ = kinematics.pointing(pose)
            self.assertGreater(fingers[2], 0.999, "the gripper still points straight up")
        forwards = kinematics.tcp(dict(start, shoulder_lift=8.0, wrist_flex=-8.0))
        self.assertGreater(forwards[0], kinematics.tcp(start)[0], "a positive lean rocks forwards")


class DanceTests(unittest.TestCase):
    def arms(self, clock, stop=None):
        buses = {"left": FakeBus(), "right": FakeBus()}
        arms = {name: Arm(name, bus, 45.0, clock=clock, sleep=clock.sleep, stop=stop) for name, bus in buses.items()}
        for arm in arms.values():
            arm.torque_on()
            arm.move({"gripper": 0.0})  # clamped on the can
        return arms, buses

    def test_both_arms_every_tick_and_back_where_they_started(self) -> None:
        clock = FakeClock()
        arms, buses = self.arms(clock)
        before = {name: len(bus.goals_streamed()) for name, bus in buses.items()}
        start = {name: bus.goals_streamed()[-1] for name, bus in buses.items()}
        dance(arms, seconds=6.0, bpm=60, rock=5, sway=3, log=lambda *_: None, clock=clock, sleep=clock.sleep)
        left, right = (bus.goals_streamed()[before[name]:] for name, bus in buses.items())
        self.assertEqual(len(left), len(right), "one goal each, every tick")
        self.assertGreater(len(left), 250)
        gripper = MOTORS["gripper"]
        self.assertTrue(all(goals[gripper] == start["left"][gripper] for goals in left), "still squeezing the can")
        for name, bus in buses.items():
            self.assertEqual(bus.goals_streamed()[-1], start[name], f"{name}: back where it stood")
        pan = MOTORS["shoulder_pan"]
        self.assertEqual([g[pan] - start["left"][pan] for g in left], [g[pan] - start["right"][pan] for g in right],
                         "the same moves for both")
        for arm in arms.values():
            self.assertEqual(arm.max_speed, 45.0, "the usual speed limit is back")
        self.assertNotEqual(DANCE_SPEED, 45.0)

    def test_a_stop_holds_both_where_they_are(self) -> None:
        stop, clock = threading.Event(), FakeClock()
        arms, buses = self.arms(clock, stop)
        sleep = clock.sleep

        def pressed_after_two_seconds(seconds):
            sleep(seconds)
            if clock.now > 2.0:
                stop.set()

        clock.sleep = pressed_after_two_seconds
        for arm in arms.values():
            arm._sleep = pressed_after_two_seconds
        with self.assertRaises(Stopped):
            dance(arms, seconds=10.0, log=lambda *_: None, clock=clock, sleep=pressed_after_two_seconds)
        for bus in buses.values():
            present = bus.positions()
            self.assertEqual(bus.goals_streamed()[-1], {MOTORS[j]: ticks for j, ticks in present.items()})

    def test_silly_numbers_move_nothing(self) -> None:
        clock = FakeClock()
        arms, buses = self.arms(clock)
        before = len(buses["left"].goals_streamed())
        with self.assertRaises(ValueError):
            dance(arms, seconds=10.0, rock=40.0, log=lambda *_: None, clock=clock, sleep=clock.sleep)
        self.assertEqual(len(buses["left"].goals_streamed()), before)


if __name__ == "__main__":
    unittest.main()
