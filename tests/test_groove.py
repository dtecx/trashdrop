"""The merch dance: four hand gestures, both arms as a person's two arms.

What is pinned down, on the model at the venue's placements and on fake
arms: it starts where the arms stand and ends in neutral, never jumping;
each move does what the team's photos show; the arms keep clear of each
other and of the table; no joint is asked for more than the servos manage
or past the venue arms' limits; and every tick writes both arms.
"""

from __future__ import annotations

import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from tests.test_arm import FakeBus, FakeClock
from trashdrop.arm import Arm, Stopped
from trashdrop.groove import (
    LEAD_BEATS,
    MAX_JOINT_SPEED,
    MIN_ABOVE_TABLE_CM,
    MIN_APART_CM,
    ORDER,
    build,
    fastest,
    outside,
    perform,
)
from trashdrop.servo import MOTORS

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
UPRIGHT = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -90.0, "wrist_flex": 0.0, "gripper": 0.0}
NEUTRAL = {"left": dict(UPRIGHT, wrist_roll=0.0), "right": dict(UPRIGHT, wrist_roll=-90.0)}  # poses.toml
# The venue's servo limits in ticks, read off the arms on 2026-09-25.
VENUE_LIMITS = {
    "left": {"shoulder_pan": (836, 2794), "shoulder_lift": (1122, 3504), "elbow_flex": (1540, 3609),
             "wrist_flex": (961, 3313), "wrist_roll": (0, 4095), "gripper": (2044, 3499)},
    "right": {"shoulder_pan": (778, 3370), "shoulder_lift": (863, 3199), "elbow_flex": (954, 3171),
              "wrist_flex": (809, 3167), "wrist_roll": (86, 3878), "gripper": (2017, 3502)},
}


def venue_arm(name: str, clock: FakeClock | None = None, stop=None) -> tuple[Arm, FakeBus]:
    bus = FakeBus()
    for joint, (low, high) in VENUE_LIMITS[name].items():
        bus.regs[MOTORS[joint]].update(min_limit=low, max_limit=high)
    clock = clock or FakeClock()
    return Arm(name, bus, 45.0, clock=clock, sleep=clock.sleep, stop=stop), bus


class RoutineTests(unittest.TestCase):
    def test_it_starts_where_the_arms_stand_and_ends_in_neutral_without_a_jump(self) -> None:
        start = {arm: dict(pose, shoulder_pan=3.0) for arm, pose in NEUTRAL.items()}
        routine = build(start, NEUTRAL)
        self.assertEqual(routine.at(0.0), start)
        self.assertEqual(routine.at(routine.total), NEUTRAL)
        ticks = [routine.at(t) for t in np.arange(0.0, routine.total, 0.02)]
        biggest = max(abs(b[arm][joint] - a[arm][joint]) for a, b in zip(ticks, ticks[1:]) for arm in a for joint in a[arm])
        self.assertLess(biggest, 3.5, "a few degrees (or percent) at most between two ticks")

    def test_every_move_in_order_each_eased_into(self) -> None:
        routine = build(NEUTRAL, NEUTRAL, reps=2, bpm=60)
        self.assertEqual(routine.labels, ["start"] + [name for name in ORDER for _ in range(4)] + ["neutral"])
        firsts = [index for index, label in enumerate(routine.labels[1:]) if routine.labels[index] != label]
        self.assertTrue(all(routine.seconds[index] == LEAD_BEATS for index in firsts), "two beats into each move")

    def test_the_servos_keep_up_and_a_beat_too_fast_is_caught(self) -> None:
        speed, _ = fastest(build(NEUTRAL, NEUTRAL))
        self.assertLess(speed, MAX_JOINT_SPEED)
        speed, where = fastest(build(NEUTRAL, NEUTRAL, bpm=120))
        self.assertGreater(speed, MAX_JOINT_SPEED)
        self.assertIn("claws", where)

    def test_nothing_past_the_venue_arms_limits(self) -> None:
        limits = {name: venue_arm(name)[0].limits_degrees() for name in NEUTRAL}
        self.assertEqual(outside(build(NEUTRAL, NEUTRAL, trims={"left": 7.8, "right": -7.8}), limits), [])
        turned_too_far = outside(build(NEUTRAL, NEUTRAL, trims={"left": -20.0, "right": 0.0}), limits)
        self.assertEqual(len(turned_too_far), 1)
        self.assertIn("left shoulder_pan", turned_too_far[0])

    def test_unknown_moves_and_silly_numbers_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            build(NEUTRAL, NEUTRAL, moves=("twerk",))
        with self.assertRaises(ValueError):
            build(NEUTRAL, NEUTRAL, bpm=0)


@unittest.skipUnless(HAS_MUJOCO, "needs the simulation extra")
class ModelTests(unittest.TestCase):
    """On the model, where the venue's arms stand (rig.toml, 2026-09-26)."""

    @classmethod
    def setUpClass(cls) -> None:
        from trashdrop.groove import Body, facing
        from trashdrop.kinematics import Kinematics
        from trashdrop.placement import Placement

        cls.bodies = {
            "left": Body(Kinematics(-80.0), Placement(11.34, -22.40, -95.14, -0.36, -0.0318, 0.0617)),
            "right": Body(Kinematics(5.0), Placement(10.98, 19.12, -79.47, -1.02, -0.0315, -0.0252)),
        }
        cls.trims, heading = facing(cls.bodies, NEUTRAL)
        cls.ahead = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading))])

    def poses(self, move: str) -> list[dict[str, dict[str, float]]]:
        return build(NEUTRAL, NEUTRAL, moves=(move,), reps=1, trims=self.trims).keyframes[1:3]

    def fingers(self, arm: str, pose: dict[str, float]) -> np.ndarray:
        """Where the fingers point, on the table's frame (z up)."""

        body = self.bodies[arm]
        way, _ = body.kinematics.pointing(pose)
        angle = np.radians(body.placement.yaw)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return np.array([*(rotation.T @ way[:2]), way[2]])

    def test_both_bases_turn_to_face_the_same_way(self) -> None:
        from trashdrop.groove import _wrap

        self.assertAlmostEqual(self.trims["left"], -self.trims["right"], delta=0.5)
        headings = [body.heading(dict(NEUTRAL[arm], shoulder_lift=45.0,
                                      shoulder_pan=NEUTRAL[arm]["shoulder_pan"] + self.trims[arm]))
                    for arm, body in self.bodies.items()]
        self.assertLess(abs(_wrap(headings[0] - headings[1])), 0.5)

    def test_claws_open_wide_then_pinch_bent_forward(self) -> None:
        spread, pinched = self.poses("claws")
        for arm in NEUTRAL:
            self.assertGreater(spread[arm]["gripper"], 50.0)
            self.assertEqual(pinched[arm]["gripper"], 0.0)
            self.assertGreater(self.fingers(arm, pinched[arm])[:2] @ self.ahead, 0.3, "the hand curls forward")

    def test_point_both_hands_level_the_same_way_then_the_other(self) -> None:
        first, second = self.poses("point")
        ways = []
        for pose in (first, second):
            flat = {arm: self.fingers(arm, pose[arm]) for arm in NEUTRAL}
            for arm, way in flat.items():
                self.assertLess(abs(way[2]), 0.25, f"{arm}: the hand about level")
                sideways = way[:2] / np.linalg.norm(way[:2])
                self.assertLess(abs(sideways @ self.ahead), 0.42, f"{arm}: pointing to the side")
            left, right = (flat[arm][:2] / np.linalg.norm(flat[arm][:2]) for arm in ("left", "right"))
            self.assertGreater(left @ right, np.cos(np.radians(5.0)), "both hands the same way")
            ways.append(left)
        self.assertLess(ways[0] @ ways[1], -0.9, "then the other way")

    def test_raise_one_arm_up_the_other_down_then_swapped(self) -> None:
        for pose, (down, up) in zip(self.poses("raise"), (("left", "right"), ("right", "left"))):
            tips = {arm: self.bodies[arm].points(pose[arm])[1][6] for arm in NEUTRAL}
            self.assertLess(tips[down], tips[up] - 20.0, f"{down} lowered, {up} up")
            self.assertGreater(pose[up]["gripper"], 50.0, "the raised hand open")

    def test_twist_turns_the_open_hands_mirrored(self) -> None:
        turns = []
        for pose in self.poses("twist"):
            left = pose["left"]["wrist_roll"] - NEUTRAL["left"]["wrist_roll"]
            right = pose["right"]["wrist_roll"] - NEUTRAL["right"]["wrist_roll"]
            self.assertNotEqual(left, 0.0)
            self.assertEqual(left, -right, "mirrored")
            self.assertTrue(all(pose[arm]["gripper"] > 50.0 for arm in NEUTRAL))
            turns.append(left)
        self.assertEqual(turns[0], -turns[1], "back and forth")

    def test_the_arms_keep_clear_of_each_other_and_of_the_table(self) -> None:
        from trashdrop.groove import check

        apart, above = check(build(NEUTRAL, NEUTRAL, reps=1, trims=self.trims), self.bodies, step_s=0.1)
        self.assertGreaterEqual(apart, MIN_APART_CM)
        self.assertGreaterEqual(above, MIN_ABOVE_TABLE_CM)

    def test_a_storyboard_of_every_pose(self) -> None:
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("needs the dataset extra")
        import cv2

        from trashdrop.groove import storyboard

        with tempfile.TemporaryDirectory() as folder:
            path = storyboard(self.bodies, NEUTRAL, self.trims, 0.0, ORDER, Path(folder) / "groove.png")
            self.assertEqual(cv2.imread(str(path)).shape, (4 * 250 + 30, 4 * 330, 3))


class PerformTests(unittest.TestCase):
    def arms(self, clock: FakeClock, stop=None) -> tuple[dict[str, Arm], dict[str, FakeBus]]:
        arms, buses = {}, {}
        for name in NEUTRAL:
            arms[name], buses[name] = venue_arm(name, clock, stop)
            arms[name].torque_on()
        return arms, buses

    def test_both_arms_every_tick_from_where_they_stand_to_neutral(self) -> None:
        clock = FakeClock()
        arms, buses = self.arms(clock)
        start = {name: arm.pose() for name, arm in arms.items()}
        routine = build(start, NEUTRAL, moves=("claws", "twist"), reps=1, bpm=120)
        before = {name: len(bus.goals_streamed()) for name, bus in buses.items()}
        perform(arms, routine, log=lambda *_: None, clock=clock, sleep=clock.sleep)
        streamed = {name: bus.goals_streamed()[before[name]:] for name, bus in buses.items()}
        self.assertEqual(len(streamed["left"]), len(streamed["right"]), "one goal each, every tick")
        self.assertGreater(len(streamed["left"]), routine.total * 50 * 0.9)
        for name, arm in arms.items():
            ticks = {pose: {MOTORS[joint]: arm.to_ticks(joint, value) for joint, value in values.items()}
                     for pose, values in (("start", start[name]), ("neutral", NEUTRAL[name]))}
            self.assertEqual(streamed[name][0], ticks["start"], f"{name}: no jump at the start")
            self.assertEqual(streamed[name][-1], ticks["neutral"], f"{name}: neutral at the end")
            self.assertEqual(arm.max_speed, 45.0, "the usual speed limit is back")

    def test_a_stop_holds_both_where_they_are(self) -> None:
        stop, clock = threading.Event(), FakeClock()
        arms, buses = self.arms(clock, stop)
        routine = build({name: arm.pose() for name, arm in arms.items()}, NEUTRAL, reps=1)
        sleep = clock.sleep

        def pressed_after_three_seconds(seconds: float) -> None:
            sleep(seconds)
            if clock.now > 3.0:
                stop.set()

        with self.assertRaises(Stopped):
            perform(arms, routine, log=lambda *_: None, clock=clock, sleep=pressed_after_three_seconds)
        for bus in buses.values():
            present = bus.positions()
            self.assertEqual(bus.goals_streamed()[-1], {MOTORS[joint]: ticks for joint, ticks in present.items()})


if __name__ == "__main__":
    unittest.main()
