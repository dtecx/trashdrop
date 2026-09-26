"""The merch dance: four hand gestures, both arms as a person's two arms.

What is pinned down, on the model at the venue's placements and on fake
arms: it starts where the arms stand and ends in neutral, never jumping;
each move does what the team's photos show, as many times and as fast as
groove.toml says; between moves both arms stand up straight; the arms keep
clear of each other and of the table, and
nothing goes behind their bases; no joint is asked for more than the
servos manage or past the venue arms' limits; and every tick writes both
arms.
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
    MAX_BEHIND_CM,
    MAX_JOINT_SPEED,
    MIN_ABOVE_TABLE_CM,
    MIN_APART_CM,
    ORDER,
    TRANSIT_SPEED,
    Part,
    build,
    fastest,
    load_parts,
    outside,
    perform,
    plan,
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
        routine = build(start, NEUTRAL, plan())
        self.assertEqual(routine.at(0.0), start)
        self.assertEqual(routine.at(routine.total), NEUTRAL)
        ticks = [routine.at(t) for t in np.arange(0.0, routine.total, 0.02)]
        biggest = max(abs(b[arm][joint] - a[arm][joint]) for a, b in zip(ticks, ticks[1:]) for arm in a for joint in a[arm])
        self.assertLessEqual(biggest, MAX_JOINT_SPEED * 0.02, "never more between two ticks than a servo turns")

    def test_every_move_in_order_each_eased_into_from_upright(self) -> None:
        trims = {"left": 7.8, "right": -7.8}
        routine = build(NEUTRAL, NEUTRAL, plan(reps=2), bpm=60, trims=trims)
        expected = ["start"]
        for number, name in enumerate(ORDER):
            expected += (["upright"] if number else []) + [name] * 4
        self.assertEqual(routine.labels, expected + ["neutral"])
        for index, label in enumerate(routine.labels[1:]):
            if label == "upright":
                upright = {arm: dict(NEUTRAL[arm], shoulder_pan=NEUTRAL[arm]["shoulder_pan"] + trims[arm])
                           for arm in NEUTRAL}
                self.assertEqual(routine.keyframes[index + 1], upright, "both straight up, facing the same way")

    def test_the_servos_keep_up_and_a_move_too_fast_is_caught_by_name(self) -> None:
        speed, _, _ = fastest(build(NEUTRAL, NEUTRAL, plan()))
        self.assertLess(speed, MAX_JOINT_SPEED)
        fast = [Part("jaws", 4, 1.0), Part("point", 4, 3.0), Part("twist", 4, 1.0)]
        speed, where, pace = fastest(build(NEUTRAL, NEUTRAL, fast))
        self.assertGreater(speed, MAX_JOINT_SPEED)
        self.assertEqual(pace, "point", "the move whose speed to turn down")
        self.assertIn("point", where)

    def test_the_most_each_move_can_speed_up_is_just_what_the_servos_manage(self) -> None:
        from trashdrop.groove import top_speed

        for part in plan(size=1.0) + plan(size=0.5):
            most = top_speed(part, NEUTRAL)
            if most < 4.0:  # not held back by MAX_SPEED
                peak, _, _ = fastest(build(NEUTRAL, NEUTRAL, [Part(part.move, 3, most, part.size)]))
                self.assertAlmostEqual(peak, MAX_JOINT_SPEED, delta=1.0, msg=part.move)
        self.assertGreater(top_speed(Part("twist", size=0.5), NEUTRAL), top_speed(Part("twist"), NEUTRAL),
                           "smaller goes faster")

    def test_each_move_as_many_times_and_as_fast_as_it_says(self) -> None:
        routine = build(NEUTRAL, NEUTRAL, [Part("jaws", 2, 1.0), Part("point", 3, 0.5)], bpm=60)
        self.assertEqual(routine.labels.count("jaws"), 2 * 2, "twice through both of its poses")
        self.assertEqual(routine.labels.count("point"), 3 * 2)
        steps = {label: [s for s, (l, pace) in zip(routine.seconds, zip(routine.labels[1:], routine.paces))
                         if l == label] for label in ("jaws", "point")}
        self.assertEqual(steps["jaws"][1:], [1.0] * 3, "a beat a pose at 60 bpm")
        self.assertEqual(steps["point"][1:], [4.0] * 5, "two beats a pose, half as fast")
        self.assertLess(steps["point"][0], 4.0, "the way in at the transit pace, not point's")

    def test_nothing_past_the_venue_arms_limits(self) -> None:
        limits = {name: venue_arm(name)[0].limits_degrees() for name in NEUTRAL}
        self.assertEqual(outside(build(NEUTRAL, NEUTRAL, plan(), trims={"left": 7.8, "right": -7.8}), limits), [])
        turned_too_far = outside(build(NEUTRAL, NEUTRAL, plan(), trims={"left": -20.0, "right": 0.0}), limits)
        self.assertEqual(len(turned_too_far), 1)
        self.assertIn("left shoulder_pan", turned_too_far[0])

    def test_a_smaller_move_swings_less_and_keeps_its_posture(self) -> None:
        # The mouth still points ahead, the flat hand stays level, the twisting hands stay open.
        kept = {"jaws": {"wrist_flex", "wrist_roll"}, "point": {"wrist_flex"}, "raise": set(), "twist": {"gripper"}}
        full, half = (build(NEUTRAL, NEUTRAL, plan(reps=1, size=size)) for size in (1.0, 0.5))
        for keyframe, smaller, label in zip(full.keyframes, half.keyframes, full.labels):
            if label not in ORDER:
                continue
            for arm in NEUTRAL:
                for joint, value in keyframe[arm].items():
                    rest = 0.0 if joint == "gripper" else NEUTRAL[arm][joint]  # the gripper from shut
                    expected = value if joint in kept[label] else rest + (value - rest) / 2
                    self.assertAlmostEqual(smaller[arm][joint], expected, msg=f"{label} {arm} {joint}")

    def test_the_ways_between_moves_keep_their_own_pace_however_fast_the_moves(self) -> None:
        for speed in (0.5, 1.0, 4.0):
            routine = build(NEUTRAL, NEUTRAL, plan(reps=2, speed=speed))
            transits = [index for index, pace in enumerate(routine.paces) if not pace]
            self.assertEqual([routine.labels[index + 1] for index in transits],
                             ["jaws", "upright", "point", "upright", "raise", "upright", "twist", "neutral"])
            for index in transits:
                a, b = routine.keyframes[index], routine.keyframes[index + 1]
                turn = max(abs(b[arm][j] - a[arm][j]) * (1.3 if j == "gripper" else 1.0) for arm in a for j in a[arm])
                self.assertLessEqual(np.pi / 2 * turn / routine.seconds[index], TRANSIT_SPEED + 1e-6)

    def test_unknown_moves_and_silly_numbers_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            build(NEUTRAL, NEUTRAL, plan(("twerk",)))
        with self.assertRaises(ValueError):
            build(NEUTRAL, NEUTRAL, plan(), bpm=0)
        with self.assertRaises(ValueError):
            build(NEUTRAL, NEUTRAL, [Part("jaws", 4, 0.0)])
        with self.assertRaises(ValueError):
            build(NEUTRAL, NEUTRAL, [Part("jaws", 0, 1.0)])
        with self.assertRaises(ValueError):
            build(NEUTRAL, NEUTRAL, [Part("jaws", 4, 1.0, 2.0)])

    def test_the_settings_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "groove.toml"
            self.assertEqual(load_parts(path), (plan(), 60.0), "without one, every move as usual")
            path.write_text('order = ["twist", "jaws"]\nbpm = 90\n[twist]\nreps = 6\nspeed = 0.5\nsize = 0.7\n')
            self.assertEqual(load_parts(path), ([Part("twist", 6, 0.5, 0.7), Part("jaws", 4, 1.0)], 90.0))
        parts, _ = load_parts(Path(__file__).resolve().parents[1] / "groove.toml")
        self.assertEqual([part.move for part in parts], list(ORDER), "the repository's has every move")


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
        cls.trims, cls.heading = facing(cls.bodies, NEUTRAL)
        cls.ahead = np.array([np.cos(np.radians(cls.heading)), np.sin(np.radians(cls.heading))])

    def poses(self, move: str) -> list[dict[str, dict[str, float]]]:
        return build(NEUTRAL, NEUTRAL, plan((move,), reps=1), trims=self.trims).keyframes[1:3]

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

    def test_jaws_a_mouth_ahead_opening_up_and_down(self) -> None:
        wide, shut = self.poses("jaws")
        for arm in NEUTRAL:
            self.assertGreater(wide[arm]["gripper"], 50.0)
            self.assertEqual(shut[arm]["gripper"], 0.0)
            for pose in (wide[arm], shut[arm]):
                self.assertGreater(self.fingers(arm, pose)[:2] @ self.ahead, 0.95, f"{arm}: pointing ahead, level")
                _, across = self.bodies[arm].kinematics.pointing(pose)
                self.assertGreater(abs(across[2]), 0.95, f"{arm}: the jaw opens up and down")

    def test_point_the_base_sweeps_a_still_flat_hand_side_to_side(self) -> None:
        first, second = self.poses("point")
        for arm in NEUTRAL:
            self.assertEqual({joint: value for joint, value in first[arm].items() if joint != "shoulder_pan"},
                             {joint: value for joint, value in second[arm].items() if joint != "shoulder_pan"},
                             f"{arm}: only the base turns")
        ways = []
        for pose in (first, second):
            flat = {arm: self.fingers(arm, pose[arm]) for arm in NEUTRAL}
            for arm, way in flat.items():
                self.assertLess(abs(way[2]), 0.05, f"{arm}: the hand level")
                _, across = self.bodies[arm].kinematics.pointing(pose[arm])
                self.assertLess(abs(across[2]), 0.2, f"{arm}: flat, the jaw closing sideways")
            left, right = (flat[arm][:2] / np.linalg.norm(flat[arm][:2]) for arm in ("left", "right"))
            self.assertGreater(left @ right, np.cos(np.radians(1.0)), "both hands the same way")
            self.assertLess(abs(left @ self.ahead), 0.3, "well to the side")
            ways.append(left)
        self.assertLess(ways[0] @ ways[1], -0.8, "then to the other side")

    def test_raise_one_arm_up_the_other_down_then_swapped(self) -> None:
        for pose, (down, up) in zip(self.poses("raise"), (("left", "right"), ("right", "left"))):
            tips = {arm: self.bodies[arm].points(pose[arm])[1][6] for arm in NEUTRAL}
            self.assertLess(tips[down], tips[up] - 20.0, f"{down} lowered, {up} up")

    def test_raise_keeps_the_grippers_shut_all_through(self) -> None:
        routine = build(NEUTRAL, NEUTRAL, plan(("point", "raise", "twist"), reps=2), trims=self.trims)
        during = [t for t in np.arange(0.0, routine.total, 0.02) if routine.label(t) == "raise"]
        self.assertGreater(len(during), 100)
        for t in during:
            self.assertEqual([routine.at(t)[arm]["gripper"] for arm in NEUTRAL], [0.0, 0.0], f"at {t:.2f} s")

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

    def test_the_arms_keep_clear_of_each_other_the_table_and_what_is_behind_them(self) -> None:
        from trashdrop.groove import check

        room = check(build(NEUTRAL, NEUTRAL, plan(reps=1), trims=self.trims), self.bodies, self.heading, step_s=0.1)
        self.assertGreaterEqual(room.apart, MIN_APART_CM)
        self.assertGreaterEqual(room.above, MIN_ABOVE_TABLE_CM)
        self.assertLessEqual(room.behind, MAX_BEHIND_CM, "nothing behind the bases")
        self.assertLess(room.out, 25.0, "a hand's reach to the sides")
        self.assertGreater(room.ahead, 35.0, "the lowered arm reaches out in front")

    def test_a_hand_bent_back_over_the_base_is_caught(self) -> None:
        from trashdrop.groove import Routine, check

        bent = {"left": dict(NEUTRAL["left"], wrist_flex=-90.0), "right": dict(NEUTRAL["right"])}
        room = check(Routine([NEUTRAL, bent], [1.0], ["start", "bent"]), self.bodies, self.heading, step_s=0.25)
        self.assertGreater(room.behind, MAX_BEHIND_CM)

    def test_a_speed_the_servos_cannot_keep_up_with_is_refused(self) -> None:
        from trashdrop.groove import main

        self.assertEqual(main(["--preview", "--speed", "3", "--moves", "jaws"]), 1)
        self.assertEqual(main(["--preview", "--speed", "0"]), 1)

    def test_a_storyboard_of_every_pose(self) -> None:
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("needs the dataset extra")
        import cv2

        from trashdrop.groove import storyboard

        with tempfile.TemporaryDirectory() as folder:
            path = storyboard(self.bodies, NEUTRAL, self.trims, 0.0, plan(), Path(folder) / "groove.png")
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
        routine = build(start, NEUTRAL, plan(("jaws", "twist"), reps=1), bpm=120)
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
        routine = build({name: arm.pose() for name, arm in arms.items()}, NEUTRAL, plan(reps=1))
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
