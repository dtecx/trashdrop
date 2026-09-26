"""Driving a real arm safely, checked against a fake servo bus.

The fake starts the way the real SO-101s were found: every goal register at
0, torque off. Enabling torque on that would throw each joint to its end stop
at full speed, so the first thing pinned down is that it cannot happen.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trashdrop.arm import (
    DEG_PER_TICK,
    TEST_ORDER,
    LIMIT_MARGIN,
    RATE_HZ,
    Arm,
    FIRST_POSE,
    load_poses,
    move_named_pose_together,
    move_together,
    resolve_arm,
    nudge_joints,
    pick_and_drop,
    save_pose,
)
from trashdrop.rig import Rig
from trashdrop.servo import MOTORS

LIMITS = {motor: (800, 3300) for motor in MOTORS.values()}
LIMITS[MOTORS["gripper"]] = (2044, 3499)


class FakeBus:
    """Six ideal servos: with torque on, each sits at its goal minus any sag."""

    def __init__(self, positions: dict[str, int] | None = None, sag: dict[str, int] | None = None) -> None:
        positions = positions or {joint: 2100 + 10 * motor for joint, motor in MOTORS.items()}
        self.regs = {
            motor: {
                "position": positions[joint],
                "goal_position": 0,
                "torque_enable": 0,
                "goal_speed": 0,
                "min_limit": LIMITS[motor][0],
                "max_limit": LIMITS[motor][1],
            }
            for joint, motor in MOTORS.items()
        }
        self.sag = {MOTORS[joint]: ticks for joint, ticks in (sag or {}).items()}
        self.log: list[tuple] = []

    def _settle(self) -> None:
        for motor, regs in self.regs.items():
            if regs["torque_enable"]:
                regs["position"] = regs["goal_position"] - self.sag.get(motor, 0)

    def read(self, motor: int, register: str) -> int:
        self._settle()
        return self.regs[motor][register]

    def write(self, motor: int, register: str, value: int) -> None:
        self.log.append(("write", motor, register, value, dict(self.regs[motor])))
        self.regs[motor][register] = value

    def write_goals(self, goals: dict[int, int]) -> None:
        self.log.append(("sync", dict(goals)))
        for motor, ticks in goals.items():
            self.regs[motor]["goal_position"] = ticks

    def positions(self) -> dict[str, int]:
        self._settle()
        return {joint: self.regs[motor]["position"] for joint, motor in MOTORS.items()}

    def goals_streamed(self) -> list[dict[int, int]]:
        return [entry[1] for entry in self.log if entry[0] == "sync"]


class FakeClock:
    def __init__(self, stop_after: int | None = None) -> None:
        self.now, self.sleeps, self.stop_after = 0.0, 0, stop_after

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        if self.stop_after is not None and self.sleeps >= self.stop_after:
            raise KeyboardInterrupt
        self.now += seconds


def make_arm(bus: FakeBus, clock: FakeClock | None = None, max_speed: float = 30.0) -> Arm:
    clock = clock or FakeClock()
    return Arm("left", bus, max_speed, clock=clock, sleep=clock.sleep)


class TorqueTests(unittest.TestCase):
    def test_torque_goes_on_only_after_every_goal_holds_the_present_position(self) -> None:
        bus = FakeBus()
        present = bus.positions()
        make_arm(bus).torque_on()
        for joint, motor in MOTORS.items():
            torque_writes = [e for e in bus.log if e[0] == "write" and e[1] == motor and e[2] == "torque_enable"]
            self.assertEqual(len(torque_writes), 1)
            # The goal register at the moment torque went on:
            self.assertEqual(torque_writes[0][4]["goal_position"], present[joint], joint)
        self.assertEqual(bus.positions(), present, "enabling torque must not move anything")

    def test_the_servos_own_speed_limit_is_set_as_a_second_line(self) -> None:
        bus = FakeBus()
        make_arm(bus, max_speed=30.0).torque_on()
        expected = round(2 * 30.0 / DEG_PER_TICK)
        self.assertTrue(all(regs["goal_speed"] == expected for regs in bus.regs.values()))

    def test_a_holding_arm_takes_a_new_speed_limit_without_letting_go(self) -> None:
        # An arm left holding from an earlier run: rig.toml says faster now.
        bus = FakeBus()
        make_arm(bus, max_speed=30.0).torque_on()
        faster = make_arm(bus, max_speed=60.0)
        faster.limit_speed()
        self.assertTrue(all(regs["goal_speed"] == round(2 * 60.0 / DEG_PER_TICK) for regs in bus.regs.values()))
        self.assertTrue(all(regs["torque_enable"] for regs in bus.regs.values()), "still holding")

    def test_a_limp_arm_refuses_to_move(self) -> None:
        with self.assertRaises(RuntimeError):
            make_arm(FakeBus()).move({"elbow_flex": 10.0})

    def test_hand_posing_releases_arm_joints_but_holds_the_gripper_where_it_is(self) -> None:
        bus = FakeBus()
        arm = make_arm(bus)
        arm.torque_on()
        gripper_position = bus.positions()["gripper"]

        arm.relax_joints_hold_gripper()

        for joint, motor in MOTORS.items():
            self.assertEqual(bool(bus.regs[motor]["torque_enable"]), joint == "gripper", joint)
        self.assertEqual(bus.regs[MOTORS["gripper"]]["goal_position"], gripper_position)
        self.assertEqual(bus.positions()["gripper"], gripper_position)


class MoveTests(unittest.TestCase):
    def test_a_move_is_smooth_and_never_faster_than_max_speed(self) -> None:
        bus, clock = FakeBus(), FakeClock()
        arm = make_arm(bus, clock, max_speed=30.0)
        arm.torque_on()
        start = bus.regs[MOTORS["shoulder_pan"]]["goal_position"]
        arm.move({"shoulder_pan": arm.from_ticks("shoulder_pan", start) + 90.0})
        path = [goals[MOTORS["shoulder_pan"]] for goals in bus.goals_streamed()]
        steps = [abs(b - a) * DEG_PER_TICK * RATE_HZ for a, b in zip(path, path[1:])]
        self.assertLessEqual(max(steps), 30.0 * 1.05, "degrees per second between two goals")
        self.assertLess(abs(path[0] - start), 3, "the first goal is where the joint already is")
        self.assertAlmostEqual((path[-1] - start) * DEG_PER_TICK, 90.0, delta=0.2)
        self.assertGreater(clock.now, 5.0, "90 degrees at 30 deg/s, minimum jerk, takes over five seconds")

    def test_a_target_past_the_joint_limit_is_clamped_inside_it(self) -> None:
        bus = FakeBus()
        arm = make_arm(bus)
        arm.torque_on()
        arm.move({"elbow_flex": 500.0})
        high = LIMITS[MOTORS["elbow_flex"]][1]
        self.assertEqual(bus.regs[MOTORS["elbow_flex"]]["goal_position"], high - LIMIT_MARGIN)

    def test_a_sagging_joint_keeps_its_goal_instead_of_creeping_down(self) -> None:
        bus = FakeBus(sag={"shoulder_lift": 12})
        arm = make_arm(bus)
        arm.torque_on()
        held = bus.regs[MOTORS["shoulder_lift"]]["goal_position"]
        for _ in range(3):
            arm.move({"wrist_roll": 10.0})
            arm.move({"wrist_roll": 0.0})
        self.assertEqual(bus.regs[MOTORS["shoulder_lift"]]["goal_position"], held)

    def test_ctrl_c_mid_move_stops_and_keeps_holding(self) -> None:
        bus, clock = FakeBus(), FakeClock(stop_after=40)
        arm = make_arm(bus, clock)
        arm.torque_on()
        with self.assertRaises(KeyboardInterrupt):
            arm.move({"shoulder_pan": 60.0})
        last = bus.goals_streamed()[-1]
        present = bus.positions()
        self.assertEqual(last, {MOTORS[joint]: ticks for joint, ticks in present.items()})
        self.assertTrue(all(regs["torque_enable"] for regs in bus.regs.values()), "still holding, not limp")


class TogetherTests(unittest.TestCase):
    def test_two_arms_move_at_once_each_on_its_own_bus(self) -> None:
        clock = FakeClock()
        left_bus, right_bus = FakeBus(), FakeBus()
        left, right = make_arm(left_bus, clock), Arm("right", right_bus, 30.0, clock=clock, sleep=clock.sleep)
        left.torque_on()
        right.torque_on()
        move_together([(left, {"shoulder_pan": 45.0}), (right, {"shoulder_pan": -45.0})])
        pan = MOTORS["shoulder_pan"]
        left_path = [goals[pan] for goals in left_bus.goals_streamed()]
        right_path = [goals[pan] for goals in right_bus.goals_streamed()]
        self.assertEqual(len(left_path), len(right_path), "one goal each, every tick of the clock")
        self.assertGreater(left_path[len(left_path) // 2], left_path[0], "both under way at the same time")
        self.assertLess(right_path[len(right_path) // 2], right_path[0])
        self.assertLess(clock.now, 2 * 1.875 * 45 / 30, "together, not one after the other")

    def test_ctrl_c_stops_both_and_both_keep_holding(self) -> None:
        clock = FakeClock(stop_after=20)
        buses = FakeBus(), FakeBus()
        arms = [Arm(name, bus, 30.0, clock=clock, sleep=clock.sleep) for name, bus in zip(("left", "right"), buses)]
        for arm in arms:
            arm.torque_on()
        with self.assertRaises(KeyboardInterrupt):
            move_together([(arm, {"elbow_flex": 40.0}) for arm in arms])
        for bus in buses:
            present = bus.positions()
            self.assertEqual(bus.goals_streamed()[-1], {MOTORS[joint]: ticks for joint, ticks in present.items()})

    def test_a_stop_from_elsewhere_holds_every_arm_where_it_is(self) -> None:
        import threading

        from trashdrop.arm import Stopped

        stop, clock = threading.Event(), FakeClock()
        bus = FakeBus()
        arm = Arm("left", bus, 30.0, clock=clock, sleep=clock.sleep, stop=stop)
        arm.torque_on()
        original_sleep = clock.sleep

        def sleep(seconds: float) -> None:  # the button is pressed a second into the move
            original_sleep(seconds)
            if clock.now > 1.0:
                stop.set()

        arm._sleep = sleep
        with self.assertRaises(Stopped):
            arm.move({"shoulder_pan": 60.0})
        present = bus.positions()
        self.assertEqual(bus.goals_streamed()[-1], {MOTORS[joint]: ticks for joint, ticks in present.items()})
        self.assertLess(clock.now, 1.1)

    def test_a_jaw_closed_on_an_item_ends_the_move_once_it_stops(self) -> None:
        # Stalled 300 ticks short of closed, the jaw never reaches its goal.
        bus, clock = FakeBus(sag={"gripper": 300}), FakeClock()
        arm = make_arm(bus, clock)
        arm.torque_on()
        before, start = clock.now, bus.regs[MOTORS["gripper"]]["goal_position"]
        arm.move({"gripper": 0.0})
        travel = abs(bus.regs[MOTORS["gripper"]]["goal_position"] - start) * DEG_PER_TICK
        trajectory = max(1.875 * travel / 30.0, 0.3)
        self.assertLess(clock.now - before, trajectory + 0.3, "stopped is done: not the 1.5 s wait on top")

    def test_both_arms_return_to_the_named_pose_without_moving_the_grippers(self) -> None:
        clock = FakeClock()
        left_bus = FakeBus(positions={joint: 2150 + 10 * motor for joint, motor in MOTORS.items()})
        right_bus = FakeBus(positions={joint: 2250 + 10 * motor for joint, motor in MOTORS.items()})
        arms = {
            "left": Arm("left", left_bus, 30.0, clock=clock, sleep=clock.sleep),
            "right": Arm("right", right_bus, 30.0, clock=clock, sleep=clock.sleep),
        }
        target = {
            "shoulder_pan": 30.0,
            "shoulder_lift": 10.0,
            "elbow_flex": -20.0,
            "wrist_flex": 5.0,
            "wrist_roll": -15.0,
            "gripper": 100.0,  # saved value must not open the current jaws
        }
        poses = {name: {FIRST_POSE: target} for name in arms}
        grippers = {"left": left_bus.positions()["gripper"], "right": right_bus.positions()["gripper"]}

        reached = move_named_pose_together(arms, poses, FIRST_POSE, speed=12.0)

        self.assertEqual(len(reached), 2)
        for name, bus in (("left", left_bus), ("right", right_bus)):
            self.assertEqual(bus.regs[MOTORS["gripper"]]["goal_position"], grippers[name])
            self.assertTrue(all(path[MOTORS["gripper"]] == grippers[name] for path in bus.goals_streamed()))
            self.assertEqual(bus.regs[MOTORS["shoulder_pan"]]["goal_position"],
                             arms[name].to_ticks("shoulder_pan", target["shoulder_pan"]))

    def test_together_pose_requires_a_positive_speed(self) -> None:
        arm = make_arm(FakeBus())
        with self.assertRaisesRegex(ValueError, "positive"):
            move_named_pose_together({"left": arm}, {}, FIRST_POSE, speed=0.0)


class BothToPoseTests(unittest.TestCase):
    def test_grippers_clamp_first_then_both_arms_move_together(self) -> None:
        from trashdrop.arm import move_both_to_pose

        clock = FakeClock()
        buses = {"left": FakeBus(), "right": FakeBus()}
        arms = {name: Arm(name, bus, 30.0, clock=clock, sleep=clock.sleep) for name, bus in buses.items()}
        pose = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -20.0, "wrist_flex": 0.0, "wrist_roll": 10.0,
                "gripper": 80.0}
        move_both_to_pose(arms, {"left": {"crab": pose}, "right": {"crab": dict(pose, wrist_roll=0.0)}}, "crab", 20.0)
        for name, arm in arms.items():
            self.assertTrue(arm.torque_is_on())
            self.assertLess(arm.pose()["gripper"], 2.0, "clamped shut (the limit's margin), not the saved 80 % open")
            self.assertAlmostEqual(arm.pose()["elbow_flex"], -20.0, delta=0.2)
        self.assertAlmostEqual(arms["left"].pose()["wrist_roll"], 10.0, delta=0.2)
        gripper = MOTORS["gripper"]
        first_joint_move = next(i for i, goals in enumerate(buses["left"].goals_streamed())
                                if goals[MOTORS["elbow_flex"]] != buses["left"].goals_streamed()[0][MOTORS["elbow_flex"]])
        self.assertLessEqual(buses["left"].goals_streamed()[first_joint_move][gripper],
                             buses["left"].goals_streamed()[0][gripper], "the jaw closed before the arm moved")

    def test_a_missing_pose_moves_nothing(self) -> None:
        from trashdrop.arm import move_both_to_pose

        bus = FakeBus()
        with self.assertRaises(ValueError):
            move_both_to_pose({"left": make_arm(bus)}, {}, "crab", 10.0)
        self.assertFalse(bus.goals_streamed())


class NudgeTests(unittest.TestCase):
    def test_every_joint_is_nudged_a_little_and_ends_where_it_started(self) -> None:
        bus = FakeBus()
        arm = make_arm(bus)
        arm.torque_on()
        start = bus.positions()
        moved = nudge_joints(arm, 6.0, 15.0, ask=lambda _: "", log=lambda *_: None)
        self.assertEqual(moved, list(TEST_ORDER))
        self.assertEqual(bus.positions(), start)
        for joint, motor in MOTORS.items():
            path = [goals[motor] for goals in bus.goals_streamed()]
            farthest = max(abs(ticks - start[joint]) for ticks in path)
            limit = 0.15 * (LIMITS[motor][1] - LIMITS[motor][0]) if joint == "gripper" else 6.0 / DEG_PER_TICK
            self.assertLessEqual(farthest, limit + 2, joint)

    def test_skip_and_stop_are_honoured(self) -> None:
        bus = FakeBus()
        arm = make_arm(bus)
        arm.torque_on()
        answers = iter(["s", "", "q"])
        moved = nudge_joints(arm, 6.0, 15.0, ask=lambda _: next(answers), log=lambda *_: None)
        self.assertEqual(moved, ["wrist_roll"])


class PickTests(unittest.TestCase):
    POSES = {
        "above": {"shoulder_lift": 10.0, "elbow_flex": -60.0, "gripper": 50.0},
        "grab": {"shoulder_lift": 20.0, "elbow_flex": -50.0, "gripper": 50.0},
        "drop": {"shoulder_pan": 80.0, "gripper": 0.0},
        "neutral": {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -90.0, "gripper": 0.0},
    }

    def arm_holding(self, stopped_at: float | None):
        """An arm whose jaws stop at ``stopped_at`` percent when told to close (None: nothing there)."""

        bus = FakeBus()
        arm = make_arm(bus, max_speed=60.0)
        arm.torque_on()
        if stopped_at is not None:
            gripper = MOTORS["gripper"]
            low, high = LIMITS[gripper]
            item_edge = round(low + stopped_at / 100 * (high - low))
            original = bus._settle

            def settle():
                original()
                regs = bus.regs[gripper]
                regs["position"] = max(regs["position"], item_edge)  # the item stops the jaws

            bus._settle = settle
        return bus, arm

    def test_an_item_is_carried_with_the_jaws_never_opening_until_the_drop(self) -> None:
        bus, arm = self.arm_holding(stopped_at=35.0)
        self.assertTrue(pick_and_drop(arm, self.POSES, log=lambda *_: None))
        gripper, pan = MOTORS["gripper"], MOTORS["shoulder_pan"]
        streamed = bus.goals_streamed()
        path = [goals[gripper] for goals in streamed]
        closed = path.index(min(path))
        first_opening = next(i for i in range(closed + 1, len(path)) if path[i] > path[i - 1])
        # The jaws start to open only once the arm is over the drop zone.
        self.assertEqual(streamed[first_opening][pan], arm.to_ticks("shoulder_pan", 80.0))
        self.assertEqual(bus.regs[pan]["goal_position"], arm.to_ticks("shoulder_pan", 0.0), "back to neutral")

    def test_jaws_that_close_on_nothing_are_a_miss_and_nothing_is_carried(self) -> None:
        bus, arm = self.arm_holding(stopped_at=None)
        self.assertFalse(pick_and_drop(arm, self.POSES, log=lambda *_: None))
        pan_goals = [goals[MOTORS["shoulder_pan"]] for goals in bus.goals_streamed()]
        self.assertLess(max(pan_goals), arm.to_ticks("shoulder_pan", 40.0), "it never swung towards the drop zone")

    def test_missing_poses_are_named(self) -> None:
        bus, arm = self.arm_holding(stopped_at=35.0)
        with self.assertRaises(ValueError) as caught:
            pick_and_drop(arm, {"neutral": {}}, log=lambda *_: None)
        self.assertIn("above", str(caught.exception))


class UnitTests(unittest.TestCase):
    def test_degrees_are_lerobots_from_the_middle_of_the_range(self) -> None:
        arm = make_arm(FakeBus())
        low, high = LIMITS[MOTORS["elbow_flex"]]
        self.assertEqual(arm.to_ticks("elbow_flex", 0.0), (low + high) // 2)
        # LeRobot: degrees = (ticks - mid) * 360 / 4095
        self.assertAlmostEqual(arm.from_ticks("elbow_flex", (low + high) // 2 + 4095 // 4), 90.0, delta=0.1)
        self.assertAlmostEqual(arm.from_ticks("elbow_flex", arm.to_ticks("elbow_flex", 37.0)), 37.0, delta=0.1)

    def test_gripper_is_percent_open_within_its_limits(self) -> None:
        arm = make_arm(FakeBus())
        low, high = LIMITS[MOTORS["gripper"]]
        self.assertEqual(arm.to_ticks("gripper", 0.0), low + LIMIT_MARGIN)
        self.assertEqual(arm.to_ticks("gripper", 100.0), high - LIMIT_MARGIN)


class PoseFileTests(unittest.TestCase):
    def test_poses_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.toml"
            save_pose("left", "rest", {"shoulder_pan": 1.25, "gripper": 4.0}, path)
            save_pose("right", "ready", {"elbow_flex": -30.0}, path)
            poses = load_poses(path)
        self.assertEqual(poses["left"]["rest"], {"shoulder_pan": 1.2, "gripper": 4.0})
        self.assertEqual(poses["right"]["ready"], {"elbow_flex": -30.0})

    def test_an_arm_is_found_by_side_or_by_the_label_written_on_it(self) -> None:
        rig = Rig()
        self.assertEqual(resolve_arm("F02", rig), "right")
        self.assertEqual(resolve_arm("left", rig), "left")
        with self.assertRaises(ValueError):
            resolve_arm("front", rig)


if __name__ == "__main__":
    unittest.main()
