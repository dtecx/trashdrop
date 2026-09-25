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
    CALIBRATION_TICK,
    DEG_PER_TICK,
    LIMIT_MARGIN,
    RATE_HZ,
    Arm,
    load_poses,
    resolve_arm,
    save_pose,
)
from trashdrop.rig import Rig
from trashdrop.servo import MOTORS

LIMITS = {motor: (800, 3300) for motor in MOTORS.values()}
LIMITS[MOTORS["gripper"]] = (2044, 3499)


class FakeBus:
    """Six ideal servos: with torque on, each sits at its goal minus any sag."""

    def __init__(self, positions: dict[str, int] | None = None, sag: dict[str, int] | None = None) -> None:
        positions = positions or {joint: 2000 + 10 * motor for joint, motor in MOTORS.items()}
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

    def test_a_limp_arm_refuses_to_move(self) -> None:
        with self.assertRaises(RuntimeError):
            make_arm(FakeBus()).move({"elbow_flex": 10.0})


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


class UnitTests(unittest.TestCase):
    def test_degrees_are_from_the_calibration_pose(self) -> None:
        arm = make_arm(FakeBus())
        self.assertEqual(arm.to_ticks("elbow_flex", 0.0), CALIBRATION_TICK)
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
