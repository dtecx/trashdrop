"""The SO-101 kinematics the real arms are driven with.

The model's angles are LeRobot's calibrated degrees; the checks here are that
the upright pose really is upright, and that solving for a point and then
computing where the gripper ends up agrees -- with the fingers straight down.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulation extra")

from trashdrop.kinematics import NEUTRAL, Kinematics  # noqa: E402


class KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = Kinematics()

    def test_the_neutral_pose_stands_straight_up(self) -> None:
        x, y, z = self.kinematics.tcp(NEUTRAL)
        self.assertAlmostEqual(y, 0.0, delta=0.005)
        self.assertGreater(z, 0.45)
        fingers, _ = self.kinematics.pointing(NEUTRAL)
        self.assertGreater(fingers[2], 0.99, "fingers point up")

    def test_a_solved_point_is_where_the_gripper_ends_up_fingers_down(self) -> None:
        for target in ((0.12, 0.0, 0.03), (0.20, 0.05, 0.05), (0.25, -0.08, 0.02), (0.18, 0.0, 0.08)):
            with self.subTest(target=target):
                solution = self.kinematics.solve(np.array(target))
                self.assertTrue(solution.reachable, solution)
                reached = self.kinematics.tcp(solution.degrees)
                self.assertLess(np.linalg.norm(reached - target), 0.002)
                fingers, _ = self.kinematics.pointing(solution.degrees)
                self.assertLess(fingers[2], -0.99)

    def test_the_jaw_closes_along_the_requested_direction(self) -> None:
        solution = self.kinematics.solve(np.array([0.18, -0.06, 0.04]), yaw_deg=60.0)
        self.assertTrue(solution.reachable, solution)
        _, across = self.kinematics.pointing(solution.degrees)
        self.assertGreater(across @ np.array([np.cos(np.radians(60)), np.sin(np.radians(60)), 0.0]), 0.99)

    def test_an_arms_own_wrist_roll_zero_is_added_on_the_way_in_and_taken_off_on_the_way_out(self) -> None:
        # The venue's left arm: its LeRobot wrist roll zero a quarter turn from the model's.
        turned = Kinematics(wrist_roll_offset=90.0)
        pose = dict(NEUTRAL, shoulder_lift=30.0, elbow_flex=-30.0, wrist_flex=70.0, wrist_roll=10.0)
        self.assertTrue(np.allclose(turned.tcp(pose), self.kinematics.tcp(dict(pose, wrist_roll=100.0))))
        target = np.array([0.18, -0.06, 0.04])
        solution = turned.solve(target, yaw_deg=60.0)
        self.assertTrue(solution.reachable, solution)
        _, across = turned.pointing(solution.degrees)
        self.assertGreater(across @ np.array([np.cos(np.radians(60)), np.sin(np.radians(60)), 0.0]), 0.99)
        plain = self.kinematics.solve(target, yaw_deg=60.0)
        difference = (solution.degrees["wrist_roll"] - plain.degrees["wrist_roll"] + 180.0) % 360.0 - 180.0
        self.assertAlmostEqual(difference, -90.0, delta=1.0)

    def test_a_point_out_of_reach_is_reported_not_approximated(self) -> None:
        self.assertFalse(self.kinematics.solve(np.array([0.60, 0.0, 0.05])).reachable)
        self.assertFalse(self.kinematics.solve(np.array([0.20, 0.0, 0.14])).reachable, "too high for fingers down")

    def test_leaning_fingers_reach_further_and_point_as_asked(self) -> None:
        target = np.array([0.34, 0.0, 0.0])
        self.assertFalse(self.kinematics.solve(target).reachable, "straight down stops short of 34 cm")
        solution = self.kinematics.solve(target, lean_deg=30.0)
        self.assertTrue(solution.reachable, solution)
        fingers, _ = self.kinematics.pointing(solution.degrees)
        self.assertGreater(fingers @ self.kinematics.approach_for(target, 30.0), 0.99)
        self.assertGreater(fingers[0], 0.4, "leaning away from the base, not towards it")

    def test_the_real_joints_limits_are_kept(self) -> None:
        tight = {"wrist_flex": (-10.0, 60.0)}
        solution = self.kinematics.solve(np.array([0.20, 0.0, 0.04]), limits=tight)
        self.assertLessEqual(solution.degrees["wrist_flex"], 60.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
