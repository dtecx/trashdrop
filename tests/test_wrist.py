"""An arm's own wrist roll zero, from where its moving jaw really opened.

At the venue the left arm's jaw closed along a screwdriver instead of across
it: its LeRobot wrist roll zero was a quarter turn from the model's. Here a
simulated person watches a simulated arm whose zero is off, answers the
questions `trashdrop rig roll` asks, and the offset has to come out right.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulation extra")

from trashdrop.kinematics import Kinematics  # noqa: E402
from trashdrop.placement import Placement  # noqa: E402
from trashdrop.wrist import (  # noqa: E402
    NUDGE_DEG,
    degrees_from_far,
    hover_pose,
    jaw_side,
    settled_offset,
    turned_by,
    wrap,
)

# The venue's arms, as the tape calibration placed them.
LEFT = Placement(x=19.12, y=-21.70, yaw=-95.36, table_z=0.88, dz_dx=-0.095, dz_dy=0.1037)
RIGHT = Placement(x=17.97, y=20.73, yaw=-84.12, table_z=-0.47, dz_dx=-0.0436, dz_dy=-0.0453)


def find_offset(true: float, placement: Placement) -> float:
    """What `rig roll` ends with, starting from no offset, for an arm whose real one is ``true``."""

    real = Kinematics(true)
    offset = 0.0
    while True:
        guess = Kinematics(offset)
        pose = hover_pose(guess, placement)
        delta = turned_by(guess, pose, placement, jaw_side(real, pose, placement))
        if delta == 0.0:
            break
        offset = wrap(offset + delta)
    # Nudged by eye, whichever way looks better, until it looks square.
    roll = pose["wrist_roll"]
    while True:
        now = abs(degrees_from_far(real, dict(pose, wrist_roll=roll), placement))
        better = min((roll + NUDGE_DEG, roll - NUDGE_DEG),
                     key=lambda r: abs(degrees_from_far(real, dict(pose, wrist_roll=r), placement)))
        if abs(degrees_from_far(real, dict(pose, wrist_roll=better), placement)) >= now:
            break
        roll = better
    return settled_offset(guess, pose, placement, roll - pose["wrist_roll"])


class WristRollTests(unittest.TestCase):
    def test_the_hover_sends_the_moving_jaw_towards_the_far_edge(self) -> None:
        kinematics = Kinematics()
        for placement in (LEFT, RIGHT):
            pose = hover_pose(kinematics, placement)
            self.assertIsNotNone(pose)
            self.assertEqual(jaw_side(kinematics, pose, placement), "far")
            self.assertLess(abs(degrees_from_far(kinematics, pose, placement)), 3.0)

    def test_the_answers_find_the_arms_own_zero(self) -> None:
        for true in (90.0, -90.0, 180.0, 0.0, 97.0, -84.0):
            for placement in (LEFT, RIGHT):
                with self.subTest(true=true, arm=placement.yaw):
                    self.assertLess(abs(wrap(find_offset(true, placement) - true)), NUDGE_DEG / 2 + 0.5)

    def test_a_touch_moves_when_the_wrist_zero_does(self) -> None:
        # The fixed fingertip is off the roll axis: touches worked out with the
        # wrong zero are wrong, which is why their joint angles are kept.
        pose = {"shoulder_pan": 20.0, "shoulder_lift": 45.0, "elbow_flex": -35.0, "wrist_flex": 80.0, "wrist_roll": 5.0}
        moved = np.linalg.norm(Kinematics(90.0).fingertip(pose) - Kinematics(0.0).fingertip(pose)) * 100
        self.assertGreater(moved, 0.8)
        self.assertLess(moved, 1.7)


if __name__ == "__main__":
    unittest.main()
