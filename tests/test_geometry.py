"""Frame transforms. The 180-degree case is the one that broke the layout."""

from __future__ import annotations

import math
import unittest

from trashdrop.geometry import Pose2D, wrap_angle
from trashdrop.station import BACK_ARM, BASE_SEPARATION, FRONT_ARM, PICK_ZONE


class Pose2DTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        pose = Pose2D.from_degrees(0.1, -0.2, 37.0)
        for x, y in ((0.0, 0.0), (0.25, -0.3), (-0.4, 0.15)):
            back = pose.to_world(*pose.to_local(x, y))
            self.assertAlmostEqual(back[0], x, places=9)
            self.assertAlmostEqual(back[1], y, places=9)

    def test_identity_frame_is_a_no_op(self) -> None:
        pose = Pose2D(0.0, 0.0, 0.0)
        self.assertEqual(pose.to_local(0.2, -0.3), (0.2, -0.3))

    def test_back_arm_sees_the_pick_zone_mirrored(self) -> None:
        # A point at the far edge for the front arm is at the near edge for
        # the back arm. If this stops holding, the two arms no longer share
        # the zone and assignment-by-material is unsound.
        far_for_front = (0.05, PICK_ZONE.bounds[2])
        local = BACK_ARM.pose.to_local(*far_for_front)
        self.assertAlmostEqual(local[0], -0.05, places=9)
        self.assertAlmostEqual(local[1], -(BASE_SEPARATION + PICK_ZONE.bounds[2]), places=9)

    def test_both_arms_reach_zone_within_the_measured_envelope(self) -> None:
        # The single-arm baseline was verified for local y in [-0.10, -0.33].
        for mount in (FRONT_ARM, BACK_ARM):
            for x, y in PICK_ZONE.corners():
                _, local_y = mount.pose.to_local(x, y)
                self.assertLess(local_y, -0.10, f"{mount.name} too close at {(x, y)}")
                self.assertGreater(local_y, -0.33, f"{mount.name} too far at {(x, y)}")


class WrapTests(unittest.TestCase):
    def test_wrap(self) -> None:
        # The range is half-open at +pi, so 3*pi folds to -pi, not +pi.
        self.assertAlmostEqual(wrap_angle(3 * math.pi), -math.pi, places=9)
        self.assertAlmostEqual(wrap_angle(0.4), 0.4, places=9)
        self.assertAlmostEqual(wrap_angle(-0.4), -0.4, places=9)

    def test_wrap_is_idempotent(self) -> None:
        for value in (-7.0, -1.0, 0.0, 1.0, 7.0):
            once = wrap_angle(value)
            self.assertAlmostEqual(wrap_angle(once), once, places=12)
            self.assertGreaterEqual(once, -math.pi)
            self.assertLess(once, math.pi)


if __name__ == "__main__":
    unittest.main()
