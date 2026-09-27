"""Where an arm stands relative to the marker sheet, from touched markers."""

from __future__ import annotations

import unittest

import numpy as np

from trashdrop.camera.markers import MARKER_SHEET_CM
from trashdrop.placement import Placement, fit_placement, fit_table


class PlacementTests(unittest.TestCase):
    def test_marker_centres_sit_on_the_zone_corners_around_the_sheet_middle(self) -> None:
        self.assertEqual(MARKER_SHEET_CM[0], (-10.0, 7.5))
        self.assertEqual(MARKER_SHEET_CM[2], (10.0, -7.5))

    def test_touches_give_back_the_arms_placement(self) -> None:
        truth = Placement(x=22.0, y=-6.0, yaw=95.0)
        sheet = [MARKER_SHEET_CM[i] for i in (0, 1, 2, 3)]
        rng = np.random.default_rng(0)
        touched = [truth.to_arm(point) + rng.normal(0, 0.2, 2) for point in sheet]  # a hand is not exact
        fitted, residuals = fit_placement(sheet, touched)
        self.assertAlmostEqual(fitted.x, truth.x, delta=0.3)
        self.assertAlmostEqual(fitted.y, truth.y, delta=0.3)
        self.assertAlmostEqual(fitted.yaw, truth.yaw, delta=1.5)
        self.assertLess(max(residuals), 0.6)

    def test_the_table_is_the_plane_through_the_touches(self) -> None:
        # The venue's left arm read the flat table 3 cm lower at full reach than beside its base.
        touches = [(33.84, -9.56, -3.33), (3.63, -35.98, -3.20), (6.54, -4.97, -0.26)]
        placement = Placement(0.0, 0.0, 0.0, *fit_table(touches))
        for x, y, z in touches:
            self.assertAlmostEqual(placement.table_height(x, y), z, places=6)

    def test_fewer_than_three_touches_give_a_level_table(self) -> None:
        self.assertEqual(fit_table([(10.0, 0.0, -2.0), (20.0, 5.0, -3.0)]), (-2.5, 0.0, 0.0))

    def test_a_direction_on_the_sheet_turns_with_the_arm(self) -> None:
        self.assertAlmostEqual(Placement(0, 0, 90.0).direction_to_arm(0.0), 90.0)
        self.assertAlmostEqual(Placement(0, 0, 170.0).direction_to_arm(30.0), -160.0)

    def test_to_sheet_undoes_to_arm(self) -> None:
        placement = Placement(11.34, -22.40, -95.14)
        for point in ([0.0, 0.0], [22.0, 0.0], [-7.5, 31.0]):
            np.testing.assert_allclose(placement.to_sheet(placement.to_arm(point)), point, atol=1e-9)
        np.testing.assert_allclose(placement.to_sheet([11.34, -22.40]), [0.0, 0.0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
