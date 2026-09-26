"""Calibrating from a taped quadrilateral that is not square.

The venue's tape zone is crooked. Nothing here assumes its shape: the check is
end to end -- a point on the table, seen by the camera, must come out where
each arm would really touch it.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

pytest.importorskip("cv2", reason="needs the dataset extra")

from trashdrop.perception.calibration import HomographyCalibration  # noqa: E402
from trashdrop.placement import Placement  # noqa: E402
from trashdrop.tape import CORNERS, solve  # noqa: E402

# The true taped corners, cm, in some frame nobody knows: skewed and uneven.
TRUE = {"far_left": (-16.0, 15.5), "far_right": (15.0, 13.0), "near_right": (14.0, -14.5), "near_left": (-15.5, -12.0)}
LEFT = Placement(x=24.0, y=-18.0, yaw=-95.0, table_z=-2.4)  # world -> left arm frame
RIGHT = Placement(x=26.0, y=19.0, yaw=88.0, table_z=-1.9)  # world -> right arm frame
REACHES = {"left": ("far_left", "near_right", "near_left"), "right": ("far_right", "near_right", "near_left")}


def camera() -> HomographyCalibration:
    """An overhead camera with a little perspective: world metres <-> pixels."""

    world = [(-0.3, 0.3), (0.3, 0.3), (0.3, -0.3), (-0.3, -0.3)]
    pixels = [(655, 238), (1262, 232), (1275, 850), (648, 842)]
    return HomographyCalibration(pixels, world)


def touches_for(arms, noise_cm: float = 0.2):
    rng = np.random.default_rng(1)
    result = {}
    for arm in arms:
        placement = LEFT if arm == "left" else RIGHT
        result[arm] = {
            name: (*(placement.to_arm(TRUE[name]) + rng.normal(0, noise_cm, 2)), placement.table_z)
            for name in REACHES[arm]
        }
    return result


def pixels_for(cam) -> dict[str, tuple[float, float]]:
    return {name: cam.world_to_pixel(TRUE[name][0] / 100, TRUE[name][1] / 100) for name in CORNERS}


class TapeTests(unittest.TestCase):
    def assert_camera_to_arm(self, solution, arm: str, cam, tolerance_cm: float) -> None:
        truth = LEFT if arm == "left" else RIGHT
        rng = np.random.default_rng(7)
        for _ in range(20):
            point = rng.uniform((-12, -10), (12, 10))  # somewhere inside the zone
            u, v = cam.world_to_pixel(point[0] / 100, point[1] / 100)
            table = np.array(solution.homography.pixel_to_world(u, v)) * 100
            error = np.linalg.norm(solution.placements[arm].to_arm(table) - truth.to_arm(point))
            self.assertLess(error, tolerance_cm, f"{arm} arm misses a seen point by {error:.2f} cm")

    def test_both_arms_and_the_camera_agree_on_a_crooked_zone(self) -> None:
        cam = camera()
        solution = solve(touches_for(("left", "right")), pixels_for(cam))
        self.assertEqual(solution.missing, [])
        self.assert_camera_to_arm(solution, "left", cam, 0.6)
        self.assert_camera_to_arm(solution, "right", cam, 0.6)
        self.assertLess(max(max(r.values()) for r in solution.residuals.values()), 0.6)
        # The zone keeps its real, uneven shape.
        near = np.linalg.norm(solution.corners["near_right"] - solution.corners["near_left"])
        self.assertAlmostEqual(near, np.linalg.norm(np.subtract(TRUE["near_right"], TRUE["near_left"])), delta=0.5)
        self.assertGreater(solution.corners["far_left"][1], 0, "far is away from the arms, +y")
        self.assertAlmostEqual(solution.placements["left"].table_z, -2.4)

    def test_one_arm_and_the_camera_are_enough_to_start(self) -> None:
        cam = camera()
        solution = solve(touches_for(("left",)), pixels_for(cam))
        self.assertEqual(solution.missing, [])
        self.assertNotIn("right", solution.placements)
        self.assert_camera_to_arm(solution, "left", cam, 0.8)

    def test_each_arm_gets_the_table_plane_through_its_own_touches(self) -> None:
        # The venue's left arm read the flat table lower the further it reached.
        touches = touches_for(("left", "right"))
        touches["left"] = {name: (x, y, 0.88 - 0.095 * x + 0.104 * y) for name, (x, y, _) in touches["left"].items()}
        solution = solve(touches, pixels_for(camera()))
        self.assertAlmostEqual(solution.placements["left"].table_height(20.0, -20.0), 0.88 - 1.9 - 2.08, places=6)
        self.assertAlmostEqual(solution.placements["right"].table_height(20.0, 20.0), -1.9, places=6)

    def test_what_is_missing_is_said(self) -> None:
        self.assertTrue(solve({}, None).missing)
        solution = solve(touches_for(("left",)), None)
        self.assertTrue(any("far right" in item for item in solution.missing))


if __name__ == "__main__":
    unittest.main()
