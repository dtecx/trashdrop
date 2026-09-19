"""Pixel-to-table conversion, including the hardware path."""

from __future__ import annotations

import unittest

import pytest

pytest.importorskip("cv2", reason="needs the dataset extra")
import numpy as np  # noqa: E402

from trashdrop.perception.calibration import (  # noqa: E402
    HomographyCalibration,
    PinholeTopDown,
)
from trashdrop.station import CAMERA, PICK_ZONE  # noqa: E402


def sim_camera() -> PinholeTopDown:
    return PinholeTopDown(
        height=CAMERA.height,
        center_x=CAMERA.center_x,
        center_y=CAMERA.center_y,
        fovy_degrees=CAMERA.fovy_degrees,
        width=CAMERA.width,
        height_px=CAMERA.height_px,
    )


class PinholeTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        camera = sim_camera()
        for x, y in PICK_ZONE.corners():
            u, v = camera.world_to_pixel(x, y)
            back = camera.pixel_to_world(u, v)
            self.assertAlmostEqual(back[0], x, places=9)
            self.assertAlmostEqual(back[1], y, places=9)

    def test_image_rows_increase_away_from_plus_y(self) -> None:
        camera = sim_camera()
        _, v_near = camera.world_to_pixel(0.0, -0.14)
        _, v_far = camera.world_to_pixel(0.0, -0.29)
        self.assertLess(v_near, v_far)


class HomographyTests(unittest.TestCase):
    def test_recovers_a_known_mapping(self) -> None:
        camera = sim_camera()
        corners = PICK_ZONE.corners()
        image_points = [camera.world_to_pixel(x, y) for x, y in corners]
        calibration = HomographyCalibration(image_points, corners)

        self.assertLess(max(calibration.residuals_mm()), 1.0)
        for (x, y), (u, v) in zip(corners, image_points):
            got = calibration.pixel_to_world(u, v)
            self.assertAlmostEqual(got[0], x, places=4)
            self.assertAlmostEqual(got[1], y, places=4)

    def test_agrees_with_the_pinhole_away_from_the_reference_points(self) -> None:
        camera = sim_camera()
        corners = PICK_ZONE.corners()
        calibration = HomographyCalibration(
            [camera.world_to_pixel(x, y) for x, y in corners], corners
        )
        u, v = camera.world_to_pixel(PICK_ZONE.center_x, PICK_ZONE.center_y)
        got = calibration.pixel_to_world(u, v)
        self.assertAlmostEqual(got[0], PICK_ZONE.center_x, places=3)
        self.assertAlmostEqual(got[1], PICK_ZONE.center_y, places=3)

    def test_collinear_points_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            HomographyCalibration(
                [(0, 0), (10, 0), (20, 0), (30, 0)],
                [(0, 0), (0.1, 0), (0.2, 0), (0.3, 0)],
            )

    def test_round_trip_survives_save_and_load(self) -> None:
        import tempfile
        from pathlib import Path

        camera = sim_camera()
        corners = PICK_ZONE.corners()
        calibration = HomographyCalibration(
            [camera.world_to_pixel(x, y) for x, y in corners], corners
        )
        with tempfile.TemporaryDirectory() as directory:
            path = calibration.save(Path(directory) / "calib.json")
            reloaded = HomographyCalibration.load(path)
        self.assertTrue(np.allclose(calibration.matrix, reloaded.matrix))


if __name__ == "__main__":
    unittest.main()
