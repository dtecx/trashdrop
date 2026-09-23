"""The metric zone follows the printed sheet while staying camera-centred."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")

from trashdrop.camera.markers import MARKER_WORLD, render_sheet  # noqa: E402
from trashdrop.dataset.autolabel import _foreground  # noqa: E402
from trashdrop.dataset.zone import calibrate_zone, load_zone, save_zone, zone_size  # noqa: E402
from trashdrop.perception.calibration import HomographyCalibration, detect_aruco_corners  # noqa: E402


def photographed_sheet():
    page = cv2.resize(render_sheet(), (960, 680), interpolation=cv2.INTER_AREA)
    source = np.float32([[0, 0], [959, 0], [959, 679], [0, 679]])
    destination = np.float32([[45, 30], [610, 12], [585, 465], [25, 445]])
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(page, matrix, (640, 480), borderValue=180)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


def test_metric_zone_centered_on_camera_with_perspective():
    frame = photographed_sheet()
    zone = calibrate_zone(frame, 20, 15)
    assert zone.corners is not None
    assert zone.matches(frame)
    assert zone.x > 0 and zone.y > 0
    calibration = HomographyCalibration(detect_aruco_corners(frame, [0, 1, 2, 3]),
                                        [MARKER_WORLD[i] for i in range(4)])
    center_world = calibration.pixel_to_world(320, 240)
    for point, offset in zip(zone.corners, [(-0.1, 0.075), (0.1, 0.075),
                                            (0.1, -0.075), (-0.1, -0.075)]):
        actual = calibration.pixel_to_world(*point)
        assert actual == pytest.approx((center_world[0] + offset[0],
                                        center_world[1] + offset[1]), abs=1e-5)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "zone.json"
        save_zone(path, zone)
        restored = load_zone(path)
        assert restored is not None
        assert np.asarray(restored.corners) == pytest.approx(np.asarray(zone.corners))


def test_polygon_mask_excludes_bounding_box_corners():
    frame = photographed_sheet()
    zone = calibrate_zone(frame, 20, 15)
    mask = zone.analysis_mask()
    assert mask.shape == zone.crop(frame).shape[:2]
    assert mask[0, 0] == 0
    assert mask[mask.shape[0] // 2, mask.shape[1] // 2] == 255


def test_exposure_compensation_precedes_polygon_clipping():
    frame = photographed_sheet()
    zone = calibrate_zone(frame, 20, 15)
    reference = np.full((zone.height, zone.width, 3), 130, np.uint8)
    live = np.full_like(reference, 165)
    cv2.circle(live, (zone.width // 2, zone.height // 2), 24, (30, 30, 30), -1)
    cv2.rectangle(live, (0, 0), (18, 18), (30, 30, 30), -1)
    mask, fit = _foreground(live, reference, valid_mask=zone.analysis_mask())
    assert fit.trusted
    assert mask[zone.height // 2, zone.width // 2] == 255
    assert mask[0, 0] == 0


def test_missing_marker_cannot_produce_zone():
    with pytest.raises(ValueError, match="four ArUco markers"):
        calibrate_zone(np.full((480, 640, 3), 255, np.uint8), 20, 15)


def test_zone_dimensions_come_from_the_selected_config(tmp_path):
    config = tmp_path / "capture_zone.toml"
    config.write_text('[zone]\nwidth_cm = 30.0\nheight_cm = 20.0\ncenter = "camera"\n')
    assert zone_size(config) == (30.0, 20.0)
