"""The camera verdict, driven with synthetic footage of cameras behaving badly.

No camera needed: each test fabricates frames with one specific defect and
checks that the defect is both measured and named in a way someone can act on.
"""

from __future__ import annotations

import unittest

import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
import numpy as np  # noqa: E402

from trashdrop.dataset.camcheck import (  # noqa: E402
    MAX_FOCUS_SWING_RATIO,
    MAX_GEOMETRIC_DRIFT_PX,
    MAX_IDLE_BRIGHTNESS_DRIFT,
    CameraReport,
    analyse_idle,
    analyse_response,
    judge,
)

WIDTH, HEIGHT = 320, 240


def scene(seed: int = 0) -> np.ndarray:
    """A textured scene, so sharpness and phase correlation have something to bite on."""

    rng = np.random.default_rng(seed)
    base = rng.integers(60, 220, (HEIGHT // 6, WIDTH // 6, 3)).astype(np.uint8)
    frame = cv2.resize(base, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(frame, (40, 40), (150, 140), (240, 240, 240), 3)
    cv2.circle(frame, (230, 170), 34, (20, 20, 20), 2)
    return frame


def steady(count: int = 12) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    base = scene()
    return [
        np.clip(base.astype(np.float32) + rng.normal(0, 1.2, base.shape), 0, 255).astype(np.uint8)
        for _ in range(count)
    ]


def report_from(frames: list[np.ndarray]) -> CameraReport:
    measured = analyse_idle(frames)
    report = CameraReport(
        idle_brightness_drift=measured["brightness_drift"],
        focus_swing_ratio=measured["focus_swing_ratio"],
        geometric_drift_px=measured["geometric_drift_px"],
    )
    return judge(report)


class SteadyCameraTests(unittest.TestCase):
    def test_a_steady_camera_passes(self) -> None:
        report = report_from(steady())
        self.assertTrue(report.usable, report.render())

    def test_measurements_are_within_limits(self) -> None:
        measured = analyse_idle(steady())
        self.assertLess(measured["brightness_drift"], MAX_IDLE_BRIGHTNESS_DRIFT)
        self.assertLess(measured["focus_swing_ratio"], MAX_FOCUS_SWING_RATIO)
        self.assertLess(measured["geometric_drift_px"], MAX_GEOMETRIC_DRIFT_PX)

    def test_two_frames_is_the_minimum(self) -> None:
        with self.assertRaises(ValueError):
            analyse_idle(steady(1))


class BadCameraTests(unittest.TestCase):
    def test_wandering_exposure_is_caught(self) -> None:
        base = scene()
        frames = [
            np.clip(base.astype(np.float32) + shift, 0, 255).astype(np.uint8)
            for shift in (0, 6, -5, 9, -8, 4, -7, 10)
        ]
        report = report_from(frames)
        self.assertFalse(report.usable)
        self.assertTrue(any("brightness wanders" in w for w in report.warnings), report.warnings)

    def test_focus_hunting_is_caught(self) -> None:
        base = scene()
        # Alternating blur is what refocusing looks like frame to frame.
        frames = []
        for index in range(10):
            radius = 1 + 2 * (index % 3)
            frames.append(cv2.GaussianBlur(base, (2 * radius + 1, 2 * radius + 1), 0))
        report = report_from(frames)
        self.assertFalse(report.usable)
        self.assertTrue(any("focus is hunting" in w for w in report.warnings), report.warnings)

    def test_a_moving_view_is_caught(self) -> None:
        base = scene()
        frames = []
        for index in range(8):
            matrix = np.float32([[1, 0, index * 1.5], [0, 1, index * 0.8]])
            frames.append(cv2.warpAffine(base, matrix, (WIDTH, HEIGHT)))
        report = report_from(frames)
        self.assertFalse(report.usable)
        self.assertTrue(any("the view moved" in w for w in report.warnings), report.warnings)

    def test_the_moving_view_warning_mentions_auto_framing(self) -> None:
        # Worth naming explicitly: a webcam that digitally pans to follow a
        # subject looks like a loose mount and is fixed somewhere else entirely.
        base = scene()
        frames = [
            cv2.warpAffine(base, np.float32([[1, 0, i * 2.0], [0, 1, 0]]), (WIDTH, HEIGHT))
            for i in range(6)
        ]
        report = report_from(frames)
        self.assertTrue(any("auto-framing" in w for w in report.warnings), report.warnings)


class ResponseTests(unittest.TestCase):
    def test_an_item_plus_re_exposure_is_still_recoverable(self) -> None:
        reference = scene()
        occupied = reference.copy()
        cv2.rectangle(occupied, (130, 95), (190, 140), (30, 30, 30), -1)
        occupied = np.clip(occupied.astype(np.float32) + 25, 0, 255).astype(np.uint8)

        response = analyse_response(reference, occupied)
        self.assertTrue(response["compensation_recovers"], response["fit"].reason)
        self.assertGreater(response["exposure_shift"], 8.0)

    def test_a_large_exposure_step_is_noted_not_failed(self) -> None:
        report = CameraReport(exposure_shift=30.0, compensation_recovers=True)
        judge(report)
        self.assertTrue(report.usable, report.warnings)
        self.assertTrue(any("re-exposed" in n for n in report.notes), report.notes)

    def test_a_failed_fit_is_a_problem(self) -> None:
        report = CameraReport(compensation_recovers=False)
        judge(report)
        self.assertFalse(report.usable)


class RenderTests(unittest.TestCase):
    def test_render_states_a_verdict(self) -> None:
        text = report_from(steady()).render()
        self.assertIn("VERDICT", text)
        self.assertIn("brightness drift", text)

    def test_a_failing_report_says_what_to_fix(self) -> None:
        report = CameraReport(idle_brightness_drift=9.0)
        judge(report)
        text = report.render()
        self.assertIn("PROBLEM", text)
        self.assertIn("auto-exposure", text)


if __name__ == "__main__":
    unittest.main()
