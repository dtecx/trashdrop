"""Exposure compensation: the thing that lets a dataset survive auto-exposure.

The tests that matter here are the pair showing that a naive difference
*fails* on an exposure jump and the compensated one *works* -- without that
contrast the module is just extra code.
"""

from __future__ import annotations

import unittest

import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
import numpy as np  # noqa: E402

from trashdrop.perception.photometric import (  # noqa: E402
    MIN_GAIN_CONTRAST,
    compensated_reference,
    estimate_photometric_fit,
)

WIDTH, HEIGHT = 320, 240


def flat_table(seed: int = 0) -> np.ndarray:
    """A plain tablecloth: almost no texture, like the real one."""

    rng = np.random.default_rng(seed)
    frame = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)
    return np.clip(frame + rng.normal(0, 2.0, frame.shape), 0, 255).astype(np.uint8)


def textured_table(seed: int = 0) -> np.ndarray:
    """A surface with enough structure that a gain is identifiable."""

    rng = np.random.default_rng(seed)
    base = rng.integers(90, 210, (HEIGHT // 8, WIDTH // 8, 3)).astype(np.uint8)
    return cv2.resize(base, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)


def expose(frame: np.ndarray, gain: float, offset: float) -> np.ndarray:
    """Simulate the camera re-exposing."""

    return np.clip(frame.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)


def place_item(frame: np.ndarray, box=(120, 90, 60, 45), colour=(35, 35, 35)) -> np.ndarray:
    x, y, w, h = box
    out = frame.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), colour, -1)
    return out


def foreground_fraction(live, reference, threshold: int = 28) -> float:
    difference = cv2.absdiff(live, reference).max(axis=2)
    return float((difference > threshold).mean())


class RecoveryTests(unittest.TestCase):
    def test_recovers_a_known_gain_on_a_textured_surface(self) -> None:
        reference = textured_table()
        live = expose(reference, gain=1.18, offset=4.0)
        fit = estimate_photometric_fit(live, reference)
        self.assertTrue(fit.trusted, fit.reason)
        self.assertTrue(np.allclose(fit.gain, 1.18, atol=0.05), fit.describe())

    def test_flat_surface_falls_back_to_offset_only(self) -> None:
        # Gain and offset are not separately identifiable without texture, so
        # the honest answer is gain 1 plus a brightness shift -- not a gain
        # fitted to sensor noise.
        reference = flat_table()
        self.assertLess(reference.std(), MIN_GAIN_CONTRAST)
        live = expose(reference, gain=1.0, offset=18.0)
        fit = estimate_photometric_fit(live, reference)
        self.assertTrue(fit.trusted, fit.reason)
        self.assertTrue(np.allclose(fit.gain, 1.0, atol=0.01), fit.describe())
        self.assertTrue(np.allclose(fit.offset, 18.0, atol=2.0), fit.describe())

    def test_no_drift_is_reported_as_no_drift(self) -> None:
        reference = textured_table()
        fit = estimate_photometric_fit(reference.copy(), reference)
        self.assertTrue(fit.trusted)
        self.assertFalse(fit.drifted, fit.describe())

    def test_changed_lighting_uses_the_whole_table_for_offset_fallback(self) -> None:
        height, width = 318, 320
        gradient = np.linspace(140, 200, height, dtype=np.float32)[:, None, None]
        reference = np.broadcast_to(gradient, (height, width, 3)).astype(np.uint8).copy()
        live = expose(reference, gain=0.2, offset=140)
        live[100:150, 120:190] = 25

        fit = estimate_photometric_fit(live, reference)
        self.assertTrue(fit.trusted, fit.reason)
        self.assertIn("offset-only", fit.reason)
        self.assertTrue(np.allclose(fit.gain, 1.0))
        # The old floor-stride sampler used only the first 40k of ~100k
        # background pixels and returned an offset biased toward the top.
        self.assertTrue(np.allclose(fit.offset, 3.0, atol=2.0), fit.describe())


class DetectionSurvivesExposureTests(unittest.TestCase):
    """The point of the module, stated as a pair of tests."""

    def setUp(self) -> None:
        self.reference = flat_table()
        # An item arrives AND the camera brightens the whole frame in response,
        # which is exactly what auto-exposure does with a dark item on a light
        # table.
        # 40 levels: comfortably past the 28-level difference threshold, and
        # well within what a webcam actually does when a dark item lands on a
        # light table.
        self.live = expose(place_item(self.reference), gain=1.0, offset=40.0)

    def test_naive_subtraction_floods(self) -> None:
        flooded = foreground_fraction(self.live, self.reference)
        # The item covers about 3.5% of the frame; a naive difference marks
        # nearly everything instead.
        self.assertGreater(flooded, 0.5, "expected the naive difference to flood")

    def test_compensated_subtraction_finds_just_the_item(self) -> None:
        corrected, fit = compensated_reference(self.live, self.reference)
        self.assertTrue(fit.trusted, fit.reason)
        self.assertTrue(fit.drifted, "a 40-level shift should register as drift")

        fraction = foreground_fraction(self.live, corrected)
        self.assertLess(fraction, 0.10, "compensated difference still floods")

        difference = cv2.absdiff(self.live, corrected).max(axis=2)
        mask = (difference > 28).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertTrue(contours, "no region survived")
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        for got, want in zip((x, y, w, h), (120, 90, 60, 45)):
            self.assertAlmostEqual(got, want, delta=8)


class RefusalTests(unittest.TestCase):
    def test_mismatched_sizes_are_refused(self) -> None:
        fit = estimate_photometric_fit(flat_table(), flat_table()[:100])
        self.assertFalse(fit.trusted)
        self.assertIn("size", fit.reason)

    def test_a_covered_frame_is_refused(self) -> None:
        # Nearly the whole view changed: the camera moved, or the reference is
        # stale. Fitting on what little is left would invent a correction.
        reference = flat_table()
        live = place_item(reference, box=(5, 5, WIDTH - 10, HEIGHT - 10), colour=(20, 20, 20))
        fit = estimate_photometric_fit(live, reference)
        self.assertFalse(fit.trusted)
        self.assertIn("stale", fit.reason)

    def test_an_untrusted_fit_leaves_the_reference_alone(self) -> None:
        reference = flat_table()
        live = place_item(reference, box=(5, 5, WIDTH - 10, HEIGHT - 10), colour=(20, 20, 20))
        corrected, fit = compensated_reference(live, reference)
        self.assertFalse(fit.trusted)
        self.assertTrue(np.array_equal(corrected, reference))


if __name__ == "__main__":
    unittest.main()
