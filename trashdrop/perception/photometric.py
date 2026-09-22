"""Cancel exposure and white-balance drift before comparing to a reference.

Background subtraction assumes the camera answers the same light with the same
pixel values. Consumer cameras do not: put a dark item on a light table and
auto-exposure brightens the *whole* frame, so a naive difference lights up
everywhere and the item is lost in it. Locking exposure is the textbook fix,
but it is frequently not available -- OpenCV on macOS talks to AVFoundation,
where most cameras ignore the exposure property entirely.

So compensate instead of demanding a lock. The camera's response over a short
interval is close to a per-channel linear map, and the reference frame can be
pushed through that map before differencing. Two passes, because which pixels
are background is exactly what we are trying to find out:

1. Difference naively to get a rough guess at what changed.
2. Fit gain and offset on the pixels that did *not* change, which are by
   definition the ones still showing the table.
3. Difference again against the corrected reference.

The result is a pipeline that does not care whether exposure was locked, which
is what makes it safe to shoot a dataset on one camera and run the demo on
another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this fraction of usable background pixels the fit is not trustworthy:
# either the item fills the frame or the camera moved.
MIN_BACKGROUND_FRACTION = 0.25
# Gains outside this range mean something other than exposure drift happened.
GAIN_LIMITS = (0.4, 2.5)
# Sample at most this many pixels per channel; a full 1080p fit is wasteful.
MAX_SAMPLES = 40_000
# Below this spread (8-bit levels) in the reference patch, gain and offset are
# not separately identifiable and only the offset is estimated. This is the
# normal case, not an edge case: a plain white tablecloth -- exactly what this
# cell is built on -- has almost no texture to pin a gain to.
MIN_GAIN_CONTRAST = 6.0


@dataclass
class PhotometricFit:
    """Per-channel linear correction mapping reference pixels onto live ones."""

    gain: np.ndarray  # shape (3,)
    offset: np.ndarray  # shape (3,)
    background_fraction: float
    trusted: bool
    reason: str = ""

    @property
    def drifted(self) -> bool:
        """Whether the camera visibly changed its response at all."""

        return bool(np.any(np.abs(self.gain - 1.0) > 0.02) or np.any(np.abs(self.offset) > 2.0))

    def describe(self) -> str:
        gains = ", ".join(f"{g:.3f}" for g in self.gain)
        offsets = ", ".join(f"{o:+.1f}" for o in self.offset)
        return f"gain=({gains}) offset=({offsets}) background={self.background_fraction:.0%}"

    def apply(self, reference: np.ndarray) -> np.ndarray:
        """Push a reference frame through the camera's current response."""

        if not self.trusted:
            return reference
        corrected = reference.astype(np.float32) * self.gain + self.offset
        return np.clip(corrected, 0, 255).astype(reference.dtype)


def _fit_channel(reference: np.ndarray, live: np.ndarray) -> tuple[float, float]:
    """Least-squares gain and offset for one channel.

    Falls back to offset-only on a low-contrast surface. Forcing a gain there
    fits the sensor noise instead of the exposure change, which produces a
    wild multiplier and destroys the very difference we are trying to measure.
    """

    ref = reference.astype(np.float32)
    lit = live.astype(np.float32)
    ref_mean, lit_mean = ref.mean(), lit.mean()
    spread = float(ref.std())
    if spread < MIN_GAIN_CONTRAST:
        return 1.0, float(lit_mean - ref_mean)
    gain = float(((ref - ref_mean) * (lit - lit_mean)).mean() / (spread * spread))
    return gain, float(lit_mean - gain * ref_mean)


def estimate_photometric_fit(
    live: np.ndarray,
    reference: np.ndarray,
    rough_threshold: int = 40,
) -> PhotometricFit:
    """Estimate how the camera's response drifted since the reference frame."""

    if live.shape != reference.shape:
        return PhotometricFit(
            gain=np.ones(3, np.float32),
            offset=np.zeros(3, np.float32),
            background_fraction=0.0,
            trusted=False,
            reason="frame and reference differ in size",
        )

    # Pass 1: a coarse difference marks anything that might be an item. The
    # threshold is deliberately generous -- excluding too much is harmless,
    # since what remains is still plenty of background to fit on.
    rough = np.abs(live.astype(np.int16) - reference.astype(np.int16)).max(axis=2)
    background = rough <= rough_threshold
    fraction = float(background.mean())

    if fraction < MIN_BACKGROUND_FRACTION:
        return PhotometricFit(
            gain=np.ones(3, np.float32),
            offset=np.zeros(3, np.float32),
            background_fraction=fraction,
            trusted=False,
            reason=(
                f"only {fraction:.0%} of the frame still looks like the table; "
                "the reference is stale or the camera moved"
            ),
        )

    indices = np.flatnonzero(background.ravel())
    if indices.size > MAX_SAMPLES:
        step = indices.size // MAX_SAMPLES
        indices = indices[::step][:MAX_SAMPLES]

    gains = np.ones(3, np.float32)
    offsets = np.zeros(3, np.float32)
    for channel in range(3):
        ref_pixels = reference[..., channel].ravel()[indices]
        live_pixels = live[..., channel].ravel()[indices]
        gain, offset = _fit_channel(ref_pixels, live_pixels)
        gains[channel], offsets[channel] = gain, offset

    if np.any(gains < GAIN_LIMITS[0]) or np.any(gains > GAIN_LIMITS[1]):
        return PhotometricFit(
            gain=np.ones(3, np.float32),
            offset=np.zeros(3, np.float32),
            background_fraction=fraction,
            trusted=False,
            reason=f"implausible gain {np.round(gains, 2).tolist()}; not exposure drift",
        )

    return PhotometricFit(
        gain=gains, offset=offsets, background_fraction=fraction, trusted=True
    )


def compensated_reference(
    live: np.ndarray,
    reference: np.ndarray,
    rough_threshold: int = 40,
) -> tuple[np.ndarray, PhotometricFit]:
    """Return the reference corrected to the live frame's exposure."""

    fit = estimate_photometric_fit(live, reference, rough_threshold)
    return fit.apply(reference), fit
