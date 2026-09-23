"""Where to pinch an item, read from its top-down silhouette.

Two facts about the SO-101 jaw decide the plan (the measured numbers are in
station.py, next to MAX_GRASP_WIDTH):

* It is not symmetric. One finger is part of the wrist and never moves; the
  other swings on the gripper servo, and the TCP sits on the fixed finger's
  inner face. So the arm lowers the FIXED finger just outside one edge of the
  item, and the moving finger sweeps the item onto it.
* The fingers are parallel at ~32 mm only. Anything wider sits in a V that
  squeezes it outward, so of two places to hold an item the narrower one wins
  even when it is off-centre: a bottle is taken by the neck, not the body.

The planner works on a binary mask and a scale rather than on a camera, so the
same code runs on live frames, dataset frames and synthetic masks, and needs
nothing beyond numpy. Widths are silhouette widths. An overhead camera sees a
tall item slightly larger than it is -- a can lying down reads ~5 % wide from
70 cm -- which errs toward opening the jaw further.

An item wider than the jaw everywhere gets an "edge" plan instead: the fixed
finger beside the middle of a long edge, the moving finger landing on the
item. That can work for a sheet of paper or cardboard, which buckles or slides
into the jaw, and cannot work for anything solid, so the caller decides
whether to try it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..station import (
    FIXED_JAW_CLEARANCE,
    JAW_OPEN_MARGIN,
    JAW_SPAN,
    MAX_GRASP_WIDTH,
    PARALLEL_GRASP_WIDTH,
)

# How far past the edge the moving finger lands in an edge grasp.
EDGE_BITE = 0.03
# Cost of holding an item at its very end, in units of one PARALLEL_GRASP_WIDTH
# of extra opening. The neck of a bottle beats its body; among equally wide
# places, the one nearest the centre of the item wins.
OFFSET_WEIGHT = 0.5
# Jaw directions tried besides the item's long axis: every 5 degrees.
ORIENTATIONS = 36
# Parts of an item thinner than this are not places to grip: to a planner that
# prefers narrow places, a shadow tail or a strand of tape left in the mask
# looks like a handle. They are removed before candidates are sought, unless
# the whole item is that thin.
MIN_PINCH_WIDTH = 0.015
# Candidates checked against the full mask before giving up on pinching.
MAX_CHECKS = 200
# The jaw may hang past the end of an item -- fingertips on the tip of a cap
# are fine -- as long as this much of the item's length is between the pads.
MIN_CONTACT = 0.012
# Under the fingers, most slices of the item (all but the thinnest quarter)
# must be at least this share of the jaw opening. Otherwise the fingers meet
# slanted faces -- the corner of a box, the tip of a flap -- and squeezing
# pushes the item out of the jaw.
PARALLEL_SIDES = 0.7
MIN_PIXELS = 30


@dataclass(frozen=True)
class GraspPlan:
    """One grasp, in the pixel frame of the mask it was planned on.

    ``center`` is the middle of the grasp line for a pinch, and the point on
    the item's edge for an edge grasp. ``across`` is a unit vector pointing
    from the fixed finger toward the moving one.
    """

    mode: str  # "pinch", "edge" or "none"
    center: tuple[float, float]
    across: tuple[float, float]
    width_m: float
    reason: str

    @property
    def opening_m(self) -> float:
        """How far apart the fingertips must be before the jaw comes down."""

        if self.mode == "edge":
            return FIXED_JAW_CLEARANCE + EDGE_BITE
        return self.width_m + FIXED_JAW_CLEARANCE + JAW_OPEN_MARGIN

    def fixed_finger(self, m_per_px: float) -> tuple[float, float]:
        """Where the fixed finger's inner face comes down: the TCP target."""

        return self._offset(-self._fixed_offset_m() / m_per_px)

    def moving_finger(self, m_per_px: float) -> tuple[float, float]:
        """Where the moving finger comes down, with the jaw at ``opening_m``."""

        return self._offset((self.opening_m - self._fixed_offset_m()) / m_per_px)

    def _fixed_offset_m(self) -> float:
        if self.mode == "edge":
            return FIXED_JAW_CLEARANCE
        return self.width_m / 2 + FIXED_JAW_CLEARANCE

    def _offset(self, distance_px: float) -> tuple[float, float]:
        return (
            self.center[0] + self.across[0] * distance_px,
            self.center[1] + self.across[1] * distance_px,
        )


def plan_grasp(
    mask: np.ndarray,
    m_per_px: float,
    *,
    max_width_m: float = MAX_GRASP_WIDTH,
    fixed_side: tuple[float, float] | None = None,
) -> GraspPlan:
    """Choose where to pinch the single item in ``mask``.

    ``fixed_side`` is a direction in pixels, (x, y). The fixed finger goes on
    that side of the item -- toward the arm's base, say, or away from a bin
    wall. By default it goes on the upper side of the image.
    """

    raw = np.asarray(mask) != 0
    if raw.sum() < MIN_PIXELS:
        return GraspPlan("none", (0.0, 0.0), (0.0, 1.0), 0.0, "no item in the mask")
    # Candidates come from the item without its thin parts; each is then
    # checked against the full mask, so nothing the camera saw ends up under
    # a finger, and the jaw opens for all of it.
    body = _opened(raw, int(MIN_PINCH_WIDTH / 2 / m_per_px))
    if body.sum() < MIN_PIXELS:
        body = raw  # the whole item is thin: a straw, a strip of foil
    points = np.column_stack(np.nonzero(body)[::-1]).astype(np.float64)
    everything = np.column_stack(np.nonzero(raw)[::-1]).astype(np.float64)
    mean = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - mean).T))
    long_axis = vectors[:, 1]  # eigh sorts ascending
    side = np.array(fixed_side if fixed_side is not None else (0.0, -1.0), dtype=np.float64)
    span = 2 * max(1, int(np.ceil(JAW_SPAN / m_per_px / 2))) + 1

    # The long axis finds a bottle's neck; the other directions find what
    # sticks out sideways, like the cap of a crushed bottle.
    angles = np.arange(ORIENTATIONS) * np.pi / ORIENTATIONS
    candidates: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    narrowest = np.inf
    for along in [long_axis] + [np.array([np.cos(a), np.sin(a)]) for a in angles]:
        across = _perpendicular(along, side)
        first, low, high, seen = _slices(points, mean, along, across)
        contact = min(max(1, round(MIN_CONTACT / m_per_px)), len(low))
        lo, hi, typical, holding, holes, middles = _windows(low, high, seen, span, contact)
        extent = hi - lo + 1.0
        widths_m = extent * m_per_px
        whole = holding == len(low)  # all of it between the fingers, like a coin
        square = (holding >= contact) & ~holes & (whole | (typical + 1.0 >= PARALLEL_SIDES * extent))
        narrowest = min(narrowest, float(np.min(np.where(square, widths_m, np.inf))))
        usable = square & (widths_m <= max_width_m)
        offset = np.abs(first + middles) / max(len(low) / 2, 1.0)
        wedge = np.maximum(widths_m - PARALLEL_GRASP_WIDTH, 0.0) / PARALLEL_GRASP_WIDTH
        for pick in np.flatnonzero(usable):
            score = float(wedge[pick] + OFFSET_WEIGHT * offset[pick])
            candidates.append((score, along, across, float(first + middles[pick])))

    candidates.sort(key=lambda candidate: candidate[0])
    for _, along, across, middle in candidates[:MAX_CHECKS]:
        s = (everything - mean) @ along
        under = (s >= middle - span / 2) & (s < middle + span / 2)
        t = (everything[under] - mean) @ across
        width = float((t.max() - t.min() + 1.0) * m_per_px)
        if width > max_width_m:
            continue
        point = mean + along * middle + across * (t.max() + t.min()) / 2
        return GraspPlan(
            "pinch",
            (float(point[0]), float(point[1])),
            (float(across[0]), float(across[1])),
            width,
            f"{width * 1000:.0f} mm under the jaw",
        )

    # Too wide everywhere: offer the middle of the long edge on the fixed side.
    across = _perpendicular(long_axis, side)
    first, low, high, seen = _slices(everything, mean, long_axis, across)
    visible = np.flatnonzero(seen)
    middle = int(visible[np.argmin(np.abs(visible + 0.5 + first))])
    point = mean + long_axis * (first + middle + 0.5) + across * low[middle]
    narrowest_text = f"{narrowest * 1000:.0f} mm" if np.isfinite(narrowest) else "unknown"
    return GraspPlan(
        "edge",
        (float(point[0]), float(point[1])),
        (float(across[0]), float(across[1])),
        float((high[middle] - low[middle] + 1.0) * m_per_px),
        f"wider than {max_width_m * 1000:.0f} mm everywhere (narrowest {narrowest_text})",
    )


def _opened(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary opening with a square: removes every part thinner than it."""

    if radius < 1:
        return mask

    def window_sum(image: np.ndarray) -> np.ndarray:
        size = 2 * radius + 1
        padded = np.pad(image.astype(np.int32), radius)
        integral = np.pad(padded.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        return (integral[size:, size:] - integral[:-size, size:]
                - integral[size:, :-size] + integral[:-size, :-size])

    eroded = window_sum(mask) == (2 * radius + 1) ** 2
    return window_sum(eroded) > 0


def _perpendicular(along: np.ndarray, side: np.ndarray) -> np.ndarray:
    """The jaw's closing direction for ``along``, with the fixed finger on ``side``."""

    across = np.array([-along[1], along[0]])
    return -across if across @ side > 0 else across


def _slices(points, mean, along, across):
    """Extent of the item across ``along``, one-pixel slice by slice."""

    s = (points - mean) @ along
    t = (points - mean) @ across
    first = np.floor(s.min())
    index = (np.floor(s) - first).astype(np.int64)
    count = int(index.max()) + 1
    low = np.full(count, np.inf)
    high = np.full(count, -np.inf)
    np.minimum.at(low, index, t)
    np.maximum.at(high, index, t)
    return first, low, high, np.bincount(index, minlength=count) > 0


def _windows(low, high, seen, span: int, contact: int):
    """Every place the jaw could sit along one direction.

    Returns, per window: the extent of the item under the fingers (low,
    high), a typical slice width there (the thinnest quarter excluded), how
    many slices of item the fingers hold, whether a slice inside the item is
    missing, and the window's middle in slice coordinates.

    The fingers are straight, so a candidate's width is the full extent of
    every slice they cover. A window with a missing slice is refused: an
    invisible stretch of item -- clear plastic, say -- has an unknown width.
    """

    count = len(low)
    pad = max(span - contact, 0)  # windows may hang this far past either end
    outside = np.zeros(pad, bool)
    low_p = np.concatenate([np.full(pad, np.inf), low, np.full(pad, np.inf)])
    high_p = np.concatenate([np.full(pad, -np.inf), high, np.full(pad, -np.inf)])
    seen_p = np.concatenate([outside, seen, outside])
    hole_p = np.concatenate([outside, ~seen, outside])

    view = np.lib.stride_tricks.sliding_window_view
    holding = view(seen_p, span).sum(axis=1)
    widths = np.sort(view(np.where(seen_p, high_p - low_p, np.inf), span), axis=1)
    quarter = (np.maximum(holding - 1, 0) // 4).astype(np.int64)
    typical = widths[np.arange(len(widths)), quarter]
    middles = np.arange(len(widths)) - pad + span / 2
    return (view(low_p, span).min(axis=1), view(high_p, span).max(axis=1), typical,
            holding, view(hole_p, span).any(axis=1), middles)
