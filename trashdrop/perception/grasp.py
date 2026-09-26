"""Where to pinch an item, read from its top-down silhouette.

Two facts about the SO-101 jaw decide the plan (the measured numbers are in
station.py, next to MAX_GRASP_WIDTH):

* It is not symmetric. One finger is part of the wrist and never moves; the
  other swings on the gripper servo, and the TCP sits on the fixed finger's
  inner face. So the arm lowers the FIXED finger just outside one edge of the
  item, and the moving finger sweeps the item onto it.
* The fingers are parallel at ~32 mm only. Anything wider sits in a V that
  squeezes it outward. A bottle's long narrow neck is worth that trade-off;
  for other shapes, a central cross-section is less likely to be a flimsy tab.

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
# Off-centre narrow tabs on miscellaneous plastic often slip out. Prefer a
# central cross-section unless a long object has a sustained narrow end like
# a bottle neck. A bottle still favours its neck over its 65 mm body.
CENTER_OFFSET_WEIGHT = 1.5
NECK_OFFSET_WEIGHT = 0.5
NECK_MIN_LENGTH_M = 0.14
NECK_MIN_ASPECT = 2.5
NECK_MAX_BODY_RATIO = 0.6
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
# Under the fingers, most slices of the item (all but a quarter) must share a
# band across at least this share of the jaw opening: both pads then press on
# faces that run along them. Otherwise the fingers meet slanted faces -- the
# corner of a box, the tip of a flap -- and squeezing pushes the item out of
# the jaw. It is the band the slices share, not how wide each one is: at the
# venue a cigarette pack lying 10 degrees askew was cut at its corner into
# slices each wide enough, but staggered, and the jaw closed along its edge.
PARALLEL_SIDES = 0.7
# Preferred: both sides of the item running along the fingers, each within
# this angle -- a straight line fitted through its edge under the jaw, so a
# ragged mask does not count as a slant. The venue's items slipped out where
# the planner took a spot that widened under the jaw: the fingers then press
# on a slanted face only. The best such grasp is taken unless it is more than
# PARALLEL_WIDER wider than the best of all, since width matters more (see
# PARALLEL_GRASP_WIDTH); without one, the best grasp is taken as before.
PARALLEL_TILT_DEG = 6.0
PARALLEL_WIDER = 0.005
# A grasp that already failed is not planned again: a candidate this close to
# one, turned this little from it, is passed over while any other will do.
AVOID_M = 0.02
AVOID_DEG = 25.0
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
    avoid=(),
) -> GraspPlan:
    """Choose where to pinch the single item in ``mask``.

    ``fixed_side`` is a direction in pixels, (x, y). The fixed finger goes on
    that side of the item -- toward the arm's base, say, or away from a bin
    wall. By default it goes on the upper side of the image.

    ``avoid`` holds grasps that already failed, as ((x, y) centre, (x, y)
    closing direction) in the mask's pixels: the next best grasp is planned
    instead -- another place, or the jaw turned -- and one of them again only
    if nothing else will do.
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
    offset_weight = (NECK_OFFSET_WEIGHT if _has_bottle_neck(points, mean, long_axis, m_per_px)
                     else CENTER_OFFSET_WEIGHT)
    side = np.array(fixed_side if fixed_side is not None else (0.0, -1.0), dtype=np.float64)
    span = 2 * max(1, int(np.ceil(JAW_SPAN / m_per_px / 2))) + 1

    # The long axis finds a bottle's neck; the other directions find what
    # sticks out sideways, like the cap of a crushed bottle.
    angles = np.arange(ORIENTATIONS) * np.pi / ORIENTATIONS
    windows = []
    narrowest = np.inf
    for along in [long_axis] + [np.array([np.cos(a), np.sin(a)]) for a in angles]:
        across = _perpendicular(along, side)
        first, low, high, seen = _slices(points, mean, along, across)
        # At least MIN_CONTACT, and never under four slices: a coarse mask's
        # ragged pixel must not pass for something to hold.
        contact = min(max(4, int(np.ceil(MIN_CONTACT / m_per_px))), len(low))
        lo, hi, shared, tilt, holding, holes, middles = _windows(low, high, seen, span, contact)
        whole = holding == len(low)  # all of it between the fingers, like a coin
        windows.append((along, across, first, len(low), lo, hi, shared, tilt, holding >= contact, holes, whole,
                        middles))

    avoided = [(np.asarray(centre, float), np.asarray(direction, float) / max(np.linalg.norm(direction), 1e-9))
               for centre, direction in avoid]

    def check(along, across, middle: float) -> GraspPlan | None:
        """The candidate against the full mask; None if too wide there."""

        s = (everything - mean) @ along
        under = (s >= middle - span / 2) & (s < middle + span / 2)
        t = (everything[under] - mean) @ across
        width = float((t.max() - t.min() + 1.0) * m_per_px)
        if width > max_width_m:
            return None
        point = mean + along * middle + across * (t.max() + t.min()) / 2
        return GraspPlan("pinch", (float(point[0]), float(point[1])), (float(across[0]), float(across[1])),
                         width, f"{width * 1000:.0f} mm under the jaw")

    def search(dodge: bool) -> GraspPlan | None:
        nonlocal narrowest
        candidates: list[tuple[float, bool, np.ndarray, np.ndarray, float]] = []
        for along, across, first, slices, lo, hi, shared, tilt, holds, holes, whole, middles in windows:
            extent = hi - lo + 1.0
            widths_m = extent * m_per_px
            square = holds & ~holes & (whole | (shared + 1.0 >= PARALLEL_SIDES * extent))
            narrowest = min(narrowest, float(np.min(np.where(square, widths_m, np.inf))))
            usable = square & (widths_m <= max_width_m)
            if dodge:  # grasps that already slipped: near one of them and turned little from it
                picks = np.flatnonzero(usable)
                points = mean + np.outer(first + middles[picks], along) + np.outer((lo[picks] + hi[picks]) / 2, across)
                for centre, direction in avoided:
                    if abs(float(across @ direction)) > np.cos(np.radians(AVOID_DEG)):
                        usable[picks[np.linalg.norm(points - centre, axis=1) * m_per_px < AVOID_M]] = False
            parallel = whole | (tilt <= PARALLEL_TILT_DEG)
            offset = np.abs(first + middles) / max(slices / 2, 1.0)
            wedge = np.maximum(widths_m - PARALLEL_GRASP_WIDTH, 0.0) / PARALLEL_GRASP_WIDTH
            for pick in np.flatnonzero(usable):
                score = float(wedge[pick] + offset_weight * offset[pick])
                candidates.append((score, bool(parallel[pick]), along, across, float(first + middles[pick])))
        candidates.sort(key=lambda candidate: candidate[0])
        best = None
        for _, parallel, along, across, middle in candidates[:MAX_CHECKS]:
            best = check(along, across, middle)
            if best is not None:
                if parallel:
                    return best
                break
        if best is None:
            return None
        for _, parallel, along, across, middle in [c for c in candidates if c[1]][:MAX_CHECKS]:
            plan = check(along, across, middle)
            if plan is not None and plan.width_m <= best.width_m + PARALLEL_WIDER:
                return plan
        return GraspPlan(best.mode, best.center, best.across, best.width_m,
                         best.reason + ", sides not quite parallel")

    for dodge in ([True, False] if avoided else [False]):  # the failed grasps again only if nothing else will do
        plan = search(dodge)
        if plan is not None:
            return plan

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


def _has_bottle_neck(points: np.ndarray, mean: np.ndarray, long_axis: np.ndarray, m_per_px: float) -> bool:
    """Recognise a long body with a sustained narrow end, not a short tab."""

    across = np.array([-long_axis[1], long_axis[0]])
    _, low, high, seen = _slices(points, mean, long_axis, across)
    length_m = len(low) * m_per_px
    width_m = (np.max(high[seen]) - np.min(low[seen]) + 1.0) * m_per_px
    if length_m < NECK_MIN_LENGTH_M or length_m < NECK_MIN_ASPECT * width_m:
        return False
    widths = (high - low + 1.0) * m_per_px
    count = len(widths)
    middle = widths[count * 3 // 10:count * 7 // 10]
    end_count = max(4, min(count * 15 // 100, int(round(0.03 / m_per_px))))
    ends = (widths[:end_count], widths[-end_count:])
    body_width = float(np.median(middle))
    end_width = min(float(np.median(end)) for end in ends)
    return body_width >= 0.045 and MIN_PINCH_WIDTH <= end_width <= NECK_MAX_BODY_RATIO * body_width


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
    high), the band across that most of its slices share (a quarter of them
    left out, for a ragged mask), how far the steeper of its two sides turns
    away from the fingers (degrees, a line fitted through each), how many
    slices of item the fingers hold, whether a slice inside the item is
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
    rows = np.arange(len(holding))
    quarter = (np.maximum(holding - 1, 0) // 4).astype(np.int64)
    # The quarter-th highest low edge and the quarter-th lowest high edge.
    lows = -np.sort(-view(np.where(seen_p, low_p, -np.inf), span), axis=1)
    highs = np.sort(view(np.where(seen_p, high_p, np.inf), span), axis=1)
    shared = highs[rows, quarter] - lows[rows, quarter]
    # Least-squares slope of each edge over the slices of item under the jaw.
    weight = view(seen_p.astype(np.float64), span)
    x = np.arange(span, dtype=np.float64)
    n, sx, sxx = weight.sum(axis=1), weight @ x, weight @ (x * x)
    below = np.maximum(n * sxx - sx * sx, 1e-9)
    slopes = [np.abs(n * (edge @ x) - sx * edge.sum(axis=1)) / below
              for edge in (view(np.where(seen_p, low_p, 0.0), span), view(np.where(seen_p, high_p, 0.0), span))]
    tilt = np.degrees(np.arctan(np.maximum(*slopes)))
    middles = rows - pad + span / 2
    return (view(low_p, span).min(axis=1), view(high_p, span).max(axis=1), shared, tilt,
            holding, view(hole_p, span).any(axis=1), middles)
