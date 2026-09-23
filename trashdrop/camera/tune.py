"""Find camera settings for this rig, so they can be frozen in camera.toml.

Three settings, three strategies -- each chosen after watching a simpler one
fail on the real C920:

* **Exposure and gain** -- on macOS, left alone. Measured on the C920 while
  QuickTime streamed: a manual exposure written over USB was replaced within
  half a second, with the camera still reporting manual mode. macOS runs its
  own auto-exposure for UVC cameras while a stream is open and rewrites both
  values. Fighting it is what produced every strange reading so far -- a tune
  that walked down to 0.3 ms while the picture stayed at median 142, a white
  QuickTime preview, "brightness wandering" in camcheck. Focus and white
  balance, tested the same way, are left untouched by macOS and stay ours.
  Elsewhere exposure is measured from the image, from 30, 20 and 10 ms --
  whole mains periods, so LED light cannot band -- after checking that the
  image actually responds to it.
* **Focus** -- autofocus gives a starting point, a sweep over the printed
  target decides. The first version measured sharpness over the whole frame
  and chose position 140, which is badly out of focus at 70 cm. At gain 159 the
  Laplacian of a mostly blank frame is sensor noise, and the curve was noise
  with spikes. Now sharpness is measured only on the Siemens star, located by
  the sheet's ArUco markers, on several frames averaged and downscaled, and
  the curve is smoothed before its peak is taken.
* **White balance** -- the C920 does not report what auto white balance chose:
  the first tune read back a value a test had written minutes earlier. So it is
  measured instead: the colour temperature is searched until the sheet's white
  paper comes out neutral, red equal to blue.

Put the marker sheet (``trashdrop camera markers``) flat in the middle of the
view first. Without it the focus sweep does not run at all -- sweeping the
frame centre is how the second real tune chose 250, the macro end of the
range -- and white balance stays automatic; both are reported.

``trashdrop camera preview`` shows what the tune sees, live.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The smoothed peak must stand this far above a typical position, or the
# sweep found noise rather than focus.
MIN_PEAK_RATIO = 1.25
COARSE_POINTS = 25
AVERAGE_FRAMES = 4
FOCUS_SETTLE_FRAMES = 12
WHITE_SETTLE_FRAMES = 10
# Gain above this fraction of its range means the scene is too dark: the
# frames are noisy, and every later measurement suffers.
DIM_GAIN_FRACTION = 0.5
# Paper brighter than this is clipped, and clipped paper looks neutral at any
# colour temperature, so it cannot be balanced on.
PAPER_CLIP_LEVEL = 245.0


class TuneError(RuntimeError):
    pass


@dataclass
class SheetView:
    """Where the marker sheet is in the image."""

    focus_roi: tuple[int, int, int, int]  # x0, y0, x1, y1 around the star
    paper_patches: list[tuple[int, int, int, int]]
    markers: int


@dataclass
class TuneResult:
    controls: dict[str, int] = field(default_factory=dict)
    focus_curve: dict[int, float] = field(default_factory=dict)
    autofocus_guess: int | None = None
    notes: list[str] = field(default_factory=list)


# --- frames -----------------------------------------------------------------


def _drain(capture, frames: int):
    """Read frames so the camera's state catches up; return the last one."""

    last = None
    for _ in range(frames):
        ok, frame = capture.read()
        if ok and frame is not None:
            last = frame
    return last


def average_frames(capture, count: int = AVERAGE_FRAMES) -> np.ndarray:
    """Mean of several frames: halves the sensor noise for every four."""

    frames = []
    for _ in range(count):
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame.astype(np.float32))
    if not frames:
        raise TuneError("the camera stopped delivering frames")
    return np.clip(np.mean(frames, axis=0), 0, 255).astype(np.uint8)


# --- the sheet --------------------------------------------------------------


def locate_sheet(frame) -> SheetView | None:
    """Find the marker sheet: the star to focus on and paper to balance on."""

    import cv2

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None
    centres = {
        int(marker): corner.reshape(4, 2).mean(axis=0)
        for marker, corner in zip(ids.flatten(), corners)
        if int(marker) in (0, 1, 2, 3)
    }
    if len(centres) < 3:
        return None

    # The star sits at the centre of the zone: the midpoint of a diagonal.
    if 0 in centres and 2 in centres:
        star = (centres[0] + centres[2]) / 2
    elif 1 in centres and 3 in centres:
        star = (centres[1] + centres[3]) / 2
    else:
        return None
    edges = [
        np.linalg.norm(centres[a] - centres[b])
        for a, b in ((0, 1), (3, 2), (0, 3), (1, 2))
        if a in centres and b in centres
    ]
    width = max(edges)  # the 20 cm side, in pixels

    height_px, width_px = frame.shape[:2]

    def box(centre, half) -> tuple[int, int, int, int]:
        x, y = centre
        return (
            max(0, int(x - half)), max(0, int(y - half)),
            min(width_px, int(x + half)), min(height_px, int(y + half)),
        )

    # The star is 7.6 cm across on a 20 cm zone; a quarter-width half-side
    # frames it with a margin. Paper patches sit halfway from the star to
    # each marker: clear of the star, the outline and the markers.
    patches = [box(star + (centre - star) / 2, 0.04 * width) for centre in centres.values()]
    return SheetView(focus_roi=box(star, 0.25 * width), paper_patches=patches, markers=len(centres))


def centre_roi(frame, fraction: float = 0.5) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    return (
        int(width * (0.5 - fraction / 2)), int(height * (0.5 - fraction / 2)),
        int(width * (0.5 + fraction / 2)), int(height * (0.5 + fraction / 2)),
    )


# --- focus ------------------------------------------------------------------


def roi_sharpness(frame, roi: tuple[int, int, int, int]) -> float:
    """Variance of the Laplacian inside ``roi``, at half resolution.

    Halving the resolution averages away pixel-level sensor noise, which the
    Laplacian would otherwise amplify, while the star's edges -- centimetres
    wide -- survive untouched.
    """

    import cv2

    x0, y0, x1, y1 = roi
    grey = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def sharpness(frame, region: float = 0.5) -> float:
    """Sharpness of the middle of the frame (used by tests and camcheck)."""

    return roi_sharpness(frame, centre_roi(frame, region))


def _smooth(curve: dict[int, float]) -> dict[int, float]:
    """Three-point moving average along the lens positions."""

    positions = sorted(curve)
    smoothed = {}
    for index, position in enumerate(positions):
        window = positions[max(0, index - 1) : index + 2]
        smoothed[position] = float(np.mean([curve[p] for p in window]))
    return smoothed


def sweep_focus(
    capture,
    camera,
    roi: tuple[int, int, int, int],
    *,
    settle_frames: int = FOCUS_SETTLE_FRAMES,
    average: int = AVERAGE_FRAMES,
) -> tuple[int, dict[int, float], float]:
    """Coarse pass over the whole range, fine pass around the smoothed peak.

    Returns the chosen position, the raw curve, and how clear the peak is.
    """

    known = camera.ranges()["focus"]
    step = max(1, known.step)
    span = known.maximum - known.minimum
    coarse = max(step, int(round(span / COARSE_POINTS / step)) * step)

    def measure(position: int) -> float:
        camera.set("focus", position)
        _drain(capture, settle_frames)
        return roi_sharpness(average_frames(capture, average), roi)

    curve = {position: measure(position) for position in range(known.minimum, known.maximum + 1, coarse)}
    smoothed = _smooth(curve)
    peak = max(smoothed, key=smoothed.get)

    for position in range(max(known.minimum, peak - coarse), min(known.maximum, peak + coarse) + 1, step):
        if position not in curve:
            curve[position] = measure(position)

    smoothed = _smooth(curve)
    best = max(smoothed, key=smoothed.get)
    typical = float(np.median(list(smoothed.values())))
    clarity = smoothed[best] / typical if typical > 0 else 0.0
    return best, dict(sorted(curve.items())), clarity


# --- white balance ----------------------------------------------------------


def paper_colour(frame, patches) -> np.ndarray:
    """Mean BGR over the paper patches."""

    samples = [frame[y0:y1, x0:x1].reshape(-1, 3) for x0, y0, x1, y1 in patches if x1 > x0 and y1 > y0]
    return np.concatenate(samples).astype(np.float32).mean(axis=0)


def balance_white(
    capture,
    camera,
    patches,
    *,
    settle_frames: int = WHITE_SETTLE_FRAMES,
    average: int = AVERAGE_FRAMES,
) -> tuple[int, float]:
    """Search the colour temperature that makes the paper neutral.

    A higher setting tells the camera the light is bluer than it is, so it
    corrects less and the image comes out warmer: red over blue rises with the
    setting, and a bisection finds where they are equal. Returns the setting
    and the remaining red/blue ratio.
    """

    known = camera.ranges()["white_balance"]
    step = max(1, known.step)
    low, high = known.minimum, known.maximum
    camera.set("white_balance_auto", 0)

    def ratio_at(value: int) -> float:
        camera.set("white_balance", value)
        _drain(capture, settle_frames)
        blue, _, red = paper_colour(average_frames(capture, average), patches)
        return float(red / max(blue, 1.0))

    while high - low > step:
        middle = low + ((high - low) // 2 // step) * step
        if middle in (low, high):
            break
        if ratio_at(middle) > 1.0:
            high = middle
        else:
            low = middle
    best = low if abs(ratio_at(low) - 1.0) <= abs(ratio_at(high) - 1.0) else high
    final = ratio_at(best)
    return best, final


# --- exposure ---------------------------------------------------------------

# The brightest percentile of the frame -- the paper when the sheet is in view
# -- should land here: bright enough to keep gain low, far enough under 255
# that paper keeps its colour for white balance and items keep contrast.
TARGET_BRIGHT = 215.0
CLIP_LIMIT = 240.0
# A tenfold exposure change must move the brightest paper at least this much,
# or exposure is not ours to set. Checked after this many frames, which gives
# a controller like macOS's time to undo the change.
MIN_EXPOSURE_RESPONSE = 40.0
RESPONSE_SETTLE_FRAMES = 30
# Exposure times that are whole multiples of the 10 ms period of 50 Hz light,
# in the camera's units of 100 microseconds; 333 is the ceiling at 30 fps.
FLICKER_SAFE = (300, 200, 100)
FRAME_TIME_LIMIT = 333


def brightness(frame) -> dict:
    """Median, 99th percentile and clipped share, on the brightest channel."""

    peak = np.asarray(frame).max(axis=2)
    return {
        "median": float(np.median(peak)),
        "p99": float(np.percentile(peak, 99)),
        "clipped": float((peak >= 250).mean()),
    }


def _bisect(measure, low: int, high: int, target: float, step: int = 1) -> int:
    """Largest value in [low, high] whose measurement stays at or under target."""

    while high - low > step:
        middle = (low + high) // 2
        if measure(middle) <= target:
            low = middle
        else:
            high = middle
    return low


def tune_exposure(
    capture,
    camera,
    *,
    settle_frames: int = WHITE_SETTLE_FRAMES,
    average: int = AVERAGE_FRAMES,
    log=print,
) -> tuple[int, int, dict, list[str]]:
    """Choose exposure and gain from the image. Returns them, stats, notes."""

    ranges = camera.ranges()
    exposure_range = ranges["exposure"]
    notes: list[str] = []
    camera.set("exposure_auto", 0)
    gain = ranges["gain"].minimum if "gain" in ranges else None
    if gain is not None:
        camera.set("gain", gain)

    def bright_at(**settings) -> float:
        for name, value in settings.items():
            camera.set(name, value)
        _drain(capture, settle_frames)
        return brightness(average_frames(capture, average))["p99"]

    ceiling = min(exposure_range.maximum, FRAME_TIME_LIMIT)
    candidates = [value for value in FLICKER_SAFE if exposure_range.minimum <= value <= ceiling]

    # Does the picture follow our exposure at all? If something else runs the
    # exposure -- as macOS does -- it undoes each change within a moment, and
    # a search would walk to a meaningless extreme. Give it time to fight back.
    long_value = candidates[0] if candidates else ceiling
    short_value = max(exposure_range.minimum, long_value // 10)
    bright_long = bright_at(exposure=long_value)
    _drain(capture, RESPONSE_SETTLE_FRAMES)
    bright_long = bright_at(exposure=long_value)
    bright_short = bright_at(exposure=short_value)
    _drain(capture, RESPONSE_SETTLE_FRAMES)
    bright_short = bright_at(exposure=short_value)
    if bright_long - bright_short < MIN_EXPOSURE_RESPONSE:
        camera.set("exposure_auto", 1)
        notes.append(
            f"the picture did not respond to exposure ({bright_long:.0f} at "
            f"{long_value / 10:.0f} ms vs {bright_short:.0f} at {short_value / 10:.1f} ms): "
            "something else controls it, so exposure stays automatic."
        )
        return None, None, brightness(average_frames(capture, average)), notes

    exposure = None
    for value in candidates:  # longest first: least gain needed, least noise
        if bright_at(exposure=value) <= CLIP_LIMIT:
            exposure = value
            break
    if exposure is None:
        # Even 10 ms clips at the lowest gain: go shorter, and accept that LED
        # light may band at these speeds.
        upper = candidates[-1] if candidates else ceiling
        exposure = _bisect(lambda v: bright_at(exposure=v), exposure_range.minimum, upper, TARGET_BRIGHT)
        notes.append(
            f"the scene is bright enough to need {exposure / 10:.1f} ms, shorter than 10 ms; "
            "under LED light that can show as bands. Dim or move the lamp if you see them."
        )
    camera.set("exposure", exposure)

    if gain is not None and bright_at() < TARGET_BRIGHT - 15:
        gain_range = ranges["gain"]
        gain = _bisect(lambda v: bright_at(gain=v), gain_range.minimum, gain_range.maximum, TARGET_BRIGHT)
        camera.set("gain", gain)
        if gain > gain_range.minimum + DIM_GAIN_FRACTION * (gain_range.maximum - gain_range.minimum):
            notes.append(
                f"needed gain {gain} of {gain_range.maximum} even at {exposure / 10:.0f} ms: "
                "the scene is too dark, so frames are noisy. Add light and tune again."
            )

    _drain(capture, settle_frames)
    stats = brightness(average_frames(capture, average))
    log(
        f"  exposure {exposure / 10:.1f} ms, gain {gain}: "
        f"brightest paper {stats['p99']:.0f}/255, median {stats['median']:.0f}, "
        f"clipped {stats['clipped']:.1%}"
    )
    return exposure, gain, stats, notes


# --- the whole tune ---------------------------------------------------------


def tune(
    capture,
    camera,
    *,
    power_line_frequency: int = 1,
    converge_frames: int = 90,
    settle_frames: int = FOCUS_SETTLE_FRAMES,
    average: int = AVERAGE_FRAMES,
    host_auto_exposure: bool | None = None,
    log=print,
) -> TuneResult:
    """Run the whole tune. Leaves the camera in the tuned state.

    ``host_auto_exposure`` says the OS runs its own auto-exposure over this
    camera (None: assume so on macOS, where it was measured).
    """

    import sys

    if host_auto_exposure is None:
        host_auto_exposure = sys.platform == "darwin"
    ranges = camera.ranges()
    result = TuneResult()

    for name, value in (
        ("power_line_frequency", power_line_frequency),
        ("zoom", ranges["zoom"].minimum if "zoom" in ranges else None),
        ("backlight_compensation", 0),
    ):
        if name in ranges and value is not None:
            camera.set(name, value)
            result.controls[name] = camera.get(name)

    # Everything automatic, streaming, on the sheet: autofocus finds a first
    # guess on the star while auto-exposure gives it a usable picture.
    for name in ("exposure_auto", "white_balance_auto", "focus_auto"):
        if name in ranges:
            camera.set(name, 1)
    log(f"  letting the camera's automatics settle ({converge_frames} frames)...")
    _drain(capture, converge_frames)
    if "focus" in ranges:
        result.autofocus_guess = camera.get("focus")

    if "exposure" in ranges and "exposure_auto" in ranges:
        if host_auto_exposure:
            camera.set("exposure_auto", 1)
            result.controls["exposure_auto"] = 1
            stats = brightness(average_frames(capture, average))
            log(
                "  exposure: left to macOS (it rewrites manual exposure while streaming); "
                f"brightest paper {stats['p99']:.0f}/255, median {stats['median']:.0f}"
            )
        else:
            exposure, gain, stats, notes = tune_exposure(
                capture, camera, settle_frames=settle_frames, average=average, log=log
            )
            result.notes.extend(notes)
            if exposure is None:
                result.controls["exposure_auto"] = 1
            else:
                result.controls["exposure_auto"] = 0
                result.controls["exposure"] = camera.get("exposure")
                if gain is not None:
                    result.controls["gain"] = camera.get("gain")
    if "exposure_priority" in ranges:
        camera.set("exposure_priority", 0)
        result.controls["exposure_priority"] = 0

    sheet = locate_sheet(average_frames(capture, average))
    if sheet is None and "focus" in ranges and "focus_auto" in ranges:
        # Autofocus may have parked the lens where the markers are too soft to
        # read. Look for the sheet across the focus range before giving up.
        known = ranges["focus"]
        stride = max(known.step, ((known.maximum - known.minimum) // 10 // known.step) * known.step)
        camera.set("focus_auto", 0)
        for position in range(known.minimum, known.maximum + 1, stride):
            camera.set("focus", position)
            _drain(capture, settle_frames)
            sheet = locate_sheet(average_frames(capture, average))
            if sheet is not None:
                log(f"  the sheet came into view at focus {position}")
                break
    if sheet is None:
        result.notes.append(
            "marker sheet NOT found in the view, so focus was not swept and white balance "
            "stays automatic. Check with `trashdrop camera preview` that the whole sheet is "
            "in the picture, then tune again."
        )
    else:
        log(f"  found the marker sheet ({sheet.markers} of 4 markers)")

    if "focus" in ranges and "focus_auto" in ranges:
        camera.set("focus_auto", 0)
        result.controls["focus_auto"] = 0
        known = ranges["focus"]
        chosen = result.autofocus_guess
        if sheet is not None:
            log(f"  autofocus suggested {result.autofocus_guess}; sweeping the lens over the star...")
            best, curve, clarity = sweep_focus(
                capture, camera, sheet.focus_roi, settle_frames=settle_frames, average=average
            )
            result.focus_curve = curve
            at_the_edge = best in (known.minimum, known.maximum)
            if clarity >= MIN_PEAK_RATIO and not at_the_edge:
                chosen = best
                log(f"  focus: {best} ({clarity:.1f}x sharper than typical)")
            else:
                why = "the best was at the end of the range" if at_the_edge else f"no clear peak ({clarity:.2f}x)"
                result.notes.append(f"focus sweep rejected: {why}; kept autofocus's position {chosen}.")
        else:
            result.notes.append(f"focus fixed at autofocus's own choice, {chosen}, without a check.")
        if chosen is None:
            raise TuneError("autofocus gave no position and the sweep could not run")
        camera.set("focus", chosen)
        result.controls["focus"] = camera.get("focus")

    if "white_balance" in ranges and "white_balance_auto" in ranges:
        paper = paper_colour(average_frames(capture, average), sheet.paper_patches) if sheet else None
        if paper is not None and float(paper.max()) < PAPER_CLIP_LEVEL:
            log("  balancing white on the sheet's paper...")
            value, ratio = balance_white(
                capture, camera, sheet.paper_patches, settle_frames=settle_frames, average=average
            )
            camera.set("white_balance", value)
            result.controls["white_balance_auto"] = 0
            result.controls["white_balance"] = camera.get("white_balance")
            log(f"  white balance: {result.controls['white_balance']} K (red/blue {ratio:.3f})")
        else:
            # Freezing an unmeasured value is worse than leaving it automatic:
            # the C920 does not report what its auto white balance chose.
            camera.set("white_balance_auto", 1)
            result.controls["white_balance_auto"] = 1
            if paper is not None:
                result.notes.append("the sheet's paper is overexposed; white balance left automatic.")

    if "focus" not in ranges:
        result.notes.append("this camera has no adjustable focus; nothing to tune there")
    return result


def focus_chart(curve: dict[int, float], width: int = 40, chosen: int | None = None) -> str:
    """The sweep as a text bar chart -- a clean single hump is what you want."""

    if not curve:
        return ""
    top = max(curve.values()) or 1.0
    mark_at = chosen if chosen is not None else max(curve, key=curve.get)
    rows = []
    for position, value in curve.items():
        bar = "#" * max(1, int(round(width * value / top)))
        mark = "  <- chosen" if position == mark_at else ""
        rows.append(f"  {position:4d} {bar}{mark}")
    return "\n".join(rows)
