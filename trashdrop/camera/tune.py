"""Find camera settings for this rig, so they can be frozen in camera.toml.

Three settings, three strategies -- each chosen after watching a simpler one
fail on the real C920:

* **Exposure and gain** -- let auto-exposure settle, read what it chose, freeze
  it. The C920 reports its live auto values for these, so this is exact.
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
table first. Without it the tune falls back to the frame centre for focus and
to the camera's own report for white balance, and says so.
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


# --- the whole tune ---------------------------------------------------------


def tune(
    capture,
    camera,
    *,
    power_line_frequency: int = 1,
    converge_frames: int = 90,
    settle_frames: int = FOCUS_SETTLE_FRAMES,
    average: int = AVERAGE_FRAMES,
    log=print,
) -> TuneResult:
    """Run the whole tune. Leaves the camera in the tuned state."""

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

    # Everything automatic, streaming, on the sheet: exposure settles, and
    # autofocus finds a first guess on the star.
    for name in ("exposure_auto", "white_balance_auto", "focus_auto"):
        if name in ranges:
            camera.set(name, 1)
    log(f"  letting the camera's automatics settle ({converge_frames} frames)...")
    _drain(capture, converge_frames)

    frozen = {name: camera.get(name) for name in ("exposure", "gain") if name in ranges}
    if "focus" in ranges:
        result.autofocus_guess = camera.get("focus")
    for name in ("exposure_auto", "exposure_priority"):
        if name in ranges:
            camera.set(name, 0)
            result.controls[name] = 0
    for name, value in frozen.items():
        camera.set(name, value)
        result.controls[name] = camera.get(name)
    log("  froze " + ", ".join(f"{name}={result.controls[name]}" for name in frozen))

    if "gain" in ranges:
        known = ranges["gain"]
        level = (result.controls["gain"] - known.minimum) / max(1, known.maximum - known.minimum)
        if level > DIM_GAIN_FRACTION:
            result.notes.append(
                f"gain is {result.controls['gain']} of {known.maximum}: the scene is too dark, "
                "so frames are noisy. Add light on the table and tune again."
            )

    sheet = locate_sheet(average_frames(capture, average))
    if sheet is None:
        result.notes.append(
            "marker sheet not found -- focus was judged on the frame centre and white "
            "balance taken from the camera. Lay the sheet flat in view and tune again."
        )
    else:
        log(f"  found the marker sheet ({sheet.markers} of 4 markers)")

    if "focus" in ranges and "focus_auto" in ranges:
        camera.set("focus_auto", 0)
        result.controls["focus_auto"] = 0
        roi = sheet.focus_roi if sheet else centre_roi(average_frames(capture, 1))
        log(f"  autofocus suggested {result.autofocus_guess}; sweeping the lens to check...")
        best, curve, clarity = sweep_focus(capture, camera, roi, settle_frames=settle_frames, average=average)
        result.focus_curve = curve
        if clarity >= MIN_PEAK_RATIO:
            chosen = best
            log(f"  focus: {best} ({clarity:.1f}x sharper than typical)")
        elif result.autofocus_guess is not None:
            chosen = result.autofocus_guess
            result.notes.append(
                f"the focus sweep had no clear peak ({clarity:.2f}x); kept autofocus's "
                f"position {chosen}. Check the image, and add light if gain is high."
            )
        else:
            raise TuneError(
                "no clear focus peak -- put the printed marker sheet flat in the middle "
                "of the table, add light, and run the tune again."
            )
        camera.set("focus", chosen)
        result.controls["focus"] = camera.get("focus")

    if "white_balance" in ranges and "white_balance_auto" in ranges:
        paper = paper_colour(average_frames(capture, average), sheet.paper_patches) if sheet else None
        if paper is not None and float(paper.max()) < PAPER_CLIP_LEVEL:
            log("  balancing white on the sheet's paper...")
            value, ratio = balance_white(
                capture, camera, sheet.paper_patches, settle_frames=settle_frames, average=average
            )
            log(f"  white balance: {value} K (red/blue {ratio:.3f})")
        else:
            if paper is not None:
                result.notes.append(
                    "the sheet's paper is overexposed, so white balance could not be measured "
                    "on it; kept the camera's own value."
                )
            value = camera.get("white_balance")
            camera.set("white_balance_auto", 0)
        camera.set("white_balance", value)
        result.controls["white_balance_auto"] = 0
        result.controls["white_balance"] = camera.get("white_balance")

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
