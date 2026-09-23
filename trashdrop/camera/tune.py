"""Find camera settings for this rig, so they can be frozen in camera.toml.

Two different strategies, because the camera's own automation is good at one
job and bad at the other:

* **Exposure and white balance** -- the camera's auto modes are sensible, the
  problem is only that they keep re-deciding. So let them converge on the
  empty table for a few seconds, read what they chose, and freeze exactly that.
* **Focus** -- the camera's autofocus is contrast-driven and a plain board
  gives it nothing to lock onto, which is why ``camcheck`` saw it hunting. So
  it is not used at all. Autofocus goes off, the lens is swept across its
  range, and the position that makes a printed target sharpest wins.

The target matters: put a printed page -- the marker sheet from
``trashdrop camera markers`` or any page of text -- flat in the middle of the
table first. Without texture every focus position looks the same, and the
sweep refuses to guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The best position must be at least this much sharper than a typical one, or
# the sweep found noise rather than a focus peak.
MIN_PEAK_RATIO = 1.3
COARSE_POINTS = 25


class TuneError(RuntimeError):
    pass


@dataclass
class TuneResult:
    controls: dict[str, int] = field(default_factory=dict)
    focus_curve: dict[int, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def sharpness(frame, region: float = 0.5) -> float:
    """Variance of the Laplacian over the middle of the frame.

    Only the middle counts: that is where the target and the pick zone are,
    and the frame edges show the desk and the wall at other distances.
    """

    import cv2

    height, width = frame.shape[:2]
    y0, y1 = int(height * (0.5 - region / 2)), int(height * (0.5 + region / 2))
    x0, x1 = int(width * (0.5 - region / 2)), int(width * (0.5 + region / 2))
    grey = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def _drain(capture, frames: int):
    """Read frames so the camera's state catches up; return the last one."""

    last = None
    for _ in range(frames):
        ok, frame = capture.read()
        if ok and frame is not None:
            last = frame
    return last


def _measure_focus(capture, camera, position: int, settle_frames: int) -> float:
    camera.set("focus", position)
    _drain(capture, settle_frames)
    readings = []
    for _ in range(2):
        ok, frame = capture.read()
        if ok and frame is not None:
            readings.append(sharpness(frame))
    if not readings:
        raise TuneError("camera stopped delivering frames during the focus sweep")
    return float(np.mean(readings))


def sweep_focus(capture, camera, *, settle_frames: int = 8, log=print) -> tuple[int, dict[int, float]]:
    """Coarse pass across the whole range, then a fine pass around the best."""

    known = camera.ranges()["focus"]
    step = max(1, known.step)
    span = known.maximum - known.minimum
    coarse = max(step, int(round(span / COARSE_POINTS / step)) * step)

    curve: dict[int, float] = {}
    for position in range(known.minimum, known.maximum + 1, coarse):
        curve[position] = _measure_focus(capture, camera, position, settle_frames)
    best = max(curve, key=curve.get)

    fine_from = max(known.minimum, best - coarse)
    fine_to = min(known.maximum, best + coarse)
    for position in range(fine_from, fine_to + 1, step):
        if position not in curve:
            curve[position] = _measure_focus(capture, camera, position, settle_frames)
    best = max(curve, key=curve.get)

    typical = float(np.median(list(curve.values())))
    if typical <= 0 or curve[best] / typical < MIN_PEAK_RATIO:
        raise TuneError(
            "no clear focus peak -- there is nothing sharp to focus on. Put a printed "
            "page (the marker sheet, or any page of text) flat in the middle of the "
            "table and run the tune again."
        )
    camera.set("focus", best)
    log(f"  focus: best position {best} ({curve[best] / typical:.1f}x sharper than typical)")
    return best, dict(sorted(curve.items()))


def tune(
    capture,
    camera,
    *,
    power_line_frequency: int = 1,
    converge_frames: int = 90,
    settle_frames: int = 8,
    log=print,
) -> TuneResult:
    """Run the whole tune. Leaves the camera in the tuned state."""

    ranges = camera.ranges()
    result = TuneResult()

    if "power_line_frequency" in ranges:
        camera.set("power_line_frequency", power_line_frequency)
        result.controls["power_line_frequency"] = camera.get("power_line_frequency")
    if "zoom" in ranges:
        camera.set("zoom", ranges["zoom"].minimum)
        result.controls["zoom"] = camera.get("zoom")
    if "backlight_compensation" in ranges:
        camera.set("backlight_compensation", 0)
        result.controls["backlight_compensation"] = camera.get("backlight_compensation")

    # Let auto exposure and white balance settle on the live scene, then
    # freeze whatever they chose. Frames must keep flowing: the camera only
    # adjusts while it is streaming.
    for name in ("exposure_auto", "white_balance_auto"):
        if name in ranges:
            camera.set(name, 1)
    log(f"  letting exposure and white balance settle ({converge_frames} frames)...")
    _drain(capture, converge_frames)

    frozen = {name: camera.get(name) for name in ("exposure", "gain", "white_balance") if name in ranges}
    if "exposure_auto" in ranges:
        camera.set("exposure_auto", 0)
        result.controls["exposure_auto"] = 0
    if "exposure_priority" in ranges:
        camera.set("exposure_priority", 0)
        result.controls["exposure_priority"] = 0
    if "white_balance_auto" in ranges:
        camera.set("white_balance_auto", 0)
        result.controls["white_balance_auto"] = 0
    for name, value in frozen.items():
        camera.set(name, value)
        result.controls[name] = camera.get(name)
    log(
        "  froze "
        + ", ".join(f"{name}={result.controls[name]}" for name in frozen)
    )

    if "focus" in ranges and "focus_auto" in ranges:
        camera.set("focus_auto", 0)
        result.controls["focus_auto"] = 0
        log("  sweeping the lens across its range...")
        best, curve = sweep_focus(capture, camera, settle_frames=settle_frames, log=log)
        result.controls["focus"] = best
        result.focus_curve = curve
    else:
        result.notes.append("this camera has no adjustable focus; nothing to tune there")

    return result


def focus_chart(curve: dict[int, float], width: int = 40) -> str:
    """The sweep as a text bar chart -- a clean single peak is what you want."""

    if not curve:
        return ""
    top = max(curve.values()) or 1.0
    best = max(curve, key=curve.get)
    rows = []
    for position, value in curve.items():
        bar = "#" * max(1, int(round(width * value / top)))
        mark = "  <- chosen" if position == best else ""
        rows.append(f"  {position:4d} {bar}{mark}")
    return "\n".join(rows)
