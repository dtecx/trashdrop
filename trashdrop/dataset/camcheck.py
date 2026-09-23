"""Is this camera good enough to shoot a dataset through? Answer in 60 seconds.

Run it on whatever camera you end up with -- the C920, a phone stream, or
whatever the venue hands you. It measures the four things that actually break
background subtraction, and says which of them this camera does.

1. **Exposure response.** Put a dark item on a light table and most cameras
   rebalance the whole frame. That is survivable, because the reference is
   compensated before differencing, but it is worth knowing how large the
   shift is.
2. **Focus hunting.** A fixed scene at a fixed distance should not refocus. If
   it does, the image sharpness swings and edges move.
3. **Geometric stability.** A camera that drifts -- a loose clamp, or digital
   "auto-framing" that pans to follow things -- voids the calibration.
4. **What can actually be controlled.** Properties are set and read back,
   because on macOS most of them are silently ignored.

The analysis is deliberately separate from the capture so it can be tested and
so recorded frames can be checked after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..perception.photometric import estimate_photometric_fit

# A dataset shot through a camera drifting more than this is not worth having:
# the reference would have to be re-shot every few frames.
MAX_IDLE_BRIGHTNESS_DRIFT = 3.0
# Sharpness is scale-free, so judge it as a relative swing.
MAX_FOCUS_SWING_RATIO = 0.25
# Sub-pixel drift is fine; a third of a pixel of jitter is not worth chasing.
MAX_GEOMETRIC_DRIFT_PX = 1.0


@dataclass
class CameraReport:
    """What the camera does, and whether that is workable."""

    backend: str = ""
    resolution: tuple[int, int] = (0, 0)
    measured_fps: float = 0.0
    controllable: dict[str, bool] = field(default_factory=dict)

    idle_brightness_drift: float = 0.0
    focus_swing_ratio: float = 0.0
    geometric_drift_px: float | None = 0.0

    exposure_shift: float | None = None
    compensation_recovers: bool | None = None

    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.warnings

    def render(self) -> str:
        lines = [
            "",
            "camera check",
            "------------",
            f"  backend          {self.backend or 'unknown'}",
            f"  resolution       {self.resolution[0]} x {self.resolution[1]}",
            f"  measured fps     {self.measured_fps:.1f}",
        ]
        if self.controllable:
            settable = [name for name, ok in self.controllable.items() if ok]
            ignored = [name for name, ok in self.controllable.items() if not ok]
            lines.append(
                f"  OpenCV controls  {', '.join(settable) or 'nothing'}"
                "   (on macOS always nothing; camera.toml does this instead)"
            )
            if ignored:
                lines.append(f"  ignored          {', '.join(ignored)}")
        lines += [
            "",
            f"  brightness drift {self.idle_brightness_drift:5.2f} levels "
            f"(limit {MAX_IDLE_BRIGHTNESS_DRIFT})",
            f"  focus swing      {self.focus_swing_ratio:5.2f} "
            f"(limit {MAX_FOCUS_SWING_RATIO})",
            (
                f"  camera drift     {self.geometric_drift_px:5.2f} px (limit {MAX_GEOMETRIC_DRIFT_PX})"
                if self.geometric_drift_px is not None
                else "  camera drift     not measurable (view too blurry or featureless)"
            ),
        ]
        if self.exposure_shift is not None:
            lines.append(f"  exposure step    {self.exposure_shift:5.1f} levels when an item appeared")
        if self.compensation_recovers is not None:
            verdict = "yes" if self.compensation_recovers else "NO"
            lines.append(f"  compensated ok   {verdict}")

        lines.append("")
        for note in self.notes:
            lines.append(f"  note: {note}")
        for warning in self.warnings:
            lines.append(f"  PROBLEM: {warning}")
        lines.append("")
        lines.append("  VERDICT: usable" if self.usable else "  VERDICT: fix the problems above first")
        lines.append("")
        return "\n".join(lines)


def _sharpness(frame) -> float:
    import cv2

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


# Below this peak value phase correlation found no real match and the shift it
# reports is noise. Measured on synthetic footage: sharp views, steady or
# moving, score 0.85-0.93; a badly defocused view at high gain -- what the first
# real camcheck saw, when it reported "233 px of drift" -- scores 0.01-0.02.
MIN_CORRELATION = 0.3
ANALYSIS_WIDTH = 640


def _shift_px(first, second) -> tuple[float, float]:
    """Shift between two frames in full-resolution pixels, and its confidence."""

    import cv2

    scale = min(1.0, ANALYSIS_WIDTH / first.shape[1])

    def prepare(frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(grey, (3, 3), 0).astype(np.float32)

    a, b = prepare(first), prepare(second)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    return float(np.hypot(dx, dy)) / scale, float(response)


def _average(frames: list) -> np.ndarray:
    return np.clip(np.mean([np.asarray(f, np.float32) for f in frames], axis=0), 0, 255).astype(np.uint8)


def analyse_idle(frames: list) -> dict:
    """Measure stability across frames of an unchanging scene."""

    if len(frames) < 2:
        raise ValueError("need at least two frames to measure stability")

    from ..camera.tune import centre_roi, locate_sheet, roi_sharpness

    brightness = [float(np.asarray(f).mean()) for f in frames]
    # Judge sharpness on the marker sheet's star when it is in view, the same
    # way the tune does: a whole noisy frame measures the sensor, not focus.
    sheet = locate_sheet(frames[0])
    roi = sheet.focus_roi if sheet else centre_roi(frames[0])
    sharpness = [roi_sharpness(f, roi) for f in frames]
    # Compare the average of the first third with the average of the last:
    # averaging cuts the sensor noise that inflated frame-by-frame shifts
    # (2.2 px of false drift on a moderately blurred view, 0.7 px this way).
    third = max(1, len(frames) // 3)
    shift, response = _shift_px(_average(frames[:third]), _average(frames[-third:]))

    median_sharpness = float(np.median(sharpness))
    swing = (
        float((np.percentile(sharpness, 90) - np.percentile(sharpness, 10)) / median_sharpness)
        if median_sharpness > 1e-6
        else 0.0
    )
    return {
        "brightness_drift": float(np.std(brightness)),
        "focus_swing_ratio": swing,
        # None when the view gave nothing to measure against.
        "geometric_drift_px": shift if response >= MIN_CORRELATION else None,
        "sheet_found": sheet is not None,
    }


def analyse_response(reference, occupied) -> dict:
    """Measure what the camera did when an item entered the scene."""

    fit = estimate_photometric_fit(occupied, reference)
    if fit.trusted:
        # How far a mid-grey BACKGROUND pixel moved: the camera's exposure
        # change alone. The frame mean would also count the item itself, or a
        # sheet taken away -- which is how the first real run read "8 levels".
        shift = float(np.max(np.abs(fit.gain * 128.0 + fit.offset - 128.0)))
    else:
        shift = float(np.abs(np.asarray(occupied, np.float32).mean() - np.asarray(reference, np.float32).mean()))
    return {
        "exposure_shift": shift,
        "compensation_recovers": bool(fit.trusted),
        "fit": fit,
    }


def judge(report: CameraReport) -> CameraReport:
    """Turn measurements into warnings a person can act on."""

    if report.idle_brightness_drift > MAX_IDLE_BRIGHTNESS_DRIFT:
        report.warnings.append(
            f"brightness wanders by {report.idle_brightness_drift:.1f} levels on a "
            "still scene. Turn off auto-exposure if you can, kill any window "
            "light, and re-shoot the background reference often."
        )
    if report.focus_swing_ratio > MAX_FOCUS_SWING_RATIO:
        report.warnings.append(
            f"focus is hunting (sharpness swings by {report.focus_swing_ratio:.0%}). "
            "Run `uv run trashdrop camera tune` with the marker sheet in view: it "
            "switches autofocus off and fixes the lens position in camera.toml."
        )
    if report.geometric_drift_px is None:
        report.notes.append(
            "drift could not be measured: the view is too blurry or featureless. "
            "Fix focus first (camera tune), with the marker sheet in view."
        )
    elif report.geometric_drift_px > MAX_GEOMETRIC_DRIFT_PX:
        report.warnings.append(
            f"the view moved {report.geometric_drift_px:.1f} px while nothing "
            "was happening. Tighten the mount, and make sure no 'auto-framing' "
            "or 'smart zoom' feature is enabled -- that digitally pans the "
            "image and voids the calibration."
        )
    if report.compensation_recovers is False:
        report.warnings.append(
            "exposure compensation could not fit this frame pair. Usually the "
            "item covers too much of the view, or the camera moved between the "
            "two shots."
        )

    if report.exposure_shift is not None and report.exposure_shift > 8.0:
        report.notes.append(
            f"the camera re-exposed by {report.exposure_shift:.0f} levels when the "
            "item appeared. That is normal and is compensated for, but re-shoot "
            "the background whenever the room light changes."
        )
    if report.measured_fps and report.measured_fps < 10:
        report.notes.append(
            f"only {report.measured_fps:.1f} fps. Fine for stills, slow for bursts. "
            "Drop the resolution if bursts feel sluggish."
        )
    return report


def probe_controls(capture) -> dict[str, bool]:
    """Try to set properties and read them back, then put them back.

    macOS routes OpenCV through AVFoundation, where most of these are accepted
    and then ignored, so asking is the only way to find out. Every probe is
    undone afterwards: on a backend that DOES honour them, leaving exposure in
    manual mode with no exposure value set can turn the image black and ruin
    the measurements that follow.
    """

    import cv2

    attempts = {
        "autofocus off": (cv2.CAP_PROP_AUTOFOCUS, 0),
        "auto-exposure manual": (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),
        "auto white balance off": (cv2.CAP_PROP_AUTO_WB, 0),
    }
    results: dict[str, bool] = {}
    for name, (prop, value) in attempts.items():
        try:
            before = capture.get(prop)
            accepted = bool(capture.set(prop, value))
            after = capture.get(prop)
            results[name] = bool(accepted and after != before)
            capture.set(prop, before)
        except Exception:
            results[name] = False
    return results


# Discard this long at the start: a webcam's first seconds are the sensor and
# auto-exposure starting up, and measuring them reports "brightness wanders"
# about a camera that is merely waking.
WARMUP_SECONDS = 2.5


def run_camcheck(
    source: str | int = 0,
    idle_seconds: float = 8.0,
    interactive: bool = True,
    width: int = 1920,
    height: int = 1080,
):
    """Open the camera, measure it, and print a verdict."""

    import time

    import cv2

    from .capture import open_camera

    capture = open_camera(source, width, height)
    report = CameraReport()

    from ..camera import apply_saved_settings

    camera_status = apply_saved_settings()
    print(camera_status)
    report.notes.append(camera_status)
    try:
        report.backend = capture.getBackendName()
        report.controllable = probe_controls(capture)

        print(f"warming up for {WARMUP_SECONDS:.0f} s...")
        warm_until = time.time() + WARMUP_SECONDS
        while time.time() < warm_until:
            capture.read()
        # Push camera.toml once more now the stream is running: starting a
        # stream reset exposure and gain on the real C920 (focus survived).
        apply_saved_settings()
        settle_until = time.time() + 1.0
        while time.time() < settle_until:
            capture.read()

        print(
            "\nPhase 1: put a printed page (the marker sheet) flat in the middle, then"
            "\n         touch nothing for a few seconds. A plain board gives focus and"
            "\n         drift nothing to measure, and makes autofocus hunt by itself..."
        )
        frames = []
        started = time.time()
        while time.time() - started < idle_seconds:
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)
        elapsed = time.time() - started
        if len(frames) < 2:
            raise RuntimeError("camera delivered almost no frames")

        report.resolution = (frames[0].shape[1], frames[0].shape[0])
        report.measured_fps = len(frames) / elapsed
        idle = analyse_idle(frames[:: max(1, len(frames) // 20)])
        report.idle_brightness_drift = idle["brightness_drift"]
        report.focus_swing_ratio = idle["focus_swing_ratio"]
        report.geometric_drift_px = idle["geometric_drift_px"]
        if not idle["sheet_found"]:
            report.notes.append("marker sheet not in view: focus was judged on the frame centre")
        reference = frames[-1]

        if interactive:
            input(
                "\nPhase 2: leave the sheet where it is and put a DARK item NEXT to it"
                "\n         (not on it), step back, then press ENTER..."
            )
            for _ in range(15):  # let auto-exposure settle on the new scene
                capture.read()
            ok, occupied = capture.read()
            if ok and occupied is not None:
                response = analyse_response(reference, occupied)
                report.exposure_shift = response["exposure_shift"]
                report.compensation_recovers = response["compensation_recovers"]
                if not response["compensation_recovers"]:
                    report.notes.append(f"fit said: {response['fit'].reason}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    judge(report)
    print(report.render())
    return report
