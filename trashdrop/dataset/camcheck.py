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
    geometric_drift_px: float = 0.0

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
            lines.append(f"  controllable     {', '.join(settable) or 'nothing'}")
            if ignored:
                lines.append(f"  ignored          {', '.join(ignored)}")
        lines += [
            "",
            f"  brightness drift {self.idle_brightness_drift:5.2f} levels "
            f"(limit {MAX_IDLE_BRIGHTNESS_DRIFT})",
            f"  focus swing      {self.focus_swing_ratio:5.2f} "
            f"(limit {MAX_FOCUS_SWING_RATIO})",
            f"  camera drift     {self.geometric_drift_px:5.2f} px "
            f"(limit {MAX_GEOMETRIC_DRIFT_PX})",
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


def _shift_px(first, second) -> float:
    import cv2

    a = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    return float(np.hypot(dx, dy))


def analyse_idle(frames: list) -> dict:
    """Measure stability across frames of an unchanging scene."""

    if len(frames) < 2:
        raise ValueError("need at least two frames to measure stability")

    brightness = [float(np.asarray(f).mean()) for f in frames]
    sharpness = [_sharpness(f) for f in frames]
    drift = [_shift_px(frames[0], f) for f in frames[1:]]

    median_sharpness = float(np.median(sharpness))
    swing = (
        float((max(sharpness) - min(sharpness)) / median_sharpness)
        if median_sharpness > 1e-6
        else 0.0
    )
    return {
        "brightness_drift": float(np.std(brightness)),
        "focus_swing_ratio": swing,
        "geometric_drift_px": float(max(drift)),
    }


def analyse_response(reference, occupied) -> dict:
    """Measure what the camera did when an item entered the scene."""

    fit = estimate_photometric_fit(occupied, reference)
    shift = float(np.abs(np.asarray(occupied, dtype=np.float32).mean() - np.asarray(reference, dtype=np.float32).mean()))
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
            "Disable autofocus -- on a C920 that is a checkbox in Logitech's "
            "software, or cv2.CAP_PROP_AUTOFOCUS=0 if the backend allows it."
        )
    if report.geometric_drift_px > MAX_GEOMETRIC_DRIFT_PX:
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
        reference = frames[-1]

        if interactive:
            input("\nPhase 2: put a DARK item in the middle of the view, then press ENTER...")
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
