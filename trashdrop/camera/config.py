"""camera.toml: the camera's settings, written for people to read and edit.

``trashdrop camera tune`` finds the values and writes this file;
``trashdrop camera apply`` pushes it to the camera, and capture pushes it on
every start when it has the rights to. Edit a number by hand and apply again --
unknown names are refused rather than ignored, so a typo cannot silently leave
autofocus on.
"""

from __future__ import annotations

import os
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .uvc import BOOLEAN, CONTROLS, ControlRange

# Apply order matters: a camera ignores or refuses a manual value while the
# matching auto mode is still on, so each mode goes before its value.
APPLY_ORDER = (
    "power_line_frequency",
    "zoom",
    "exposure_auto",
    "exposure_priority",
    "exposure",
    "gain",
    "white_balance_auto",
    "white_balance",
    "focus_auto",
    "focus",
    "brightness",
    "contrast",
    "saturation",
    "sharpness",
    "backlight_compensation",
)

EXPLAIN = {
    "power_line_frequency": "1 = 50 Hz (Europe, Poland), 2 = 60 Hz. The wrong one shows as rolling bands under LED light.",
    "zoom": "Digital zoom. Keep at the minimum: zooming crops the view and voids the calibration.",
    "exposure_auto": "true on macOS: it runs its own auto-exposure while streaming and rewrites manual exposure within half a second (measured on the C920). Focus and white balance are left alone.",
    "exposure_priority": "false = keep the frame rate constant instead of slowing down in dim light.",
    "exposure": "Exposure time in units of 100 microseconds (156 = 15.6 ms). Frozen from what auto-exposure chose.",
    "gain": "Sensor gain. Lower means less noise. Frozen from what auto-exposure chose.",
    "white_balance_auto": "false = colours are fixed and do not drift with what is on the table.",
    "white_balance": "Colour temperature in kelvin. Frozen from what auto white balance chose.",
    "focus_auto": "Autofocus hunts on a plain board, which blurs frames and shifts the image. Keep false.",
    "focus": "Lens position, found by sweeping the lens over a printed target on the table.",
    "brightness": "Image processing; left at the camera default unless you have a reason.",
    "contrast": "Image processing; left at the camera default unless you have a reason.",
    "saturation": "Image processing; left at the camera default unless you have a reason.",
    "sharpness": "In-camera sharpening; left at the camera default unless you have a reason.",
    "backlight_compensation": "Brightens the subject against a bright background; 0 keeps exposure predictable.",
}


@dataclass
class CameraSettings:
    backend: str
    device: str
    controls: dict[str, int] = field(default_factory=dict)
    tuned_at: str = ""
    note: str = ""


@dataclass
class Applied:
    name: str
    wanted: int
    got: int | None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.got is not None and self.got == self.wanted


def load(path: Path) -> CameraSettings:
    path = Path(path)
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    camera = payload.get("camera", {})
    raw_controls = payload.get("controls", {})
    unknown = sorted(set(raw_controls) - set(CONTROLS))
    if unknown:
        raise ValueError(
            f"{path.name}: unknown control(s) {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(CONTROLS))}"
        )
    controls: dict[str, int] = {}
    for name, value in raw_controls.items():
        if isinstance(value, bool):
            controls[name] = int(value)
        elif isinstance(value, int):
            controls[name] = value
        else:
            raise ValueError(f"{path.name}: {name} must be a whole number or true/false, got {value!r}")
    return CameraSettings(
        backend=str(camera.get("backend", "uvc")),
        device=str(camera.get("device", "")),
        controls=controls,
        tuned_at=str(camera.get("tuned_at", "")),
        note=str(camera.get("note", "")),
    )


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render(settings: CameraSettings, ranges: dict[str, ControlRange] | None = None) -> str:
    ranges = ranges or {}
    lines = [
        "# TrashDrop camera settings -- applied every time the camera is opened.",
        "#",
        "# Written by `trashdrop camera tune`. Edit any value by hand, then push it",
        "# to the camera with:",
        "#     uv run trashdrop camera apply",
        "# Capture and camcheck also push it every time they open the camera.",
        "#",
        "# Re-tune whenever the camera height, the lighting or the table changes --",
        "# on site that is the first thing to do after mounting the camera.",
        "",
        "[camera]",
        f"backend = {_quote(settings.backend)}",
        f"device = {_quote(settings.device)}",
        f"tuned_at = {_quote(settings.tuned_at)}",
        f"note = {_quote(settings.note)}",
        "",
        "[controls]",
    ]
    for name in APPLY_ORDER:
        if name not in settings.controls:
            continue
        value = settings.controls[name]
        lines.append(f"# {EXPLAIN.get(name, '')}".rstrip())
        known = ranges.get(name)
        if known is not None and name not in BOOLEAN:
            lines.append(
                f"# camera range {known.minimum}..{known.maximum}, step {known.step}, "
                f"default {known.default}"
            )
        rendered = ("true" if value else "false") if name in BOOLEAN else str(int(value))
        lines.append(f"{name} = {rendered}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def hand_back(path: Path) -> None:
    """If run under sudo anyway, give written files back to the real user.

    Camera control no longer needs root on macOS, but someone will still try
    it with sudo, and a root-owned camera.toml or dataset breaks the next edit.
    """

    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if hasattr(os, "geteuid") and os.geteuid() == 0 and uid and gid:
        path = Path(path)
        targets = [path] + (list(path.rglob("*")) if path.is_dir() else [])
        for target in targets:
            try:
                os.chown(target, int(uid), int(gid))
            except OSError:
                pass


def save(settings: CameraSettings, path: Path, ranges: dict[str, ControlRange] | None = None) -> Path:
    path = Path(path)
    if path.exists():
        backup = path.with_suffix(f".{time.strftime('%Y%m%d-%H%M%S')}.toml.bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        hand_back(backup)
    path.write_text(render(settings, ranges), encoding="utf-8")
    hand_back(path)
    return path


def _fit(value: int, known: ControlRange) -> tuple[int, str]:
    """Clamp to the camera's range and snap to its step; say if it had to."""

    fitted = min(max(value, known.minimum), known.maximum)
    if known.step > 1:
        fitted = known.minimum + round((fitted - known.minimum) / known.step) * known.step
        fitted = min(fitted, known.maximum)
    note = "" if fitted == value else f"adjusted from {value} to fit the camera"
    return fitted, note


def apply(camera, settings: CameraSettings, *, pause=time.sleep) -> list[Applied]:
    """Push settings to the camera in a safe order, then read every one back."""

    ranges = camera.ranges()
    results: list[Applied] = []
    for name in APPLY_ORDER:
        if name not in settings.controls:
            continue
        wanted = settings.controls[name]
        known = ranges.get(name)
        if known is None:
            results.append(Applied(name, wanted, None, "not supported by this camera"))
            continue
        wanted, note = _fit(int(wanted), known)
        camera.set(name, wanted)
        if name.endswith("_auto"):
            # Give the camera a moment to leave its auto mode before the
            # manual value arrives.
            pause(0.05)
        results.append(Applied(name, wanted, None, note))

    for result in results:
        if result.note == "not supported by this camera":
            continue
        result.got = camera.get(result.name)
        if not result.ok and not result.note:
            result.note = "camera kept a different value"
    return results


def format_applied(results: list[Applied]) -> str:
    rows = []
    for result in results:
        mark = "ok " if result.ok else "-- "
        got = "-" if result.got is None else str(result.got)
        rows.append(f"  {mark}{result.name:24s} wanted {result.wanted:>6}  camera {got:>6}  {result.note}")
    return "\n".join(rows)
