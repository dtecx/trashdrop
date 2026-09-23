"""Camera control: freeze focus, exposure and white balance in camera.toml.

    uv run trashdrop camera markers               # printable focus target
    uv run trashdrop camera tune --camera 1       # find values, write camera.toml
    uv run trashdrop camera apply                 # push camera.toml again

Why not OpenCV, and how it reaches the camera on macOS: see ``uvc.py``.
Nothing here is imported by the simulator.
"""

from __future__ import annotations

from pathlib import Path

from ..station import repository_root
from .config import APPLY_ORDER, Applied, CameraSettings, apply, format_applied, hand_back, load, save
from .tune import TuneError, TuneResult, focus_chart, tune
from .uvc import CONTROLS, CameraPermissionError, ControlRange, UvcCamera

CONFIG_FILE = "camera.toml"


def config_path() -> Path:
    return repository_root() / CONFIG_FILE


def apply_saved_settings() -> str:
    """Push camera.toml to the camera if there is one. Never raises.

    Returns a one-line status for the caller to print; a broken or missing
    config must never stop a capture session.
    """

    path = config_path()
    if not path.is_file():
        return (
            "camera: no camera.toml yet -- autofocus and auto-exposure are on. "
            "Tune it: uv run trashdrop camera tune --camera 1"
        )
    try:
        settings = load(path)
        camera = UvcCamera.find(settings.device if ":" in settings.device else None)
        results = apply(camera, settings)
    except Exception as error:
        return f"camera: camera.toml not pushed: {error}"
    differ = [result.name for result in results if not result.ok]
    return "camera: camera.toml applied" + (f" (camera kept its own {', '.join(differ)})" if differ else "")


__all__ = [
    "APPLY_ORDER",
    "Applied",
    "CONTROLS",
    "CONFIG_FILE",
    "CameraPermissionError",
    "CameraSettings",
    "ControlRange",
    "TuneError",
    "TuneResult",
    "UvcCamera",
    "apply",
    "apply_saved_settings",
    "config_path",
    "focus_chart",
    "format_applied",
    "hand_back",
    "load",
    "save",
    "tune",
]
