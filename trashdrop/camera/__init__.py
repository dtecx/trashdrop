"""Camera control: freeze focus, exposure and white balance in camera.toml.

    sudo .venv/bin/python -B -m trashdrop camera tune --camera 1   # find values
    sudo .venv/bin/python -B -m trashdrop camera apply             # push them
    uv run trashdrop camera markers                                # printable target

Why sudo, and why not OpenCV: see ``uvc.py``. Nothing here is imported by the
simulator, and pyusb is only needed once a command actually talks to a camera.
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

    Returns a one-line status for the caller to print. Capture runs as the
    normal user, so on macOS this usually reports that it could not push --
    which is fine if ``camera apply`` ran with sudo since the camera was
    plugged in, because the camera keeps its settings until it loses power.
    """

    path = config_path()
    if not path.is_file():
        return (
            "camera: no camera.toml yet -- autofocus and auto-exposure are on. "
            "Tune it: sudo .venv/bin/python -B -m trashdrop camera tune --camera 1"
        )
    try:
        settings = load(path)
        camera = UvcCamera.find(settings.device if ":" in settings.device else None)
        results = apply(camera, settings)
    except CameraPermissionError:
        return (
            "camera: camera.toml not pushed from here (macOS needs sudo). If you ran "
            "`sudo .venv/bin/python -B -m trashdrop camera apply` since plugging the "
            "camera in, it is already set."
        )
    except Exception as error:  # a broken config must not stop a capture session
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
