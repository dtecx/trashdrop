"""Which camera index is which? Open each one and take a snapshot.

On a MacBook the built-in camera is usually index 0 and a USB webcam index 1,
but "usually" is not good enough at a venue with other people's devices
around. This opens every index that responds, saves a thumbnail of what it
sees, and reports the resolution it actually delivers. The webcam is the one
whose thumbnail shows the table from above.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraInfo:
    index: int
    resolution: tuple[int, int]
    backend: str
    snapshot: Path


def device_names() -> list[str]:
    """Camera names from the OS, where that is cheap to ask. macOS only."""

    if sys.platform != "darwin":
        return []
    try:
        output = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    # Device names sit at exactly four spaces of indent and end with a colon.
    return [
        line.strip().rstrip(":")
        for line in output.splitlines()
        if line.startswith("    ") and not line.startswith("     ") and line.strip().endswith(":")
    ]


def list_cameras(
    max_index: int = 5,
    width: int = 1920,
    height: int = 1080,
    out_dir: Path = Path("out/cameras"),
) -> list[CameraInfo]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    found: list[CameraInfo] = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        try:
            if not capture.isOpened():
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            frame = None
            # A few reads: the first frames are often black while the sensor
            # and auto-exposure start up.
            for _ in range(15):
                ok, candidate = capture.read()
                if ok and candidate is not None:
                    frame = candidate
            if frame is None:
                continue
            thumbnail = cv2.resize(frame, (640, int(round(frame.shape[0] * 640 / frame.shape[1]))))
            path = out_dir / f"camera_{index}.jpg"
            cv2.imwrite(str(path), thumbnail)
            found.append(
                CameraInfo(
                    index=index,
                    resolution=(frame.shape[1], frame.shape[0]),
                    backend=capture.getBackendName(),
                    snapshot=path,
                )
            )
        finally:
            capture.release()
    return found
