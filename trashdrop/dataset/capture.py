"""Capture bursts of one item at a time, on the real rig.

The workflow this supports is the one that makes labelling nearly free:
shoot a reference photo of the empty table, then put ONE item down and take a
burst while nudging it between frames. ``autolabel`` then recovers the mask and
box by differencing against the reference, and the class comes from the folder,
so nobody draws a box by hand.

Camera source can be a device index (``0``) or a URL. The URL form is what to
use when the camera turns out to be an Android phone -- install any "IP webcam"
app and pass e.g. ``http://192.168.1.5:8080/video``.

Layout written:

    data/raw/<session>/<class>/<object_id>/frame_0000.jpg
    data/bg/<session>/<lighting>.jpg
    data/raw/<session>/manifest.csv

Keys in the live window:

    SPACE  capture a burst          b  (re)shoot the background reference
    n      next object id           c  change class
    l      change lighting label    q  finish
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ..station import ALL_CATEGORIES
from .manifest import ManifestWriter

BURST_FRAMES = 25
BURST_INTERVAL_S = 0.12


@dataclass
class CaptureConfig:
    session: str
    root: Path = Path("data")
    source: str | int = 0
    width: int = 1280
    height: int = 720
    burst: int = BURST_FRAMES
    lighting: str = "default"


def open_camera(source: str | int, width: int, height: int):
    """Open a camera and verify it actually delivers a frame."""

    import cv2

    capture = cv2.VideoCapture(source)
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {source!r}. For a phone, pass the full "
            "stream URL, e.g. http://192.168.1.5:8080/video . On macOS, grant "
            "the terminal camera permission in System Settings > Privacy."
        )
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(f"Camera {source!r} opened but returned no frame")
    return capture


def _overlay(frame, lines: list[str]):
    import cv2

    canvas = frame.copy()
    for index, text in enumerate(lines):
        y = 28 + index * 26
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 1, cv2.LINE_AA)
    return canvas


def _prompt(current: str, options: list[str] | None = None) -> str:
    """Ask on the terminal; the OpenCV window cannot take text input."""

    if options:
        print(f"  options: {', '.join(options)}")
    answer = input(f"  new value (enter keeps {current!r}): ").strip()
    if not answer:
        return current
    if options and answer not in options:
        print(f"  {answer!r} is not one of the options; keeping {current!r}")
        return current
    return answer


def run_capture(config: CaptureConfig, category: str, object_id: str) -> Path:
    """Interactive capture loop. Returns the session's raw folder."""

    import cv2

    if category not in ALL_CATEGORIES:
        raise ValueError(f"{category!r} is not a station category: {ALL_CATEGORIES}")

    raw_root = config.root / "raw" / config.session
    bg_root = config.root / "bg" / config.session
    raw_root.mkdir(parents=True, exist_ok=True)
    bg_root.mkdir(parents=True, exist_ok=True)
    manifest = ManifestWriter(raw_root / "manifest.csv")

    capture = open_camera(config.source, config.width, config.height)
    lighting = config.lighting
    captured = 0
    print(
        "\nSPACE burst | b background | n next object | c class | l lighting | q quit\n"
        "Shoot the background FIRST, with the table empty.\n"
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("  dropped frame")
                continue

            has_bg = (bg_root / f"{lighting}.jpg").is_file()
            window = _overlay(
                frame,
                [
                    f"class={category}  object={object_id}  light={lighting}",
                    f"captured={captured}  background={'yes' if has_bg else 'MISSING'}",
                    "SPACE burst | b bg | n next | c class | l light | q quit",
                ],
            )
            cv2.imshow("trashdrop capture", window)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("b"):
                path = bg_root / f"{lighting}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"  background for {lighting!r} -> {path}")
            elif key == ord("c"):
                category = _prompt(category, list(ALL_CATEGORIES))
            elif key == ord("l"):
                lighting = _prompt(lighting)
            elif key == ord("n"):
                object_id = _prompt(object_id)
            elif key == ord(" "):
                if not has_bg:
                    print("  shoot the background first (b), with the table empty")
                    continue
                target = raw_root / category / object_id
                target.mkdir(parents=True, exist_ok=True)
                start = len(list(target.glob("frame_*.jpg")))
                print(f"  burst of {config.burst} into {target} ...")
                for index in range(config.burst):
                    ok, burst_frame = capture.read()
                    if not ok or burst_frame is None:
                        continue
                    name = f"frame_{start + index:04d}.jpg"
                    cv2.imwrite(str(target / name), burst_frame)
                    manifest.append(
                        image=str((target / name).relative_to(config.root)),
                        category=category,
                        object_id=object_id,
                        lighting=lighting,
                        session=config.session,
                    )
                    captured += 1
                    preview = _overlay(burst_frame, [f"capturing {index + 1}/{config.burst}"])
                    cv2.imshow("trashdrop capture", preview)
                    cv2.waitKey(1)
                    time.sleep(BURST_INTERVAL_S)
                print(f"  done, {captured} frames this session")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        manifest.close()

    print(f"\n{captured} frames -> {raw_root}")
    print(f"manifest: {manifest.path}")
    print("next: trashdrop autolabel --session " + config.session)
    return raw_root
