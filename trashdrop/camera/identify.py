"""Make sure the video we look at comes from the webcam we are adjusting.

Two independent handles reach the camera: OpenCV opens a *video stream* by
index, and the UVC controls reach a *USB device*. Nothing ties them together.
On this project's MacBook the webcam was index 0 and the built-in camera index
1, and an afternoon of tuning ran with ``--camera 1``: the C920's lens was
swept while sharpness was measured on the laptop camera looking at the person
at the keyboard. Every result was noise, and nothing said so.

So before trusting a stream, wiggle the webcam and watch the stream. Digital
zoom is the wiggle: it changes the whole picture at once, macOS does not
override it, and it is undone without trace. If the stream does not change
when the webcam zooms, it is some other camera.
"""

from __future__ import annotations

import numpy as np

# Mean absolute difference, in 8-bit levels on a downscaled grey frame, that a
# 2x zoom must produce -- and that zooming back must remove.
MIN_CHANGE = 6.0
MAX_RESIDUAL = 4.0
SETTLE_FRAMES = 12


def _snapshot(capture, frames: int = 3) -> np.ndarray | None:
    import cv2

    grabbed = []
    for _ in range(frames):
        ok, frame = capture.read()
        if ok and frame is not None:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grabbed.append(cv2.resize(grey, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32))
    return np.mean(grabbed, axis=0) if grabbed else None


def _drain(capture, frames: int) -> None:
    for _ in range(frames):
        capture.read()


def stream_follows_camera(capture, camera, *, settle_frames: int = SETTLE_FRAMES) -> tuple[bool | None, str]:
    """Does this OpenCV stream show the webcam behind ``camera``?

    Returns True, False, or None when it cannot tell (no zoom control, or the
    picture changed on its own -- someone moving in front of the lens).
    """

    ranges = camera.ranges()
    if "zoom" not in ranges:
        return None, "the webcam has no zoom control to test with"
    known = ranges["zoom"]
    original = camera.get("zoom")
    zoomed = min(known.maximum, max(known.minimum * 2, original + (known.maximum - known.minimum) // 2))

    _drain(capture, settle_frames)
    before = _snapshot(capture)
    try:
        camera.set("zoom", zoomed)
        _drain(capture, settle_frames)
        during = _snapshot(capture)
    finally:
        camera.set("zoom", original)
    _drain(capture, settle_frames)
    after = _snapshot(capture)

    if before is None or during is None or after is None:
        return None, "the stream delivered no frames"
    change = float(np.abs(during - before).mean())
    residual = float(np.abs(after - before).mean())
    if change < MIN_CHANGE:
        return False, f"zooming the webcam changed this picture by only {change:.1f} levels"
    if residual > MAX_RESIDUAL:
        return None, (
            f"the picture changed with the zoom ({change:.1f}) but did not settle back "
            f"({residual:.1f}) -- is something moving in front of the camera?"
        )
    return True, f"zoom changed the picture by {change:.1f} levels and back to {residual:.1f}"


def find_stream_index(camera, open_stream, max_index: int = 4, log=print) -> int:
    """The OpenCV index whose picture follows the webcam's zoom.

    ``open_stream(index)`` returns an opened capture or raises; each stream is
    released after its test.
    """

    tried = []
    for index in range(max_index + 1):
        try:
            capture = open_stream(index)
        except Exception:
            continue
        try:
            verdict, detail = stream_follows_camera(capture, camera)
        finally:
            capture.release()
        tried.append(f"  index {index}: {detail}")
        if verdict:
            log(f"camera index {index} is the webcam ({detail})")
            return index
    raise RuntimeError(
        "could not find which camera index shows the webcam:\n" + "\n".join(tried or ["  no camera opened"])
    )
