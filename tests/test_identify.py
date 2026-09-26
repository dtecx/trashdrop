"""Telling the webcam's stream apart from every other camera on the machine.

The real failure: tuning ran with ``--camera 1``, which was the MacBook's own
camera, while the settings went to the USB webcam. These fakes are the two
streams of that afternoon -- one that follows the webcam, one that does not.
"""

from __future__ import annotations

import unittest

import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
import numpy as np  # noqa: E402

from test_camera import fake_camera  # noqa: E402
from trashdrop.camera.identify import find_stream_index, stream_follows_camera  # noqa: E402


def scene() -> np.ndarray:
    """A table from above: a board with some structure on it."""

    rng = np.random.default_rng(3)
    frame = np.full((360, 640, 3), 150, np.uint8)
    for _ in range(25):
        x, y = int(rng.integers(0, 600)), int(rng.integers(0, 330))
        cv2.rectangle(frame, (x, y), (x + 40, y + 25), tuple(int(v) for v in rng.integers(20, 240, 3)), -1)
    return frame


class WebcamStream:
    """Frames from the webcam itself: they follow its digital zoom."""

    def __init__(self, device) -> None:
        self.device, self.base, self.rng = device, scene(), np.random.default_rng(1)
        self.released = False

    def read(self):
        factor = self.device.current["zoom"] / 100.0
        frame = self.base
        if factor > 1.0:
            height, width = frame.shape[:2]
            h, w = int(height / factor), int(width / factor)
            y0, x0 = (height - h) // 2, (width - w) // 2
            frame = cv2.resize(frame[y0 : y0 + h, x0 : x0 + w], (width, height))
        noisy = frame.astype(np.float32) + self.rng.normal(0, 2, frame.shape)
        return True, np.clip(noisy, 0, 255).astype(np.uint8)

    def release(self) -> None:
        self.released = True


class LaptopStream(WebcamStream):
    """The laptop's own camera: whatever the webcam does, this picture stays."""

    def read(self):
        noisy = self.base.astype(np.float32) + self.rng.normal(0, 2, self.base.shape)
        return True, np.clip(noisy, 0, 255).astype(np.uint8)


class RestlessStream(WebcamStream):
    """Someone moving in front of the lens: the picture changes on its own."""

    def __init__(self, device) -> None:
        super().__init__(device)
        self.frames = 0

    def read(self):
        self.frames += 1
        frame = scene()
        cv2.circle(frame, (40 + (self.frames * 23) % 560, 180), 60, (30, 30, 30), -1)
        return True, frame


class WristCamera:
    """A small wrist camera: a token zoom range, so brightness is the wiggle."""

    usb_id = "2993:0858"

    def __init__(self) -> None:
        self.current = {"zoom": 0, "brightness": 0, "saturation": 64}

    def ranges(self):
        from trashdrop.camera.uvc import ControlRange

        return {
            "zoom": ControlRange(0, 3, 1, 0),
            "brightness": ControlRange(-128, 127, 1, 0),
            "saturation": ControlRange(0, 128, 1, 64),
        }

    def get(self, name: str) -> int:
        return self.current[name]

    def set(self, name: str, value: int) -> None:
        self.current[name] = value


class WristStream(WebcamStream):
    """Frames from the wrist camera: they follow its brightness."""

    def read(self):
        frame = self.base.astype(np.float32) + self.device.current["brightness"] * 0.5
        noisy = frame + self.rng.normal(0, 2, frame.shape)
        return True, np.clip(noisy, 0, 255).astype(np.uint8)


class IdentityTests(unittest.TestCase):
    def test_the_webcam_stream_is_recognised(self) -> None:
        camera, device = fake_camera()
        verdict, detail = stream_follows_camera(WebcamStream(device), camera, settle_frames=1)
        self.assertTrue(verdict, detail)

    def test_the_laptop_camera_is_rejected(self) -> None:
        camera, device = fake_camera()
        verdict, detail = stream_follows_camera(LaptopStream(device), camera, settle_frames=1)
        self.assertIs(verdict, False, detail)

    def test_zoom_is_always_put_back(self) -> None:
        camera, device = fake_camera()
        stream_follows_camera(LaptopStream(device), camera, settle_frames=1)
        self.assertEqual(device.current["zoom"], 100)

    def test_movement_in_view_is_inconclusive_not_a_verdict(self) -> None:
        camera, device = fake_camera()
        verdict, detail = stream_follows_camera(RestlessStream(device), camera, settle_frames=1)
        self.assertIsNone(verdict, detail)

    def test_the_right_index_is_found_among_several(self) -> None:
        camera, device = fake_camera()
        streams = {0: LaptopStream(device), 1: WebcamStream(device), 2: LaptopStream(device)}

        def open_stream(index: int):
            if index not in streams:
                raise RuntimeError("no such camera")
            return streams[index]

        self.assertEqual(find_stream_index(camera, open_stream, log=lambda *_: None), 1)
        self.assertTrue(all(stream.released for stream in list(streams.values())[:2]))

    def test_no_matching_stream_is_an_error_that_lists_what_was_tried(self) -> None:
        camera, device = fake_camera()

        def open_stream(index: int):
            if index > 1:
                raise RuntimeError("no such camera")
            return LaptopStream(device)

        with self.assertRaises(RuntimeError) as caught:
            find_stream_index(camera, open_stream, log=lambda *_: None)
        self.assertIn("index 0", str(caught.exception))
        self.assertIn("index 1", str(caught.exception))

    def test_a_camera_without_real_zoom_is_found_by_its_brightness(self) -> None:
        wrist = WristCamera()
        verdict, detail = stream_follows_camera(WristStream(wrist), wrist, settle_frames=1)
        self.assertTrue(verdict, detail)
        self.assertIn("brightness", detail)
        self.assertEqual(wrist.current["brightness"], 0, "brightness must be put back")

        verdict, detail = stream_follows_camera(LaptopStream(wrist), wrist, settle_frames=1)
        self.assertIs(verdict, False, detail)
        self.assertEqual(wrist.current, {"zoom": 0, "brightness": 0, "saturation": 64})

    def test_no_camera_opening_at_all_points_at_the_camera_permission(self) -> None:
        camera, _ = fake_camera()

        def open_stream(index: int):
            raise RuntimeError("not authorized")

        with self.assertRaises(RuntimeError) as caught:
            find_stream_index(camera, open_stream, log=lambda *_: None)
        self.assertIn("permission", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
