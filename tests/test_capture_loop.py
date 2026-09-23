"""The capture loop end to end, with scripted frames and key presses.

The loop is GUI glue around the auto-shutter, which is exactly the kind of code
that works in every unit and fails when assembled. This drives a whole short
session -- background, two poses, a class switch, a manual burst -- through the
real loop with a fake camera, a fake window and a fake clock.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
import numpy as np  # noqa: E402

from trashdrop.dataset.capture import CaptureConfig, next_object_id, run_capture  # noqa: E402

WIDTH, HEIGHT = 640, 360


def table(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((HEIGHT, WIDTH, 3), (150, 175, 160), dtype=np.float32)
    return np.clip(frame + rng.normal(0, 1.5, frame.shape), 0, 255).astype(np.uint8)


def item(seed: int, x: int, y: int, w: int, h: int, colour) -> np.ndarray:
    frame = table(seed)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, -1)
    return frame


class FakeClock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


class FakeCamera:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class ScriptedWindow:
    """Stands in for cv2.imshow / waitKey: presses keys on given frames."""

    def __init__(self, camera: FakeCamera, clock: FakeClock, keys: dict[int, str]) -> None:
        self.camera, self.clock, self.keys = camera, clock, keys
        self.shown = 0
        self.closed = False

    def show(self, image) -> None:
        assert image.shape[1] == 960, "preview should be downscaled for a laptop screen"
        self.shown += 1
        self.clock.now += 0.1

    def key(self) -> int:
        if not self.camera.frames and self.shown:
            return ord("q")
        pressed = self.keys.get(self.shown - 1)
        return ord(pressed) if pressed else 255

    def close(self) -> None:
        self.closed = True


class CaptureLoopTests(unittest.TestCase):
    def test_a_short_session(self) -> None:
        frames = (
            [table(s) for s in range(4)]  # 0-3 empty table; background on frame 2
            + [item(s, 280, 150, 70, 45, (40, 40, 180)) for s in range(10, 30)]  # 4-23 pose A
            + [item(s, 120, 220, 45, 70, (40, 40, 180)) for s in range(30, 50)]  # 24-43 pose B
            + [item(s, 330, 100, 60, 60, (190, 190, 200)) for s in range(50, 75)]  # 44-68 new item
        )
        camera = FakeCamera(frames)
        clock = FakeClock()
        window = ScriptedWindow(camera, clock, keys={2: "b", 44: "4", 64: " "})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = CaptureConfig(session="s1", root=root, burst=3, burst_interval=0.0)
            run_capture(config, category="plastic", capture=camera, ui=window, clock=clock)

            self.assertTrue(camera.released)
            self.assertTrue(window.closed)
            self.assertTrue((root / "bg" / "s1" / "default.jpg").is_file())

            with (root / "raw" / "s1" / "manifest.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            by_object: dict[str, int] = {}
            for row in rows:
                key = f"{row['category']}/{row['object_id']}"
                by_object[key] = by_object.get(key, 0) + 1

            # Two settled poses of the first item, taken by the shutter alone.
            self.assertEqual(by_object.get("plastic/plastic_01"), 2, by_object)
            # Key 4 switched to metal with a fresh id: one auto frame, then a
            # manual burst of three.
            self.assertEqual(by_object.get("metal/metal_01"), 4, by_object)
            for row in rows:
                self.assertTrue((root / row["image"]).is_file(), row["image"])

    def test_without_a_background_nothing_is_taken(self) -> None:
        frames = [item(s, 280, 150, 70, 45, (40, 40, 180)) for s in range(30)]
        camera = FakeCamera(frames)
        clock = FakeClock()
        window = ScriptedWindow(camera, clock, keys={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_capture(
                CaptureConfig(session="s1", root=root, burst_interval=0.0),
                capture=camera,
                ui=window,
                clock=clock,
            )
            with (root / "raw" / "s1" / "manifest.csv").open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])

    def test_an_unknown_class_is_refused_before_the_camera_opens(self) -> None:
        with self.assertRaises(ValueError):
            run_capture(CaptureConfig(session="s1"), category="batteries")


class ObjectIdTests(unittest.TestCase):
    def test_ids_continue_from_what_is_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / "metal" / "metal_01").mkdir(parents=True)
            (raw / "metal" / "metal_04").mkdir(parents=True)
            (raw / "metal" / "cola_can").mkdir(parents=True)
            self.assertEqual(next_object_id(raw, "metal"), "metal_05")

    def test_next_moves_past_an_unsaved_current_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(next_object_id(Path(directory), "paper", "paper_03"), "paper_04")

    def test_first_id_of_a_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(next_object_id(Path(directory), "bio"), "bio_01")


if __name__ == "__main__":
    unittest.main()
