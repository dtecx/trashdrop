"""Manual capture must save exactly one photo per press and undo cleanly."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
import numpy as np  # noqa: E402

from trashdrop.dataset.capture import (  # noqa: E402
    CaptureConfig, _OpenCvUi, _capture_key, next_object_id, run_capture,
)
from trashdrop.dataset.manifest import read_manifest  # noqa: E402
from trashdrop.dataset.zone import DetectionZone, load_zone, save_zone  # noqa: E402

WIDTH, HEIGHT = 640, 360


def frame(item=False):
    image = np.full((HEIGHT, WIDTH, 3), 170, np.uint8)
    if item:
        cv2.rectangle(image, (280, 145), (370, 225), (40, 40, 190), -1)
    return image


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        self.released = True


class ScriptedWindow:
    def __init__(self, camera, keys):
        self.camera = camera
        self.keys = keys
        self.shown = 0
        self.closed = False

    def show(self, image):
        assert image.shape[:2] == (900, 1280)
        self.shown += 1

    def key(self):
        value = self.keys.pop(self.shown - 1, None)
        if value:
            return ord(value)
        if not self.camera.frames and self.shown:
            return 27
        return 255

    def close(self):
        self.closed = True


def configured(root):
    path = root / "camera_zone.json"
    save_zone(path, DetectionZone(WIDTH, HEIGHT, 180, 70, 280, 220))
    return CaptureConfig(session="s1", root=root, zone_calibration=path)


class CaptureLoopTests(unittest.TestCase):
    def test_manual_one_press_one_frame_class_switch_and_undo(self):
        camera = FakeCamera([frame(), frame(True), frame(True), frame(True),
                             frame(True), frame(True), frame(True), frame(True)])
        window = ScriptedWindow(camera, {0: "r", 1: "q", 2: "w", 3: "q",
                                         4: "e", 5: "2", 6: "q"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_capture(configured(root), capture=camera, ui=window)
            rows = read_manifest(root / "raw" / "s1" / "manifest.csv")
            self.assertEqual([(r.category, r.object_id) for r in rows],
                             [("plastic", "plastic_01"), ("paper", "paper_01")])
            self.assertTrue((root / "raw" / "s1" / "_undone" / "plastic" / "plastic_01" / "frame_0000.jpg").is_file())
            self.assertTrue((root / "raw" / "s1" / "plastic" / "plastic_01" / "frame_0001.jpg").is_file())
            self.assertTrue((root / "raw" / "s1" / "zone.json").is_file())
            self.assertTrue(camera.released)
            self.assertTrue(window.closed)

    def test_no_automatic_capture_even_with_settled_item(self):
        camera = FakeCamera([frame()] + [frame(True)] * 8)
        window = ScriptedWindow(camera, {0: "r"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_capture(configured(root), capture=camera, ui=window)
            self.assertEqual(read_manifest(root / "raw" / "s1" / "manifest.csv"), [])

    def test_light_change_requires_its_own_background(self):
        camera = FakeCamera([frame(), frame(True), frame(), frame(True)])
        window = ScriptedWindow(camera, {0: "r", 1: "t", 2: "r", 3: "q"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_capture(configured(root), capture=camera, ui=window)
            rows = read_manifest(root / "raw" / "s1" / "manifest.csv")
            self.assertEqual([row.lighting for row in rows], ["daylight"])
            self.assertTrue((root / "bg" / "s1" / "daylight.jpg").is_file())

    def test_re_shooting_background_keeps_the_reference_used_by_older_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = FakeCamera([frame(), frame(True), frame(), frame(True)])
            run_capture(configured(root), capture=camera,
                        ui=ScriptedWindow(camera, {0: "r", 1: "q", 2: "r", 3: "q"}))
            rows = read_manifest(root / "raw" / "s1" / "manifest.csv")
            self.assertEqual([row.lighting for row in rows], ["default", "default_v01"])
            self.assertTrue((root / "bg" / "s1" / "default.jpg").is_file())
            self.assertTrue((root / "bg" / "s1" / "default_v01.jpg").is_file())

            camera = FakeCamera([frame(True)])
            run_capture(configured(root), capture=camera,
                        ui=ScriptedWindow(camera, {0: "q"}))
            rows = read_manifest(root / "raw" / "s1" / "manifest.csv")
            self.assertEqual(rows[-1].lighting, "default_v01")

    def test_missing_background_or_zone_blocks_a_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = FakeCamera([frame(True), frame(True)])
            run_capture(configured(root), capture=camera, ui=ScriptedWindow(camera, {0: "q"}))
            self.assertEqual(read_manifest(root / "raw" / "s1" / "manifest.csv"), [])
            camera = FakeCamera([frame(), frame(True)])
            config = CaptureConfig(session="s2", root=root, zone_calibration=root / "missing.json")
            run_capture(config, capture=camera, ui=ScriptedWindow(camera, {0: "r", 1: "q"}))
            self.assertEqual(read_manifest(root / "raw" / "s2" / "manifest.csv"), [])

    def test_resume_keeps_session_zone_and_recent_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configured(root)
            camera = FakeCamera([frame(), frame(True)])
            run_capture(config, capture=camera, ui=ScriptedWindow(camera, {0: "r", 1: "q"}))
            config.zone_calibration.unlink()
            camera = FakeCamera([frame(True), frame(True)])
            run_capture(config, capture=camera, ui=ScriptedWindow(camera, {0: "w", 1: "q"}))
            rows = read_manifest(root / "raw" / "s1" / "manifest.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].object_id, "plastic_02")

    def test_undo_after_resume_removes_the_last_photo_of_the_same_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configured(root)
            camera = FakeCamera([frame(), frame(True), frame(True)])
            run_capture(config, category="metal", object_id="metal_01", capture=camera,
                        ui=ScriptedWindow(camera, {0: "r", 1: "q", 2: "q"}))
            camera = FakeCamera([frame(True)])
            run_capture(config, category="metal", object_id="metal_01", capture=camera,
                        ui=ScriptedWindow(camera, {0: "w"}))
            rows = read_manifest(root / "raw" / "s1" / "manifest.csv")
            self.assertEqual([Path(row.image).name for row in rows], ["frame_0000.jpg"])
            self.assertTrue((root / "raw" / "s1" / "_undone" / "metal" / "metal_01" /
                             "frame_0001.jpg").is_file())

    def test_shortcuts_accept_uppercase_and_russian_keyboard_layout(self):
        self.assertEqual(_capture_key(ord("W")), ord("w"))
        self.assertEqual(_capture_key(ord("ц")), ord("w"))
        self.assertEqual(_capture_key(ord("Ч")), ord("x"))
        self.assertEqual(_capture_key(ord(" ")), ord(" "))

    def test_clicking_undo_button_queues_the_same_action(self):
        from trashdrop.dataset.capture import UNDO_BUTTON

        ui = _OpenCvUi.__new__(_OpenCvUi)
        ui._clicked_key = None
        x0, y0, x1, y1 = UNDO_BUTTON
        ui._on_mouse(cv2.EVENT_LBUTTONDOWN, (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
        self.assertEqual(ui._clicked_key, ord("w"))

    def test_only_three_partner_item_keys_select_a_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = FakeCamera([frame(), frame(True), frame(True), frame(True),
                                 frame(True), frame(True), frame(True)])
            window = ScriptedWindow(camera, {0: "r", 1: "4", 2: "q", 3: "5",
                                             4: "q", 5: "3", 6: "q"})
            run_capture(configured(root), capture=camera, ui=window)
            rows = read_manifest(root / "raw" / "s1" / "manifest.csv")
            self.assertEqual([(row.category, row.object_id) for row in rows],
                             [("plastic", "plastic_01"), ("plastic", "plastic_01"),
                              ("metal", "metal_01")])

    def test_recalibration_updates_an_empty_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configured(root)
            camera = FakeCamera([frame()])
            run_capture(config, capture=camera, ui=ScriptedWindow(camera, {}))
            original = load_zone(root / "raw" / "s1" / "zone.json")
            revised = DetectionZone(WIDTH, HEIGHT, 110, 40, 420, 280)
            save_zone(config.zone_calibration, revised)

            camera = FakeCamera([frame()])
            run_capture(config, capture=camera, ui=ScriptedWindow(camera, {}))

            self.assertNotEqual(original, revised)
            self.assertEqual(load_zone(root / "raw" / "s1" / "zone.json"), revised)
            self.assertEqual(read_manifest(root / "raw" / "s1" / "manifest.csv"), [])

    def test_recalibration_requires_new_session_after_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configured(root)
            camera = FakeCamera([frame(), frame(True)])
            run_capture(config, capture=camera, ui=ScriptedWindow(camera, {0: "r", 1: "q"}))
            original = load_zone(root / "raw" / "s1" / "zone.json")
            save_zone(config.zone_calibration, DetectionZone(WIDTH, HEIGHT, 110, 40, 420, 280))

            with self.assertRaisesRegex(ValueError, "Start a new --session"):
                run_capture(config)

            self.assertEqual(load_zone(root / "raw" / "s1" / "zone.json"), original)
            self.assertEqual(len(read_manifest(root / "raw" / "s1" / "manifest.csv")), 1)

    def test_an_unknown_class_is_refused_before_camera_opens(self):
        with self.assertRaises(ValueError):
            run_capture(CaptureConfig(session="s1"), category="batteries")


class ObjectIdTests(unittest.TestCase):
    def test_ids_continue_from_what_is_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / "metal" / "metal_01").mkdir(parents=True)
            (raw / "metal" / "metal_04").mkdir(parents=True)
            (raw / "metal" / "cola_can").mkdir(parents=True)
            self.assertEqual(next_object_id(raw, "metal"), "metal_05")

    def test_next_moves_past_an_unsaved_current_id(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(next_object_id(Path(directory), "paper", "paper_03"), "paper_04")


if __name__ == "__main__":
    unittest.main()
