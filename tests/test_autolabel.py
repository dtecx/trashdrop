"""The capture-to-crops pipeline, exercised on synthetic frames.

No camera and no MuJoCo: frames are drawn with OpenCV so the test can run
anywhere, including the Windows laptops. What it checks is the contract the
team depends on tomorrow -- that a burst shot against an empty-table reference
comes back as crops with a correct box, and that the frames which should be
refused actually are.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
import numpy as np  # noqa: E402

from trashdrop.dataset.autolabel import _foreground, autolabel_session  # noqa: E402
from trashdrop.dataset.manifest import ManifestWriter, read_manifest, split_by_object  # noqa: E402
from trashdrop.dataset.review import reject_frames  # noqa: E402
from trashdrop.dataset.zone import DetectionZone, save_zone  # noqa: E402
from trashdrop.perception.regions import find_item_region  # noqa: E402

WIDTH, HEIGHT = 320, 240


def table(noise_seed: int = 0) -> np.ndarray:
    """An empty, slightly noisy work surface."""

    rng = np.random.default_rng(noise_seed)
    frame = np.full((HEIGHT, WIDTH, 3), 210, dtype=np.uint8)
    noise = rng.normal(0, 2.0, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def with_item(seed: int, box: tuple[int, int, int, int], colour=(40, 40, 200)) -> np.ndarray:
    x, y, w, h = box
    frame = table(seed)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, -1)
    return frame


class AutolabelTests(unittest.TestCase):
    def _session(self, root: Path, frames: list[tuple[str, np.ndarray]], category="plastic",
                 object_id="bottle_01", lighting="default") -> None:
        raw = root / "raw" / "s1" / category / object_id
        raw.mkdir(parents=True, exist_ok=True)
        bg_dir = root / "bg" / "s1"
        bg_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(bg_dir / f"{lighting}.jpg"), table(0))
        with ManifestWriter(root / "raw" / "s1" / "manifest.csv") as manifest:
            for name, image in frames:
                cv2.imwrite(str(raw / name), image)
                manifest.append(
                    image=str((raw / name).relative_to(root)),
                    category=category,
                    object_id=object_id,
                    lighting=lighting,
                    session="s1",
                )

    def test_a_clean_burst_becomes_crops_with_correct_boxes(self) -> None:
        boxes = [(90 + i * 6, 70 + i * 4, 52, 38) for i in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._session(
                root, [(f"frame_{i:04d}.jpg", with_item(i + 1, b)) for i, b in enumerate(boxes)]
            )
            report = autolabel_session("s1", root=root)

            self.assertEqual(report.labelled, len(boxes))
            self.assertEqual(report.rejected, 0)
            records = [
                json.loads(line)
                for line in Path(report.labels_path).read_text(encoding="utf-8").splitlines()
            ]
            for record, expected in zip(records, boxes):
                x, y, w, h = record["bbox_xywh"]
                # JPEG and blurring move the edges a little; a few pixels of
                # slack still pins the box to the right object.
                self.assertAlmostEqual(x, expected[0], delta=6)
                self.assertAlmostEqual(y, expected[1], delta=6)
                self.assertAlmostEqual(w, expected[2], delta=10)
                self.assertAlmostEqual(h, expected[3], delta=10)
                self.assertTrue((root / record["crop"]).is_file())

    def test_frames_with_two_objects_are_refused(self) -> None:
        frame = with_item(1, (40, 40, 50, 40))
        cv2.rectangle(frame, (200, 140), (260, 190), (30, 180, 60), -1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._session(root, [("frame_0000.jpg", frame)])
            report = autolabel_session("s1", root=root)
            self.assertEqual(report.labelled, 0)
            self.assertIn("more_than_one_object", report.reasons)

    def test_selected_zone_ignores_other_objects_and_keeps_full_frame_coordinates(self) -> None:
        frame = with_item(1, (90, 70, 50, 40))
        cv2.rectangle(frame, (245, 145), (300, 200), (30, 180, 60), -1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._session(root, [("frame_0000.jpg", frame)])
            save_zone(root / "raw" / "s1" / "zone.json",
                      DetectionZone(WIDTH, HEIGHT, 60, 45, 120, 105))
            report = autolabel_session("s1", root=root)
            self.assertEqual(report.labelled, 1, report.reasons)
            record = json.loads(Path(report.labels_path).read_text().splitlines()[0])
            x, y, w, h = record["bbox_xywh"]
            self.assertAlmostEqual(x, 90, delta=6)
            self.assertAlmostEqual(y, 70, delta=6)
            self.assertAlmostEqual(w, 50, delta=10)
            self.assertAlmostEqual(h, 40, delta=10)
            self.assertTrue(all(60 <= point[0] <= 180 for point in record["rotated_box"]))
            crop = cv2.imread(str(root / record["crop"]))
            self.assertLessEqual(crop.shape[1], 120)

    def test_rejected_frame_stays_excluded_after_autolabel_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._session(root, [
                ("frame_0000.jpg", with_item(1, (90, 70, 50, 40))),
                ("frame_0001.jpg", with_item(2, (100, 75, 50, 40))),
            ])
            self.assertEqual(autolabel_session("s1", root=root).labelled, 2)
            self.assertEqual(reject_frames("s1", ["plastic/bottle_01/frame_0000.jpg"], root), 1)
            self.assertTrue((root / "raw/s1/plastic/bottle_01/frame_0000.jpg").is_file())
            self.assertFalse((root / "crops/s1/plastic/bottle_01/frame_0000.jpg").exists())

            report = autolabel_session("s1", root=root)
            self.assertEqual(report.labelled, 1)
            self.assertEqual(report.reasons, {"manual_reject": 1})
            self.assertFalse((root / "crops/s1/plastic/bottle_01/frame_0000.jpg").exists())

    def test_an_empty_table_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._session(root, [("frame_0000.jpg", table(7))])
            report = autolabel_session("s1", root=root)
            self.assertEqual(report.labelled, 0)
            self.assertIn("nothing_changed", report.reasons)

    def test_an_item_running_off_frame_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._session(root, [("frame_0000.jpg", with_item(1, (0, 90, 60, 40)))])
            report = autolabel_session("s1", root=root)
            self.assertEqual(report.labelled, 0)
            self.assertIn("touches_frame_edge", report.reasons)

    def test_cast_shadow_does_not_expand_the_item_crop(self) -> None:
        reference = np.full((300, 400, 3), (150, 175, 160), np.uint8)
        for colour in ((35, 40, 45), (228, 230, 225)):
            with self.subTest(colour=colour):
                live = reference.copy()
                shadow = np.zeros(reference.shape[:2], np.uint8)
                cv2.fillPoly(shadow, [np.array([[55, 90], [230, 110],
                                                 [230, 220], [85, 230]])], 255)
                live[shadow != 0] = (live[shadow != 0].astype(np.float32) * 0.72).astype(np.uint8)
                cv2.rectangle(live, (220, 120), (290, 205), colour, -1)

                mask, _ = _foreground(live, reference)
                region, reason = find_item_region(mask)
                self.assertEqual(reason, "ok")
                self.assertAlmostEqual(region.box[0], 220, delta=8)
                self.assertAlmostEqual(region.box[2], 71, delta=10)

    def test_neutral_grey_item_is_not_erased_as_a_shadow(self) -> None:
        reference = np.full((300, 400, 3), (150, 175, 160), np.uint8)
        live = reference.copy()
        cv2.rectangle(live, (220, 120), (290, 205), (112, 132, 120), -1)
        mask, _ = _foreground(live, reference)
        region, reason = find_item_region(mask)
        self.assertEqual(reason, "ok")
        self.assertAlmostEqual(region.box[2], 71, delta=10)

    def test_smooth_grey_item_is_labelled_with_its_whole_outline(self) -> None:
        reference = np.full((360, 640, 3), (160, 170, 165), np.uint8)
        live = reference.copy()
        cv2.rectangle(live, (245, 110), (355, 245), (124, 132, 128), -1)
        cv2.rectangle(live, (280, 150), (320, 190), (235, 238, 235), -1)
        for y in (130, 210, 235):
            cv2.line(live, (260, y), (340, y), (90, 100, 95), 2)
        mask, fit = _foreground(live, reference)
        region, reason = find_item_region(mask)
        self.assertTrue(fit.trusted)
        self.assertEqual(reason, "ok")
        self.assertLessEqual(region.box[0], 245)
        self.assertGreaterEqual(region.box[0] + region.box[2], 355)

    def test_pale_item_outline_is_kept_below_strong_foreground_threshold(self) -> None:
        reference = np.full((360, 640, 3), 155, np.uint8)
        live = reference.copy()
        # The outer paper boundary changes by less than the strong foreground
        # threshold; the two folds are the only strong foreground pixels.
        cv2.rectangle(live, (180, 70), (360, 260), (180, 180, 180), -1)
        cv2.line(live, (250, 140), (305, 185), (90, 90, 90), 3)
        cv2.line(live, (260, 210), (318, 165), (210, 210, 210), 2)

        mask, _ = _foreground(live, reference)
        region, reason = find_item_region(mask)
        self.assertEqual(reason, "ok")
        x, y, w, h = region.box
        self.assertLessEqual(x, 180)
        self.assertLessEqual(y, 70)
        self.assertGreaterEqual(x + w, 360)
        self.assertGreaterEqual(y + h, 260)

    def test_old_bio_photos_do_not_enter_the_three_material_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._session(root, [("frame_0000.jpg", with_item(1, (90, 70, 50, 40)))],
                          category="bio")
            old_crop = root / "crops/s1/bio/bottle_01/frame_0000.jpg"
            old_crop.parent.mkdir(parents=True)
            old_crop.write_bytes(b"stale")
            report = autolabel_session("s1", root=root)
            self.assertEqual(report.labelled, 0)
            self.assertEqual(report.reasons, {"unsupported_category": 1})
            self.assertTrue((root / "raw/s1/bio/bottle_01/frame_0000.jpg").is_file())
            self.assertFalse(old_crop.exists())

    def test_missing_background_is_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._session(root, [("frame_0000.jpg", with_item(1, (90, 70, 50, 40)))])
            (root / "bg" / "s1" / "default.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                autolabel_session("s1", root=root)


class SplitTests(unittest.TestCase):
    def test_all_frames_of_one_object_land_in_the_same_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            with ManifestWriter(path) as manifest:
                for object_id in (f"obj_{i:02d}" for i in range(12)):
                    for frame in range(5):
                        manifest.append(
                            image=f"{object_id}/{frame}.jpg",
                            category="plastic",
                            object_id=object_id,
                            lighting="default",
                            session="s1",
                        )
            rows = read_manifest(path)
            split = split_by_object(rows)
            # One decision per object, never per frame: a burst is made of
            # near-duplicates, so splitting inside it inflates accuracy.
            self.assertEqual(len(split), 12)
            self.assertEqual(set(split.values()) - {"train", "test"}, set())


if __name__ == "__main__":
    unittest.main()
