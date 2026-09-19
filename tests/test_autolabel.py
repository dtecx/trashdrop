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

from trashdrop.dataset.autolabel import autolabel_session  # noqa: E402
from trashdrop.dataset.manifest import ManifestWriter, read_manifest, split_by_object  # noqa: E402

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
