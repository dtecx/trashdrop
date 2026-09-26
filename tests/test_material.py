"""What an item is made of, and so which side of the table it goes to.

The classifier's own weights are not in the repository (training/export.py
writes them into models/, 350 MB), so most of this runs on a stand-in
encoder: what is pinned down is the arithmetic around it -- the crop reaches
CLIP letterboxed, not stretched, the head's answer is read the way it was
fitted, and only a sure answer sends an item anywhere. Where the real model
and the training cache are present, the cell's preprocessing is also checked
against the one the head was fitted on.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")

from trashdrop.perception.classifier import MATERIAL_MODEL_DIR, ClipMaterialClassifier  # noqa: E402
from trashdrop.sorter import MIN_CONFIDENCE, item_crop, side_for  # noqa: E402

CLASSES = ("metal", "other", "paper", "plastic")
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
PAD_RGB = (122, 116, 104)
TRAINING_CACHE = Path(__file__).resolve().parents[1] / "training" / "cache" / "ViT-B-32_laion2b_s34b_b79k_local_test_1.npz"


def write_head(directory: Path) -> None:
    """A head whose class k is embedding axis k, with CLIP's real preprocessing constants."""

    coef = np.zeros((len(CLASSES), 8), np.float32)
    coef[np.arange(len(CLASSES)), np.arange(len(CLASSES))] = 20.0
    np.savez(directory / "head.npz", coef=coef, intercept=np.zeros(len(CLASSES), np.float32),
             classes=np.array(CLASSES), mean=MEAN, std=STD, pad_rgb=np.array(PAD_RGB, np.uint8),
             size=np.array(224), encoder=np.array("stand-in"))


class ClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        write_head(self.directory)
        self.seen: list[np.ndarray] = []

    def classifier(self, axis: int) -> ClipMaterialClassifier:
        def encoder(pixels: np.ndarray) -> np.ndarray:
            self.seen.append(pixels)
            embedding = np.full((1, 8), 0.01, np.float32)
            embedding[0, axis] = 1.0
            return embedding

        return ClipMaterialClassifier(self.directory, encoder=encoder)

    def test_a_long_item_is_letterboxed_not_stretched(self) -> None:
        crop = np.zeros((60, 180, 3), np.uint8)  # a bottle lying down, black
        pixels = self.classifier(0).preprocess(crop)
        self.assertEqual(pixels.shape, (1, 3, 224, 224))
        pad = (np.array(PAD_RGB, np.float32) / 255 - MEAN) / STD
        self.assertTrue(np.allclose(pixels[0, :, 5, 112], pad, atol=0.02), "above the item: CLIP's mean colour")
        self.assertTrue(np.allclose(pixels[0, :, 112, 112], -MEAN / STD, atol=0.02), "the item itself, unchanged")
        self.assertEqual(pixels.dtype, np.float32)

    def test_the_head_is_read_as_it_was_fitted(self) -> None:
        for axis, name in enumerate(CLASSES):
            with self.subTest(name=name):
                probabilities = self.classifier(axis).probabilities(np.zeros((40, 40, 3), np.uint8))
                self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)
                self.assertEqual(max(probabilities, key=probabilities.get), name)

    def test_other_is_the_mixed_category(self) -> None:
        category, confidence = self.classifier(CLASSES.index("other")).classify(np.zeros((40, 40, 3), np.uint8))
        self.assertEqual(category, "mixed")
        self.assertGreater(confidence, 0.9)

    @unittest.skipUnless((MATERIAL_MODEL_DIR / "clip_image.onnx").is_file() and TRAINING_CACHE.is_file(),
                         "needs the exported model and the training cache")
    def test_the_cell_sees_our_crops_as_the_head_was_fitted_on_them(self) -> None:
        classifier = ClipMaterialClassifier()
        self.addCleanup(classifier.close)
        with np.load(TRAINING_CACHE) as cache:
            paths, embeddings = cache["paths"], cache["embeddings"].astype(np.float32)
        cosines = []
        for index in np.random.default_rng(0).choice(len(paths), 12, replace=False):
            embedding = np.asarray(classifier._encode(classifier.preprocess(cv2.imread(str(paths[index])))))[0]
            cosines.append(embedding @ embeddings[index] / np.linalg.norm(embedding) / np.linalg.norm(embeddings[index]))
        self.assertGreater(np.median(cosines), 0.99)
        self.assertGreater(min(cosines), 0.93)


class SideTests(unittest.TestCase):
    def test_plastic_and_metal_go_left_paper_right(self) -> None:
        self.assertEqual(side_for({"plastic": 0.95, "paper": 0.05})[0], "left")
        self.assertEqual(side_for({"metal": 0.9, "plastic": 0.05, "paper": 0.05})[0], "left")
        self.assertEqual(side_for({"paper": 0.92, "plastic": 0.08})[0], "right")

    def test_plastic_or_metal_between_them_is_sure_of_the_left(self) -> None:
        side, sure = side_for({"plastic": 0.5, "metal": 0.45, "paper": 0.05})
        self.assertEqual(side, "left")
        self.assertAlmostEqual(sure, 0.95)

    def test_the_likelier_side_wins_by_default(self) -> None:
        # The venue: an unsure item standing still stopped the demo.
        self.assertEqual(MIN_CONFIDENCE, 0.0)
        self.assertEqual(side_for({"plastic": 0.55, "paper": 0.45}), ("left", 0.55))
        self.assertEqual(side_for({"paper": 0.3, "plastic": 0.2, "other": 0.5})[0], "right")

    def test_a_threshold_leaves_unsure_items_where_they_are(self) -> None:
        self.assertIsNone(side_for({"plastic": 0.6, "paper": 0.4}, min_confidence=0.8)[0])
        self.assertIsNone(side_for({"other": 0.9, "paper": 0.1}, min_confidence=0.8)[0])
        self.assertEqual(side_for({"plastic": 0.85, "paper": 0.15}, min_confidence=0.8)[0], "left")

    def test_the_crop_holds_the_item_and_a_little_table(self) -> None:
        frame = np.zeros((1080, 1920, 3), np.uint8)
        item = np.zeros((540, 960), bool)
        item[100:120, 200:260] = True  # at half resolution
        crop = item_crop(frame, item, 2.0)
        self.assertEqual(crop.shape[:2], (40 + 20, 120 + 20))


if __name__ == "__main__":
    unittest.main()
