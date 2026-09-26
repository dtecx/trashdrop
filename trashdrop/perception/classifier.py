"""Crop classifiers: name the material in an image of one item.

The interface is one method, so whatever gets trained this week can be dropped
in without touching the motion stack. Training happens outside this repo --
keep the heavy stack (torch, ultralytics) out of the cell's environment and
export to ONNX, which runs anywhere including the Windows laptops on the team.

Confidence is part of the contract. The dispatcher routes anything below its
floor to the mixed bin, so a classifier must return a calibrated-ish score
rather than always 1.0. A softmax maximum is good enough.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..station import ALL_CATEGORIES, MIXED_CATEGORY, repository_root

MATERIAL_MODEL_DIR = repository_root() / "models" / "material"


class ConstantClassifier:
    """Always returns the same answer. For tests and for a wiring check."""

    def __init__(self, category: str = MIXED_CATEGORY, confidence: float = 0.0) -> None:
        if category not in ALL_CATEGORIES:
            raise ValueError(f"{category!r} is not a station category")
        self.category = category
        self.confidence = confidence

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        return self.category, self.confidence


class OnnxCropClassifier:
    """Run an exported image classifier over a crop.

    Expects a model taking ``(1, 3, size, size)`` float32 in 0..1, normalised
    with ImageNet statistics, and emitting one logit per class in ``labels``
    order. That is what a fine-tuned torchvision or timm backbone exports by
    default.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: Path,
        labels: list[str],
        *,
        size: int = 224,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        try:
            import onnxruntime
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "onnxruntime is optional. Install it with: "
                "uv sync --extra classifier"
            ) from error

        unknown = sorted(set(labels) - set(ALL_CATEGORIES))
        if unknown:
            raise ValueError(f"Model labels are not station categories: {unknown}")

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Classifier weights not found: {model_path}")

        self.session = onnxruntime.InferenceSession(str(model_path), providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.labels = list(labels)
        self.size = size

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        import cv2

        # Pad to square before resizing so the item is not stretched -- aspect
        # ratio is a real cue between a flattened carton and a bottle.
        height, width = crop.shape[:2]
        side = max(height, width)
        square = np.zeros((side, side, 3), dtype=crop.dtype)
        y0, x0 = (side - height) // 2, (side - width) // 2
        square[y0 : y0 + height, x0 : x0 + width] = crop

        resized = cv2.resize(square, (self.size, self.size), interpolation=cv2.INTER_AREA)
        array = resized.astype(np.float32) / 255.0
        array = (array - self.MEAN) / self.STD
        return np.transpose(array, (2, 0, 1))[None, ...]

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        if crop is None or crop.size == 0:
            return MIXED_CATEGORY, 0.0
        logits = self.session.run(None, {self.input_name: self._preprocess(crop)})[0][0]
        shifted = logits - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        index = int(probabilities.argmax())
        return self.labels[index], float(probabilities[index])


class ClipMaterialClassifier:
    """What an item is made of, from a crop: CLIP's image encoder and a small head.

    training/export.py writes both into models/material/: the encoder as ONNX
    (so the cell needs only onnxruntime), and a logistic regression fitted on
    CLIP embeddings of the public datasets and our own crops, with the
    preprocessing it was fitted with. Judged on objects it had never seen
    (training/evaluate.py), it named plastic, paper, metal or "other".

    The crop is letterboxed to a square with CLIP's mean colour, as the
    training images were: a bottle lying down is three times longer than it
    is wide, and stretching or cropping it would change what CLIP sees.
    """

    def __init__(self, directory: Path = MATERIAL_MODEL_DIR, *, encoder=None) -> None:
        directory = Path(directory)
        head_file = directory / "head.npz"
        if not head_file.is_file():
            raise FileNotFoundError(f"no material classifier in {directory}: cd training && uv run python export.py")
        with np.load(head_file, allow_pickle=False) as head:
            self.coef = head["coef"].astype(np.float32)
            self.intercept = head["intercept"].astype(np.float32)
            self.classes = [str(name) for name in head["classes"]]
            self.mean = head["mean"].astype(np.float32)
            self.std = head["std"].astype(np.float32)
            self.pad_rgb = tuple(int(v) for v in head["pad_rgb"])
            self.size = int(head["size"])
        if encoder is None:
            try:
                import onnxruntime
            except ImportError as error:  # pragma: no cover - optional dependency
                raise RuntimeError("onnxruntime is optional. Install it with: uv sync --inexact --extra classifier") from error
            session = onnxruntime.InferenceSession(str(directory / "clip_image.onnx"),
                                                   providers=["CPUExecutionProvider"])
            name = session.get_inputs()[0].name

            def encoder(pixels: np.ndarray) -> np.ndarray:
                return session.run(None, {name: pixels})[0]

        self._encode = encoder

    def close(self) -> None:
        """Let the ONNX session go now: torn down while Python exits, onnxruntime can abort."""

        self._encode = None

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """(1, 3, size, size) float32, as CLIP expects it."""

        import cv2

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        side = max(height, width)
        square = np.empty((side, side, 3), np.uint8)
        square[:] = self.pad_rgb
        y0, x0 = (side - height) // 2, (side - width) // 2
        square[y0 : y0 + height, x0 : x0 + width] = rgb
        # Shrinking averages, like the antialiased resize the training images had.
        interpolation = cv2.INTER_AREA if side > self.size else cv2.INTER_CUBIC
        resized = cv2.resize(square, (self.size, self.size), interpolation=interpolation)
        array = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        return np.ascontiguousarray(np.transpose(array, (2, 0, 1))[None, ...], dtype=np.float32)

    def probabilities(self, crop_bgr: np.ndarray) -> dict[str, float]:
        """Every material the head knows, with its probability."""

        embedding = np.asarray(self._encode(self.preprocess(crop_bgr)), np.float32)[0]
        embedding = embedding / np.linalg.norm(embedding)
        logits = self.coef @ embedding + self.intercept
        shifted = np.exp(logits - logits.max())
        return {name: float(p) for name, p in zip(self.classes, shifted / shifted.sum())}

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        if crop is None or crop.size == 0:
            return MIXED_CATEGORY, 0.0
        probabilities = self.probabilities(crop)
        name = max(probabilities, key=probabilities.get)
        return (name if name in ALL_CATEGORIES else MIXED_CATEGORY), probabilities[name]
