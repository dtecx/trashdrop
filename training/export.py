"""Export the material classifier for the cell: CLIP's image encoder as ONNX, plus a head.

The cell runs it with onnxruntime alone (`uv sync --inexact --extra classifier`
in the repository root), no torch. The head is the variant evaluate.py found
best on objects it had never seen -- a logistic regression on CLIP embeddings
of the public datasets and our own crops -- fitted here on all of them.

    cd training && uv run python export.py        # writes ../models/material/

The encoder takes a (1, 3, 224, 224) float32 image, letterboxed to a square
with CLIP's mean colour and normalised with CLIP's statistics (the constants
are saved with the head), and returns the unnormalised embedding.
"""

from __future__ import annotations

import argparse

import numpy as np

from embed import MODEL, PAD_RGB, PRETRAINED, ROOT, ROUTES, load_model
from evaluate import LOCAL_WEIGHT, fit_probe, load

OUT = ROOT / "models" / "material"
PUBLIC = ("trashnet", "realwaste", "drinking_waste")
SIZE = 224


def main() -> None:
    import torch
    from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", action="append", default=[], help="local capture session(s), embedded already")
    args = parser.parse_args()
    sessions = args.session or ["test_1"]

    sets = [load(name) for name in PUBLIC]
    local = [load(f"local_{session}") for session in sessions]
    X = np.concatenate([s[0] for s in sets + local])
    y = np.concatenate([s[1] for s in sets + local])
    weights = np.concatenate([np.ones(len(s[0])) for s in sets] + [np.full(len(s[0]), LOCAL_WEIGHT) for s in local])
    probe = fit_probe(X, y, weights)
    print(f"head fitted on {len(X)} images ({sum(len(s[0]) for s in local)} of them ours), classes {list(probe.classes_)}")
    assert set(probe.classes_) <= set(ROUTES)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "head.npz",
        coef=probe.coef_.astype(np.float32),
        intercept=probe.intercept_.astype(np.float32),
        classes=np.array(probe.classes_),
        mean=np.array(OPENAI_DATASET_MEAN, np.float32),
        std=np.array(OPENAI_DATASET_STD, np.float32),
        pad_rgb=np.array(PAD_RGB, np.uint8),
        size=np.array(SIZE),
        encoder=np.array(f"{MODEL} {PRETRAINED}"),
    )

    model, _, _ = load_model("cpu")
    visual = model.visual.eval()
    # PyTorch's fused attention kernel has no ONNX form; the plain one does.
    torch.backends.mha.set_fastpath_enabled(False)
    target = OUT / "clip_image.onnx"
    with torch.no_grad():
        torch.onnx.export(
            visual,
            torch.zeros(1, 3, SIZE, SIZE),
            str(target),
            input_names=["pixels"],
            output_names=["embedding"],
            dynamic_axes={"pixels": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    print(f"wrote {target} ({target.stat().st_size / 1e6:.0f} MB) and {OUT / 'head.npz'}")

    # The exported encoder must agree with the one the head was fitted on.
    import onnxruntime

    session = onnxruntime.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    probe_input = torch.rand(2, 3, SIZE, SIZE)
    with torch.no_grad():
        expected = visual(probe_input).numpy()
    got = session.run(None, {"pixels": probe_input.numpy()})[0]
    cosine = (expected * got).sum(1) / np.linalg.norm(expected, axis=1) / np.linalg.norm(got, axis=1)
    print(f"onnx vs torch cosine: {cosine.min():.6f}")
    assert cosine.min() > 0.9999


if __name__ == "__main__":
    main()
