"""CLIP image embeddings for the public datasets and our own crops, cached.

Embeddings are computed once and stored under ``training/cache``; every
experiment after that is a few seconds of scikit-learn on top.

Every image is letterboxed to a square before CLIP sees it, padded with CLIP's
own mean colour. Our crops are elongated -- a bottle lying down is three times
longer than it is wide -- and CLIP's default centre crop would cut its ends
off. The ONNX export does the same padding, so training and the live cell see
identical inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = Path(__file__).resolve().parent / "cache"

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
# CLIP's normalisation mean, as 8-bit RGB: padding with it becomes ~0 after
# normalisation, so it adds no signal of its own.
PAD_RGB = (122, 116, 104)

ROUTES = ("plastic", "paper", "metal", "other")

PUBLIC = {
    "trashnet": (
        DATA / "public" / "trashnet",
        {"cardboard": "paper", "glass": "other", "metal": "metal", "paper": "paper",
         "plastic": "plastic", "trash": "other"},
    ),
    "realwaste": (
        DATA / "public" / "realwaste" / "RealWaste",
        {"Cardboard": "paper", "Food Organics": "other", "Glass": "other", "Metal": "metal",
         "Miscellaneous Trash": "other", "Paper": "paper", "Plastic": "plastic",
         "Textile Trash": "other", "Vegetation": "other"},
    ),
    "drinking_waste": (
        DATA / "public" / "drinking_waste" / "Images_of_Waste" / "rawimgs",
        {"AluCan": "metal", "Glass": "other", "HDPEM": "plastic", "PET": "plastic"},
    ),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def letterbox(image):
    """Pad a PIL image to a square with CLIP's mean colour."""

    from PIL import Image

    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), PAD_RGB)
    canvas.paste(image.convert("RGB"), ((side - width) // 2, (side - height) // 2))
    return canvas


def load_model(device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, pretrained=PRETRAINED, cache_dir=str(CACHE / "weights")
    )
    return model.eval().to(device), preprocess, open_clip.get_tokenizer(MODEL)


def pick_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def embed_paths(paths: list[Path], model, preprocess, device: str, batch: int = 64) -> np.ndarray:
    import torch
    from PIL import Image

    out = []
    for start in range(0, len(paths), batch):
        images = []
        for path in paths[start : start + batch]:
            with Image.open(path) as source:
                images.append(preprocess(letterbox(source)))
        with torch.no_grad():
            features = model.encode_image(torch.stack(images).to(device))
            features = features / features.norm(dim=-1, keepdim=True)
        out.append(features.float().cpu().numpy())
        print(f"    {min(start + batch, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return np.concatenate(out).astype(np.float16)


def public_items(name: str) -> tuple[list[Path], list[str], list[str]]:
    root, mapping = PUBLIC[name]
    paths, routes, sources = [], [], []
    for folder, route in mapping.items():
        images = sorted(p for p in (root / folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        paths += images
        routes += [route] * len(images)
        sources += [folder] * len(images)
    return paths, routes, sources


def local_items(session: str) -> tuple[list[Path], list[str], list[str]]:
    rows = [json.loads(line) for line in (DATA / "labels" / f"{session}.jsonl").read_text().splitlines() if line.strip()]
    return (
        [DATA / row["crop"] for row in rows],
        [row["category"] for row in rows],
        [row["object_id"] for row in rows],
    )


def cache_path(name: str) -> Path:
    return CACHE / f"{MODEL}_{PRETRAINED}_{name}.npz"


def build(name: str, paths, routes, groups, model, preprocess, device: str) -> Path:
    target = cache_path(name)
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            if list(stored["paths"]) == [str(p) for p in paths]:
                print(f"  {name}: cached ({len(paths)})")
                return target
    print(f"  {name}: embedding {len(paths)} images")
    embeddings = embed_paths(paths, model, preprocess, device)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, paths=np.array([str(p) for p in paths]), embeddings=embeddings,
             routes=np.array(routes), groups=np.array(groups))
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", action="append", default=[], help="local capture session(s)")
    args = parser.parse_args()

    device = pick_device()
    print(f"loading {MODEL} ({PRETRAINED}) on {device}")
    model, preprocess, _ = load_model(device)
    for name in PUBLIC:
        paths, routes, groups = public_items(name)
        if paths:
            build(name, paths, routes, groups, model, preprocess, device)
        else:
            print(f"  {name}: no images under {PUBLIC[name][0]}")
    for session in args.session or ["test_1"]:
        paths, routes, groups = local_items(session)
        build(f"local_{session}", paths, routes, groups, model, preprocess, device)


if __name__ == "__main__":
    main()
