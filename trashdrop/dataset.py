"""Dependency-free indexing and routing rules for TrashNet-style datasets.

This module deliberately does not download, copy, or train on images. It turns
an existing local class-folder dataset into a deterministic JSONL manifest that
can later be consumed by a training project, while keeping the simulator light.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

# TrashNet has six visual classes. The station's bin contract is intentionally
# narrower: cardboard is routed to paper; glass and unknown "trash" are sent to
# a reject lane until a locally approved disposal route is configured. TrashNet
# does not teach the bio class, so it must never be used to route bio-waste.
TRASHNET_TO_STATION: dict[str, str | None] = {
    "cardboard": "paper",
    "glass": None,
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": None,
}


@dataclass(frozen=True)
class ManifestRecord:
    """One image reference, source class, station route, and stable split."""

    image: str
    source_label: str
    station_category: str | None
    split: str


@dataclass(frozen=True)
class IndexReport:
    """Summary emitted after writing an image manifest."""

    dataset_root: str
    manifest: str
    images: int
    source_counts: dict[str, int]
    station_counts: dict[str, int]
    reject_images: int


def _split_for(relative_image: Path) -> str:
    """Assign a reproducible 70/15/15 split without modifying source files."""

    bucket = int(hashlib.sha256(relative_image.as_posix().encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def index_trashnet_style_dataset(dataset_root: Path) -> list[ManifestRecord]:
    """Index ``<root>/<class>/<image>`` folders without copying any image.

    The folder set must consist entirely of the six TrashNet class names. This
    prevents a newly added or misspelled label from silently reaching a bin.
    """

    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_root}")
    labels = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    unknown = sorted(set(labels) - set(TRASHNET_TO_STATION))
    missing = sorted(set(TRASHNET_TO_STATION) - set(labels))
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown folders: {', '.join(unknown)}")
        if missing:
            details.append(f"missing expected folders: {', '.join(missing)}")
        raise ValueError("Not a complete TrashNet-style dataset (" + "; ".join(details) + ")")

    records: list[ManifestRecord] = []
    for label in labels:
        for image in sorted(dataset_root.joinpath(label).rglob("*")):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative_image = image.relative_to(dataset_root)
            records.append(
                ManifestRecord(
                    image=str(image),
                    source_label=label,
                    station_category=TRASHNET_TO_STATION[label],
                    split=_split_for(relative_image),
                )
            )
    if not records:
        raise ValueError(f"No supported image files found under {dataset_root}")
    return records


def write_manifest(dataset_root: Path, manifest_path: Path) -> IndexReport:
    """Write a JSONL manifest and return counts needed for a data review."""

    records = index_trashnet_style_dataset(dataset_root)
    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    source_counts = {label: 0 for label in TRASHNET_TO_STATION}
    station_counts: dict[str, int] = {}
    reject_images = 0
    for record in records:
        source_counts[record.source_label] += 1
        if record.station_category is None:
            reject_images += 1
        else:
            station_counts[record.station_category] = station_counts.get(record.station_category, 0) + 1
    return IndexReport(
        dataset_root=str(dataset_root.expanduser().resolve()),
        manifest=str(manifest_path),
        images=len(records),
        source_counts=source_counts,
        station_counts=station_counts,
        reject_images=reject_images,
    )
