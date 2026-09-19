"""TACO COCO indexing and conservative route mapping for TrashDrop.

TACO supplies object locations, unlike a class-folder classifier dataset. The
adapter preserves those boxes so a detector can be trained or evaluated without
throwing away the information an arm needs to choose a grasp point.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def route_taco_label(label: str) -> str:
    """Map a TACO taxonomy label to a station route, failing closed to reject.

    TACO's detailed labels are intentionally consolidated only where material
    is clear. Food *containers* are not classified as bio-waste, and ambiguous
    litter stays on the reject route for a person to decide.
    """

    normalized = label.casefold().replace("_", " ").replace("-", " ")
    if any(token in normalized for token in ("food waste", "organic", "vegetation")):
        return "bio"
    if any(token in normalized for token in ("paper", "carton", "cardboard", "corrugated", "tissue")):
        return "paper"
    if any(token in normalized for token in ("aluminium", "aluminum", "metal", "drink can", "pop tab")):
        return "metal"
    if any(token in normalized for token in ("plastic", "styrofoam", "foam")):
        return "plastic"
    return "reject"


def _split_for(relative_image: Path) -> str:
    bucket = int(hashlib.sha256(relative_image.as_posix().encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


@dataclass(frozen=True)
class TacoObjectRecord:
    """One COCO annotation retained as an object-detection training example."""

    image: str
    image_id: int
    annotation_id: int
    source_label: str
    station_category: str
    bbox_xywh: tuple[float, float, float, float]
    area: float
    split: str


@dataclass(frozen=True)
class TacoIndexReport:
    annotation_file: str
    image_root: str
    manifest: str
    objects: int
    route_counts: dict[str, int]
    missing_images: int


def index_taco_coco(annotation_file: Path, image_root: Path) -> tuple[list[TacoObjectRecord], int]:
    """Read a TACO COCO JSON file and preserve every non-crowd object box."""

    annotation_file = annotation_file.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    if not annotation_file.is_file():
        raise FileNotFoundError(f"TACO annotations not found: {annotation_file}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"TACO image root not found: {image_root}")
    try:
        payload: dict[str, Any] = json.loads(annotation_file.read_text(encoding="utf-8"))
        categories = {int(entry["id"]): str(entry["name"]) for entry in payload["categories"]}
        images = {int(entry["id"]): str(entry["file_name"]) for entry in payload["images"]}
        annotations = payload["annotations"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid COCO annotation file: {annotation_file}") from error

    records: list[TacoObjectRecord] = []
    missing_images = 0
    for annotation in annotations:
        if annotation.get("iscrowd", 0):
            continue
        try:
            image_id = int(annotation["image_id"])
            category = categories[int(annotation["category_id"])]
            relative_image = Path(images[image_id])
            raw_box = annotation["bbox"]
            if len(raw_box) != 4:
                raise ValueError("bbox has the wrong length")
            bbox = tuple(float(value) for value in raw_box)
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError("bbox must have positive dimensions")
        except (KeyError, TypeError, ValueError) as error:
            identifier = annotation.get("id", "unknown") if isinstance(annotation, dict) else "unknown"
            raise ValueError(f"Invalid TACO annotation {identifier!r}") from error
        image_path = image_root / relative_image
        if not image_path.is_file():
            missing_images += 1
        records.append(
            TacoObjectRecord(
                image=str(image_path),
                image_id=image_id,
                annotation_id=int(annotation.get("id", len(records))),
                source_label=category,
                station_category=route_taco_label(category),
                bbox_xywh=bbox,
                area=float(annotation.get("area", bbox[2] * bbox[3])),
                split=_split_for(relative_image),
            )
        )
    if not records:
        raise ValueError("TACO annotations contain no non-crowd objects")
    return records, missing_images


def write_taco_manifest(
    annotation_file: Path,
    image_root: Path,
    manifest_path: Path,
) -> TacoIndexReport:
    """Write COCO object records as JSONL; source images remain untouched."""

    records, missing_images = index_taco_coco(annotation_file, image_root)
    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    route_counts: dict[str, int] = {}
    for record in records:
        route_counts[record.station_category] = route_counts.get(record.station_category, 0) + 1
    return TacoIndexReport(
        annotation_file=str(annotation_file.expanduser().resolve()),
        image_root=str(image_root.expanduser().resolve()),
        manifest=str(manifest_path),
        objects=len(records),
        route_counts=route_counts,
        missing_images=missing_images,
    )
