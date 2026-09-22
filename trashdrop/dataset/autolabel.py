"""Turn one-item-at-a-time bursts into labelled masks, boxes and crops.

This is the step that removes manual annotation from the critical path. Each
frame is differenced against the empty-table reference shot under the same
lighting; the largest resulting region is the item; its class is the folder it
was shot into. What comes out is a crop per frame, ready to train a classifier
on, plus a box and mask if a detector is wanted later.

It also refuses frames it cannot label confidently -- more than one large
region, or a region touching the frame edge, or one far from the median size
of its burst. Those are usually a hand still in shot or the item rolling out of
view. Rejecting them is cheaper than training on them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..perception.photometric import compensated_reference
from .manifest import ManifestRow, read_manifest, split_by_object

MIN_AREA_FRACTION = 0.0008
MAX_AREA_FRACTION = 0.35
EDGE_MARGIN_PX = 4
CROP_PADDING_PX = 10


@dataclass
class LabelRecord:
    """One auto-labelled frame."""

    image: str
    crop: str
    category: str
    object_id: str
    lighting: str
    session: str
    split: str
    bbox_xywh: tuple[int, int, int, int]
    area_px: int
    rotated_box: list[list[int]]
    angle_degrees: float


@dataclass
class AutolabelReport:
    session: str
    frames: int
    labelled: int
    rejected: int
    reasons: dict[str, int]
    crops_root: str
    labels_path: str
    per_category: dict[str, int]
    per_category_objects: dict[str, int]
    # Frames where the camera had visibly re-exposed since the reference was
    # shot. Compensated for, but a high count means the reference is going
    # stale -- re-shoot it more often.
    exposure_drifted_frames: int = 0


def _foreground(frame, background, threshold: int = 28):
    """Difference against the reference, after cancelling exposure drift.

    Returns the mask and the photometric fit, so a caller can report frames
    where the camera had visibly rebalanced -- that is worth knowing during a
    shoot, while it can still be fixed.
    """

    import cv2

    live = cv2.GaussianBlur(frame, (5, 5), 0)
    reference, fit = compensated_reference(live, cv2.GaussianBlur(background, (5, 5), 0))
    difference = cv2.absdiff(live, reference).max(axis=2)
    mask = (difference > threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2), fit


def _largest_region(mask, frame_area: int):
    """Return (contour, area) for the single item, or (None, reason)."""

    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sizeable = [c for c in contours if cv2.contourArea(c) > frame_area * MIN_AREA_FRACTION]
    if not sizeable:
        return None, "nothing_changed"
    sizeable.sort(key=cv2.contourArea, reverse=True)
    largest = sizeable[0]
    area = cv2.contourArea(largest)
    if area > frame_area * MAX_AREA_FRACTION:
        return None, "region_too_large"
    if len(sizeable) > 1 and cv2.contourArea(sizeable[1]) > area * 0.35:
        return None, "more_than_one_object"
    return largest, area


def autolabel_session(
    session: str,
    root: Path = Path("data"),
    *,
    threshold: int = 28,
    holdout_fraction: float = 0.25,
) -> AutolabelReport:
    """Label every frame of one capture session."""

    import cv2

    root = Path(root).expanduser().resolve()
    raw_root = root / "raw" / session
    bg_root = root / "bg" / session
    crops_root = root / "crops" / session
    labels_path = root / "labels" / f"{session}.jsonl"

    rows: list[ManifestRow] = read_manifest(raw_root / "manifest.csv")
    if not rows:
        raise ValueError(f"Manifest for session {session!r} is empty")

    backgrounds: dict[str, np.ndarray] = {}
    for candidate in sorted(bg_root.glob("*.jpg")):
        image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
        if image is not None:
            backgrounds[candidate.stem] = image
    if not backgrounds:
        raise FileNotFoundError(
            f"No background references in {bg_root}. Shoot one per lighting "
            "condition with the table empty (press 'b' during capture)."
        )

    split = split_by_object(rows, holdout_fraction)
    crops_root.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[LabelRecord] = []
    reasons: dict[str, int] = {}
    objects_seen: dict[str, set] = {}
    drifted_frames = 0

    for row in rows:
        frame_path = root / row.image
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            reasons["unreadable_frame"] = reasons.get("unreadable_frame", 0) + 1
            continue
        background = backgrounds.get(row.lighting)
        if background is None:
            reasons["no_background_for_lighting"] = reasons.get("no_background_for_lighting", 0) + 1
            continue
        if background.shape != frame.shape:
            reasons["background_size_mismatch"] = reasons.get("background_size_mismatch", 0) + 1
            continue

        height, width = frame.shape[:2]
        mask, fit = _foreground(frame, background, threshold)
        if not fit.trusted:
            reasons["exposure_fit_failed"] = reasons.get("exposure_fit_failed", 0) + 1
            continue
        if fit.drifted:
            drifted_frames += 1
        contour, info = _largest_region(mask, height * width)
        if contour is None:
            reasons[info] = reasons.get(info, 0) + 1
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if (
            x <= EDGE_MARGIN_PX
            or y <= EDGE_MARGIN_PX
            or x + w >= width - EDGE_MARGIN_PX
            or y + h >= height - EDGE_MARGIN_PX
        ):
            reasons["touches_frame_edge"] = reasons.get("touches_frame_edge", 0) + 1
            continue

        (_, _), (_, _), angle = cv2.minAreaRect(contour)
        rotated = np.intp(cv2.boxPoints(cv2.minAreaRect(contour)))

        x0 = max(x - CROP_PADDING_PX, 0)
        y0 = max(y - CROP_PADDING_PX, 0)
        x1 = min(x + w + CROP_PADDING_PX, width)
        y1 = min(y + h + CROP_PADDING_PX, height)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            reasons["empty_crop"] = reasons.get("empty_crop", 0) + 1
            continue

        key = f"{row.category}/{row.object_id}"
        crop_dir = crops_root / row.category / row.object_id
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / Path(row.image).name
        cv2.imwrite(str(crop_path), crop)

        objects_seen.setdefault(row.category, set()).add(row.object_id)
        records.append(
            LabelRecord(
                image=row.image,
                crop=str(crop_path.relative_to(root)),
                category=row.category,
                object_id=row.object_id,
                lighting=row.lighting,
                session=row.session,
                split=split.get(key, "train"),
                bbox_xywh=(int(x), int(y), int(w), int(h)),
                area_px=int(info),
                rotated_box=[[int(a), int(b)] for a, b in rotated],
                angle_degrees=float(angle),
            )
        )

    with labels_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    per_category: dict[str, int] = {}
    for record in records:
        per_category[record.category] = per_category.get(record.category, 0) + 1

    return AutolabelReport(
        session=session,
        frames=len(rows),
        labelled=len(records),
        rejected=len(rows) - len(records),
        reasons=dict(sorted(reasons.items())),
        crops_root=str(crops_root),
        labels_path=str(labels_path),
        per_category=dict(sorted(per_category.items())),
        per_category_objects={k: len(v) for k, v in sorted(objects_seen.items())},
        exposure_drifted_frames=drifted_frames,
    )
