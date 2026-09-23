"""Turn one-item-at-a-time bursts into boxes and classifier crops.

This is the step that removes manual annotation from the critical path. Each
frame is differenced against its empty-table reference, with exposure drift
compensated. Shadow removal and edge recovery form a region from the item's
fragments; its class is the folder it was shot into. What comes out is a crop
per frame, ready to train a classifier, plus a box for inspection or training.

It also refuses frames it cannot label confidently -- a second object, a
region touching the frame edge, or a region too large to be an item. Those are
usually a hand still in shot or the item rolling out of view. Rejecting them is
cheaper than training on them.

Fragments of one item are grouped before any of that is judged; see
``perception/regions.py`` for why transparent bottles need it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..perception.photometric import compensated_reference
from ..perception.regions import ANALYSIS_WIDTH, find_item_region
from ..perception.shadows import recover_object_edges, suppress_cast_shadows
from ..station import SORT_CATEGORIES
from .manifest import ManifestRow, read_manifest, split_by_object
from .review import read_rejected_frames
from .zone import load_zone

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


def _foreground(frame, background, threshold: int = 28, valid_mask=None):
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
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if valid_mask is not None:
        mask[valid_mask == 0] = 0
    shadow_free = suppress_cast_shadows(live, reference, mask)
    shadow_free = cv2.morphologyEx(shadow_free, cv2.MORPH_OPEN, kernel)
    return recover_object_edges(live, reference, mask, shadow_free, valid_mask), fit


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
    zone = load_zone(raw_root / "zone.json")

    rows: list[ManifestRow] = read_manifest(raw_root / "manifest.csv")
    if not rows:
        raise ValueError(f"Manifest for session {session!r} is empty")
    manually_rejected = read_rejected_frames(root, session)

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
        if row.image in manually_rejected:
            reasons["manual_reject"] = reasons.get("manual_reject", 0) + 1
            (root / row.image.replace("raw/", "crops/", 1)).unlink(missing_ok=True)
            continue
        if row.category not in SORT_CATEGORIES:
            reasons["unsupported_category"] = reasons.get("unsupported_category", 0) + 1
            (root / row.image.replace("raw/", "crops/", 1)).unlink(missing_ok=True)
            continue
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

        if zone is not None:
            if not zone.matches(frame):
                reasons["zone_size_mismatch"] = reasons.get("zone_size_mismatch", 0) + 1
                continue
            analysis_frame = zone.crop(frame)
            analysis_background = zone.crop(background)
            offset_x, offset_y = zone.x, zone.y
            valid_mask = zone.analysis_mask()
        else:
            analysis_frame, analysis_background = frame, background
            offset_x, offset_y = 0, 0
            valid_mask = None

        height, width = analysis_frame.shape[:2]
        # Use the same pixel scale as the live preview. At full resolution a
        # transparent bottle produces many tiny disconnected highlights; at
        # this scale its cap, label and outline form one physical item.
        small_width = min(width, ANALYSIS_WIDTH)
        small_height = max(1, round(height * small_width / width))
        size = (small_width, small_height)
        small_frame = cv2.resize(analysis_frame, size, interpolation=cv2.INTER_AREA)
        small_background = cv2.resize(analysis_background, size, interpolation=cv2.INTER_AREA)
        small_valid = (cv2.resize(valid_mask, size, interpolation=cv2.INTER_NEAREST)
                       if valid_mask is not None else None)
        mask, fit = _foreground(small_frame, small_background, threshold, small_valid)
        if not fit.trusted:
            reasons["exposure_fit_failed"] = reasons.get("exposure_fit_failed", 0) + 1
            continue
        if fit.drifted:
            drifted_frames += 1
        region, reason = find_item_region(mask, edge_margin_px=EDGE_MARGIN_PX,
                                           valid_mask=small_valid)
        if region is None:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue

        scale_x, scale_y = width / small_width, height / small_height
        sx, sy, sw, sh = region.box
        x, y = int(np.floor(sx * scale_x)), int(np.floor(sy * scale_y))
        x1 = int(np.ceil((sx + sw) * scale_x))
        y1 = int(np.ceil((sy + sh) * scale_y))
        w, h = x1 - x, y1 - y
        hull = np.rint(region.hull.astype(np.float32) * (scale_x, scale_y)).astype(np.int32)
        (_, _), (_, _), angle = cv2.minAreaRect(hull)
        rotated = np.intp(cv2.boxPoints(cv2.minAreaRect(hull)))
        rotated += (offset_x, offset_y)

        x0 = max(x - CROP_PADDING_PX, 0)
        y0 = max(y - CROP_PADDING_PX, 0)
        x1 = min(x + w + CROP_PADDING_PX, width)
        y1 = min(y + h + CROP_PADDING_PX, height)
        crop = analysis_frame[y0:y1, x0:x1].copy()
        if valid_mask is not None:
            # Photometric fitting saw the unaltered pixels above. Only the
            # exported crop hides unrelated objects beyond the physical zone.
            crop_mask = valid_mask[y0:y1, x0:x1]
            reference_crop = analysis_background[y0:y1, x0:x1]
            crop[crop_mask == 0] = reference_crop[crop_mask == 0]
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
                bbox_xywh=(int(x + offset_x), int(y + offset_y), int(w), int(h)),
                area_px=int(round(region.area * scale_x * scale_y)),
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
