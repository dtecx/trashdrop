"""Look at what was auto-labelled, and throw away what is wrong.

Autolabelling is cheap but not perfect, and a handful of bad crops in a class
of two hundred is enough to confuse a small classifier. This builds contact
sheets so a person can scan a whole burst in one screen, and removes rejected
crops from both the folder and the label file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

THUMBNAIL = 128
COLUMNS = 10


def contact_sheets(session: str, root: Path = Path("data"), out_dir: Path | None = None) -> list[Path]:
    """Write one contact sheet per category. Returns the paths written."""

    import cv2

    root = Path(root).expanduser().resolve()
    labels_path = root / "labels" / f"{session}.jsonl"
    if not labels_path.is_file():
        raise FileNotFoundError(f"No labels for session {session!r}; run autolabel first")

    out_dir = Path(out_dir) if out_dir else root / "review" / session
    out_dir.mkdir(parents=True, exist_ok=True)

    by_category: dict[str, list[dict]] = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            by_category.setdefault(record["category"], []).append(record)

    written: list[Path] = []
    for category, records in sorted(by_category.items()):
        tiles = []
        for record in records:
            crop = cv2.imread(str(root / record["crop"]), cv2.IMREAD_COLOR)
            if crop is None:
                continue
            tile = cv2.resize(crop, (THUMBNAIL, THUMBNAIL), interpolation=cv2.INTER_AREA)
            # Label each tile with its object id so a bad burst is identifiable.
            cv2.putText(
                tile,
                record["object_id"][:12],
                (3, THUMBNAIL - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                record["object_id"][:12],
                (3, THUMBNAIL - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (0, 255, 120),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        if not tiles:
            continue

        rows = []
        for start in range(0, len(tiles), COLUMNS):
            chunk = tiles[start : start + COLUMNS]
            while len(chunk) < COLUMNS:
                chunk.append(np.zeros((THUMBNAIL, THUMBNAIL, 3), np.uint8))
            rows.append(np.hstack(chunk))
        sheet = np.vstack(rows)
        path = out_dir / f"{category}.jpg"
        cv2.imwrite(str(path), sheet)
        written.append(path)

    return written


def drop_objects(session: str, object_ids: list[str], root: Path = Path("data")) -> int:
    """Remove every crop and label belonging to the given object ids.

    Use after looking at a contact sheet and finding a burst where the item
    rolled away or a hand stayed in shot. Raw frames are left alone so the
    decision can be revisited.
    """

    root = Path(root).expanduser().resolve()
    labels_path = root / "labels" / f"{session}.jsonl"
    if not labels_path.is_file():
        raise FileNotFoundError(f"No labels for session {session!r}")

    drop = set(object_ids)
    kept: list[str] = []
    removed = 0
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["object_id"] in drop:
            crop = root / record["crop"]
            if crop.is_file():
                crop.unlink()
            removed += 1
        else:
            kept.append(line)

    labels_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed
