"""The capture manifest: one CSV row per frame, written as it is shot.

Kept as plain CSV on purpose -- it opens in any spreadsheet, and the rest of
the team is on Windows. The manifest is the record of *how* each frame was
shot, which is what makes a split honest later: holding out by ``object_id``
rather than by frame is the difference between a real generalisation number
and a model that memorised twenty photos of the same bottle.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ("image", "category", "object_id", "lighting", "session", "captured_at")


@dataclass
class ManifestRow:
    image: str
    category: str
    object_id: str
    lighting: str
    session: str
    captured_at: str


class ManifestWriter:
    """Append rows to a CSV, creating it with a header when new."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=FIELDS)
        if is_new:
            self._writer.writeheader()
            self._handle.flush()

    def append(self, **row) -> None:
        row.setdefault("captured_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._writer.writerow({field: row.get(field, "") for field in FIELDS})
        # Flush every row: a capture session that crashes should not lose the
        # record of frames that are already on disk.
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def read_manifest(path: Path) -> list[ManifestRow]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No manifest at {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [ManifestRow(**{f: row.get(f, "") for f in FIELDS}) for row in csv.DictReader(handle)]


def summarise(rows: list[ManifestRow]) -> dict:
    """Counts that answer 'have we shot enough, and enough variety?'."""

    by_category: dict[str, int] = {}
    objects: dict[str, set] = {}
    lighting: dict[str, int] = {}
    for row in rows:
        by_category[row.category] = by_category.get(row.category, 0) + 1
        objects.setdefault(row.category, set()).add(row.object_id)
        lighting[row.lighting] = lighting.get(row.lighting, 0) + 1
    return {
        "frames": len(rows),
        "frames_per_category": dict(sorted(by_category.items())),
        "distinct_objects_per_category": {k: len(v) for k, v in sorted(objects.items())},
        "frames_per_lighting": dict(sorted(lighting.items())),
    }


def split_by_object(rows: list[ManifestRow], holdout_fraction: float = 0.25) -> dict[str, str]:
    """Assign each object_id to train or test, never splitting one object.

    Frames from one burst are near-duplicates. Splitting them across train and
    test inflates accuracy badly -- the model sees the test bottle in training
    from a slightly different angle. Holding out whole objects is the only
    split that measures anything.
    """

    import hashlib

    assignment: dict[str, str] = {}
    for row in rows:
        key = f"{row.category}/{row.object_id}"
        if key in assignment:
            continue
        bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
        assignment[key] = "test" if bucket < holdout_fraction * 100 else "train"
    return assignment
