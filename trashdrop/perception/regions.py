"""Turn a foreground mask into the region of ONE item, fragments included.

Shooting one item at a time means everything that changed against the empty
table should belong to that item. The trouble is transparent packaging: a clear
PET bottle shows the table straight through itself, so background subtraction
does not see a bottle -- it sees the cap, the label, and a few highlights along
the edges, as separate blobs with table in between. Treating the largest blob
as "the item" and refusing the frame because other blobs exist throws away
exactly the items the pitch promises to handle.

So fragments are clustered by proximity: anything within ``gap_px`` of a
cluster joins it, repeatedly, so a chain of fragments along a bottle merges into
one item. Clusters are then compared by their *summed* area. Comparing single
fragments instead is a trap -- a second, opaque object can easily be larger than
every individual piece of a transparent bottle, and would then be taken for
"the item" while the bottle was waved off as noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fraction of the frame a region must cover to count at all.
MIN_AREA_FRACTION = 0.0008
# Fraction of the frame above which a "region" is really a lighting change or a
# stale background, not an item.
MAX_AREA_FRACTION = 0.35
# A leftover region larger than this fraction of the grouped item is a second
# object rather than noise.
SECOND_OBJECT_RATIO = 0.35
# Default grouping distance as a fraction of the frame width. At 60 cm a C920
# frame is ~85 cm wide, so 5 % is roughly 4 cm: enough to bridge the clear body
# between a bottle's cap and its label, not enough to swallow a second item
# lying elsewhere in the zone.
DEFAULT_GAP_FRACTION = 0.05


@dataclass
class ItemRegion:
    """The grouped region of a single item."""

    hull: np.ndarray  # convex hull of every fragment, OpenCV contour format
    box: tuple[int, int, int, int]  # x, y, w, h
    area: float  # summed area of the fragments, not of the hull
    fragments: int


def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Distance between two axis-aligned boxes; zero when they overlap."""

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0, bx - (ax + aw), ax - (bx + bw))
    dy = max(0, by - (ay + ah), ay - (by + bh))
    return float(np.hypot(dx, dy))


def _union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return x0, y0, x1 - x0, y1 - y0


def find_item_region(
    mask: np.ndarray,
    *,
    gap_px: float | None = None,
    edge_margin_px: int = 4,
) -> tuple[ItemRegion | None, str]:
    """Return the single item in ``mask``, or ``None`` and the reason why not.

    Reasons match what autolabel reports, so a refused frame is explained the
    same way whether it was refused live or afterwards.
    """

    import cv2

    height, width = mask.shape[:2]
    frame_area = float(height * width)
    gap = gap_px if gap_px is not None else DEFAULT_GAP_FRACTION * width

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sizeable = [c for c in contours if cv2.contourArea(c) > frame_area * MIN_AREA_FRACTION]
    if not sizeable:
        return None, "nothing_changed"

    sizeable.sort(key=cv2.contourArea, reverse=True)
    boxes = [cv2.boundingRect(c) for c in sizeable]
    areas = [float(cv2.contourArea(c)) for c in sizeable]

    # Single-linkage clustering on box gaps. Seeds are taken largest-first, so
    # a cluster grows outward from its most substantial fragment.
    clusters: list[tuple[list[int], tuple[int, int, int, int], float]] = []
    unassigned = list(range(len(sizeable)))
    while unassigned:
        seed = unassigned.pop(0)
        members, box = [seed], boxes[seed]
        grew = True
        while grew:
            grew = False
            for index in list(unassigned):
                if _rect_gap(boxes[index], box) <= gap:
                    members.append(index)
                    box = _union(box, boxes[index])
                    unassigned.remove(index)
                    grew = True
        clusters.append((members, box, sum(areas[i] for i in members)))

    clusters.sort(key=lambda cluster: cluster[2], reverse=True)
    group, group_box, group_area = clusters[0]
    for _, _, other_area in clusters[1:]:
        if other_area > group_area * SECOND_OBJECT_RATIO:
            return None, "more_than_one_object"

    if group_area > frame_area * MAX_AREA_FRACTION:
        return None, "region_too_large"

    x, y, w, h = group_box
    if (
        x <= edge_margin_px
        or y <= edge_margin_px
        or x + w >= width - edge_margin_px
        or y + h >= height - edge_margin_px
    ):
        # Usually a hand still in shot, or the item rolled out of view.
        return None, "touches_frame_edge"

    points = np.vstack([sizeable[i] for i in group])
    hull = cv2.convexHull(points)
    return ItemRegion(hull=hull, box=group_box, area=group_area, fragments=len(group)), "ok"
