"""Discard cast shadows from a one-frame foreground mask.

A shadow darkens the table while keeping nearly the same colour. Compare the
live frame to the exposure-corrected empty table, so camera re-exposure is not
confused with a shadow. Very dark pixels stay in the mask: on this rig they can
be the body of a black can, indistinguishable from a contact shadow by colour
alone. The remaining small contact shadow is safer than erasing the item.
"""

from __future__ import annotations

import numpy as np

# Measured on the home rig's six black-can poses and two paper objects. A cast
# shadow is usually 60-98% of the corrected table brightness with a near-zero
# Lab chroma change; darker pixels can belong to the object itself.
SHADOW_BRIGHTNESS_RANGE = (0.60, 0.98)
MAX_CHROMA_CHANGE = 12
# A neutral grey object can have the same brightness ratio and chroma as a
# shadow. If almost everything would disappear, keep the original foreground
# and let the operator improve the light or reject the crop during review.
MIN_RETAINED_FRACTION = 0.05
# A straight cast-shadow border contributes about one Canny pixel per row of
# this window (1/17 = 5.9%). Folds, can rims and bottle ribs make denser edges.
EDGE_DENSITY_WINDOW = 17
MIN_REMOTE_SHADOW_EDGE_DENSITY = 0.08
MIN_ENCLOSING_EDGE_DENSITY = 0.07
WEAK_EDGE_CHANGE = 12


def _shadow_like_pixels(live: np.ndarray, reference: np.ndarray) -> np.ndarray:
    import cv2

    live_grey = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY).astype(np.float32)
    reference_grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    brightness_ratio = (live_grey + 1.0) / (reference_grey + 1.0)

    live_lab = cv2.cvtColor(live, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    chroma_change = cv2.absdiff(live_lab[..., 1:], reference_lab[..., 1:]).max(axis=2)
    return (
        (brightness_ratio >= SHADOW_BRIGHTNESS_RANGE[0])
        & (brightness_ratio < SHADOW_BRIGHTNESS_RANGE[1])
        & (chroma_change < MAX_CHROMA_CHANGE)
    )


def suppress_cast_shadows(live: np.ndarray, reference: np.ndarray,
                          foreground: np.ndarray) -> np.ndarray:
    """Return a mask without pixels that look like illumination-only changes."""

    import cv2

    if live.shape != reference.shape or foreground.shape != live.shape[:2]:
        raise ValueError("Shadow suppression needs aligned images and mask")

    shadow = _shadow_like_pixels(live, reference)
    result = foreground.copy()
    result[shadow] = 0
    original_pixels = np.count_nonzero(foreground)
    if original_pixels and np.count_nonzero(result) < original_pixels * MIN_RETAINED_FRACTION:
        return foreground.copy()
    return result


def recover_object_edges(live: np.ndarray, reference: np.ndarray,
                         changed: np.ndarray, shadow_free: np.ndarray,
                         valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Restore item outlines lost when illumination-like pixels were removed.

    A silver can side, white paper fold, or clear bottle wall can have the same
    brightness ratio as a cast shadow. The original change mask provides a
    search area; only edges new relative to the empty table are restored.
    Smooth cast shadows have little edge evidence away from their boundary.
    """

    import cv2

    if live.shape != reference.shape or changed.shape != live.shape[:2] or shadow_free.shape != changed.shape:
        raise ValueError("Edge recovery needs aligned images and masks")
    if valid_mask is not None and valid_mask.shape != changed.shape:
        raise ValueError("Valid mask does not match the image")

    grey = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
    reference_grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, 25, 60)
    reference_edges = cv2.Canny(reference_grey, 25, 60)
    nearby = np.ones((5, 5), np.uint8)
    edges[cv2.dilate(reference_edges, nearby) != 0] = 0
    # Strong colour changes seed the item. Its pale outline may change by less
    # than that threshold, so allow a weaker difference only where a genuinely
    # new edge also exists.
    weak_change = cv2.absdiff(live, reference).max(axis=2) > WEAK_EDGE_CHANGE
    weak_change |= changed != 0
    edges[cv2.dilate(weak_change.astype(np.uint8), nearby) == 0] = 0
    if valid_mask is not None:
        edges[valid_mask == 0] = 0

    # A smooth grey can may have no texture away from its printed label. Its
    # outer contour still encloses the reliable core, unlike a cast shadow
    # projected to one side. Preserve such a contour before filtering long,
    # smooth shadow borders below.
    enclosing = np.zeros(changed.shape, np.uint8)
    core_contours, _ = cv2.findContours(shadow_free, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if core_contours:
        core = max(core_contours, key=cv2.contourArea)
        moments = cv2.moments(core)
        if moments["m00"]:
            centre = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            frame_area = float(changed.size)
            for contour in contours:
                area = cv2.contourArea(contour)
                if (frame_area * 0.002 < area < frame_area * 0.35
                        and cv2.pointPolygonTest(contour, centre, False) >= 0):
                    inside = np.zeros(changed.shape, np.uint8)
                    cv2.drawContours(inside, [contour], -1, 255, -1)
                    pixels = inside != 0
                    # A projected shadow can partly surround the object too,
                    # but its interior has just a smooth border. A real can
                    # has a rim, label or folds inside the enclosing outline.
                    if np.mean(edges[pixels] != 0) >= MIN_ENCLOSING_EDGE_DENSITY:
                        cv2.drawContours(enclosing, [contour], -1, 255, 2)

    # An isolated, smooth line around a dark patch is a cast-shadow border.
    # Keep nearby edges (item silhouettes) and remote textured edges (paper
    # folds, bottle ribs), but do not reconnect a long shadow to the item.
    near_size = max(5, int(round(live.shape[1] * 0.03)) | 1)
    near_item = cv2.dilate(shadow_free, np.ones((near_size, near_size), np.uint8)) != 0
    density = cv2.boxFilter((edges != 0).astype(np.float32), -1,
                            (EDGE_DENSITY_WINDOW, EDGE_DENSITY_WINDOW))
    shadow_border = cv2.dilate(_shadow_like_pixels(live, reference).astype(np.uint8), nearby) != 0
    edges[shadow_border & ~near_item & (density < MIN_REMOTE_SHADOW_EDGE_DENSITY)] = 0

    recovered = cv2.bitwise_or(shadow_free, cv2.dilate(edges, np.ones((3, 3), np.uint8)))
    recovered = cv2.bitwise_or(recovered, enclosing)
    if valid_mask is not None:
        recovered[valid_mask == 0] = 0
    return recovered
