"""Draw the jury page's camera annotations into a Spectacles video frame."""

from __future__ import annotations


def draw(frame, state: dict):
    """Return a separate BGR frame with the cell's current sorting annotations."""

    import cv2
    import numpy as np

    scene = state.get("scene") or {}
    look = state.get("last") or {}
    source_width, source_height = scene.get("size", (frame.shape[1], frame.shape[0]))
    sx = frame.shape[1] / max(source_width, 1)
    sy = frame.shape[0] / max(source_height, 1)
    scale = min(sx, sy)
    canvas = frame.copy()

    def point(value):
        return round(float(value[0]) * sx), round(float(value[1]) * sy)

    def polygon(points, color, thickness=2, closed=True):
        if points and len(points) >= 2:
            cv2.polylines(canvas, [np.array([point(p) for p in points], dtype=np.int32)],
                          closed, color, max(1, round(thickness * scale * 2)))

    def label(words, position, color=(255, 255, 255)):
        x, y = point(position)
        font_scale = max(0.45, 1.15 * scale * 2)
        weight = max(1, round(2 * scale * 2))
        cv2.putText(canvas, words, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), weight + 3, cv2.LINE_AA)
        cv2.putText(canvas, words, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, weight, cv2.LINE_AA)

    for outline in scene.get("searched") or []:
        polygon(outline, (99, 140, 42), 2)
    polygon(scene.get("zone") or [], (151, 220, 61), 3)
    for name, position in (scene.get("bases") or {}).items():
        cv2.circle(canvas, point(position), max(4, round(12 * scale * 2)), (223, 211, 200), 2)
        label(name.upper(), (position[0], position[1] - 25))
    sides = scene.get("sides") or {}
    for name, position in (scene.get("drops") or {}).items():
        materials = [material for material, side in sides.items() if side == name]
        color = (246, 130, 59) if "paper" in materials else (24, 197, 245)
        cv2.circle(canvas, point(position), max(5, round(18 * scale * 2)), color, 2)
        label(" / ".join(materials).upper() or name.upper(),
              (position[0], position[1] + 38), color)

    outline = look.get("outline") or []
    if len(outline) >= 2:
        destination = (scene.get("drops") or {}).get(look.get("arm"))
        if destination:
            moving = look.get("moving")
            origin = moving or (sum(p[0] for p in outline) / len(outline),
                                sum(p[1] for p in outline) / len(outline))
            # The path is a plan, not a live robot trajectory; draw it behind
            # the grasp and contour so the pick remains legible.
            cv2.arrowedLine(canvas, point(origin), point(destination),
                            (255, 175, 56), max(2, round(2 * scale * 2)),
                            cv2.LINE_AA, tipLength=0.07)
        polygon(outline, (240, 208, 63), 3)
        fixed, moving = look.get("fixed"), look.get("moving")
        if fixed and moving:
            cv2.line(canvas, point(fixed), point(moving), (61, 197, 255), 3)
            cv2.circle(canvas, point(fixed), max(4, round(10 * scale * 2)), (95, 90, 255), -1)
            cv2.circle(canvas, point(moving), max(4, round(10 * scale * 2)), (255, 141, 77), 2)
        probabilities = look.get("probabilities") or {}
        if probabilities:
            material, confidence = max(probabilities.items(), key=lambda item: item[1])
            top = min(outline, key=lambda p: p[1])
            arm = look.get("arm")
            words = f"{material.upper()} {confidence:.0%}" + (f"  {arm.upper()} ARM" if arm else "")
            label(words, (top[0], max(35, top[1] - 24)), (240, 208, 63))
    return canvas
