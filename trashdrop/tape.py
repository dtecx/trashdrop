"""A taped quadrilateral on the table: the pick zone and the calibration in one.

Nobody tapes a perfect square, so its shape is never assumed. Each arm
touches the corners it reaches with the tip of its fixed finger; the two near
corners, which both arms reach, tie the two arms' frames together; and the
camera is told where the same four corners are in its picture. From those:

* the corners in one shared table frame, in cm: origin at their centroid,
  x from the near left corner to the near right one, y away from the arms;
* each arm's placement in that frame, fitted to its own touches;
* the homography from camera pixels to that frame, through the four corners;
* the pick zone: the quadrilateral itself.

With only one arm touched so far, its three corners and the camera's four
clicks fill in the fourth corner -- the camera is close enough to overhead
for an affine map across the zone -- so one arm can work before the other is
calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .placement import Placement, fit_placement

CORNERS = ("far_left", "far_right", "near_right", "near_left")
LABELS = {"far_left": "far left", "far_right": "far right", "near_right": "near right", "near_left": "near left"}
SHARED = ("near_right", "near_left")


@dataclass
class TapeSolution:
    corners: dict[str, np.ndarray] = field(default_factory=dict)  # table frame, cm
    placements: dict[str, Placement] = field(default_factory=dict)
    residuals: dict[str, dict[str, float]] = field(default_factory=dict)  # arm -> corner -> cm
    homography: object | None = None  # HomographyCalibration, pixels -> table metres
    missing: list[str] = field(default_factory=list)  # what is still needed, for people


def solve(touches: dict[str, dict[str, tuple[float, float, float]]],
          pixels: dict[str, tuple[float, float]] | None) -> TapeSolution:
    """Everything the touches and the camera clicks determine so far."""

    solution = TapeSolution()
    arms = [arm for arm in ("left", "right") if len(touches.get(arm, {})) >= 3]
    if not arms:
        solution.missing.append("an arm touching at least three corners: uv run trashdrop rig touch left --tape")
        return solution

    # All corners in the first arm's frame; other arms joined through the shared near corners.
    reference = arms[0]
    known = {name: np.array(point[:2], float) for name, point in touches[reference].items()}
    for other in arms[1:]:
        theirs = touches[other]
        shared = [name for name in SHARED if name in theirs and name in known]
        if len(shared) < 2:
            solution.missing.append(f"both near corners touched by the {reference} and the {other} arm")
            continue
        to_reference, _ = fit_placement([theirs[name][:2] for name in shared], [known[name] for name in shared])
        for name, point in theirs.items():
            mapped = to_reference.to_arm(point[:2])
            known[name] = (known[name] + mapped) / 2 if name in known else mapped

    if len(known) < 4:
        if pixels and all(name in pixels for name in CORNERS) and len(known) == 3:
            # Three corners known in cm and all four in pixels: an affine map fills the fourth.
            names = list(known)
            source = np.array([pixels[name] for name in names], float)
            target = np.array([known[name] for name in names], float)
            affine = np.linalg.solve(np.c_[source, np.ones(3)], target)  # 3x2
            for name in CORNERS:
                if name not in known:
                    known[name] = np.r_[np.asarray(pixels[name], float), 1.0] @ affine
        else:
            missing = [LABELS[name] for name in CORNERS if name not in known]
            solution.missing.append(f"the {', '.join(missing)} corner touched by an arm, or the camera's clicks")
            return solution

    # The shared table frame.
    centre = np.mean([known[name] for name in CORNERS], axis=0)
    x_axis = known["near_right"] - known["near_left"]
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.array([-x_axis[1], x_axis[0]])  # a quarter turn left of "left to right": away from the arms
    solution.corners = {name: np.array([(known[name] - centre) @ x_axis, (known[name] - centre) @ y_axis])
                        for name in CORNERS}

    for arm in arms:
        touched = touches[arm]
        names = list(touched)
        table_z = float(np.mean([touched[name][2] for name in names]))
        placement, residuals = fit_placement([solution.corners[name] for name in names],
                                             [touched[name][:2] for name in names], table_z)
        solution.placements[arm] = placement
        solution.residuals[arm] = dict(zip(names, residuals))

    if pixels and all(name in pixels for name in CORNERS):
        from .perception.calibration import HomographyCalibration

        solution.homography = HomographyCalibration(
            [pixels[name] for name in CORNERS], [solution.corners[name] / 100.0 for name in CORNERS]
        )
    else:
        solution.missing.append("the corners clicked in the camera picture: uv run trashdrop camera tape")
    return solution
