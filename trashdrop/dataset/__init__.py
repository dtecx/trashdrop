"""Dataset tooling: capture on the rig, label without drawing boxes, review.

Two sources of data, kept apart on purpose:

* ``capture`` + ``autolabel`` -- our own rig. This is the data that decides
  whether the cell works on the day, because it is the only data shot through
  our camera, at our angle, under our lights, on our table.
* ``trashnet`` + ``taco`` -- public datasets, indexed in place and never
  copied or modified. Useful to pretrain and to sanity-check a class, but
  every one of them was shot in a different domain, so none of them replaces
  the rig data.

Both paths end in the same shape: an image, a category, and a split that was
assigned by object rather than by frame.
"""

from __future__ import annotations

from .autolabel import AutolabelReport, LabelRecord, autolabel_session
from .camcheck import CameraReport, analyse_idle, analyse_response, judge, run_camcheck
from .capture import CaptureConfig, open_camera, run_capture
from .manifest import ManifestRow, ManifestWriter, read_manifest, split_by_object, summarise
from .taco import TacoIndexReport, index_taco_coco, route_taco_label, write_taco_manifest
from .trashnet import (
    TRASHNET_TO_STATION,
    IndexReport,
    index_trashnet_style_dataset,
    write_manifest,
)

__all__ = [
    "AutolabelReport",
    "CameraReport",
    "CaptureConfig",
    "analyse_idle",
    "analyse_response",
    "IndexReport",
    "LabelRecord",
    "ManifestRow",
    "ManifestWriter",
    "TRASHNET_TO_STATION",
    "TacoIndexReport",
    "autolabel_session",
    "index_taco_coco",
    "index_trashnet_style_dataset",
    "open_camera",
    "judge",
    "read_manifest",
    "run_camcheck",
    "route_taco_label",
    "run_capture",
    "split_by_object",
    "summarise",
    "write_manifest",
    "write_taco_manifest",
]
