from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trashdrop.dataset import TRASHNET_TO_STATION, write_manifest


class TrashNetManifestTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> None:
        for label in TRASHNET_TO_STATION:
            directory = root / label
            directory.mkdir()
            (directory / f"{label}.jpg").write_bytes(b"placeholder")

    def test_manifest_maps_only_approved_station_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset-resized"
            root.mkdir()
            self._make_dataset(root)
            manifest = Path(temporary_directory) / "manifest.jsonl"
            report = write_manifest(root, manifest)
            records = [json.loads(line) for line in manifest.read_text().splitlines()]

        self.assertEqual(report.images, 6)
        self.assertEqual(report.mixed_images, 2)
        routes = {record["source_label"]: record["station_category"] for record in records}
        self.assertEqual(routes["cardboard"], "paper")
        # Glass is routed to mixed on purpose: a glass bottle is heavier than
        # this arm should lift, so it must never reach a material bin.
        self.assertEqual(routes["glass"], "mixed")
        self.assertEqual(routes["trash"], "mixed")

    def test_missing_class_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset-resized"
            root.mkdir()
            (root / "paper").mkdir()
            with self.assertRaisesRegex(ValueError, "missing expected folders"):
                write_manifest(root, root / "manifest.jsonl")


if __name__ == "__main__":
    unittest.main()
