from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trashdrop.dataset.taco import route_taco_label, write_taco_manifest


class TacoIndexTests(unittest.TestCase):
    def test_route_mapping_is_conservative(self) -> None:
        self.assertEqual(route_taco_label("clear_plastic_bottle"), "plastic")
        self.assertEqual(route_taco_label("drink_can"), "metal")
        self.assertEqual(route_taco_label("corrugated_carton"), "paper")
        self.assertEqual(route_taco_label("food_container"), "mixed")
        self.assertEqual(route_taco_label("unlabeled_litter"), "mixed")

    def test_coco_records_keep_box_and_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            images.mkdir()
            (images / "sample.jpg").write_bytes(b"placeholder")
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "categories": [
                            {"id": 1, "name": "clear_plastic_bottle"},
                            {"id": 2, "name": "unlabeled_litter"},
                        ],
                        "images": [{"id": 5, "file_name": "sample.jpg"}],
                        "annotations": [
                            {"id": 11, "image_id": 5, "category_id": 1, "bbox": [1, 2, 3, 4]},
                            {"id": 12, "image_id": 5, "category_id": 2, "bbox": [5, 6, 7, 8]},
                        ],
                    }
                )
            )
            manifest = root / "taco.jsonl"
            report = write_taco_manifest(annotations, images, manifest)
            records = [json.loads(line) for line in manifest.read_text().splitlines()]

        self.assertEqual(report.objects, 2)
        self.assertEqual(report.missing_images, 0)
        self.assertEqual(records[0]["bbox_xywh"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(records[0]["station_category"], "plastic")
        self.assertEqual(records[1]["station_category"], "mixed")


if __name__ == "__main__":
    unittest.main()
