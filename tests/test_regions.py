"""Grouping foreground fragments into one item.

The case that motivated this: a clear PET bottle is mostly invisible to
background subtraction, so what shows up is its cap, its label and a few edge
highlights as separate blobs. That must come back as one item, while a genuine
second object must still be refused.
"""

from __future__ import annotations

import unittest

import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
import numpy as np  # noqa: E402

from trashdrop.perception.regions import find_item_region  # noqa: E402

WIDTH, HEIGHT = 640, 360


def blank() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), np.uint8)


def clear_bottle(mask: np.ndarray, x: int = 220, y: int = 160) -> np.ndarray:
    """What a transparent bottle lying on its side looks like after differencing.

    Cap, label band and the base highlight, with clear body -- and therefore
    no detection -- between them.
    """

    cv2.rectangle(mask, (x, y + 6), (x + 22, y + 34), 255, -1)  # cap
    cv2.rectangle(mask, (x + 40, y), (x + 95, y + 40), 255, -1)  # label band
    cv2.rectangle(mask, (x + 118, y + 4), (x + 150, y + 36), 255, -1)  # base highlight
    return mask


class FragmentTests(unittest.TestCase):
    def test_a_transparent_bottle_comes_back_as_one_item(self) -> None:
        region, reason = find_item_region(clear_bottle(blank()))
        self.assertEqual(reason, "ok")
        self.assertEqual(region.fragments, 3)
        x, y, w, h = region.box
        # The box must span cap to base, not just the largest fragment.
        self.assertLessEqual(x, 220)
        self.assertGreaterEqual(x + w, 370)

    def test_a_long_transparent_bottle_keeps_distant_cap_and_base(self) -> None:
        mask = blank()
        cv2.rectangle(mask, (100, 160), (120, 190), 255, -1)
        cv2.rectangle(mask, (165, 155), (200, 195), 255, -1)
        cv2.rectangle(mask, (245, 160), (270, 190), 255, -1)
        region, reason = find_item_region(mask)
        self.assertEqual(reason, "ok")
        self.assertEqual(region.fragments, 3)
        self.assertLessEqual(region.box[0], 100)
        self.assertGreaterEqual(region.box[0] + region.box[2], 271)

    def test_an_opaque_item_is_a_single_fragment(self) -> None:
        mask = blank()
        cv2.rectangle(mask, (250, 150), (330, 210), 255, -1)
        region, reason = find_item_region(mask)
        self.assertEqual(reason, "ok")
        self.assertEqual(region.fragments, 1)

    def test_a_distant_second_object_is_refused(self) -> None:
        mask = clear_bottle(blank(), x=60, y=60)
        cv2.rectangle(mask, (470, 250), (560, 320), 255, -1)
        region, reason = find_item_region(mask)
        self.assertIsNone(region)
        self.assertEqual(reason, "more_than_one_object")

    def test_a_small_distant_speck_is_ignored_not_refused(self) -> None:
        mask = blank()
        cv2.rectangle(mask, (250, 150), (330, 210), 255, -1)
        cv2.rectangle(mask, (560, 40), (570, 50), 255, -1)  # crumb, below the area floor
        region, reason = find_item_region(mask)
        self.assertEqual(reason, "ok")
        self.assertEqual(region.fragments, 1)


class RefusalTests(unittest.TestCase):
    def test_live_view_can_show_a_box_for_an_item_at_the_border(self) -> None:
        mask = blank()
        cv2.rectangle(mask, (0, 120), (100, 200), 255, -1)
        region, reason = find_item_region(mask, allow_edge=True)
        self.assertEqual(reason, "touches_frame_edge")
        self.assertIsNotNone(region)

    def test_polygon_border_rejects_a_truncated_item(self) -> None:
        mask = blank()
        cv2.rectangle(mask, (170, 120), (220, 180), 255, -1)
        valid = np.zeros_like(mask)
        valid[:, :222] = 255
        region, reason = find_item_region(mask, valid_mask=valid)
        self.assertIsNone(region)
        self.assertEqual(reason, "touches_frame_edge")

    def test_empty_mask(self) -> None:
        self.assertEqual(find_item_region(blank())[1], "nothing_changed")

    def test_a_hand_at_the_edge(self) -> None:
        mask = blank()
        cv2.rectangle(mask, (0, 120), (140, 190), 255, -1)
        self.assertEqual(find_item_region(mask)[1], "touches_frame_edge")

    def test_a_fragment_at_the_edge_refuses_the_whole_group(self) -> None:
        # A bottle whose base has rolled out of view is not a usable crop.
        mask = clear_bottle(blank(), x=WIDTH - 150, y=160)
        self.assertEqual(find_item_region(mask)[1], "touches_frame_edge")

    def test_a_region_too_large_to_be_an_item(self) -> None:
        mask = blank()
        cv2.rectangle(mask, (40, 30), (600, 330), 255, -1)
        self.assertEqual(find_item_region(mask)[1], "region_too_large")


if __name__ == "__main__":
    unittest.main()
