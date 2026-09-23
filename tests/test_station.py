"""Cell geometry invariants that a layout edit must not silently break."""

from __future__ import annotations

import unittest

from trashdrop.station import (
    ALL_CATEGORIES,
    ARMS,
    BIN_HALF_WIDTH,
    BINS,
    CATEGORY_OWNER,
    MAX_GRASP_WIDTH,
    MIXED_CATEGORY,
    PICK_ZONE,
    SORT_CATEGORIES,
    bin_for,
    is_graspable,
    owner_of,
)


class BinTests(unittest.TestCase):
    def test_every_category_has_exactly_one_bin(self) -> None:
        self.assertEqual(set(BINS), set(ALL_CATEGORIES))

    def test_sort_bins_are_owned_by_one_arm_each(self) -> None:
        for category in SORT_CATEGORIES:
            self.assertEqual(BINS[category].arm, CATEGORY_OWNER[category])

    def test_mixed_bin_is_shared(self) -> None:
        self.assertEqual(BINS[MIXED_CATEGORY].arm, "any")

    def test_both_arms_own_target_materials(self) -> None:
        counts = {mount.name: 0 for mount in ARMS}
        for category in SORT_CATEGORIES:
            counts[owner_of(category)] += 1
        self.assertEqual(sorted(counts.values()), [1, 2])
        self.assertEqual(set(SORT_CATEGORIES), {"plastic", "paper", "metal"})

    def test_no_bin_overlaps_the_pick_zone(self) -> None:
        min_x, max_x, min_y, max_y = PICK_ZONE.bounds
        for key, spec in BINS.items():
            clear_in_x = spec.x + BIN_HALF_WIDTH < min_x or spec.x - BIN_HALF_WIDTH > max_x
            clear_in_y = spec.y + BIN_HALF_WIDTH < min_y or spec.y - BIN_HALF_WIDTH > max_y
            self.assertTrue(clear_in_x or clear_in_y, f"bin {key} overlaps the pick zone")

    def test_bins_do_not_overlap_each_other(self) -> None:
        keys = list(BINS)
        for i, first in enumerate(keys):
            for second in keys[i + 1 :]:
                a, b = BINS[first], BINS[second]
                apart = (
                    abs(a.x - b.x) >= 2 * BIN_HALF_WIDTH
                    or abs(a.y - b.y) >= 2 * BIN_HALF_WIDTH
                )
                self.assertTrue(apart, f"bins {first} and {second} overlap")

    def test_unknown_category_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            bin_for("batteries")


class GraspabilityTests(unittest.TestCase):
    def test_wide_items_are_refused(self) -> None:
        self.assertFalse(is_graspable(MAX_GRASP_WIDTH + 0.001))
        self.assertTrue(is_graspable(MAX_GRASP_WIDTH - 0.001))

    def test_heavy_items_are_refused(self) -> None:
        # An empty 0.5 L glass bottle is around 0.35 kg.
        self.assertFalse(is_graspable(0.02, mass_kg=0.35))
        # An empty aluminium can is around 15 g.
        self.assertTrue(is_graspable(0.02, mass_kg=0.015))


if __name__ == "__main__":
    unittest.main()
