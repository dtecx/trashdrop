"""The auto-shutter, driven frame by frame with a fake clock.

Each test is one moment of a real shoot: an empty table, a hand in shot, an
item settling, the same pose held, the item moved, the camera re-exposing.
"""

from __future__ import annotations

import unittest

import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
import numpy as np  # noqa: E402

from trashdrop.dataset.shutter import AutoShutter  # noqa: E402

WIDTH, HEIGHT = 640, 360
FRAME_DT = 0.1


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def table(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((HEIGHT, WIDTH, 3), (150, 175, 160), dtype=np.float32)  # pale green board
    return np.clip(frame + rng.normal(0, 1.5, frame.shape), 0, 255).astype(np.uint8)


def with_item(seed: int, x: int = 280, y: int = 150, w: int = 70, h: int = 45, colour=(40, 40, 180)):
    frame = table(seed)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, -1)
    return frame


class ShutterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.shutter = AutoShutter(table(0), clock=self.clock)
        self.seed = 1

    def feed(self, make_frame, frames: int) -> list:
        """Feed ``frames`` frames; save whatever the shutter approves."""

        views = []
        for _ in range(frames):
            self.seed += 1
            view = self.shutter.update(make_frame(self.seed))
            if view.capture:
                self.shutter.mark_captured()
            views.append(view)
            self.clock.now += FRAME_DT
        return views


class DecisionTests(ShutterTestCase):
    def test_smooth_grey_item_uses_its_rim_to_complete_the_box(self) -> None:
        reference = np.full((HEIGHT, WIDTH, 3), (160, 170, 165), np.uint8)
        live = reference.copy()
        # The body looks like a shadow; only the bright label is a strong
        # foreground seed. The rim and printed bands reveal the whole item.
        cv2.rectangle(live, (245, 110), (355, 245), (124, 132, 128), -1)
        cv2.rectangle(live, (280, 150), (320, 190), (235, 238, 235), -1)
        for y in (130, 210, 235):
            cv2.line(live, (260, y), (340, y), (90, 100, 95), 2)
        shutter = AutoShutter(reference, clock=self.clock)
        shutter.update(live)
        view = shutter.update(live)
        self.assertIsNotNone(view.box, view.status)
        x, y, w, h = view.box
        self.assertLessEqual(x, 245)
        self.assertLessEqual(y, 110)
        self.assertGreaterEqual(x + w, 355)
        self.assertGreaterEqual(y + h, 245)

    def test_motion_outside_polygon_does_not_delay_inside_item(self) -> None:
        valid = np.zeros((HEIGHT, WIDTH), np.uint8)
        valid[70:290, 200:470] = 255
        views = []
        for seed in range(25):
            frame = with_item(seed)
            cv2.rectangle(frame, (10 + seed * 4, 90), (100 + seed * 4, 170),
                          (30, 180, 60), -1)
            view = self.shutter.update(frame, valid)
            if view.capture:
                self.shutter.mark_captured()
            views.append(view)
            self.clock.now += FRAME_DT
        self.assertEqual(sum(view.capture for view in views), 1)

    def test_an_empty_table_is_never_captured(self) -> None:
        views = self.feed(table, 20)
        self.assertFalse(any(v.capture for v in views))
        self.assertIn("empty table", views[-1].status)

    def test_a_settled_item_is_captured_exactly_once(self) -> None:
        views = self.feed(with_item, 25)
        self.assertEqual(sum(v.capture for v in views), 1)
        # Not before the settle time has passed.
        first = next(i for i, v in enumerate(views) if v.capture)
        self.assertGreaterEqual(first * FRAME_DT, self.shutter.settle_seconds - 1e-9)
        self.assertIn("new pose", views[-1].status)

    def test_moving_the_item_earns_another_frame(self) -> None:
        first = self.feed(with_item, 20)
        moved = self.feed(lambda s: with_item(s, x=120, y=220, w=45, h=70), 20)
        self.assertEqual(sum(v.capture for v in first), 1)
        self.assertEqual(sum(v.capture for v in moved), 1)

    def test_a_hand_at_the_edge_blocks_the_shutter(self) -> None:
        def hand(seed: int) -> np.ndarray:
            frame = with_item(seed)
            cv2.rectangle(frame, (0, 120), (200, 190), (120, 150, 200), -1)  # forearm from the side
            return frame

        views = self.feed(hand, 20)
        self.assertFalse(any(v.capture for v in views))

    def test_the_box_is_reported_in_full_resolution_pixels(self) -> None:
        views = self.feed(with_item, 25)
        box = next(v.box for v in views if v.capture)
        x, y, w, h = box
        self.assertAlmostEqual(x, 280, delta=8)
        self.assertAlmostEqual(y, 150, delta=8)
        self.assertAlmostEqual(w, 70, delta=10)
        self.assertAlmostEqual(h, 45, delta=10)

    def test_shadow_does_not_expand_the_live_box(self) -> None:
        def shadowed_item(seed: int) -> np.ndarray:
            image = table(seed)
            shadow = np.zeros((HEIGHT, WIDTH), np.uint8)
            cv2.fillPoly(shadow, [np.array([[80, 95], [285, 130],
                                             [285, 235], [100, 245]])], 255)
            image[shadow != 0] = (image[shadow != 0].astype(np.float32) * 0.72).astype(np.uint8)
            cv2.rectangle(image, (280, 150), (350, 215), (35, 40, 45), -1)
            return image

        views = self.feed(shadowed_item, 25)
        box = next(v.box for v in views if v.capture)
        self.assertAlmostEqual(box[0], 280, delta=12)
        self.assertAlmostEqual(box[2], 71, delta=15)


class ExposureTests(ShutterTestCase):
    def test_changed_lighting_still_draws_the_item_box(self) -> None:
        gradient = np.linspace(140, 200, HEIGHT, dtype=np.float32)[:, None, None]
        reference = np.broadcast_to(gradient, (HEIGHT, WIDTH, 3)).astype(np.uint8).copy()
        live = np.clip(reference.astype(np.float32) * 0.2 + 140, 0, 255).astype(np.uint8)
        cv2.rectangle(live, (280, 150), (350, 215), (25, 25, 25), -1)
        shutter = AutoShutter(reference, clock=self.clock)
        shutter.update(live)
        view = shutter.update(live)
        self.assertIsNotNone(view.box, view.status)
        self.assertAlmostEqual(view.box[0], 280, delta=12)

    def test_re_exposure_is_not_motion_and_not_a_new_pose(self) -> None:
        self.feed(with_item, 20)  # captured once, now holding

        def brighter(seed: int) -> np.ndarray:
            frame = with_item(seed).astype(np.float32) + 25.0
            return np.clip(frame, 0, 255).astype(np.uint8)

        views = self.feed(brighter, 20)
        self.assertFalse(any(v.capture for v in views), [v.status for v in views])
        self.assertFalse(any(v.status == "moving" for v in views[2:]))


class StateTests(ShutterTestCase):
    def test_no_background_asks_for_one(self) -> None:
        shutter = AutoShutter(None, clock=self.clock)
        view = None
        for seed in range(5):
            view = shutter.update(with_item(seed))
            self.clock.now += FRAME_DT
        self.assertFalse(view.capture)
        self.assertIn("press b", view.status)

    def test_a_new_object_is_captured_even_in_the_same_pose(self) -> None:
        self.feed(with_item, 20)
        self.shutter.forget_last()
        views = self.feed(with_item, 12)
        self.assertEqual(sum(v.capture for v in views), 1)

    def test_a_background_at_another_resolution_is_flagged(self) -> None:
        shutter = AutoShutter(cv2.resize(table(0), (320, 180)), clock=self.clock)
        view = None
        for seed in range(4):
            view = shutter.update(with_item(seed))
            self.clock.now += FRAME_DT
        self.assertIn("resolution", view.status)


if __name__ == "__main__":
    unittest.main()
