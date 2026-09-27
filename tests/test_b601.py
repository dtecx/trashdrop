"""The central-arm mode must stay an input preview until hardware is commissioned."""

import unittest
import threading
import time
from types import SimpleNamespace

from trashdrop.b601 import B601Preview
from trashdrop.web.manual import ManualBridge


class B601PreviewTests(unittest.TestCase):
    def test_pinch_is_a_clutch_and_loss_holds_the_preview(self) -> None:
        preview = B601Preview(scale=0.5)
        hand = {"tracked": True, "thumb": [0, 0, 0], "index": [2, 0, 0], "pinch": True}
        preview.update({"right": hand}, 0)
        hand["thumb"] = [4, 2, 0]
        hand["index"] = [6, 2, 0]
        guide = preview.update({"right": hand}, 0)
        self.assertEqual(guide["offset"], [2.0, 1.0, 0.0])
        self.assertEqual(preview.update({"right": hand}, 0.6)["offset"], [2.0, 1.0, 0.0])
        self.assertEqual(preview.state, "hand lost: held")
        hand["pinch"] = False
        preview.update({"right": hand}, 0)
        self.assertEqual(preview.offset, [2.0, 1.0, 0.0])

    def test_invalid_points_cannot_create_a_target(self) -> None:
        preview = B601Preview()
        guide = preview.update({"right": {"tracked": True, "thumb": [float("nan"), 0, 0],
                                           "index": [0, 0, 0], "pinch": True}}, 0)
        self.assertEqual(guide["offset"], [0.0, 0.0, 0.0])
        self.assertEqual(guide["mode"], "holding")

    def test_knuckle_frame_previews_wrist_orientation(self) -> None:
        preview = B601Preview()
        hand = {"tracked": True, "thumb": [0, 0, 0], "index": [1, 0, 0], "pinch": True,
                "wrist": [0, -1, 0], "indexKnuckle": [1, 0, 0],
                "middleKnuckle": [0, 0, 0], "pinkyKnuckle": [-1, 0, 0]}
        preview.update({"right": hand}, 0)
        hand.update(wrist=[1, 0, 0], indexKnuckle=[0, 1, 0],
                    middleKnuckle=[0, 0, 0], pinkyKnuckle=[0, -1, 0])
        self.assertEqual(preview.update({"right": hand}, 0)["rotation"], [0.0, 0.0, 90.0])


class B601GateTests(unittest.TestCase):
    def test_real_b601_is_refused_before_touching_any_bus(self) -> None:
        bridge = ManualBridge.__new__(ManualBridge)
        bridge.cell = SimpleNamespace(options=SimpleNamespace(dry_run=False))
        bridge.network_error = None
        bridge.hands = SimpleNamespace(connected=True)
        error = bridge.start(target="b601")
        self.assertIn("not commissioned", error)

    def test_lens_b601_request_does_not_interrupt_real_so101_mode(self) -> None:
        bridge = ManualBridge.__new__(ManualBridge)
        bridge.cell = SimpleNamespace(options=SimpleNamespace(dry_run=False))
        bridge.hands = SimpleNamespace(presentation=True)
        bridge.active, bridge.starting, bridge.target = True, False, "so101"
        bridge.stop = lambda: self.fail("SO-101 mode must not be interrupted")
        error = bridge._run_lens_command({"command": "b601", "enabled": True})
        self.assertIn("not commissioned", error)

    def test_preview_reserves_the_cell_but_never_touches_so101(self) -> None:
        class UntouchableArm:
            def __getattribute__(self, name):
                raise AssertionError(f"B601 preview touched SO-101: {name}")

        bridge = ManualBridge.__new__(ManualBridge)
        bridge.cell = SimpleNamespace(options=SimpleNamespace(dry_run=True), busy=None,
                                      stop_event=threading.Event(), _lock=threading.Lock(),
                                      version=0, arms={"left": UntouchableArm(), "right": UntouchableArm()},
                                      log=lambda message: None)
        bridge.hands = SimpleNamespace(connected=True, take_commands=lambda: [],
                                       playout=lambda: ({"right": {"tracked": True,
                                                                   "thumb": [0, 0, 0],
                                                                   "index": [1, 0, 0],
                                                                   "pinch": True}}, 0),
                                       report=None, status="")
        bridge.network_error = None
        bridge._lock = threading.Lock()
        bridge._shutdown = threading.Event()
        bridge.active = bridge.starting = False
        self.assertIsNone(bridge.start(target="b601"))
        self.assertEqual(bridge.cell.busy, "spectacles-b601")
        time.sleep(0.06)
        self.assertIn("preview only", bridge.hands.status)
        bridge.stop()
        self.assertIsNone(bridge.cell.busy)


if __name__ == "__main__":
    unittest.main()
