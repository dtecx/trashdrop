"""The page's server: routes, JSON, the stream, and refusing what it should refuse.

A stand-in cell answers, so this needs neither camera nor arms nor MuJoCo:
what is pinned down is the HTTP surface the page relies on.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2", reason="needs the dataset extra")

from trashdrop.web.server import make_server  # noqa: E402
from trashdrop.web.manual import ManualBridge  # noqa: E402


class StandInCell:
    def __init__(self) -> None:
        self.camera = self
        self.started, self.stopped, self.options = [], 0, {}
        self.frame = np.full((1080, 1920, 3), 128, np.uint8)

    def latest(self):
        return self.frame, 1.0

    def state(self):
        return {"version": np.int64(3), "scene": {"size": [1920, 1080], "zone": [[1.5, 2.5]]}, "log": []}

    def begin(self, action):
        if action == "pick":
            return "photograph the empty zone first"
        self.started.append(action)
        return None

    def stop(self):
        self.stopped += 1
        return "STOP: nothing was moving"

    def set_options(self, changes):
        if "material" in changes and changes["material"] == "glass":
            raise ValueError("material must be one of plastic, metal, paper")
        self.options.update(changes)

    def set_speeds(self, arm, max_speed, descent_speed):
        self.options["speeds"] = (arm, max_speed, descent_speed)
        return f"{arm} arm: {max_speed:g} deg/s"


class StandInSpectacles:
    def __init__(self, folder: Path, jpeg: bytes) -> None:
        self.folder = folder
        self.active = False
        self.presentation = False
        self.manual_options = None
        self.commands = []
        self.spectator = type("Spectator", (), {"latest": lambda _self: (jpeg, 4)})()

    def state(self):
        return {"available": True, "active": self.active, "connected": True, "presentation": self.presentation,
                "view_sequence": 4, "snapshots": [], "hands": {}, "arms": {}}

    def start(self, **options):
        self.active = True
        return None

    def stop(self):
        self.active = False
        return "Spectacles manual mode stopped"

    def command(self, message):
        self.commands.append(message)
        return None

    def request_snapshot(self):
        return None

    def set_presentation(self, enabled):
        self.presentation = enabled
        return None

    def configure(self, **options):
        self.manual_options = options
        return None


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = StandInCell()
        self.server = make_server(self.cell, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def post(self, path, body=None):
        request = urllib.request.Request(self.base + path, data=json.dumps(body or {}).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_the_page_and_its_state(self) -> None:
        with urllib.request.urlopen(self.base + "/", timeout=5) as response:
            self.assertIn(b"TrashDrop", response.read())
        with urllib.request.urlopen(self.base + "/api/state", timeout=5) as response:
            state = json.loads(response.read())
        self.assertEqual(state["version"], 3, "numpy numbers come out as plain JSON")

    def test_actions_stop_and_settings(self) -> None:
        self.assertEqual(self.post("/api/action/look"), (200, {"ok": True, "error": None}))
        status, reply = self.post("/api/action/pick")
        self.assertEqual((status, reply["ok"]), (409, False))
        self.assertEqual(self.post("/api/stop")[1], {"ok": True, "message": "STOP: nothing was moving"})
        self.assertEqual(self.cell.stopped, 1)
        self.assertEqual(self.post("/api/options", {"dry_run": True})[0], 200)
        self.assertTrue(self.cell.options["dry_run"])
        self.assertEqual(self.post("/api/options", {"material": "glass"})[0], 400)
        self.assertEqual(self.post("/api/speeds", {"arm": "left", "max_speed": 60, "descent_speed": 25})[0], 200)
        self.assertEqual(self.cell.options["speeds"], ("left", 60.0, 25.0))

    def test_a_single_frame_is_a_jpeg(self) -> None:
        with urllib.request.urlopen(self.base + "/frame.jpg", timeout=5) as response:
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            self.assertTrue(response.read(3).startswith(b"\xff\xd8"))

    def test_the_stream_is_mjpeg(self) -> None:
        with urllib.request.urlopen(self.base + "/stream.mjpg", timeout=5) as response:
            self.assertIn("multipart/x-mixed-replace", response.headers["Content-Type"])
            head = response.read(200)
        self.assertTrue(head.startswith(b"--frame"))
        self.assertIn(b"image/jpeg", head)


class SpectaclesServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.cell = StandInCell()
        ok, jpeg = __import__("cv2").imencode(".jpg", self.cell.frame)
        assert ok
        self.spectacles = StandInSpectacles(Path(self.temporary.name), jpeg.tobytes())
        self.server = make_server(self.cell, "127.0.0.1", 0, self.spectacles)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.temporary.cleanup()

    def post(self, path, body=None):
        request = urllib.request.Request(self.base + path, data=json.dumps(body or {}).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())

    def test_presentation_opens_the_glasses_ui_without_moving_the_arms(self) -> None:
        status, reply = self.post("/api/spectacles/presentation", {"enabled": True, "mode": "pinch", "scale": 0.5})
        self.assertEqual((status, reply["ok"]), (200, True))
        self.assertFalse(self.spectacles.active)
        self.assertTrue(self.spectacles.presentation)
        self.assertEqual(self.spectacles.manual_options, {"mode": "pinch", "scale": 0.5, "facing": "same"})
        self.post("/api/spectacles/presentation", {"enabled": False})
        self.assertFalse(self.spectacles.active)
        self.assertFalse(self.spectacles.presentation)

    def test_web_controls_send_commands_to_the_lens_session(self) -> None:
        self.spectacles.active = True
        self.post("/api/spectacles/command", {"command": "home"})
        self.assertEqual(self.spectacles.commands, [{"command": "home"}])


class SpectaclesControlsTests(unittest.TestCase):
    def test_glasses_switch_between_hand_control_auto_neutral_and_web(self) -> None:
        actions = []

        class Cell:
            busy = None
            auto = False

            def begin(self, action):
                actions.append(action)
                self.busy = action
                self.auto = action == "auto"
                return None

            def stop(self):
                actions.append("stop")
                self.busy = None
                self.auto = False

        bridge = ManualBridge.__new__(ManualBridge)
        bridge.cell = Cell()
        bridge.hands = SimpleNamespace(presentation=True)
        bridge.active = bridge.starting = False
        bridge.mode, bridge.scale, bridge.facing = "pinch", 0.5, "same"
        bridge._wait_for_cell = lambda timeout=5.0: None
        bridge.set_presentation = lambda enabled: setattr(bridge.hands, "presentation", enabled)

        def start(**options):
            actions.append(("manual", options))
            bridge.active, bridge.cell.busy = True, "spectacles"

        def stop():
            actions.append("manual stop")
            bridge.active, bridge.cell.busy = False, None

        bridge.start, bridge.stop = start, stop
        self.assertIsNone(bridge._run_lens_command({"command": "manual", "enabled": True}))
        self.assertIsNone(bridge._run_lens_command({"command": "auto", "enabled": True}))
        self.assertIsNone(bridge._run_lens_command({"command": "neutral"}))
        self.assertIsNone(bridge._run_lens_command({"command": "presentation", "enabled": False}))
        self.assertEqual(actions, [("manual", {"mode": "pinch", "scale": 0.5, "facing": "same"}),
                                   "manual stop", "auto", "stop", "neutral"])
        self.assertFalse(bridge.hands.presentation)

    def test_spectacles_video_draws_the_same_sorting_information_as_the_page(self) -> None:
        source = np.zeros((120, 240, 3), dtype=np.uint8)
        state = {"scene": {"size": [240, 120], "zone": [[40, 25], [200, 25], [200, 100], [40, 100]],
                           "searched": [], "bases": {}, "drops": {}, "sides": {}},
                 "last": {"outline": [[80, 45], [125, 45], [125, 80], [80, 80]],
                          "fixed": [90, 60], "moving": [115, 60],
                          "probabilities": {"plastic": 0.92}, "arm": "left"}}
        bridge = ManualBridge.__new__(ManualBridge)
        bridge.active = bridge.starting = False
        bridge.cell = SimpleNamespace(state=lambda: state)
        annotated = bridge._decorate_video(source)
        self.assertTrue(np.any(annotated != source))
        self.assertEqual(int(source.sum()), 0, "the camera frame shared with the web page is untouched")
        bridge.active = True
        self.assertIs(bridge._decorate_video(source), source, "manual control needs an unobscured view")


if __name__ == "__main__":
    unittest.main()
