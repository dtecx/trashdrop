"""Serve the cell's page: the overhead stream, a JSON view of the cell, and its controls.

Standard library only, like api.py: nothing to install and nothing fetched
from the internet, so the page works on the venue's wifi or with none. The
page (page.html) polls /api/state and draws the overlays itself from the
data there; the stream is MJPEG, which every browser shows in an <img>.

    GET  /               the page
    GET  /frame.jpg      the overhead camera's newest frame; the page asks ~10 times a second
    GET  /stream.mjpg    the same as one MJPEG stream, for anything else that wants it
    GET  /api/state      everything the page shows, as JSON
    POST /api/action/X   start empty | look | pick | neutral | auto | relax
    POST /api/stop       hold every arm where it is, stop auto; says what it stopped
    POST /api/options    {"fingertips_cm": .., "material": .., ...}
    POST /api/speeds     {"arm": "left", "max_speed": .., "descent_speed": ..}

The page fetches single frames rather than holding the MJPEG stream open:
Safari keeps only a few connections to a host, a stream that reconnected
filled them, and the page's buttons then waited forever behind it.

It listens on 127.0.0.1 unless told otherwise: whoever opens the page can
move the arms.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("page.html")
STREAM_WIDTH = 1280
STREAM_FPS = 12
JPEG_QUALITY = 75


def _plain(value):
    """What json cannot write by itself: numpy numbers and arrays."""

    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON")


def encode(frame, *, width: int = STREAM_WIDTH, quality: int = JPEG_QUALITY) -> bytes:
    import cv2

    height, source_width = frame.shape[:2]
    if source_width > width:
        frame = cv2.resize(frame, (width, round(height * width / source_width)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode a frame")
    return jpeg.tobytes()


class FrameCache:
    """The newest frame as JPEG, encoded once however many pages ask for it."""

    def __init__(self, camera) -> None:
        self._camera, self._lock = camera, threading.Lock()
        self._at, self._jpeg = None, b""

    def jpeg(self) -> bytes | None:
        frame, at = self._camera.latest()
        if frame is None:
            return None
        with self._lock:
            if at != self._at:
                self._jpeg, self._at = encode(frame), at
            return self._jpeg


def calibration_state(cell) -> list[dict]:
    """The setup checklist shown on the jury page, with the exact recovery command."""

    if not hasattr(cell, "rig"):
        return []
    root = PAGE.parents[2]
    rig = cell.rig
    arms_identified = all(devices.bus for devices in rig.arms.values())
    arms_placed = all(devices.sheet and len(devices.touches) >= 4 for devices in rig.arms.values())
    rolls = all(devices.wrist_roll_offset is not None for devices in rig.arms.values())
    return [
        {"key": "cameras", "label": "Camera identity", "ready": bool(rig.overhead),
         "detail": rig.overhead or "no overhead camera", "command": "uv run trashdrop cameras"},
        {"key": "camera-settings", "label": "Camera settings", "ready": (root / "camera.toml").is_file(),
         "detail": "focus, exposure and white balance", "command": "uv run trashdrop camera tune --camera auto"},
        {"key": "camera-zone", "label": "Camera zone", "ready": (root / "camera_zone.json").is_file(),
         "detail": "capture rectangle and markers", "command": "uv run trashdrop camera zone --camera auto"},
        {"key": "tape", "label": "Tape and table frame",
         "ready": len(rig.tape_pixels) >= 4 and (root / "camera_sheet.json").is_file(),
         "detail": f"{len(rig.tape_pixels)}/4 tape corners", "command": "uv run trashdrop camera tape"},
        {"key": "rig", "label": "Rig identity", "ready": arms_identified,
         "detail": " · ".join(f"{name}: {devices.label}" for name, devices in rig.arms.items()),
         "command": "uv run trashdrop rig identify --quick"},
        {"key": "placements", "label": "Arm placements", "ready": arms_placed,
         "detail": "both fingertips touched the four tape corners",
         "command": "uv run trashdrop rig touch left --tape  # then right"},
        {"key": "roll", "label": "Wrist-roll zero", "ready": rolls,
         "detail": "jaw angle aligned to the item", "command": "uv run trashdrop rig roll left  # then right"},
    ]


def make_handler(cell, spectacles=None):
    frames = FrameCache(cell.camera)

    class Handler(BaseHTTPRequestHandler):
        server_version = "TrashDrop"

        def log_message(self, *args) -> None:  # every request would flood the terminal
            pass

        def log_error(self, format, *args) -> None:  # but not a failing one
            cell.log("web: " + format % args)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                state = cell.state()
                state["calibrations"] = calibration_state(cell)
                state["spectacles"] = spectacles.state() if spectacles is not None else {
                    "available": False, "network_error": "not started", "active": False, "connected": False,
                    "snapshots": [], "hands": {}, "arms": {},
                }
                self._json(state)
            elif path == "/frame.jpg":
                jpeg = frames.jpeg()
                if jpeg is None:
                    self._send(503, b"no frame yet", "text/plain")
                else:
                    self._send(200, jpeg, "image/jpeg")
            elif path == "/stream.mjpg":
                self._stream()
            elif path == "/spectacles/view.jpg" and spectacles is not None:
                jpeg, _ = spectacles.spectator.latest()
                if jpeg is None:
                    self._send(503, b"no Spectacles frame yet", "text/plain")
                else:
                    self._send(200, jpeg, "image/jpeg")
            elif path.startswith("/spectacles/snaps/") and spectacles is not None:
                name = Path(path).name
                picture = spectacles.folder / "snaps" / name
                if name != path.rsplit("/", 1)[-1] or picture.suffix.lower() != ".jpg" or not picture.is_file():
                    self._send(404, b"not found", "text/plain")
                else:
                    self._send(200, picture.read_bytes(), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                if path == "/api/stop":
                    if spectacles is not None:
                        spectacles.set_presentation(False)
                    manual = spectacles.stop() if spectacles is not None and spectacles.active else None
                    stopped = cell.stop()
                    return self._json({"ok": True, "message": manual or stopped})
                if path == "/api/spectacles/start" and spectacles is not None:
                    error = spectacles.start(mode=body.get("mode", "pinch"), scale=float(body.get("scale", 1.0)),
                                             facing=body.get("facing", "same"),
                                             target=body.get("target", "so101"))
                    return self._json({"ok": error is None, "error": error}, 200 if error is None else 409)
                if path == "/api/spectacles/stop" and spectacles is not None:
                    spectacles.set_presentation(False)
                    return self._json({"ok": True, "message": spectacles.stop()})
                if path == "/api/spectacles/presentation" and spectacles is not None:
                    enabled = bool(body.get("enabled"))
                    if not enabled:
                        spectacles.set_presentation(False)
                        message = spectacles.stop() if spectacles.active else "returned to the web controls"
                        return self._json({"ok": True, "message": message})
                    error = spectacles.configure(mode=body.get("mode", "pinch"),
                                                 scale=float(body.get("scale", 1.0)),
                                                 facing=body.get("facing", "same"),
                                                 target=body.get("target", "so101"))
                    if error is None:
                        error = spectacles.set_presentation(True)
                    if error is not None:
                        spectacles.set_presentation(False)
                    return self._json({"ok": error is None, "error": error}, 200 if error is None else 409)
                if path == "/api/spectacles/command" and spectacles is not None:
                    error = spectacles.command(body)
                    return self._json({"ok": error is None, "error": error}, 200 if error is None else 409)
                if path == "/api/spectacles/snapshot" and spectacles is not None:
                    error = spectacles.request_snapshot()
                    return self._json({"ok": error is None, "error": error}, 200 if error is None else 409)
                if path.startswith("/api/action/"):
                    error = cell.begin(path.rsplit("/", 1)[1])
                    return self._json({"ok": error is None, "error": error}, 200 if error is None else 409)
                if path == "/api/options":
                    cell.set_options(body)
                    return self._json({"ok": True})
                if path == "/api/speeds":
                    message = cell.set_speeds(body["arm"], float(body["max_speed"]), float(body["descent_speed"]))
                    return self._json({"ok": True, "message": message})
            except (ValueError, KeyError, TypeError) as error:
                return self._json({"ok": False, "error": str(error)}, 400)
            self._send(404, b"not found", "text/plain")

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser gave up on this picture; the next poll asks again

        def _json(self, payload, status: int = 200) -> None:
            self._send(status, json.dumps(payload, default=_plain).encode(), "application/json")

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            shown = 0.0
            try:
                while True:
                    frame, at = cell.camera.latest()
                    if frame is None or at == shown:
                        time.sleep(0.02)
                        continue
                    shown = at
                    jpeg = encode(frame)
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                    time.sleep(1.0 / STREAM_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page closed or reloaded

    return Handler


def make_server(cell, host: str = "127.0.0.1", port: int = 8000, spectacles=None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(cell, spectacles))
    server.daemon_threads = True
    return server
