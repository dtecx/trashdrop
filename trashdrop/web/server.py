"""Serve the cell's page: the overhead stream, a JSON view of the cell, and its controls.

Standard library only, like api.py: nothing to install and nothing fetched
from the internet, so the page works on the venue's wifi or with none. The
page (page.html) polls /api/state and draws the overlays itself from the
data there; the stream is MJPEG, which every browser shows in an <img>.

    GET  /               the page
    GET  /stream.mjpg    the overhead camera, ~12 frames a second
    GET  /api/state      everything the page shows, as JSON
    POST /api/action/X   start empty | look | pick | neutral | auto
    POST /api/stop       hold every arm where it is, stop auto
    POST /api/options    {"fingertips_cm": .., "material": .., ...}
    POST /api/speeds     {"arm": "left", "max_speed": .., "descent_speed": ..}

It listens on 127.0.0.1 unless told otherwise: whoever opens the page can
move the arms.
"""

from __future__ import annotations

import json
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


def encode(frame) -> bytes:
    import cv2

    height, width = frame.shape[:2]
    if width > STREAM_WIDTH:
        frame = cv2.resize(frame, (STREAM_WIDTH, round(height * STREAM_WIDTH / width)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("could not encode a frame")
    return jpeg.tobytes()


def make_handler(cell):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TrashDrop"

        def log_message(self, *args) -> None:  # the cell keeps its own log
            pass

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(cell.state())
            elif path == "/stream.mjpg":
                self._stream()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                if path == "/api/stop":
                    cell.stop()
                    return self._json({"ok": True})
                if path.startswith("/api/action/"):
                    error = cell.begin(path.rsplit("/", 1)[1])
                    return self._json({"ok": error is None, "error": error}, 200 if error is None else 409)
                if path == "/api/options":
                    cell.set_options(body)
                    return self._json({"ok": True})
                if path == "/api/speeds":
                    cell.set_speeds(body["arm"], float(body["max_speed"]), float(body["descent_speed"]))
                    return self._json({"ok": True})
            except (ValueError, KeyError, TypeError) as error:
                return self._json({"ok": False, "error": str(error)}, 400)
            self._send(404, b"not found", "text/plain")

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

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


def make_server(cell, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(cell))
    server.daemon_threads = True
    return server
