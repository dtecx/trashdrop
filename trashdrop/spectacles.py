"""Snap Spectacles drive the arms: each hand drives the arm on its side, by pinches or like a joystick.

A Lens on the glasses (spectacles/spectacles, a Lens Studio 5.15 project; its
script is Assets/Scripts/HandStream.ts) streams both hands' fingertips,
knuckles and the wearer's head over a WebSocket, some 30 times a second.
This module serves that socket and has each arm follow a hand. Kept apart
from the sorting code, like the dance: it reads the arms, the model and where
the arms stand, and changes none of it.

    uv run python -m trashdrop.spectacles --dry-run   # no arms: the link, and where they would go
    uv run python -m trashdrop.spectacles             # the arms follow the hands; Ctrl+C holds them

The glasses reach the Mac over Wi-Fi, or over their USB-C cable: with them
plugged in, the port on the glasses is forwarded to the Mac's (adb reverse,
as the Spectacles answer adb), and the Lens connects to ws://127.0.0.1:8765
-- no network needed, and none in the way.

How a hand drives its arm, --mode pinch (the default; PinchFollower), the
way VR teleoperation does it with a grip button:

* Thumb and index together grab the jaw: while pinched it goes where it was
  plus --scale times as far as the pinch point goes, and the wrist roll
  turns as the hand does. Let go and the arm stays; pinch again anywhere to
  go on. Thumb and pinky together open a closed jaw or close an open one.
  No calibration; forward is where the wearer looked when pinching, level.
  The pinch point is steadied by a 1-euro filter.

--mode joystick (Follower):

* First, calibration: both hands up in the air, level with each other, held
  still for --hold seconds (5; the glasses count down). Where they rest is
  each hand's neutral, and right is the way from the left hand to the right
  one -- wherever the wearer happened to look (with one arm, --arm, it is
  where the wearer looked, level).
* Then each hand is a joystick. Within --dead-zone (3 cm, on each axis) of
  its neutral a hand moves nothing; past it, the arm's jaw moves the same
  way -- the hand a little lower, the jaw goes down; lower and to the right,
  down and to the right -- at --gain cm/s for every cm past the dead zone,
  --top-speed at most. Back to neutral, the arm stops where it is.
* A fist turns the jaw: close the hand, turn the fist, open it -- the wrist
  roll turned as far as the fist did, and stays there. While the fist is
  closed nothing else moves; turn it again to go on turning.
* Thumb and index tip apart open the jaw; together they close it -- JAW_DELAY_S
  late, so that thumb and index meeting on the way into a fist grip nothing.
  After a fist or a hand out of sight, the jaw waits until thumb and index
  agree with how open it is, so opening the hand does not drop what it holds.
* The glasses show what each arm makes of its hand: holding, moving (and
  which way: forward, back, right, left, up, down), turning the jaw.
* Every message from the glasses is kept in out/spectacles/ (--no-record:
  not), one JSON line each, to replay what the hands really did.
* A hand out of sight holds its arm. Back within a moment, it carries on;
  after longer, its arm waits until the hand is back at its neutral (the
  glasses say which way). Both hands out of sight for RECALIBRATE_S:
  calibrate again.
* The wearer stands behind the arms, facing their way; with --facing them,
  in front of them facing back, each hand drives the arm on its side as the
  wearer sees it, and a hand moved towards the arms moves its jaw towards
  their bases.
* A target stays where its arm can reach, above the table, and never more
  than LEAD_CM ahead of the jaw, so a hand brought back stops it at once.
  The jaw keeps pointing down.
* The two arms keep CLEARANCE_CM between their centre lines (foot to jaw,
  on the sheet both are placed on), so either may reach into the middle, or
  past it, while the other is elsewhere; a step that would bring them nearer
  is not taken. Unplaced, or with one arm driven alone, each keeps to its
  own side (SIDE_CM). When an arm cannot go where its hand says, the glasses
  say why: "moving left: at the other arm", "... at the table", "... at full
  reach", "... at a joint limit".

Each tick (50 Hz) every arm takes one damped least-squares step towards its
target on the model -- far quicker than solving afresh, and a target out of
reach just leaves the arm at the nearest it gets -- no joint turning faster
than --speed.
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import math
import socket
import socketserver
import struct
import threading
import time
from pathlib import Path

import numpy as np

PORT = 8765
VIDEO_PORT = 8766
VIDEO_FPS = 30  # at most; the webcam's own rate, and the glasses show as many as they decode
VIDEO_CAPTURE_WIDTH = 800
VIDEO_CAPTURE_HEIGHT = 600
VIDEO_WIDTH = 640
VIDEO_QUALITY = 60
VIDEO_REPORT_S = 10.0  # how often the bridge says what the video does
RATE_HZ = 50
SPEED = 120.0  # deg/s a joint turns at most while following
GRIPPER_SPEED = 150.0  # percent a second
STALE_S = 0.3  # a hand not heard of for this long holds its arm
SMOOTHING = 0.5  # the share of each new hand position; the rest is the last: steadies the tracking's jitter
ORIENTATION_SMOOTHING = 0.4  # knuckles jitter while pinching; filter angles without averaging across the 180-degree seam
OPEN = 60.0  # percent open with thumb and index well apart
PINCH_CM = (2.0, 9.0)  # thumb tip to index tip: shut at the first or closer, open at the second or wider
HOLD_S = 5.0  # calibration: the hands held level and still this long
HOLD_CM = 5.0  # still: every wrist within this of where the hold began (3 kept restarting the count)
LEVEL_CM = 15.0  # level: the two wrists at most this far apart in height (8 took half a minute to meet)
SPAN_CM = 10.0  # hands at least this far apart at calibration say which way right is; closer, the gaze does
DEAD_ZONE_CM = 3.0  # on each axis: a hand this close to its neutral moves nothing
GAIN = 1.5  # cm/s the jaw moves for every cm the hand is past the dead zone
TOP_SPEED = 6.0  # cm/s the jaw moves at most: gentle, for trying it out
LEAD_CM = 2.0  # a target runs at most this far ahead of the jaw: bring the hand back and it stops
GRACE_S = 0.5  # a hand back within this carries on; later, its arm waits for it at the neutral
RECALIBRATE_S = 3.0  # every hand out of sight this long: calibrate again
FIST = (1.1, 1.35)  # curl (see curl()): a fist closes below the first, opens above the second
# --mode pinch. Thumb and index tips: pinched below the first, let go above the second (a recorded session:
# pinches under 2.5 cm, open hands 10-15). Thumb and pinky tips: under the first the jaw toggles, over the
# second it may toggle again (never under 6 cm in that session, fists included).
GRAB_CM = (2.5, 4.0)
TOGGLE_CM = (3.5, 5.5)
# Each pinch does one thing, whichever the hand does first: turning past TWIST_DEG makes it a twist (the
# jaw turns, and stays put), moving the pinch past DRAG_CM makes it a drag (the jaw moves, and does not
# turn). A twist about the forearm swings the pinch ~9 cm round it -- 12 degrees is 1.9 cm, past DRAG_CM
# -- so a twist that began as a drag becomes one after all past LATE_TWIST (degrees, while the pinch has gone
# less than cm), and the jaw goes back to where the pinch found it. Replayed on three sessions: left-hand
# twists 8 of 20 came out as twists before (12 degrees, no late switch), 15 of 20 now; drags unchanged (78 of
# 83 left, 119 of 130 right).
TWIST_DEG = 10.0
DRAG_CM = 1.5
LATE_TWIST = (25.0, 4.0)
SCALE = 1.0  # --mode pinch: jaw cm per hand cm
# 1-euro filter (Casiez, Roussel and Vogel, CHI 2012) on the pinch point: cutoff Hz at rest, and how much
# faster it gets per cm/s -- steady when the hand is still, little lag when it moves.
EURO = (1.5, 0.05)
CATCH_PCT = 10.0  # a held jaw follows the pinch again once the two are this close
# A pinch reaches the jaw this late. Closing the hand into a fist brings thumb and index
# together before the fist is told; what they did in the meantime is dropped, not gripped.
JAW_DELAY_S = 0.4
READY_CM = (22.0, 0.0, 10.0)  # where each arm waits, in its own frame: ahead of its base, over the table
REACH_CM = (12.0, 38.0)  # from the base's turning axis
HEIGHT_CM = (3.0, 35.0)  # the TCP above the table
# Placed on the sheet, the two arms' centre lines (foot, shoulder, elbow, wrist, jaw) are kept this far apart,
# so either may reach into the middle while the other is not there. Unplaced, or driving one arm alone,
# each keeps to its own side instead: no more than SIDE_CM past its base's line towards the other.
CLEARANCE_CM = 10.0
SIDE_CM = 14.0  # the bases stand ~38 cm apart: the middle 10 cm is then neither arm's
POINTING_WEIGHT = 0.01  # metres a radian: the fingers pointing down gives way to where the hand is
DAMPING = 0.02
READY_ROOM_DEG = 10.0  # the ready pose keeps every joint this far from its limits, room to follow
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
STEERED = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")  # place the jaw, fingers down; the fist sets the roll
# Which hand drives which arm: standing behind the arms, or in front facing them.
HAND_FOR = {"same": {"left": "left", "right": "right"}, "them": {"left": "right", "right": "left"}}
UP = np.array([0.0, 1.0, 0.0])  # the glasses' world: y up, x right, z back, centimetres

# --- the socket: RFC 6455, just what one Lens needs ---------------------------------------

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key.strip() + GUID).encode()).digest()).decode()


def read_frame(stream) -> tuple[bool, int, bytes]:
    """(final, opcode, payload) of the next frame; a close (8) when the stream ends."""

    head = stream.read(2)
    if len(head) < 2:
        return True, 8, b""
    final, opcode, length = bool(head[0] & 0x80), head[0] & 0x0F, head[1] & 0x7F
    if length == 126:
        length = struct.unpack(">H", stream.read(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", stream.read(8))[0]
    mask = stream.read(4) if head[1] & 0x80 else b""
    payload = stream.read(length)
    if mask:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return final, opcode, payload


def frame(payload: bytes, opcode: int = 1) -> bytes:
    """A server's frame: final and unmasked."""

    size = len(payload)
    if size < 126:
        head = bytes([0x80 | opcode, size])
    elif size < 1 << 16:
        head = bytes([0x80 | opcode, 126]) + struct.pack(">H", size)
    else:
        head = bytes([0x80 | opcode, 127]) + struct.pack(">Q", size)
    return head + payload


class Hands:
    """The newest message from the glasses, and what to tell them back; shared between threads.

    With ``record`` (a text file), every message goes into it too, one JSON line
    each with the bridge's time as "at": what the hands really did, to replay.
    """

    def __init__(self, clock=time.monotonic, record=None, snapshots=None, spectator=None) -> None:
        self._lock = threading.Lock()
        self._message, self._at = None, -math.inf
        self._commands: collections.deque[dict] = collections.deque()
        self.clock = clock
        self.record = record
        self.snapshots = snapshots
        self.spectator = spectator
        self.command_handler = None
        self.context = None  # optional current cell mode supplied by the unified web bridge
        self.status = "waiting for the arms"  # what the arms do, in words
        self.report: str | None = None  # what the glasses are sent: JSON with the status, boxes and stops (report())
        self.connected = False
        self.presentation: bool | None = None  # web demo mode; omitted by the standalone bridge

    def receive(self, message: dict) -> None:
        """Route a Lens message without letting a multi-megabyte snapshot become hand state."""

        if self.spectator is not None and isinstance(message, dict) and "spectator" in message:
            self.spectator.receive(message)
            return
        if self.snapshots is not None and isinstance(message, dict) and "snap" in message:
            self.snapshots.receive(message)
            return
        if isinstance(message, dict) and isinstance(message.get("command"), str):
            if self.command_handler is not None and self.command_handler(message):
                return
            with self._lock:
                self._commands.append(message)
            return
        self.put(message)

    def put(self, message: dict) -> None:
        with self._lock:
            self._message, self._at = message, self.clock()
            if self.record is not None and isinstance(message, dict):
                self.record.write(json.dumps({"at": round(self._at, 3), **message}, separators=(",", ":")) + "\n")

    def stop_recording(self) -> None:
        with self._lock:
            record, self.record = self.record, None
        if record is not None:
            record.close()

    def latest(self) -> tuple[dict | None, float]:
        """(the newest message, how many seconds old)."""

        with self._lock:
            return self._message, self.clock() - self._at

    def take_commands(self) -> list[dict]:
        """Commands pressed in the Lens since the last control tick."""

        with self._lock:
            commands = list(self._commands)
            self._commands.clear()
        return commands

    def answer(self) -> str:
        """Current arm report, with a snapshot request while the trigger is pending."""

        answer = self.report or self.status
        context = self.context() if self.context is not None else {}
        snap = self.snapshots.requested() if self.snapshots is not None else None
        spectator = self.spectator.requested() if self.spectator is not None else None
        if snap is None and spectator is None and self.presentation is None and not context:
            return answer
        try:
            report = json.loads(answer)
        except ValueError:
            report = {"status": answer}
        if not isinstance(report, dict):
            report = {"status": answer}
        report.update(context)
        if snap is not None:
            report["snap"] = snap
        if spectator is not None:
            report["spectator"] = spectator
        if self.presentation is not None:
            report["presentation"] = self.presentation
        return json.dumps(report, separators=(",", ":"))


class Snapshots:
    """Request, validate and save the optical camera and rendered Lens view."""

    def __init__(self, folder: Path, trigger: Path, *, wall_clock=time.time, log=print) -> None:
        self.folder, self.trigger = Path(folder), Path(trigger)
        self.wall_clock, self.log = wall_clock, log
        self._lock = threading.Lock()
        self._pending: int | None = None
        self._last_id = 0

    def requested(self) -> int | None:
        """Consume a trigger file and return one stable request id until its pictures arrive."""

        with self._lock:
            if self._pending is None and self.trigger.exists():
                self.trigger.unlink(missing_ok=True)
                self._last_id = max(self._last_id + 1, round(self.wall_clock() * 1000))
                self._pending = self._last_id
                self.log(f"requesting Spectacles snapshot {self._pending}")
            return self._pending

    @staticmethod
    def _jpeg(message: dict, name: str) -> bytes:
        value = message.get(name)
        if not isinstance(value, str):
            raise ValueError(f"snapshot has no {name} image")
        try:
            picture = base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as error:
            raise ValueError(f"snapshot {name} is not base64") from error
        if not picture.startswith(b"\xff\xd8") or not picture.endswith(b"\xff\xd9"):
            raise ValueError(f"snapshot {name} is not a JPEG")
        return picture

    @staticmethod
    def _composite(camera_jpeg: bytes, view_jpeg: bytes) -> bytes:
        """Add the emissive Lens view to the colour camera, as the eye sees both."""

        import cv2

        camera = cv2.imdecode(np.frombuffer(camera_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        view = cv2.imdecode(np.frombuffer(view_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if camera is None or view is None:
            raise ValueError("snapshot JPEG cannot be decoded")
        if camera.shape[:2] != view.shape[:2]:
            camera = cv2.resize(camera, (view.shape[1], view.shape[0]), interpolation=cv2.INTER_AREA)
        composite = cv2.add(camera, view)  # saturated addition: black Lens pixels stay transparent
        ok, encoded = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError("snapshot composite cannot be encoded")
        return encoded.tobytes()

    def receive(self, message: dict) -> tuple[Path, Path, Path] | None:
        """Save a response for the outstanding id; ignore late duplicate responses."""

        snap = message.get("snap")
        with self._lock:
            if not isinstance(snap, int) or snap != self._pending:
                return None
            try:
                view, camera = self._jpeg(message, "view"), self._jpeg(message, "camera")
                composite = self._composite(camera, view)
            except (ValueError, ImportError) as error:
                self.log(f"cannot save Spectacles snapshot {snap}: {error}")
                return None
            self.folder.mkdir(parents=True, exist_ok=True)
            stem = f"snap-{snap}"
            view_path = self.folder / f"{stem}-view.jpg"
            camera_path = self.folder / f"{stem}-camera.jpg"
            composite_path = self.folder / f"{stem}-composite.jpg"
            view_path.write_bytes(view)
            camera_path.write_bytes(camera)
            composite_path.write_bytes(composite)
            self._pending = None
        self.log(f"Spectacles snapshot {snap}: {view_path}, {camera_path}, {composite_path}")
        return view_path, camera_path, composite_path


class SpectatorFrames:
    """A low-rate optical-view stream for the jury page, requested one frame at a time."""

    def __init__(self, *, fps: float = 2.0, clock=time.monotonic, log=print) -> None:
        self.fps, self.clock, self.log = fps, clock, log
        self.enabled = False
        self._lock = threading.Lock()
        self._pending: int | None = None
        self._last_request = -math.inf
        self._last_id = 0
        self.sequence = 0
        self.jpeg: bytes | None = None
        self.captured_at = 0.0

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = enabled
            if not enabled:
                self._pending = None

    def requested(self) -> int | None:
        with self._lock:
            now = self.clock()
            if not self.enabled:
                return None
            if self._pending is None and now - self._last_request >= 1 / self.fps:
                self._last_id += 1
                self._pending = self._last_id
                self._last_request = now
            return self._pending

    def receive(self, message: dict) -> bool:
        request = message.get("spectator")
        with self._lock:
            if not isinstance(request, int) or request != self._pending:
                return False
        try:
            if isinstance(message.get("composite"), str):
                composite = Snapshots._jpeg(message, "composite")
            else:  # older Lens: camera and transparent render target arrived separately
                view = Snapshots._jpeg(message, "view")
                camera = Snapshots._jpeg(message, "camera")
                composite = Snapshots._composite(camera, view)
        except (ValueError, ImportError) as error:
            self.log(f"cannot decode Spectacles spectator frame {request}: {error}")
            with self._lock:
                self._pending = None
            return False
        with self._lock:
            if request != self._pending:
                return False
            self.jpeg = composite
            self.sequence += 1
            self.captured_at = time.time()
            self._pending = None
        return True

    def latest(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self.jpeg, self.sequence


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_server(hands: Hands, host: str = "0.0.0.0", port: int = PORT) -> socketserver.ThreadingTCPServer:
    """A WebSocket server that puts every JSON message into ``hands`` and answers with its status."""

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            self.rfile.readline()  # GET / HTTP/1.1
            headers = {}
            while True:
                line = self.rfile.readline().decode("latin-1").strip()
                if not line:
                    break
                name, _, value = line.partition(":")
                headers[name.strip().lower()] = value.strip()
            if "sec-websocket-key" not in headers:  # a browser, say: tell it what this is
                body = b"TrashDrop's Spectacles bridge: the Lens connects with ws://<this address>:<port>\n"
                self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: "
                                 + str(len(body)).encode() + b"\r\n\r\n" + body)
                return
            self.wfile.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                              f"Sec-WebSocket-Accept: {accept_key(headers['sec-websocket-key'])}\r\n\r\n").encode())
            hands.connected, told, parts = True, None, []
            try:
                while True:
                    final, opcode, payload = read_frame(self.rfile)
                    if opcode == 8:
                        self.wfile.write(frame(b"", 8))
                        return
                    if opcode == 9:
                        self.wfile.write(frame(payload, 10))
                        continue
                    if opcode not in (0, 1, 2):
                        continue
                    parts.append(payload)
                    if not final:
                        continue
                    text, parts = b"".join(parts), []
                    try:
                        hands.receive(json.loads(text))
                    except ValueError:
                        continue
                    answer = hands.answer()
                    if answer != told:
                        told = answer
                        self.wfile.write(frame(told.encode()))
            except (OSError, struct.error):
                pass  # the glasses went away
            finally:
                hands.connected = False

    return _Server((host, port), Handler)


def crop_video(frame, tape_pixels: dict[str, tuple[float, float]],
               reference_size: tuple[int, int]):
    """Trim only above the taped square, keeping the full width and both arms."""

    height = frame.shape[0]
    _, reference_height = reference_size
    if len(tape_pixels) < 4 or reference_height <= 0:
        return frame
    points = np.asarray(list(tape_pixels.values()), dtype=float)
    center_y = float(points[:, 1].mean()) * height / reference_height
    # If the tape centre is below the image centre, discard the same excess
    # above it so the retained picture is centred vertically on that tape.
    top = min(max(round(2 * center_y - height), 0), height - 40)
    return frame[top:, :]


class VideoFrames:
    """Read continuously; publish only the newest encoded frame to every viewer."""

    def __init__(self, capture=None, *, fps: float = VIDEO_FPS, width: int = VIDEO_WIDTH,
                 quality: int = VIDEO_QUALITY, tape_pixels=None, reference_size=(0, 0),
                 clock=time.monotonic, wall_clock=time.time, decorate=None) -> None:
        self.capture = capture
        self.fps, self.width, self.quality = fps, width, quality
        self.tape_pixels = tape_pixels or {}
        self.reference_size = reference_size
        self.clock, self.wall_clock = clock, wall_clock
        self.decorate = decorate
        self.condition = threading.Condition()
        self.sequence = 0
        self.packet: bytes | None = None
        self.running = True
        self.thread: threading.Thread | None = None
        self.counted_since, self.read_count, self.sent_count, self.sent_bytes = clock(), 0, 0, 0

    def report(self) -> str:
        """What the video did since the last report: the camera's rate, the rate sent, the size of a frame."""

        with self.condition:
            now = self.clock()
            seconds = max(now - self.counted_since, 1e-9)
            text = (f"video: camera {self.read_count / seconds:.1f} fps, sent {self.sent_count / seconds:.1f} fps, "
                    f"{self.sent_bytes / max(self.sent_count, 1) / 1024:.0f} KB a frame")
            self.counted_since, self.read_count, self.sent_count, self.sent_bytes = now, 0, 0, 0
        return text

    def start(self) -> None:
        if self.capture is None:
            raise RuntimeError("video capture is not open")
        self.thread = threading.Thread(target=self._read, name="spectacles-video", daemon=True)
        self.thread.start()

    def _read(self) -> None:
        from .web.server import encode

        next_encode = self.clock()
        while self.running:
            ok, picture = self.capture.read()
            if not ok or picture is None:
                time.sleep(0.05)
                continue
            self.read_count += 1
            captured_ms = round(self.wall_clock() * 1000)
            now = self.clock()
            # Keep draining the camera so its buffer cannot grow stale. A quarter frame's slack:
            # a camera at just --video-fps must not lose every other frame to its jitter.
            if now < next_encode - 0.25 / self.fps:
                continue
            if self.decorate is not None:
                picture = self.decorate(picture)
            picture = crop_video(picture, self.tape_pixels, self.reference_size)
            jpeg = encode(picture, width=self.width, quality=self.quality)
            self.publish(jpeg, captured_ms)
            next_encode = max(next_encode + 1 / self.fps, now)

    def publish(self, jpeg: bytes, captured_ms: int) -> None:
        with self.condition:
            self.sequence += 1
            self.packet = json.dumps({"seq": self.sequence, "capturedMs": captured_ms,
                                      "jpeg": base64.b64encode(jpeg).decode("ascii")}, separators=(",", ":")).encode()
            self.sent_count += 1
            self.sent_bytes += len(jpeg)
            self.condition.notify_all()

    def after(self, sequence: int) -> tuple[int, bytes] | None:
        with self.condition:
            self.condition.wait_for(lambda: not self.running or self.sequence > sequence, timeout=1.0)
            if not self.running or self.packet is None or self.sequence <= sequence:
                return None
            return self.sequence, self.packet

    def close(self) -> None:
        with self.condition:
            self.running = False
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()


def make_video_server(video: VideoFrames, host: str = "0.0.0.0",
                      port: int = VIDEO_PORT) -> socketserver.ThreadingTCPServer:
    """A separate WebSocket; slow clients skip frames instead of building a queue."""

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            self.rfile.readline()
            headers = {}
            while True:
                line = self.rfile.readline().decode("latin-1").strip()
                if not line:
                    break
                name, _, value = line.partition(":")
                headers[name.strip().lower()] = value.strip()
            if "sec-websocket-key" not in headers:
                body = b"TrashDrop Spectacles video: connect by WebSocket.\n"
                self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: "
                                 + str(len(body)).encode() + b"\r\n\r\n" + body)
                return
            self.wfile.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                              f"Sec-WebSocket-Accept: {accept_key(headers['sec-websocket-key'])}\r\n\r\n").encode())
            sequence = 0
            send_lock = threading.Lock()

            def receive() -> None:
                while video.running:
                    try:
                        _, opcode, payload = read_frame(self.rfile)
                        if opcode == 8:
                            return
                        if opcode != 1:
                            continue
                        message = json.loads(payload)
                        if not isinstance(message, dict) or not isinstance(message.get("pingMs"), (int, float)):
                            continue
                        answer = json.dumps({"pongMs": message["pingMs"],
                                             "serverMs": round(time.time() * 1000)}).encode()
                        with send_lock:
                            self.wfile.write(frame(answer))
                    except (OSError, ValueError, struct.error):
                        return

            threading.Thread(target=receive, daemon=True).start()
            try:
                while video.running:
                    latest = video.after(sequence)
                    if latest is None:
                        continue
                    sequence, packet = latest
                    with send_lock:
                        self.wfile.write(frame(packet))
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    return _Server((host, port), Handler)


def usb_tunnel(port: int) -> str | None:
    """Spectacles on a USB cable: their ``port`` forwarded to this Mac's (adb reverse). The Lens's address, or None."""

    import shutil
    import subprocess

    if not shutil.which("adb"):
        return None
    try:
        listed = subprocess.run(["adb", "devices", "-l"], capture_output=True, text=True, timeout=15).stdout
        glasses = [line.split()[0] for line in listed.splitlines()[1:]
                   if " device " in f"{line} " and "Snap" in line]
        if not glasses:
            return None
        done = subprocess.run(["adb", "-s", glasses[0], "reverse", f"tcp:{port}", f"tcp:{port}"],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return f"ws://127.0.0.1:{port}" if done.returncode == 0 else None


def addresses() -> list[str]:
    """This computer's addresses on the local network: what to type into the Lens."""

    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # sends nothing: only picks the way out
            found.append(probe.getsockname()[0])
    except OSError:
        pass
    try:
        found += [address for address in socket.gethostbyname_ex(socket.gethostname())[2]
                  if not address.startswith("127.") and address not in found]
    except OSError:
        pass
    return found or ["<this Mac's address>"]


# --- a hand to its arm -------------------------------------------------------------------


def facing_frame(look, head, point) -> np.ndarray:
    """The wearer's forward, right and up as rows, in the glasses' world.

    Forward is ``look`` made level. The hands are always in front of the
    face, so if ``point`` is behind ``head`` along it, the camera's axis came
    the other way round, and forward is turned about.
    """

    head, point = np.asarray(head, dtype=float), np.asarray(point, dtype=float)
    ahead = (point - head) * [1.0, 0.0, 1.0]
    forward = np.asarray(look, dtype=float) * [1.0, 0.0, 1.0]
    if np.linalg.norm(forward) < 1e-6:
        forward = ahead
    forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
    if forward @ ahead < 0:
        forward = -forward
    return np.array([forward, np.cross(forward, UP), UP])  # right-handed, y up: forward x up is right


def body_frame(left, right, look, head) -> np.ndarray:
    """The wearer's forward, right and up from the hands at rest: right runs from the left hand to the right.

    Where the wearer happened to look while calibrating -- at the arms, at the
    video -- does not turn it, as it would a frame taken from the gaze. Hands
    closer together than SPAN_CM cannot tell; the gaze (facing_frame) does.
    """

    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    gaze = facing_frame(look, head, (left + right) / 2)
    across = (right - left) * [1.0, 0.0, 1.0]
    if np.linalg.norm(across) < SPAN_CM:
        return gaze
    across /= np.linalg.norm(across)
    if across @ gaze[1] < 0:  # hands crossed over: the gaze still knows which way is right
        across = -across
    return np.array([np.cross(UP, across), across, UP])


def gripper_for(gap_cm: float) -> float:
    """Percent open for the gap between thumb and index tip."""

    low, high = PINCH_CM
    return OPEN * min(max((gap_cm - low) / (high - low), 0.0), 1.0)


def angle_delta(current: float, start: float) -> float:
    """Shortest signed rotation in degrees, including across the +/-180 seam."""

    return (current - start + 180.0) % 360.0 - 180.0


def hand_angles(hand: dict, wearer: np.ndarray) -> tuple[float, float] | None:
    """Palm roll about the fingers and pitch above the wearer's level gaze.

    The wrist-to-middle-knuckle line is stable during a pinch, unlike the
    fingertips. Index-to-pinky supplies an across-palm direction for roll.
    """

    try:
        wrist = np.asarray(hand["wrist"], dtype=float)
        middle = np.asarray(hand["middleKnuckle"], dtype=float)
        across = np.asarray(hand["indexKnuckle"], dtype=float) - np.asarray(hand["pinkyKnuckle"], dtype=float)
        if any(vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in (wrist, middle, across)):
            return None
        fingers = middle - wrist
        if np.linalg.norm(fingers) < 1.0:
            return None
        fingers /= np.linalg.norm(fingers)
        across -= fingers * float(across @ fingers)
        reference_right = wearer[1] - fingers * float(wearer[1] @ fingers)
        if np.linalg.norm(across) < 1.0 or np.linalg.norm(reference_right) < 0.15:
            return None
        across /= np.linalg.norm(across)
        reference_right /= np.linalg.norm(reference_right)
        roll = math.degrees(math.atan2(float(fingers @ np.cross(reference_right, across)),
                                       float(reference_right @ across)))
        # Height above the horizontal is independent of hand yaw; turning the
        # palm sideways must not make the jaw tilt farther.
        pitch = math.degrees(math.asin(min(max(float(fingers @ wearer[2]), -1.0), 1.0)))
        return roll, pitch
    except (KeyError, TypeError, ValueError):
        return None


def curl(hand: dict) -> float | None:
    """How far the fingers fold: middle, ring and pinky tip to the wrist, over the palm's length.

    About 2 with the fingers straight, about 1 or less in a fist; thumb and
    index -- the pinch -- are left out of it.
    """

    try:
        wrist = np.asarray(hand["wrist"], dtype=float)
        palm = float(np.linalg.norm(np.asarray(hand["middleKnuckle"], dtype=float) - wrist))
        tips = [np.asarray(hand[name], dtype=float) for name in ("middleTip", "ringTip", "pinkyTip")]
    except (KeyError, TypeError, ValueError):
        return None
    if wrist.shape != (3,) or palm < 1.0 or any(tip.shape != (3,) for tip in tips):
        return None
    return float(np.mean([np.linalg.norm(tip - wrist) for tip in tips])) / palm


def past_dead_zone(offset: np.ndarray, dead_zone: float) -> np.ndarray:
    """How far past ``dead_zone`` each axis is, signed; nothing inside it."""

    return np.sign(offset) * np.maximum(np.abs(offset) - dead_zone, 0.0)


def way_back(seen: np.ndarray) -> str:
    """Which way a hand ``seen`` (forward, right, up of its neutral) should go to reach it."""

    axis = int(np.argmax(np.abs(seen)))
    words = (("closer", "farther"), ("left", "right"), ("down", "up"))  # for a positive offset, a negative one
    return f"{words[axis][0 if seen[axis] > 0 else 1]} {abs(float(seen[axis])):.0f} cm"


def heading_words(seen: np.ndarray, dead_zone: float) -> list[str]:
    """Which ways a hand ``seen`` (forward, right, up of its neutral) is past the dead zone, in the wearer's words."""

    words = (("forward", "back"), ("right", "left"), ("up", "down"))
    return [pair[0 if value > 0 else 1] for value, pair in zip(past_dead_zone(seen, dead_zone), words) if value]


def heading(seen: np.ndarray, dead_zone: float) -> str:
    return ", ".join(heading_words(seen, dead_zone))


def segment_gap(p0, p1, q0, q1) -> float:
    """How close the segments p0-p1 and q0-q1 come (Ericson, Real-Time Collision Detection, 5.1.9)."""

    p0, p1, q0, q1 = (np.asarray(point, dtype=float) for point in (p0, p1, q0, q1))
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e < 1e-12:
            s, t = min(max(-c / a, 0.0), 1.0), 0.0
        else:
            b = float(d1 @ d2)
            denominator = a * e - b * b
            s = min(max((b * f - c * e) / denominator, 0.0), 1.0) if denominator > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = min(max(-c / a, 0.0), 1.0), 0.0
            elif t > 1.0:
                s, t = min(max((b - c) / a, 0.0), 1.0), 1.0
    return float(np.linalg.norm(p0 + d1 * s - (q0 + d2 * t)))


def lines_apart(one: np.ndarray, other: np.ndarray) -> float:
    """How close two polylines come: two arms' centre lines, say."""

    return min(segment_gap(one[i], one[i + 1], other[j], other[j + 1])
               for i in range(len(one) - 1) for j in range(len(other) - 1))


class OneEuro:
    """The 1-euro filter for a point (Casiez, Roussel and Vogel, CHI 2012): a low-pass whose cutoff rises with speed."""

    def __init__(self, min_cutoff: float = EURO[0], beta: float = EURO[1], d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))

    def __call__(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.x is None or dt <= 0:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return self.x.copy()
        a = self._alpha(self.d_cutoff, dt)
        self.dx = a * (x - self.x) / dt + (1 - a) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * float(np.linalg.norm(self.dx)), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()

    def reset(self) -> None:
        self.x = self.dx = None


class Calibration:
    """Where each hand rests: held up, level and still, for ``hold_s``."""

    def __init__(self, sides, hold_s: float = HOLD_S) -> None:
        self.sides, self.hold_s = tuple(sides), hold_s
        self.reset()

    def reset(self, why: str | None = None) -> None:
        self.start: dict[str, np.ndarray] | None = None
        self.total: dict[str, np.ndarray] = {}
        self.count, self.held = 0, 0.0
        hands = "both hands" if len(self.sides) > 1 else f"the {self.sides[0]} hand"
        self.status = why or f"calibrate: {hands} up in the air, level, still"

    def update(self, wrists: dict[str, object], dt: float) -> dict[str, np.ndarray] | None:
        """Each hand's neutral once the hold is done; None until then (``status`` says why)."""

        if any(wrists.get(side) is None for side in self.sides):
            self.reset()
            return None
        points = {side: np.asarray(wrists[side], dtype=float) for side in self.sides}
        if len(self.sides) > 1:
            first, second = self.sides[:2]
            gap = float(points[first][1] - points[second][1])  # y is up
            if abs(gap) > LEVEL_CM:
                self.reset(f"calibrate: {first if gap > 0 else second} hand {abs(gap):.0f} cm higher, level them")
                return None
        if self.start is None or any(np.linalg.norm(points[side] - self.start[side]) > HOLD_CM
                                     for side in self.sides):
            self.start, self.total, self.count, self.held = points, {side: np.zeros(3) for side in self.sides}, 0, 0.0
        for side in self.sides:
            self.total[side] += points[side]
        self.count += 1
        self.held += dt
        if self.held < self.hold_s:
            self.status = f"calibrate: hold still {math.ceil(self.hold_s - self.held)}"
            return None
        return {side: self.total[side] / self.count for side in self.sides}


class Follower:
    """One arm steered by one hand like a joystick, on the model: joint targets for the arm, tick by tick."""

    calibrates = True  # its hand needs a neutral first (Calibration, engage)

    def __init__(self, name: str, kinematics, placement, pose: dict[str, float], *, speed: float = SPEED,
                 facing: str = "same", limits: dict[str, tuple[float, float]] | None = None,
                 dead_zone: float = DEAD_ZONE_CM, gain: float = GAIN, top_speed: float = TOP_SPEED):
        self.name, self.kinematics, self.placement = name, kinematics, placement
        self.q = dict(pose)  # joint -> degrees (gripper: percent open): what the arm is sent
        self.speed, self.facing, self.limits = speed, facing, limits or {}
        self.dead_zone, self.gain, self.top_speed = dead_zone, gain, top_speed
        self.toward_other = -1.0 if name == "left" else 1.0  # the other arm is to the left arm's right (-y)
        self.neutral: tuple[np.ndarray, np.ndarray] | None = None  # where the hand rests, the wearer's frame
        self.point: np.ndarray | None = None  # the wrist, steadied
        self.target: np.ndarray | None = None  # where the jaw is headed, cm in the arm's frame
        self.roll = self.q["wrist_roll"]  # where the fist left the wrist roll
        self.hand_roll: float | None = None  # the hand's roll about its fingers, steadied
        self.grip: tuple[float, float] | None = None  # hand roll and wrist roll when the fist closed
        self.fist = False
        self.jaw_held = False  # waiting for thumb and index to agree with how open the jaw is
        self.pinches: collections.deque[tuple[float, float]] = collections.deque()  # (when, percent) on the way
        self.jaw_want: float | None = None  # the pinch that has reached the jaw
        self.clock_s = 0.0  # this follower's own time, in ticks' dt
        self.centred = True  # False: back after a while out of sight, waiting for the hand at its neutral
        self.lost_s = 0.0
        self.curl: float | None = None
        self.pinned: list[str] = []  # joints the last step found at a limit, pushed further
        self.blocked: list[str] = []  # the ways the hand pushes that its arm cannot go (heading_words)
        self.mode = "calibrating"  # calibrating, holding, moving, turning, lost, centring: for the glasses' colours
        self.state = "calibrating"
        self.home_q = {joint: self.q[joint] for joint in ARM_JOINTS}
        self.homing = False
        self.menu_jaw: float | None = None

    def tcp_cm(self) -> np.ndarray:
        return self.kinematics.tcp({joint: self.q[joint] for joint in ARM_JOINTS}) * 100

    def tip_height(self) -> float:
        """How far the tip of the fixed finger is above the table, cm: what the camera from above cannot show."""

        tip = self.kinematics.fingertip({joint: self.q[joint] for joint in ARM_JOINTS}) * 100
        return float(tip[2] - (self.placement.table_height(tip[0], tip[1]) if self.placement is not None else 0.0))

    def guide(self) -> dict | None:
        """What the glasses draw for this arm's hand, None before calibrating.

        The box is the dead zone round the hand's neutral, in the glasses' own
        world (cm), its axes the wearer's forward, right and up; ``blocked`` are
        the faces the hand pushes on in vain; ``tip`` the finger's height above
        the table.
        """

        if self.neutral is None:
            return None
        start, wearer = self.neutral
        return {"centre": [round(float(value), 1) for value in start], "axes": np.round(wearer, 3).tolist(),
                "half": self.dead_zone, "mode": self.mode, "state": self.state,
                "blocked": list(self.blocked), "tip": round(self.tip_height()),
                "jaw": round(self.q["gripper"]), "roll": round(self.q["wrist_roll"])}

    def command_jaw(self, opened: bool | None = None) -> None:
        """Open, close, or toggle the jaw from the Lens menu, at the normal jaw speed."""

        if opened is None:
            opened = self.q["gripper"] < OPEN / 2
        self.menu_jaw = OPEN if opened else 0.0

    def finish_commands(self, dt: float) -> None:
        if self.menu_jaw is None:
            return
        most = GRIPPER_SPEED * dt
        delta = self.menu_jaw - self.q["gripper"]
        self.q["gripper"] += min(max(delta, -most), most)
        if abs(self.menu_jaw - self.q["gripper"]) < 1e-6:
            self.menu_jaw = None

    def hold(self) -> None:
        """Cancel every in-flight menu or hand motion at the current pose."""

        let_go = getattr(self, "let_go", None)
        if let_go is not None:
            let_go()
        self.homing = False
        self.menu_jaw = None
        if hasattr(self, "jaw"):
            self.jaw = self.q["gripper"]

    def begin_home(self) -> None:
        """Leave the current gesture and start a speed-capped return to the ready pose."""

        self.hold()
        self.homing = True

    def step_home(self, dt: float, others: list[np.ndarray] | None = None) -> dict[str, float]:
        """One collision-aware joint step home; the jaw stays as it is so an item is not dropped."""

        most = self.speed * dt
        candidate = dict(self.q)
        for joint in ARM_JOINTS:
            delta = self.home_q[joint] - self.q[joint]
            candidate[joint] += min(max(delta, -most), most)
        stopped = None
        if others and self.placement is not None:
            gap = min(lines_apart(self.centre_line(candidate), other) for other in others)
            now = min(lines_apart(self.centre_line(), other) for other in others)
            if gap < CLEARANCE_CM and gap < now:
                candidate, stopped = dict(self.q), "the other arm"
        self.q = candidate
        remaining = max(abs(self.home_q[joint] - self.q[joint]) for joint in ARM_JOINTS)
        self.homing = remaining > 1e-6
        self.mode = "moving" if self.homing else "holding"
        self.state = "returning home" + (f": at {stopped}" if stopped else "") if self.homing else "home"
        self.blocked = [] if stopped is None else ["home"]
        if not self.homing:
            self.target = self.tcp_cm()
        return self.q

    def engage(self, neutral, wearer) -> None:
        """Calibrated: the hand rests at ``neutral``, and ``wearer`` is forward, right and up."""

        self.neutral = (np.asarray(neutral, dtype=float), np.asarray(wearer, dtype=float))
        self.point, self.target = None, self.tcp_cm()
        self.hand_roll, self.grip, self.fist = None, None, False
        self.pinches.clear()
        self.jaw_want = None
        self.centred, self.lost_s, self.state, self.mode = True, 0.0, "holding", "holding"

    def release(self) -> None:
        """Back to calibrating: the arm holds."""

        self.neutral = self.point = self.target = None
        self.hand_roll, self.grip, self.fist = None, None, False
        self.pinches.clear()
        self.state = self.mode = "calibrating"

    def centre_line(self, q: dict[str, float] | None = None) -> np.ndarray | None:
        """This arm's centre line (see Kinematics.links) on the sheet, cm; z is its own. None unplaced."""

        if self.placement is None:
            return None
        pose = self.q if q is None else q
        points = self.kinematics.links({joint: pose[joint] for joint in ARM_JOINTS}) * 100
        return np.array([[*self.placement.to_sheet(point[:2]), point[2]] for point in points])

    def update(self, hand: dict | None, dt: float, others: list[np.ndarray] | None = None,
               head: dict | None = None) -> dict[str, float]:
        """Where to send the arm now, given the hand (None or untracked: hold where it is).

        ``others``: the other arms' centre lines on the sheet (centre_line), kept
        CLEARANCE_CM away; without them the arm keeps to its own side. ``head``:
        the glasses' pose (unused here: the calibration fixed the wearer's frame).
        """

        if self.neutral is None:
            return self.q
        self.clock_s += dt
        if not hand or not hand.get("tracked"):
            self.lost_s += dt
            self.point, self.hand_roll, self.grip, self.fist = None, None, None, False
            self.pinches.clear()
            self.state, self.mode, self.blocked = "no hand", "lost", []
            return self.q
        if self.lost_s > GRACE_S:
            self.centred, self.jaw_held, self.jaw_want = False, True, None
        self.lost_s = 0.0
        wrist = np.asarray(hand["wrist"], dtype=float)
        self.point = wrist if self.point is None else SMOOTHING * wrist + (1 - SMOOTHING) * self.point
        start, wearer = self.neutral
        seen = wearer @ (self.point - start)  # forward, right, up of the neutral
        forward, right, up = seen
        offset = np.array([forward, -right, up] if self.facing == "same" else [-forward, right, up])
        if not self.centred:
            if np.any(np.abs(offset) > self.dead_zone):
                self.state, self.mode, self.blocked = f"back to the middle: {way_back(seen)}", "centring", []
                return self.q
            self.centred, self.target = True, self.tcp_cm()
        self.curl = curl(hand)
        if self.curl is not None and self.curl < FIST[0] and not self.fist:
            self.fist, self.grip, self.jaw_held, self.jaw_want = True, None, True, None
            self.pinches.clear()  # thumb and index met on the way into the fist: that was no pinch
        elif self.curl is not None and self.curl > FIST[1] and self.fist:
            self.fist, self.grip = False, None
        measured = hand_angles(hand, wearer)
        if measured is not None:
            self.hand_roll = measured[0] if self.hand_roll is None else \
                self.hand_roll + ORIENTATION_SMOOTHING * angle_delta(measured[0], self.hand_roll)
        velocity = np.zeros(3)
        if self.fist:
            if self.grip is None and self.hand_roll is not None:
                self.grip = (self.hand_roll, self.roll)
            if self.grip is not None and self.hand_roll is not None:
                low, high = self.limits.get("wrist_roll", (-math.inf, math.inf))
                self.roll = min(max(self.grip[1] + angle_delta(self.hand_roll, self.grip[0]), low), high)
            self.state, self.mode = "turning the jaw", "turning"
        else:
            velocity = self.gain * past_dead_zone(offset, self.dead_zone)
            fastest = float(np.linalg.norm(velocity))
            if fastest > self.top_speed:
                velocity *= self.top_speed / fastest
            self.state = f"moving {heading(seen, self.dead_zone)}" if fastest > 0 else "holding"
            self.mode = "moving" if fastest > 0 else "holding"
        tcp = self.tcp_cm()
        own_side = not (others and self.placement is not None)
        target, stopped = self.within_reach(self.target + velocity * dt, own_side=own_side)
        lead = float(np.linalg.norm(target - tcp))
        if lead > LEAD_CM:
            target, _ = self.within_reach(tcp + (target - tcp) * (LEAD_CM / lead), own_side=own_side)
        stopped = self.advance(target, dt, others, stopped, float(np.linalg.norm(velocity)) * dt)
        self.blocked = []
        if self.state.startswith("moving") and stopped:
            self.state += f": at {stopped}"
            self.blocked = {"the table": ["down"], "the top": ["up"]}.get(stopped, heading_words(seen, self.dead_zone))
        if not self.fist:
            self.pinches.append((self.clock_s, gripper_for(float(np.linalg.norm(
                np.asarray(hand["thumb"], dtype=float) - np.asarray(hand["index"], dtype=float))))))
            while self.pinches and self.clock_s - self.pinches[0][0] >= JAW_DELAY_S - 1e-9:
                self.jaw_want = self.pinches.popleft()[1]
            if self.jaw_want is not None:
                if self.jaw_held and abs(self.jaw_want - self.q["gripper"]) <= CATCH_PCT:
                    self.jaw_held = False
                if not self.jaw_held:
                    most = GRIPPER_SPEED * dt
                    self.q["gripper"] += min(max(self.jaw_want - self.q["gripper"], -most), most)
        if self.jaw_held:
            self.state += ", jaw waits for a pinch" if not self.fist else ""
        return self.q

    def advance(self, target: np.ndarray, dt: float, others: list[np.ndarray] | None, stopped: str | None,
                expected_cm: float) -> str | None:
        """One step towards ``target`` (within reach already) -- none that brings the arms nearer than
        CLEARANCE_CM -- and what stopped the arm, if anything: ``stopped`` so far, the other arm, or a
        joint at its limit where the jaw should have moved ``expected_cm`` (a noticeable step) and barely did."""

        tcp = self.tcp_cm()
        self.target = target
        step = self.step_to(self.target, dt, roll=self.roll)
        if others and self.placement is not None:
            gap = min(lines_apart(self.centre_line(step), other) for other in others)
            if gap < CLEARANCE_CM and gap < min(lines_apart(self.centre_line(), other) for other in others):
                step, self.target, stopped = dict(self.q), tcp, "the other arm"  # no nearer: hold
        moved = float(np.linalg.norm(self.kinematics.tcp({joint: step[joint] for joint in ARM_JOINTS}) * 100 - tcp))
        if not stopped and self.pinned and expected_cm > 0.02 and moved < 0.2 * expected_cm:
            stopped = "a joint limit"
        self.q = step
        return stopped

    def within_reach(self, target: np.ndarray, own_side: bool = True) -> tuple[np.ndarray, str | None]:
        """The nearest point the arm may go -- its reach from the base, above the table, and with
        ``own_side`` its own side -- and what stopped the target there, if anything did."""

        stopped = None
        axis = self.kinematics.pan_axis * 100
        radial = target[:2] - axis
        distance = float(np.linalg.norm(radial))
        if distance > 1e-6:
            kept = min(max(distance, REACH_CM[0]), REACH_CM[1])
            if kept != distance:
                stopped = "full reach" if distance > REACH_CM[1] else "the base"
            radial = radial / distance * kept
        x, y = axis + radial
        if own_side:
            side = max(y, -SIDE_CM) if self.toward_other < 0 else min(y, SIDE_CM)
            if side != y:
                stopped = "the other arm's side"
            y = side
        table = self.placement.table_height(x, y) if self.placement is not None else 0.0
        z = min(max(float(target[2]), table + HEIGHT_CM[0]), table + HEIGHT_CM[1])
        if z != float(target[2]):
            stopped = "the table" if float(target[2]) < z else "the top"
        return np.array([x, y, z]), stopped

    def step_to(self, target_cm: np.ndarray, dt: float, roll: float | None = None) -> dict[str, float]:
        """One damped least-squares step of the steered joints towards ``target_cm``, fingers down, speed capped.

        ``roll`` is where the wrist roll goes, at the same speed; it turns the
        jaw about its own axis, which pointing down does not see.
        """

        arm = {joint: self.q[joint] for joint in ARM_JOINTS}
        most = self.speed * dt
        if roll is not None:
            if "wrist_roll" in self.limits:
                roll = min(max(roll, self.limits["wrist_roll"][0]), self.limits["wrist_roll"][1])
            arm["wrist_roll"] += min(max(roll - arm["wrist_roll"], -most), most)
        goal = np.asarray(target_cm, dtype=float) / 100
        down = np.array([0.0, 0.0, -1.0])

        def residual(pose: dict[str, float]) -> np.ndarray:
            fingers, _ = self.kinematics.pointing(pose)
            return np.concatenate([self.kinematics.tcp(pose) - goal, POINTING_WEIGHT * (fingers - down)])

        now = residual(arm)
        jacobian = np.zeros((len(now), len(STEERED)))
        for column, joint in enumerate(STEERED):
            nudged = dict(arm, **{joint: arm[joint] + 0.5})
            jacobian[:, column] = (residual(nudged) - now) / math.radians(0.5)
        # A joint at a limit that the step would push past it sits the step out, and the
        # others make up for it: clipped afterwards, it would leave them overshooting.
        free = np.ones(len(STEERED), dtype=bool)
        self.pinned = []
        for _ in range(len(STEERED)):
            used = jacobian * free
            step = -np.linalg.solve(used.T @ used + DAMPING ** 2 * np.eye(len(STEERED)), used.T @ now) * free
            pinned = [column for column, joint in enumerate(STEERED) if free[column] and joint in self.limits
                      and ((arm[joint] <= self.limits[joint][0] and step[column] < 0)
                           or (arm[joint] >= self.limits[joint][1] and step[column] > 0))]
            if not pinned:
                break
            free[pinned] = False
            self.pinned += [STEERED[column] for column in pinned]
        q = dict(self.q, wrist_roll=arm["wrist_roll"])
        for column, joint in enumerate(STEERED):
            value = arm[joint] + min(max(math.degrees(step[column]), -most), most)
            if joint in self.limits:
                value = min(max(value, self.limits[joint][0]), self.limits[joint][1])
            q[joint] = value
        return q


def roll_sense(kinematics, pose: dict[str, float]) -> float:
    """+1 if raising wrist_roll turns the jaw about where it points by the right-hand rule, -1 the other way.

    Pointing down, a right-hand turn is clockwise seen from above -- as the
    overhead camera shows it (on the venue's arms raising wrist_roll turns
    the jaw anticlockwise: -1).
    """

    arm = {joint: pose[joint] for joint in ARM_JOINTS}
    pointing, across = kinematics.pointing(arm)
    _, turned = kinematics.pointing(dict(arm, wrist_roll=arm["wrist_roll"] + 5.0))
    return 1.0 if float(np.cross(across, turned) @ pointing) > 0 else -1.0


class PinchFollower(Follower):
    """One arm dragged by one hand's pinch, on the model: thumb and index together grab its jaw.

    A pinch does one thing, whichever the hand does first (TWIST_DEG, DRAG_CM):

    * Moved, it drags: the jaw goes where it was when the pinch closed, plus
      --scale times as far as the pinch has gone since -- forward, right and
      up being where the wearer then looked, level.
    * Twisted in place, like a screwdriver, it turns the jaw the same way
      about where it points, the first TWIST_DEG aside: clockwise as the
      wearer sees the back of the hand is clockwise from above, as the
      overhead camera shows the jaw. The jaw stays where it is meanwhile.

    Let go, and the arm stays; pinch again anywhere to go on (a clutch, as VR
    teleoperation does it) -- to drag further, or turn further. Thumb and
    pinky together open a closed jaw, or close an open one.
    """

    calibrates = False

    def __init__(self, name: str, kinematics, placement, pose: dict[str, float], *, scale: float = SCALE,
                 **options) -> None:
        super().__init__(name, kinematics, placement, pose, **options)
        self.scale = scale
        self.smooth = OneEuro()
        self.pinched = False
        self.drag: tuple | None = None  # pinch point, wearer's frame, jaw, hand roll, wrist roll: when it closed
        self.jaw = self.q["gripper"]  # where the jaw is going, open or closed
        self.toggle_armed = True  # thumb and pinky parted since the last toggle
        self.sense = roll_sense(kinematics, self.q)
        self.start_roll = self.q["wrist_roll"]
        self.gesture: str | None = None  # what this pinch does: "dragging" or "turning", once it is clear
        self.target = self.tcp_cm()
        self.state = self.mode = "free"

    def let_go(self) -> None:
        if self.drag is not None:
            self.drag, self.target = None, self.tcp_cm()  # stop where it is, not where it was headed
        self.gesture = None

    def update(self, hand: dict | None, dt: float, others: list[np.ndarray] | None = None,
               head: dict | None = None) -> dict[str, float]:
        """Where to send the arm now, given the hand and the glasses' pose (see Follower.update)."""

        self.blocked, stopped = [], None
        if not hand or not hand.get("tracked"):
            self.let_go()
            self.pinched = False
            self.smooth.reset()
            self.state, self.mode = "no hand", "lost"
        else:
            thumb, index = (np.asarray(hand[key], dtype=float) for key in ("thumb", "index"))
            gap = float(np.linalg.norm(thumb - index))
            told = hand.get("pinch")  # the glasses' own pinch detection, when the Lens sends it
            self.pinched = told if isinstance(told, bool) else gap < GRAB_CM[1 if self.pinched else 0]
            point = self.smooth((thumb + index) / 2, dt)
            pinky = hand.get("pinkyTip")
            if pinky is not None and not self.pinched:
                reach = float(np.linalg.norm(thumb - np.asarray(pinky, dtype=float)))
                if self.toggle_armed and reach < TOGGLE_CM[0]:
                    self.jaw, self.toggle_armed = (0.0 if self.jaw > OPEN / 2 else OPEN), False
                elif reach > TOGGLE_CM[1]:
                    self.toggle_armed = True
            if self.pinched and self.drag is None and head:
                wearer = facing_frame(head["look"], head["p"], point)
                measured = hand_angles(hand, wearer)
                self.drag = (point, wearer, self.tcp_cm(), None if measured is None else measured[0], self.roll)
            if self.pinched and self.drag is not None:
                start, wearer, jaw_at, hand_roll_at, roll_at = self.drag
                seen = wearer @ (point - start)
                measured = hand_angles(hand, wearer)
                turn = angle_delta(measured[0], hand_roll_at) if measured is not None and hand_roll_at is not None \
                    else 0.0
                if self.gesture is None:  # the pinch's first clear motion says what it does
                    if abs(turn) > TWIST_DEG:
                        self.gesture = "turning"
                    elif float(np.linalg.norm(seen)) > DRAG_CM:
                        self.gesture = "dragging"
                elif self.gesture == "dragging" and abs(turn) > LATE_TWIST[0] \
                        and float(np.linalg.norm(seen)) < LATE_TWIST[1]:
                    self.gesture = "turning"  # a twist after all: the "drag" was the pinch swinging with it
                wanted, wrist_stopped = jaw_at, False
                if self.gesture == "turning":
                    low, high = self.limits.get("wrist_roll", (-math.inf, math.inf))
                    past = math.copysign(max(abs(turn) - TWIST_DEG, 0.0), turn)
                    self.roll = min(max(roll_at + self.sense * past, low), high)
                    wrist_stopped = self.roll != roll_at + self.sense * past
                elif self.gesture == "dragging":
                    forward, right, up = seen
                    moved = np.array([forward, -right, up] if self.facing == "same" else [-forward, right, up])
                    wanted = jaw_at + self.scale * moved
                target, stopped = self.within_reach(wanted, own_side=not (others and self.placement is not None))
                tcp = self.tcp_cm()
                stopped = self.advance(target, dt, others, stopped, min(float(np.linalg.norm(target - tcp)), 0.2))
                self.state = self.gesture or "pinched"
                self.mode = {"turning": "turning", "dragging": "moving"}.get(self.gesture, "holding")
                if wrist_stopped:  # the left arm, say, turns only 77 degrees clockwise from its ready pose
                    self.state += ": at the wrist's limit"
                if stopped and self.gesture == "dragging":
                    self.state += f": at {stopped}"
                    push = wanted - tcp  # the arm's frame; back into the wearer's words
                    seen = np.array([push[0], -push[1], push[2]] if self.facing == "same" else [-push[0], push[1], push[2]])
                    self.blocked = {"the table": ["down"], "the top": ["up"]}.get(stopped, heading_words(seen, 1.0))
            else:
                self.let_go()
                self.state, self.mode = "free", "holding"
        most = GRIPPER_SPEED * dt
        self.q["gripper"] += min(max(self.jaw - self.q["gripper"], -most), most)
        clockwise = self.sense * (self.q["wrist_roll"] - self.start_roll)  # seen from above
        if abs(clockwise) >= 5.0:
            self.state += f", turned {abs(clockwise):.0f}° {'cw' if clockwise > 0 else 'ccw'}"
        if self.jaw < OPEN / 2:
            self.state += ", jaw closed"
        return self.q

    def guide(self) -> dict:
        """What the glasses show for this arm's hand: how it fares, where it cannot go, the finger's height."""

        clockwise = self.sense * (self.q["wrist_roll"] - self.start_roll)
        return {"mode": self.mode, "state": self.state, "blocked": list(self.blocked),
                "tip": round(self.tip_height()), "jaw": round(self.q["gripper"]),
                "roll": round(self.q["wrist_roll"]), "turn": round(clockwise)}

    def command_jaw(self, opened: bool | None = None) -> None:
        if opened is None:
            opened = self.jaw < OPEN / 2
        self.jaw = OPEN if opened else 0.0
        self.menu_jaw = None


def ready_pose(kinematics, placement, neutral: dict[str, float],
               limits: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
    """Where an arm waits for its hand: fingers down over the table, READY_CM ahead of its base."""

    x, y, z = READY_CM
    height = (placement.table_height(x, y) if placement is not None else 0.0) + z
    bounds = limits or kinematics.own_limits()

    def room(degrees: dict[str, float]) -> float:
        return min(min(value - bounds[joint][0], bounds[joint][1] - value) for joint, value in degrees.items()
                   if joint in bounds and joint in STEERED)

    best = None
    for lean in (0.0, 15.0, 30.0, 45.0):  # leaning a little reaches this high, and leaves the wrist room
        solution = kinematics.solve((x / 100, y / 100, height / 100), start=neutral, limits=limits, lean_deg=lean)
        score = (solution.reachable, min(room(solution.degrees), READY_ROOM_DEG), -solution.position_error_m)
        if best is None or score > best[0]:
            best = (score, solution)
        if solution.reachable and room(solution.degrees) >= READY_ROOM_DEG:
            break
    return dict(neutral, **best[1].degrees)


def follow(followers: dict[str, Follower], hands: Hands, arms: dict | None = None, *, facing: str = "same",
           hold_s: float = HOLD_S, log=print, clock=time.monotonic, sleep=time.sleep,
           shutdown: threading.Event | None = None) -> None:
    """Forever, RATE_HZ times a second: calibrate if the followers need it, then every arm a step as its hand
    steers. Ctrl+C ends it."""

    from .servo import MOTORS

    sides = {name: HAND_FOR[facing][name] for name in followers}  # arm -> the hand that drives it
    calibration = Calibration(sorted(set(sides.values())), hold_s)
    period, last, said, reported, unseen = 1.0 / RATE_HZ, clock(), None, -math.inf, 0.0
    stopped, home_queue = False, []
    while shutdown is None or not shutdown.is_set():
        now = clock()
        dt, last = min(max(now - last, 0.0), 0.1), now
        for command in hands.take_commands():
            action = command["command"]
            if action == "stop":
                stopped, home_queue = True, []
                for follower in followers.values():
                    follower.hold()
            elif action == "hold":
                stopped, home_queue = bool(command.get("enabled")), []
                if stopped:
                    for follower in followers.values():
                        follower.hold()
            elif not stopped and action == "precision":
                precise = bool(command.get("enabled"))
                for follower in followers.values():
                    if hasattr(follower, "scale"):
                        follower.scale = 0.5 if precise else SCALE
            elif not stopped and action == "jaw":
                opened = command.get("open")
                for follower in followers.values():
                    follower.command_jaw(opened if isinstance(opened, bool) else None)
            elif not stopped and action == "home":
                home_queue = list(followers)
                for follower in followers.values():
                    follower.begin_home()
        message, age = hands.latest()
        fresh = message if message is not None and age <= STALE_S else {}
        hand_of = {name: fresh.get(side) for name, side in sides.items()}
        seen = {name: bool(hand and hand.get("tracked") and hand.get("wrist")) for name, hand in hand_of.items()}
        engaged = all(not follower.calibrates or follower.neutral is not None for follower in followers.values())
        if engaged and any(follower.calibrates for follower in followers.values()):
            unseen = 0.0 if any(seen.values()) else unseen + dt
            if unseen >= RECALIBRATE_S:
                for follower in followers.values():
                    follower.release()
                calibration.reset()
                engaged = False
        if not engaged:
            neutral = calibration.update({sides[name]: hand_of[name]["wrist"] if seen[name] else None
                                          for name in followers}, dt)
            head = fresh.get("head")
            if neutral is not None and head:
                if "left" in neutral and "right" in neutral:
                    wearer = body_frame(neutral["left"], neutral["right"], head["look"], head["p"])
                else:  # one hand says nothing of which way right is: the gaze does
                    wearer = facing_frame(head["look"], head["p"], np.mean(list(neutral.values()), axis=0))
                for name, follower in followers.items():
                    follower.engage(neutral[sides[name]], wearer)
                engaged, unseen = True, 0.0
        lines = {name: follower.centre_line() for name, follower in followers.items()}
        active_home = home_queue[0] if home_queue else None
        for name, follower in followers.items():
            others = [line for other, line in lines.items() if other != name and line is not None]
            if stopped:
                follower.state, follower.mode, follower.blocked = "stopped", "holding", []
                q = follower.q
            elif active_home is not None:
                if name == active_home:
                    q = follower.step_home(dt, others=others or None)
                else:
                    follower.state, follower.mode, follower.blocked = "waiting while the other arm goes home", "holding", []
                    q = follower.q
            else:
                q = follower.update(hand_of[name], dt, others=others or None, head=fresh.get("head"))
            follower.finish_commands(dt)
            q = follower.q
            lines[name] = follower.centre_line()
            if arms:
                arm = arms[name]
                arm.bus.write_goals({MOTORS[joint]: arm.to_ticks(joint, value) for joint, value in q.items()})
        if active_home is not None and not followers[active_home].homing:
            home_queue.pop(0)
        if engaged:
            status = " | ".join(f"{name}: {follower.state}" for name, follower in followers.items())
        else:
            status = calibration.status
        if not arms:
            status += " (dry run)"
        hands.status = status
        arm_guides = {name: follower.guide() for name, follower in followers.items()}
        guides = {sides[name]: arm_guides[name] for name in followers}
        scale = next((follower.scale for follower in followers.values() if hasattr(follower, "scale")), None)
        hands.report = json.dumps({"status": status, "hands": {side: guide for side, guide in guides.items() if guide},
                                   "arms": {name: guide for name, guide in arm_guides.items() if guide},
                                   "mode": "pinch" if scale is not None else "joystick", "scale": scale,
                                   "stopped": stopped}, separators=(",", ":"))
        if status != said:
            said = status
            log(status + ("" if hands.connected else "; the glasses are not connected"))
        if not arms and engaged and any(seen.values()) and now - reported >= 0.5:
            reported = now  # a dry run says where the arms would go, and how folded each hand is
            log("  " + ", ".join(f"{name}: jaw at {f.tcp_cm()[0]:.0f} {f.tcp_cm()[1]:.0f} {f.tcp_cm()[2]:.0f} cm, "
                                 f"roll {f.q['wrist_roll']:.0f}°, {f.q['gripper']:.0f}% open"
                                 + (f", curl {f.curl:.2f}" if f.curl is not None else "")
                                 for name, f in followers.items() if seen[name]))
        sleep(max(0.0, period - (clock() - now)))


def _others_on(port: str) -> list[str]:
    """Other programs with this serial port open (lsof, where there is one)."""

    import os
    import shutil
    import subprocess

    if not shutil.which("lsof"):
        return []
    found = subprocess.run(["lsof", "-t", port], capture_output=True, text=True).stdout.split()
    return [pid for pid in found if pid != str(os.getpid())]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trashdrop.spectacles", description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="no arms: the link, and where the arms would go")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--video", action="store_true", help="stream the overhead webcam to the glasses")
    parser.add_argument("--video-port", type=int, default=VIDEO_PORT)
    parser.add_argument("--camera", default="auto", help="video source; auto identifies the USB webcam")
    parser.add_argument("--video-fps", type=float, default=VIDEO_FPS,
                        help=f"frames a second sent at most (default {VIDEO_FPS})")
    parser.add_argument("--video-width", type=int, default=VIDEO_WIDTH, help=f"pixels (default {VIDEO_WIDTH})")
    parser.add_argument("--video-quality", type=int, default=VIDEO_QUALITY,
                        help=f"JPEG quality (default {VIDEO_QUALITY})")
    parser.add_argument("--mode", choices=("pinch", "joystick"), default="pinch",
                        help="pinch: thumb and index grab the jaw and drag it, thumb and pinky open or close it; "
                             "joystick: calibrate, then a hand past its neutral drives its jaw (default pinch)")
    parser.add_argument("--scale", type=float, default=SCALE,
                        help=f"--mode pinch: jaw cm per hand cm (default {SCALE:g})")
    parser.add_argument("--hold", type=float, default=HOLD_S,
                        help=f"seconds the hands are held level and still to calibrate (default {HOLD_S:g})")
    parser.add_argument("--dead-zone", type=float, default=DEAD_ZONE_CM,
                        help=f"cm a hand may stray from its neutral, on each axis, moving nothing (default {DEAD_ZONE_CM:g})")
    parser.add_argument("--gain", type=float, default=GAIN,
                        help=f"cm/s the jaw moves per cm the hand is past the dead zone (default {GAIN:g})")
    parser.add_argument("--top-speed", type=float, default=TOP_SPEED,
                        help=f"cm/s the jaw moves at most (default {TOP_SPEED:g})")
    parser.add_argument("--speed", type=float, default=SPEED, help=f"deg/s a joint turns at most (default {SPEED:g})")
    parser.add_argument("--facing", choices=("same", "them"), default="same",
                        help="same: standing behind the arms, facing their way; them: in front, facing them")
    parser.add_argument("--arm", choices=("left", "right"), default=None, help="only this arm follows")
    parser.add_argument("--no-record", action="store_true",
                        help="do not keep what the glasses send (kept by default in out/spectacles/, to replay)")
    args = parser.parse_args(argv)
    if not (0 < args.speed <= 200 and 0 < args.gain <= 10 and 0 < args.top_speed <= 30
            and 0 <= args.dead_zone <= 10 and 0.5 <= args.hold <= 30 and 0 < args.scale <= 3):
        print("--speed above 0 and at most 200, --gain above 0 and at most 10, --top-speed above 0 and at most 30, "
              "--dead-zone 0 to 10, --hold 0.5 to 30, --scale above 0 and at most 3")
        return 1
    if not (1 <= args.video_fps <= 60 and 160 <= args.video_width <= 1920 and 20 <= args.video_quality <= 95):
        print("--video-fps 1 to 60, --video-width 160 to 1920, --video-quality 20 to 95")
        return 1
    if args.video and args.video_port == args.port:
        print("hand and video WebSockets need different ports")
        return 1

    from .arm import connect, load_poses, move_together
    from .kinematics import Kinematics
    from .placement import Placement
    from .rig import load_rig

    rig, poses = load_rig(), load_poses()
    names = [args.arm] if args.arm else ["left", "right"]
    kinematics = {name: Kinematics(rig.arms[name].wrist_roll_offset) for name in names}
    placements = {name: Placement(*rig.arms[name].sheet) if rig.arms[name].sheet else None for name in names}
    folder = Path(__file__).resolve().parent.parent / "out" / "spectacles"
    folder.mkdir(parents=True, exist_ok=True)
    record = None
    if not args.no_record:
        path = folder / f"session-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        record = path.open("w", buffering=1)  # a line at a time: a crash loses nothing
        print(f"keeping what the glasses send in {path}")
    hands = Hands(record=record, snapshots=Snapshots(folder / "snaps", folder / "snap"))
    video = video_server = None
    if args.video:
        from .__main__ import _require_same_camera, _resolve_camera, _webcam
        from .dataset.capture import open_camera
        from .dataset.zone import DEFAULT_CALIBRATION, load_zone

        try:
            webcam = _webcam()
            if args.camera == "auto" and webcam is None:
                raise RuntimeError("cannot identify the USB webcam; check rig.toml or pass an explicit --camera")
            source = _resolve_camera(args.camera, VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT)
            capture = open_camera(source, VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT)
            _require_same_camera(capture, source, webcam, refuse=True)
            zone = load_zone(DEFAULT_CALIBRATION)
            reference_size = (zone.frame_width, zone.frame_height) if zone else (0, 0)
            video = VideoFrames(capture, fps=args.video_fps, width=args.video_width, quality=args.video_quality,
                                tape_pixels=rig.tape_pixels, reference_size=reference_size)
            video_server = make_video_server(video, "0.0.0.0", args.video_port)
            video.start()
            threading.Thread(target=video_server.serve_forever, daemon=True).start()

            def report_video() -> None:
                while video.running:
                    time.sleep(VIDEO_REPORT_S)
                    if video.running:
                        print(video.report())

            threading.Thread(target=report_video, name="spectacles-video-report", daemon=True).start()
        except (OSError, RuntimeError, ValueError) as error:
            if video_server is not None:
                video_server.server_close()
            if video is not None:
                video.close()
            print(f"cannot start video: {error}")
            return 1
    try:
        server = make_server(hands, "0.0.0.0", args.port)
    except OSError as error:
        if video_server is not None:
            video_server.shutdown()
            video_server.server_close()
        if video is not None:
            video.close()
        print(f"cannot listen on port {args.port}: {error}")
        return 1
    threading.Thread(target=server.serve_forever, daemon=True).start()
    wifi = " or ".join(f"ws://{ip}:{args.port}" for ip in addresses())
    tunnel = usb_tunnel(args.port)
    if tunnel:
        print(f"listening. The glasses are on the USB cable: in the Lens, set Url to {tunnel} "
              f"(over Wi-Fi instead: {wifi})")
    else:
        print(f"listening. In the Lens, set Url to {wifi} (or plug the glasses in by USB and start again)")
    if video_server is not None:
        video_tunnel = usb_tunnel(args.video_port)
        video_url = video_tunnel or " or ".join(f"ws://{ip}:{args.video_port}" for ip in addresses())
        print(f"video: {video_url} (up to {args.video_width}px, {args.video_fps:g} fps, "
              f"JPEG quality {args.video_quality}; what it does every {VIDEO_REPORT_S:g} s)")
    arms, speeds = {}, {}
    try:
        if args.dry_run:
            limits = {name: kinematics[name].own_limits() for name in names}
        else:
            arms = {name: connect(name, rig) for name in names}
            for name, arm in arms.items():
                others = _others_on(arm.bus.port)
                if others:
                    print(f"another program has the {name} arm's bus open (process {', '.join(others)} -- "
                          "`trashdrop web`?): stop it first")
                    return 1
            limits = {name: arm.limits_degrees() for name, arm in arms.items()}
            for name, arm in arms.items():
                speeds[name], arm.max_speed = arm.max_speed, args.speed
                arm.limit_speed()  # The first move into READY must obey --speed too.
        ready = {name: ready_pose(kinematics[name], placements[name], poses[name]["neutral"], limits[name])
                 for name in names}
        if arms:
            then = ("then pinch thumb and index to grab a jaw and drag it; thumb and pinky open or close it"
                    if args.mode == "pinch" else
                    f"then hold your hands up, level and still, for {args.hold:g} s; after that each steers its arm")
            input(f"{' and '.join(names)} go to their ready pose, fingers down over the table: keep clear; "
                  f"{then}. Ctrl+C holds them. Enter...")
            for arm in arms.values():
                if not arm.torque_is_on():
                    arm.torque_on()
            move_together([(arms[name], ready[name]) for name in names])
            ready = {name: arm.pose() for name, arm in arms.items()}
        if args.mode == "pinch":
            followers = {name: PinchFollower(name, kinematics[name], placements[name], ready[name], scale=args.scale,
                                             speed=args.speed, facing=args.facing, limits=limits[name])
                         for name in names}
        else:
            followers = {name: Follower(name, kinematics[name], placements[name], ready[name], speed=args.speed,
                                        facing=args.facing, limits=limits[name], dead_zone=args.dead_zone,
                                        gain=args.gain, top_speed=args.top_speed) for name in names}
        follow(followers, hands, arms or None, facing=args.facing, hold_s=args.hold)
    except KeyboardInterrupt:
        for arm in arms.values():
            arm.hold()
        print("\nstopped; the arms hold where they are")
        return 130
    except (ValueError, RuntimeError) as error:
        print(error)
        return 1
    finally:
        server.shutdown()
        server.server_close()
        hands.stop_recording()
        if video_server is not None:
            video_server.shutdown()
            video_server.server_close()
        if video is not None:
            video.close()
        for name, arm in arms.items():
            if name in speeds:
                arm.max_speed = speeds[name]
                try:
                    arm.limit_speed()
                except Exception:  # a bus that went away must not hide what happened first
                    pass
            arm.bus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
