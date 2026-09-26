"""Snap Spectacles drive the arms: each hand moves the arm on its side, a pinch closes its jaw.

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

How a hand drives its arm:

* Relative, like a mouse. When a hand comes into view, it and its arm are
  anchored where they are; from then on the arm's fingertips move as the
  point between thumb and index tip moves (--scale times as far). A hand
  that drops out of view holds its arm, and is anchored afresh when it comes
  back, so nothing jumps.
* Forward is where the wearer looked, level, when the hand was anchored;
  the hands are always in front of the face, which settles which way round
  the camera's axis is. The wearer stands behind the arms, facing their
  way; with --facing them, in front of them facing back, each hand drives
  the arm on its side as the wearer sees it, and a hand moved towards the
  arms moves its jaw towards their bases.
* Thumb and index tip apart open the jaw; together they close it.
* Tilt the fingers up or down to tilt the jaw (wrist flex); turn the palm
  about the finger direction to turn the jaw (wrist roll). Both are relative
  to the hand orientation when it first appears, so the arm does not jump.
* A target stays where its arm can reach, above the table and on its own
  side, clear of the other arm.

Each tick (50 Hz) every arm takes one damped least-squares step towards its
target on the model -- far quicker than solving afresh, and a target out of
reach just leaves the arm at the nearest it gets -- no joint turning faster
than --speed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import socket
import socketserver
import struct
import threading
import time

import numpy as np

PORT = 8765
VIDEO_PORT = 8766
VIDEO_FPS = 10
VIDEO_CAPTURE_WIDTH = 800
VIDEO_CAPTURE_HEIGHT = 600
VIDEO_WIDTH = 640
VIDEO_QUALITY = 60
RATE_HZ = 50
SPEED = 120.0  # deg/s a joint turns at most while following
GRIPPER_SPEED = 150.0  # percent a second
SCALE = 1.0  # arm cm per hand cm
STALE_S = 0.3  # a hand not heard of for this long holds its arm
SMOOTHING = 0.5  # the share of each new hand position; the rest is the last: steadies the tracking's jitter
ORIENTATION_SMOOTHING = 0.4  # knuckles jitter while pinching; filter angles without averaging across the 180-degree seam
OPEN = 60.0  # percent open with thumb and index well apart
PINCH_CM = (2.0, 9.0)  # thumb tip to index tip: shut at the first or closer, open at the second or wider
READY_CM = (22.0, 0.0, 10.0)  # where each arm waits, in its own frame: ahead of its base, over the table
REACH_CM = (12.0, 38.0)  # from the base's turning axis
HEIGHT_CM = (3.0, 35.0)  # the TCP above the table
SIDE_CM = 14.0  # how far past its base's line towards the other arm each may go (the bases stand ~38 cm apart)
POINTING_WEIGHT = 0.01  # metres a radian: the fingers pointing down gives way to where the hand is
DAMPING = 0.02
READY_ROOM_DEG = 10.0  # the ready pose keeps every joint this far from its limits, room to follow
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
STEERED = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")  # positional fallback for an older Lens without knuckles
POSITION_JOINTS = STEERED[:3]  # once the hand commands the wrist, these three compensate to hold the TCP
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
    """The newest message from the glasses, and what to tell them back; shared between threads."""

    def __init__(self, clock=time.monotonic) -> None:
        self._lock = threading.Lock()
        self._message, self._at = None, -math.inf
        self.clock = clock
        self.status = "waiting for the arms"
        self.connected = False

    def put(self, message: dict) -> None:
        with self._lock:
            self._message, self._at = message, self.clock()

    def latest(self) -> tuple[dict | None, float]:
        """(the newest message, how many seconds old)."""

        with self._lock:
            return self._message, self.clock() - self._at


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
                        hands.put(json.loads(text))
                    except ValueError:
                        continue
                    if hands.status != told:
                        told = hands.status
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

    def __init__(self, capture=None, *, tape_pixels=None, reference_size=(0, 0),
                 clock=time.monotonic, wall_clock=time.time) -> None:
        self.capture = capture
        self.tape_pixels = tape_pixels or {}
        self.reference_size = reference_size
        self.clock, self.wall_clock = clock, wall_clock
        self.condition = threading.Condition()
        self.sequence = 0
        self.packet: bytes | None = None
        self.running = True
        self.thread: threading.Thread | None = None

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
            captured_ms = round(self.wall_clock() * 1000)
            now = self.clock()
            if now < next_encode:
                continue  # Keep draining the camera so its buffer cannot grow stale.
            picture = crop_video(picture, self.tape_pixels, self.reference_size)
            jpeg = encode(picture, width=VIDEO_WIDTH, quality=VIDEO_QUALITY)
            self.publish(jpeg, captured_ms)
            next_encode = max(next_encode + 1 / VIDEO_FPS, now)

    def publish(self, jpeg: bytes, captured_ms: int) -> None:
        with self.condition:
            self.sequence += 1
            self.packet = json.dumps({"seq": self.sequence, "capturedMs": captured_ms,
                                      "jpeg": base64.b64encode(jpeg).decode("ascii")}, separators=(",", ":")).encode()
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


class Follower:
    """One arm following one hand, on the model: joint targets for the arm, tick by tick."""

    def __init__(self, name: str, kinematics, placement, pose: dict[str, float], *, scale: float = SCALE,
                 speed: float = SPEED, facing: str = "same", limits: dict[str, tuple[float, float]] | None = None):
        self.name, self.kinematics, self.placement = name, kinematics, placement
        self.q = dict(pose)  # joint -> degrees (gripper: percent open): what the arm is sent
        self.scale, self.speed, self.facing, self.limits = scale, speed, facing, limits or {}
        self.toward_other = -1.0 if name == "left" else 1.0  # the other arm is to the left arm's right (-y)
        self.anchor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None  # hand point, wearer's frame, TCP
        self.orientation_anchor: tuple[float, float, float, float] | None = None  # hand roll/pitch, joint roll/flex
        self.orientation: tuple[float, float] | None = None  # filtered hand roll/pitch
        self.point: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self.state = "no hand"

    def tcp_cm(self) -> np.ndarray:
        return self.kinematics.tcp({joint: self.q[joint] for joint in ARM_JOINTS}) * 100

    def update(self, hand: dict | None, head: dict | None, dt: float) -> dict[str, float]:
        """Where to send the arm now, given the hand (None or untracked: hold where it is)."""

        if not hand or not hand.get("tracked") or not head:
            self.anchor = self.point = self.target = None
            self.orientation_anchor = self.orientation = None
            self.state = "no hand"
            return self.q
        thumb, index = np.asarray(hand["thumb"], dtype=float), np.asarray(hand["index"], dtype=float)
        seen = (thumb + index) / 2
        self.point = seen if self.point is None else SMOOTHING * seen + (1 - SMOOTHING) * self.point
        if self.anchor is None:
            self.anchor = (self.point.copy(), facing_frame(head["look"], head["p"], self.point), self.tcp_cm())
        start, wearer, tcp = self.anchor
        forward, right, up = wearer @ (self.point - start)
        moved = [forward, -right, up] if self.facing == "same" else [-forward, right, up]
        self.target = self.within_reach(tcp + self.scale * np.array(moved))
        measured = hand_angles(hand, wearer)
        wrist_targets = None
        if measured is not None:
            if self.orientation_anchor is None:
                self.orientation_anchor = (measured[0], measured[1], self.q["wrist_roll"], self.q["wrist_flex"])
                self.orientation = measured
            else:
                self.orientation = tuple(previous + ORIENTATION_SMOOTHING * angle_delta(current, previous)
                                         for previous, current in zip(self.orientation, measured))
            roll_start, pitch_start, joint_roll, joint_flex = self.orientation_anchor
            wrist_targets = {"wrist_roll": joint_roll + angle_delta(self.orientation[0], roll_start),
                             "wrist_flex": joint_flex - angle_delta(self.orientation[1], pitch_start)}
        elif self.orientation_anchor is not None:
            wrist_targets = {joint: self.q[joint] for joint in ("wrist_flex", "wrist_roll")}
        self.q = self.step_to(self.target, dt, wrist_targets)
        most = GRIPPER_SPEED * dt
        want = gripper_for(float(np.linalg.norm(thumb - index)))
        self.q["gripper"] += min(max(want - self.q["gripper"], -most), most)
        self.state = "following"
        return self.q

    def within_reach(self, target: np.ndarray) -> np.ndarray:
        """The nearest point the arm may go: its reach from the base, its own side, above the table."""

        axis = self.kinematics.pan_axis * 100
        radial = target[:2] - axis
        distance = float(np.linalg.norm(radial))
        if distance > 1e-6:
            radial = radial / distance * min(max(distance, REACH_CM[0]), REACH_CM[1])
        x, y = axis + radial
        y = max(y, -SIDE_CM) if self.toward_other < 0 else min(y, SIDE_CM)
        table = self.placement.table_height(x, y) if self.placement is not None else 0.0
        z = min(max(float(target[2]), table + HEIGHT_CM[0]), table + HEIGHT_CM[1])
        return np.array([x, y, z])

    def step_to(self, target_cm: np.ndarray, dt: float,
                wrist_targets: dict[str, float] | None = None) -> dict[str, float]:
        """One damped least-squares step of the steered joints towards ``target_cm``, their speed capped."""

        arm = {joint: self.q[joint] for joint in ARM_JOINTS}
        most = self.speed * dt
        if wrist_targets is not None:
            for joint in ("wrist_flex", "wrist_roll"):
                wanted = wrist_targets[joint]
                if joint in self.limits:
                    wanted = min(max(wanted, self.limits[joint][0]), self.limits[joint][1])
                arm[joint] += min(max(wanted - arm[joint], -most), most)
        goal = np.asarray(target_cm, dtype=float) / 100
        down = np.array([0.0, 0.0, -1.0])
        steered = STEERED if wrist_targets is None else POSITION_JOINTS

        def residual(pose: dict[str, float]) -> np.ndarray:
            if wrist_targets is not None:
                return self.kinematics.tcp(pose) - goal
            fingers, _ = self.kinematics.pointing(pose)
            return np.concatenate([self.kinematics.tcp(pose) - goal, POINTING_WEIGHT * (fingers - down)])

        now = residual(arm)
        jacobian = np.zeros((len(now), len(steered)))
        for column, joint in enumerate(steered):
            nudged = dict(arm, **{joint: arm[joint] + 0.5})
            jacobian[:, column] = (residual(nudged) - now) / math.radians(0.5)
        # A joint at a limit that the step would push past it sits the step out, and the
        # others make up for it: clipped afterwards, it would leave them overshooting.
        free = np.ones(len(steered), dtype=bool)
        for _ in range(len(steered)):
            used = jacobian * free
            step = -np.linalg.solve(used.T @ used + DAMPING ** 2 * np.eye(len(steered)), used.T @ now) * free
            pinned = [column for column, joint in enumerate(steered) if free[column] and joint in self.limits
                      and ((arm[joint] <= self.limits[joint][0] and step[column] < 0)
                           or (arm[joint] >= self.limits[joint][1] and step[column] > 0))]
            if not pinned:
                break
            free[pinned] = False
        q = dict(self.q, wrist_flex=arm["wrist_flex"], wrist_roll=arm["wrist_roll"])
        for column, joint in enumerate(steered):
            value = arm[joint] + min(max(math.degrees(step[column]), -most), most)
            if joint in self.limits:
                value = min(max(value, self.limits[joint][0]), self.limits[joint][1])
            q[joint] = value
        return q


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
           log=print, clock=time.monotonic, sleep=time.sleep) -> None:
    """Forever, RATE_HZ times a second: every arm a step towards its hand. KeyboardInterrupt ends it."""

    from .servo import MOTORS

    period, last, said, reported = 1.0 / RATE_HZ, clock(), None, -math.inf
    while True:
        now = clock()
        dt, last = min(max(now - last, 0.0), 0.1), now
        message, age = hands.latest()
        fresh = message if message is not None and age <= STALE_S else {}
        for name, follower in followers.items():
            q = follower.update(fresh.get(HAND_FOR[facing][name]), fresh.get("head"), dt)
            if arms:
                arm = arms[name]
                arm.bus.write_goals({MOTORS[joint]: arm.to_ticks(joint, value) for joint, value in q.items()})
        status = ", ".join(f"{name} arm: {follower.state}" for name, follower in followers.items())
        if not arms:
            status += " (dry run)"
        hands.status = status
        if status != said:
            said = status
            log(status + ("" if hands.connected else "; the glasses are not connected"))
        if not arms and now - reported >= 0.5 and any(f.target is not None for f in followers.values()):
            reported = now  # a dry run says where the arms would go
            log("  " + ", ".join(f"{name}: jaw at {f.tcp_cm()[0]:.0f} {f.tcp_cm()[1]:.0f} {f.tcp_cm()[2]:.0f} cm, "
                                 f"flex {f.q['wrist_flex']:.0f}°, roll {f.q['wrist_roll']:.0f}°, "
                                 f"{f.q['gripper']:.0f}% open" for name, f in followers.items()
                                 if f.target is not None))
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
    parser.add_argument("--scale", type=float, default=SCALE, help=f"arm cm per hand cm (default {SCALE:g})")
    parser.add_argument("--speed", type=float, default=SPEED, help=f"deg/s a joint turns at most (default {SPEED:g})")
    parser.add_argument("--facing", choices=("same", "them"), default="same",
                        help="same: standing behind the arms, facing their way; them: in front, facing them")
    parser.add_argument("--arm", choices=("left", "right"), default=None, help="only this arm follows")
    args = parser.parse_args(argv)
    if not (0 < args.scale <= 3 and 0 < args.speed <= 200):
        print("--scale above 0 and at most 3, --speed above 0 and at most 200")
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
    hands = Hands()
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
            video = VideoFrames(capture, tape_pixels=rig.tape_pixels, reference_size=reference_size)
            video_server = make_video_server(video, "0.0.0.0", args.video_port)
            video.start()
            threading.Thread(target=video_server.serve_forever, daemon=True).start()
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
        print(f"video: {video_url} (up to {VIDEO_WIDTH}px, {VIDEO_FPS} fps, JPEG quality {VIDEO_QUALITY})")
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
            input(f"{' and '.join(names)} go to their ready pose, fingers down over the table: keep clear. "
                  "Then each follows its hand; Ctrl+C holds them. Enter...")
            for arm in arms.values():
                if not arm.torque_is_on():
                    arm.torque_on()
            move_together([(arms[name], ready[name]) for name in names])
            ready = {name: arm.pose() for name, arm in arms.items()}
        followers = {name: Follower(name, kinematics[name], placements[name], ready[name], scale=args.scale,
                                    speed=args.speed, facing=args.facing, limits=limits[name]) for name in names}
        follow(followers, hands, arms or None, facing=args.facing)
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
