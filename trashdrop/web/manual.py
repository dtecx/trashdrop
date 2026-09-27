"""Spectacles teleoperation owned by the web process.

The sorting cell and hand bridge must share the same arm objects: a Feetech
bus cannot safely be opened twice.  This manager keeps the Lens sockets and
overhead video alive, and reserves the cell worker while a manual session is
active.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from ..spectacles import (
    HOLD_S,
    PORT,
    SCALE,
    SPEED,
    VIDEO_FPS,
    VIDEO_PORT,
    VIDEO_QUALITY,
    VIDEO_WIDTH,
    Follower,
    Hands,
    PinchFollower,
    Snapshots,
    SpectatorFrames,
    VideoFrames,
    addresses,
    follow,
    make_server,
    make_video_server,
    ready_pose,
    usb_tunnel,
)


class _LatestFrameCapture:
    """Present a cell Camera as the small ``read/release`` API VideoFrames needs."""

    def __init__(self, camera) -> None:
        self.camera = camera
        self.shown_at = 0.0
        self.running = True

    def read(self):
        deadline = time.monotonic() + 0.1
        while self.running and time.monotonic() < deadline:
            picture, at = self.camera.latest()
            if picture is not None and at != self.shown_at:
                self.shown_at = at
                return True, picture
            time.sleep(0.005)
        return False, None

    def release(self) -> None:
        self.running = False


class ManualBridge:
    """The Lens sockets, video and one exclusive manual-control session."""

    def __init__(self, cell, folder: Path, *, hand_port: int = PORT, video_port: int = VIDEO_PORT,
                 video_fps: float = VIDEO_FPS, video_width: int = VIDEO_WIDTH,
                 video_quality: int = VIDEO_QUALITY) -> None:
        self.cell, self.folder = cell, Path(folder)
        self.hand_port, self.video_port = hand_port, video_port
        self.video_fps, self.video_width, self.video_quality = video_fps, video_width, video_quality
        self.folder.mkdir(parents=True, exist_ok=True)
        record_path = self.folder / f"session-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self.spectator = SpectatorFrames(log=self.cell.log)
        self.hands = Hands(record=record_path.open("w", buffering=1), snapshots=Snapshots(
            self.folder / "snaps", self.folder / "snap", log=self.cell.log), spectator=self.spectator)
        self.hands.presentation = False
        self.hands.command_handler = self._lens_command
        self.hands.context = self._report_context
        self.hand_server = self.video_server = self.video = None
        self._servers_started = False
        self.network_error: str | None = None
        self.urls: dict[str, str] = {}
        self._lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self.active = False
        self.starting = False
        self.mode = "pinch"
        self.scale = SCALE
        self.facing = "same"
        self.dry_run = True
        self.error: str | None = None
        self.control_error: str | None = None
        self._overlay_error_at = 0.0

    def _report_context(self) -> dict:
        """Tell the Lens which mode actually owns the arms, even without a follower."""

        manual = self.active or self.starting
        auto = self.cell.auto or self.cell.busy == "auto"
        context = {"manual": manual, "auto": auto, "busy": self.cell.busy,
                   "controlError": self.control_error,
                   "emptyPhotographed": getattr(self.cell, "background", None) is not None}
        if not manual:
            context.update(arms={}, hands={}, status=("auto sort: running" if auto else
                           "moving arms to neutral" if self.cell.busy == "neutral" else "ready"))
        return context

    def _lens_command(self, message: dict) -> bool:
        """Run Lens controls through the same cell owner as the web page."""

        action = message.get("command")
        if action not in ("presentation", "manual", "auto", "neutral", "empty"):
            return False
        with self._command_lock:
            self.control_error = self._run_lens_command(message)
        if self.control_error:
            self.cell.log("Spectacles control: " + self.control_error)
        return True

    def _wait_for_cell(self, timeout: float = 5.0) -> str | None:
        deadline = time.monotonic() + timeout
        while self.cell.busy is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        return f"busy: {self.cell.busy}" if self.cell.busy is not None else None

    def _run_lens_command(self, message: dict) -> str | None:
        action = message["command"]
        if action == "presentation":
            if message.get("enabled") is not False:
                return "use the web page to enter Spectacles UI"
            self.set_presentation(False)
            if self.active or self.starting:
                self.stop()
            return None
        if not self.hands.presentation:
            return "enter Spectacles UI from the web page first"
        if action == "manual":
            if message.get("enabled") is False:
                if self.active or self.starting:
                    self.stop()
                return None
            if self.active or self.starting:
                return None
            if self.cell.busy == "auto":
                self.cell.stop()
                error = self._wait_for_cell()
                if error:
                    return error
            return self.start(mode=self.mode, scale=self.scale, facing=self.facing)
        if action == "auto":
            if message.get("enabled") is False:
                if self.cell.busy == "auto":
                    self.cell.stop()
                return None
            if self.cell.busy == "auto":
                return None
            if self.active or self.starting:
                self.stop()
                error = self._wait_for_cell()
                if error:
                    return error
            return self.cell.begin("auto")
        if action == "empty":
            if self.active or self.starting:
                self.stop()
            if self.cell.busy == "auto":
                self.cell.stop()
            error = self._wait_for_cell()
            return error or self.cell.begin("empty")
        if self.active or self.starting:
            self.stop()
        if self.cell.busy == "auto":
            self.cell.stop()
        error = self._wait_for_cell()
        return error or self.cell.begin("neutral")

    def configure(self, *, mode: str = "pinch", scale: float = SCALE, facing: str = "same") -> str | None:
        """Save the manual settings selected before entering the glasses UI."""

        if mode not in ("pinch", "joystick"):
            return "mode must be pinch or joystick"
        if facing not in ("same", "them"):
            return "facing must be same or them"
        if not 0 < float(scale) <= 3:
            return "scale must be above 0 and at most 3"
        self.mode, self.scale, self.facing = mode, float(scale), facing
        return None

    def start_network(self) -> None:
        """Start both WebSockets without taking control of either arm."""

        try:
            self.hand_server = make_server(self.hands, "0.0.0.0", self.hand_port)
            capture = _LatestFrameCapture(self.cell.camera)
            reference_size = tuple(self.cell.scene()["size"])
            self.video = VideoFrames(capture, fps=self.video_fps, width=self.video_width,
                                     quality=self.video_quality, tape_pixels=self.cell.rig.tape_pixels,
                                     reference_size=reference_size, decorate=self._decorate_video)
            self.video_server = make_video_server(self.video, "0.0.0.0", self.video_port)
            self.video.start()
            threading.Thread(target=self.hand_server.serve_forever, name="spectacles-hands", daemon=True).start()
            threading.Thread(target=self.video_server.serve_forever, name="spectacles-video-server", daemon=True).start()
            self._servers_started = True
            hand_tunnel = usb_tunnel(self.hand_port)
            video_tunnel = usb_tunnel(self.video_port)
            wifi = addresses()
            self.urls = {
                "hands": hand_tunnel or (f"ws://{wifi[0]}:{self.hand_port}" if wifi else "unavailable"),
                "video": video_tunnel or (f"ws://{wifi[0]}:{self.video_port}" if wifi else "unavailable"),
            }
            self.cell.log(f"Spectacles bridge ready: {self.urls['hands']}; video {self.urls['video']}")
        except (OSError, RuntimeError, ValueError) as error:
            self.network_error = str(error)
            self.cell.log(f"Spectacles bridge unavailable: {error}")
            self._close_network()

    def _decorate_video(self, frame):
        """Mirror the web overlays during sorting; hand control gets a clear view."""

        if self.active or self.starting:
            return frame
        from .overlay import draw

        try:
            return draw(frame, self.cell.state())
        except Exception as error:
            # A bad annotation must not tear down the glasses' video socket.
            now = time.monotonic()
            if now - self._overlay_error_at > 10:
                self._overlay_error_at = now
                self.cell.log(f"Spectacles video overlay unavailable: {error}")
            return frame

    def start(self, *, mode: str = "pinch", scale: float = SCALE, facing: str = "same") -> str | None:
        """Reserve the arms and start following hands; return why that was refused."""

        if mode not in ("pinch", "joystick"):
            return "mode must be pinch or joystick"
        if facing not in ("same", "them"):
            return "facing must be same or them"
        if not 0 < float(scale) <= 3:
            return "scale must be above 0 and at most 3"
        if self.network_error:
            return f"Spectacles bridge unavailable: {self.network_error}"
        if not self.hands.connected:
            return "the glasses are not connected"
        with self._lock:
            if self.active or self.starting:
                return "manual mode is already running"
            with self.cell._lock:
                if self.cell.busy:
                    return f"busy: {self.cell.busy}"
                self.cell.busy = "spectacles"
                self.cell.stop_event.clear()
            self.active, self.starting = True, True
            self.mode, self.scale, self.facing = mode, float(scale), facing
            self.dry_run = bool(self.cell.options.dry_run)
            self.error = None
            self._shutdown.clear()
            self._thread = threading.Thread(target=self._run, name="spectacles-follow", daemon=True)
            self._thread.start()
        self.cell.log(f"Spectacles manual mode starting: {mode}, scale {scale:g}" +
                      (" (dry run)" if self.dry_run else ""))
        return None

    def _run(self) -> None:
        from ..arm import move_together

        names = list(self.cell.arms)
        hardware = not self.dry_run and all(hasattr(arm, "to_ticks") for arm in self.cell.arms.values())
        try:
            ready = {name: ready_pose(self.cell.kinematics[name], self.cell.placements[name],
                                      self.cell.poses[name]["neutral"], self.cell.limits[name]) for name in names}
            if hardware:
                for arm in self.cell.arms.values():
                    if not arm.torque_is_on():
                        arm.torque_on()
                move_together([(self.cell.arms[name], ready[name]) for name in names])
                ready = {name: self.cell.arms[name].pose() for name in names}
            followers = {}
            for name in names:
                options = dict(speed=SPEED, facing=self.facing, limits=self.cell.limits[name])
                followers[name] = (PinchFollower(name, self.cell.kinematics[name], self.cell.placements[name],
                                                 ready[name], scale=self.scale, **options)
                                   if self.mode == "pinch" else
                                   Follower(name, self.cell.kinematics[name], self.cell.placements[name],
                                            ready[name], **options))
            self.starting = False
            follow(followers, self.hands, self.cell.arms if hardware else None, facing=self.facing,
                   hold_s=HOLD_S, log=self.cell.log, shutdown=self._shutdown)
        except (KeyboardInterrupt, RuntimeError, ValueError) as error:
            if not self._shutdown.is_set():
                self.error = str(error)
                self.cell.log(f"Spectacles manual mode failed: {error}")
        finally:
            if hardware:
                for arm in self.cell.arms.values():
                    try:
                        arm.hold()
                    except Exception as error:
                        self.cell.log(f"cannot hold {arm.name} arm after manual mode: {error}")
            self.starting = False
            self.active = False
            self.cell._refresh_poses()
            with self.cell._lock:
                if self.cell.busy == "spectacles":
                    self.cell.busy = None
            self.cell.version += 1
            self.cell.log("Spectacles manual mode stopped; the arms hold where they are")

    def stop(self) -> str:
        """Stop following and hold the arms; safe to call when already stopped."""

        with self._lock:
            if not self.active and not self.starting:
                return "Spectacles manual mode is already stopped"
            self.cell.stop_event.set()
            self.hands.receive({"command": "stop"})
            self._shutdown.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
        return "Spectacles manual mode stopped; the arms hold where they are"

    def command(self, message: dict) -> str | None:
        if message.get("command") in ("presentation", "manual", "auto", "neutral", "empty"):
            self._lens_command(message)
            return self.control_error
        if not self.active:
            return "manual mode is not running"
        if message.get("command") not in ("jaw", "precision", "home", "hold", "stop"):
            return "unknown Spectacles command"
        self.hands.receive(message)
        return None

    def request_snapshot(self) -> str | None:
        if not self.hands.connected:
            return "the glasses are not connected"
        self.hands.snapshots.trigger.parent.mkdir(parents=True, exist_ok=True)
        self.hands.snapshots.trigger.touch()
        return None

    def set_presentation(self, enabled: bool) -> str | None:
        """Show or hide the glasses-controlled jury mode from web or the Lens."""

        if enabled and not self.hands.connected:
            return "the glasses are not connected"
        self.control_error = None
        self.hands.presentation = bool(enabled)
        # Raw optical streaming is parked while the on-device UI is tuned.
        # Keep the encoder idle; the jury page uses the stable overhead camera.
        self.spectator.set_enabled(False)
        self.cell.version += 1
        self.cell.log("Spectacles presentation " + ("started" if enabled else "returned to the web controls"))
        return None

    def state(self) -> dict:
        try:
            report = json.loads(self.hands.report) if self.hands.report else {}
        except ValueError:
            report = {"status": self.hands.status}
        context = self._report_context()
        snaps = []
        folder = self.folder / "snaps"
        if folder.is_dir():
            for path in sorted(folder.glob("*-composite.jpg"), key=lambda item: item.stat().st_mtime, reverse=True)[:6]:
                stem = path.name.removesuffix("-composite.jpg")
                snaps.append({"id": stem, "composite": f"/spectacles/snaps/{path.name}",
                              "camera": f"/spectacles/snaps/{stem}-camera.jpg",
                              "view": f"/spectacles/snaps/{stem}-view.jpg"})
        _, view_sequence = self.spectator.latest()
        return {
            "available": self.network_error is None,
            "network_error": self.network_error,
            "active": self.active,
            "starting": self.starting,
            "connected": self.hands.connected,
            "mode": self.mode,
            "scale": self.scale,
            "facing": self.facing,
            "dry_run": self.dry_run,
            "error": self.error,
            "status": context.get("status", report.get("status", self.hands.status)),
            "hands": report.get("hands", {}) if context["manual"] else {},
            "arms": report.get("arms", {}) if context["manual"] else {},
            "stopped": report.get("stopped", False),
            "presentation": bool(self.hands.presentation),
            "optical_stream": False,
            "view_sequence": view_sequence,
            "urls": self.urls,
            "snapshots": snaps,
        }

    def close(self) -> None:
        self.set_presentation(False)
        if self.active or self.starting:
            self.stop()
        self._close_network()
        self.hands.stop_recording()

    def _close_network(self) -> None:
        if self.hand_server is not None:
            if self._servers_started:
                self.hand_server.shutdown()
            self.hand_server.server_close()
            self.hand_server = None
        if self.video_server is not None:
            if self._servers_started:
                self.video_server.shutdown()
            self.video_server.server_close()
            self.video_server = None
        if self.video is not None:
            self.video.close()
            self.video = None
        self._servers_started = False
