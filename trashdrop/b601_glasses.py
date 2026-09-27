"""Single-bus B601-RS Spectacles bridge; no SO-101 bus is opened.

Run from the official SDK's isolated uv environment on the Mac. Without
``--live``, the LIVE button is refused and this is a hand-tracking preview.
With ``--live``, a separate press of B601 LIVE is still required before the
seven motors are enabled. The operator parks with NEUTRAL before exit.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from .b601 import B601HandMotion, B601Preview, heading_and_pitch, jaw_frame
from .b601_motor import B601_SDK, JOINT_SPEED, B601Motor
from .spectacles import (
    VIDEO_CAPTURE_HEIGHT, VIDEO_CAPTURE_WIDTH, VIDEO_FPS, VIDEO_PORT, VIDEO_QUALITY, VIDEO_WIDTH,
    Hands, VideoFrames, make_server, make_video_server, usb_tunnel,
)


class B601GlassesBridge:
    """Keep Lens messages and all CAN commands on separate owning threads."""

    def __init__(self, *, live_allowed: bool, sdk_root: Path, channel: str = "can0@1000000",
                 video_enabled: bool = False, camera: str = "auto", video_port: int = VIDEO_PORT,
                 video_capture=None, tape_pixels=None, reference_size=(0, 0)) -> None:
        self.live_allowed = live_allowed
        self.sdk_root = Path(sdk_root)
        self.channel = channel
        self.video_enabled, self.camera, self.video_port = video_enabled, camera, video_port
        self.video_capture = video_capture
        self.tape_pixels = tape_pixels
        self.reference_size = reference_size
        self.park_file = Path(__file__).resolve().parents[1] / "b601_park.toml"
        self.hands = Hands()
        self.hands.presentation = True
        self.web_mode = False
        self.preview = B601Preview(scale=1.0)
        self.motion = B601HandMotion(scale=1.0)
        self.mode = "off"
        self.error: str | None = None
        self.driver: B601Motor | None = None
        self.anchor_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.anchor_generation = 0
        self.gripping = False  # the thumb-pinky gesture was on last tick
        self.server = None
        self.video = self.video_server = None
        self.telemetry = None
        self._reported_at = self._video_reported_at = 0.0
        self._last_state = ""
        self._worker_stop = threading.Event()
        self._worker: threading.Thread | None = None

    def start_network(self, port: int = 8765, *, background: bool = False) -> None:
        if self.video_enabled and self.video_port == port:
            raise ValueError("B601 hand and video sockets need separate ports")
        folder = Path(__file__).resolve().parents[1] / "out" / "spectacles"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.hands.record = (folder / f"b601-hands-{stamp}.jsonl").open("w", buffering=1)
        self.telemetry = (folder / f"b601-control-{stamp}.jsonl").open("w", buffering=1)
        # The live B601 socket is USB-forwarded through adb; do not expose a
        # motor-enable command to everyone on the venue Wi-Fi.
        self.server = make_server(self.hands, "127.0.0.1", port)
        threading.Thread(target=self.server.serve_forever, name="b601-hands", daemon=True).start()
        link = usb_tunnel(port)
        print(f"B601 Spectacles bridge: {link or 'USB tunnel unavailable; reconnect Spectacles'}", flush=True)
        if self.video_enabled:
            self._start_video()
        self._report()
        if background:
            self._worker = threading.Thread(target=self._run_ticks, name="b601-control", daemon=True)
            self._worker.start()

    def _run_ticks(self) -> None:
        while not self._worker_stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                self.error = f"B601 control loop held: {exc}"
                if self.driver is not None:
                    self.driver.hold()
                    self.mode = "hold"
                self._report()
                print(self.error, flush=True)
            self._worker_stop.wait(1 / 30)

    def _start_video(self) -> None:
        """Use the same cropped overhead stream and UIKit Frame as the main Lens."""

        from .__main__ import _require_same_camera, _resolve_camera, _webcam
        from .dataset.capture import open_camera
        from .dataset.zone import DEFAULT_CALIBRATION, load_zone
        from .rig import load_rig

        capture = self.video_capture
        try:
            if capture is None:
                webcam = _webcam()
                if self.camera == "auto" and webcam is None:
                    raise RuntimeError("cannot identify the USB webcam; pass --camera")
                source = _resolve_camera(self.camera, VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT)
                capture = open_camera(source, VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT)
                _require_same_camera(capture, source, webcam, refuse=True)
                zone = load_zone(DEFAULT_CALIBRATION)
                size = (zone.frame_width, zone.frame_height) if zone else (0, 0)
                tape = load_rig().tape_pixels
            else:
                size = self.reference_size
                tape = self.tape_pixels
            self.video = VideoFrames(capture, fps=VIDEO_FPS, width=VIDEO_WIDTH, quality=VIDEO_QUALITY,
                                     tape_pixels=tape, reference_size=size)
            self.video_server = make_video_server(self.video, "127.0.0.1", self.video_port)
            self.video.start()
            threading.Thread(target=self.video_server.serve_forever, name="b601-video-server", daemon=True).start()
            link = usb_tunnel(self.video_port)
            print(f"B601 overhead video: {link or 'USB tunnel unavailable; reconnect Spectacles'}", flush=True)
        except Exception as exc:
            if self.video_server is not None:
                self.video_server.server_close()
                self.video_server = None
            if self.video is not None:
                self.video.close()
                self.video = None
            elif capture is not None:
                capture.release()
            self.error = f"overhead camera unavailable: {exc}"
            print(self.error, flush=True)

    def _command(self, message: dict) -> None:
        action = message.get("command")
        enabled = message.get("enabled") is not False
        self.error = None
        if action == "b601":
            if self.driver is not None:
                self.error = "Park B601 with NEUTRAL before preview"
            else:
                self.mode = "preview" if enabled else "off"
                self.preview = B601Preview(scale=1.0)
        elif action == "b601_live":
            if not enabled:
                if self.driver is not None:
                    self.driver.hold()
                    self.mode = "hold"
                return
            if not self.hands.presentation:
                self.error = "Enter Spectacles UI before enabling B601"
                return
            if not self.live_allowed:
                self.mode = "preview"
                self.preview = B601Preview(scale=1.0)
                self.error = "B601 preview only: motor enabling is locked"
                return
            if self.driver is None:
                urdf = self.sdk_root / "urdf/RS/urdf/ReBot_Arm_RS.urdf"
                motor = B601Motor(channel=self.channel, urdf=urdf, park=self.park_file)
                try:
                    motor.connect()
                except Exception as exc:
                    self.error = f"B601 enable refused: {exc}"
                    return
                self.driver = motor
            self.motion = self._fresh_motion()
            self.anchor_pose = None
            self.anchor_generation = 0
            self.mode = "live"
            print(f"B601 LIVE enabled: seven present-position holds; {math.degrees(JOINT_SPEED):.0f} deg/s joint cap", flush=True)
        elif action == "neutral":
            if self.driver is not None:
                if self.driver.start_park():
                    self.mode = "parking"
                    self.anchor_pose = None
                else:
                    self.error = self.driver.state
            else:
                self.mode = "off"
        elif action in ("stop", "hold"):
            if self.driver is not None:
                self.driver.hold()
                self.mode = "hold"
        elif action == "presentation":
            if enabled:
                self.hands.presentation = True
            else:
                if self.driver is not None:
                    self.driver.hold()
                    self.mode = "hold"
                # The standalone bridge has no web page to return to. In its
                # menu this button is labelled HOLD; never hide the only PARK.
                self.hands.presentation = True
        elif action in ("manual", "auto", "empty"):
            self.error = "SO-101 is unplugged; B601 remains available in the same Lens"

    def _fresh_motion(self) -> B601HandMotion:
        """New clutches that keep the jaw where it is: a new one would otherwise close it."""

        motion = B601HandMotion(scale=1.0)
        if self.driver is not None:
            motion.jaw = self.driver.grip_fraction()
        return motion

    def tick(self) -> None:
        for command in self.hands.take_commands():
            self._command(command)
        packet, age = self.hands.playout()
        if self.mode == "preview":
            self.preview.update(packet, age)
        elif self.mode == "live" and self.driver is not None:
            if not self.hands.connected:
                self.driver.hold()
                self.motion = self._fresh_motion()
                self.anchor_pose = None
                self.anchor_generation = 0
            else:
                displacement = self.motion.update(packet, age)
                gripping = self.motion.gesture == "grip"
                if self.gripping and not gripping:
                    # Let go: the jaw stops where it is. It used to go on to where the hand had set it.
                    self.motion.jaw = self.driver.grip_fraction(now=True)
                self.gripping = gripping
                if displacement is None:
                    self.driver.hold()
                    self.anchor_pose = None
                else:
                    if self.anchor_generation != self.motion.anchor_generation:
                        position, rotation = self.driver.tool_pose()
                        self.anchor_pose = (position, *heading_and_pitch(rotation))
                        self.anchor_generation = self.motion.anchor_generation
                    if self.anchor_pose is not None:
                        delta, turn = displacement
                        start, heading, pitch = self.anchor_pose
                        heading += math.degrees(math.atan2(turn[1, 0], turn[0, 0]))  # thumb-middle
                        if self.motion.gesture == "drag" and self.motion.pitch is not None:
                            # The jaw points as far down as the hand does: level to straight down.
                            pitch = min(max(self.motion.pitch, -90.0), 0.0)
                        self.driver.target_tool(start + delta, jaw_frame(heading, pitch))
                # Thumb-pinky sets how far the jaw is open; every tick, so a hold keeps it there.
                self.driver.set_grip_fraction(self.motion.jaw)
        if self.driver is not None:
            try:
                parked = self.driver.step()
            except Exception as exc:
                # Never disable on a fault: disabled, the motors let the arm fall. It did, 14:55, after
                # one position read timed out. Hold instead: the motors keep their last set point,
                # the next tick tries again, and the hand has to re-pinch.
                self.error = f"B601 CAN hiccup, holding: {exc}"
                self.driver.hold()
                self.motion = self._fresh_motion()
                self.anchor_pose = None
            else:
                if parked:
                    self.driver = None
                    self.mode = "off"
                    self.motion = B601HandMotion(scale=1.0)
                    print("B601 parked; all motors disabled", flush=True)
        self._report()
        self._log_control(age)

    def _log_control(self, hand_age: float) -> None:
        """Retain the reason for each hold; startup-only logs hid the lift limit."""

        now = time.monotonic()
        if self.video is not None and now - self._video_reported_at >= 10:
            self._video_reported_at = now
            print(self.video.report(), flush=True)
        if self.telemetry is None or now - self._reported_at < 0.2:
            return
        self._reported_at = now
        motor = self.driver
        state = motor.state if motor is not None else self.mode
        record = {"at": round(now, 3), "mode": self.mode, "gesture": self.motion.gesture,
                  "hand": self.motion.state, "state": state, "hand_age": round(hand_age, 3),
                  "connected": self.hands.connected}
        if motor is not None:
            record["command_deg"] = np.degrees(motor.command).round(2).tolist()
            record["feedback_deg"] = np.degrees(motor.feedback).round(2).tolist()
            record["desired_deg"] = np.degrees(motor.desired).round(2).tolist()
            record["temperature_c"] = np.round(getattr(motor, "temperature", np.zeros(7)), 1).tolist()
            record["status"] = [int(code) for code in getattr(motor, "status", np.zeros(7))]
        self.telemetry.write(json.dumps(record, separators=(",", ":")) + "\n")
        if state != self._last_state:
            self._last_state = state
            print(f"B601 {self.mode}: {self.motion.state}; {state}", flush=True)

    def _report(self) -> None:
        if self.driver is not None:
            state = self.driver.state
            joints = [round(math.degrees(angle), 1) for angle in self.driver.feedback]
            guide = {"mode": "moving" if self.mode in ("live", "parking") else "holding",
                     "state": self.motion.state if self.mode == "live" and state.startswith("holding:") else state,
                     "joints": joints, "jaw": round(self.motion.jaw * 100),
                     "blocked": ["down"] if "table" in state else []}
        elif self.mode == "preview":
            guide = self.preview.guide()
            state = "B601 input preview · no motors"
        else:
            guide = {"mode": "holding", "state": "motors off", "blocked": []}
            state = "B601 ready · palm menu: B601 LIVE or PREVIEW"
        if self.error:
            state = self.error
        report = {"status": state, "manual": self.mode in ("live", "preview"),
                  "manualTarget": "b601" if self.mode == "preview" else "b601_live",
                  "standaloneB601": not self.web_mode, "so101Offline": True,
                  "videoAvailable": self.video is not None,
                  "b601Holding": self.mode == "hold", "b601Parking": self.mode == "parking",
                  "auto": False, "busy": "b601" if self.driver is not None else None,
                  "controlError": self.error, "emptyPhotographed": True,
                  "hands": {"right": guide}, "arms": {"b601": guide}}
        self.hands.status = state
        self.hands.report = json.dumps(report, separators=(",", ":"))

    def close(self) -> None:
        self._worker_stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2)
            self._worker = None
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.driver is not None:
            self.driver.close()
            self.driver = None
        if self.video_server is not None:
            self.video_server.shutdown()
            self.video_server.server_close()
            self.video_server = None
        if self.video is not None:
            self.video.close()
            self.video = None
        self.hands.stop_recording()
        if self.telemetry is not None:
            self.telemetry.close()
            self.telemetry = None


def main() -> int:
    parser = argparse.ArgumentParser(description="B601-RS Spectacles bridge (SO-101 disconnected)")
    parser.add_argument("--sdk-root", type=Path, default=B601_SDK)
    parser.add_argument("--channel", default="can0@1000000")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--live", action="store_true", help="allow the separate B601 LIVE button to enable motors")
    parser.add_argument("--video", action="store_true", help="stream the cropped overhead camera to the Lens Frame")
    parser.add_argument("--camera", default="auto", help="overhead camera source; auto identifies the USB webcam")
    parser.add_argument("--video-port", type=int, default=VIDEO_PORT)
    args = parser.parse_args()
    bridge = B601GlassesBridge(live_allowed=args.live, sdk_root=args.sdk_root, channel=args.channel,
                               video_enabled=args.video, camera=args.camera, video_port=args.video_port)
    bridge.start_network(args.port)
    interrupted = False
    try:
        while True:
            try:
                bridge.tick()
                time.sleep(1 / 30)
            except KeyboardInterrupt:
                if bridge.driver is None or interrupted:
                    if bridge.driver is not None:
                        print("Emergency release: support the arm; motors disabling", flush=True)
                    break
                interrupted = True
                bridge.driver.hold()
                bridge.mode = "hold"
                print("B601 holding. Tap NEUTRAL to park and disable; second Ctrl+C releases motors.", flush=True)
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
