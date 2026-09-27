"""Single-bus B601-RS Spectacles bridge; no SO-101 or camera is opened.

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

from .b601 import B601HandMotion, B601Preview
from .b601_motor import B601Motor
from .spectacles import Hands, make_server, usb_tunnel


class B601GlassesBridge:
    """Keep Lens messages and all CAN commands on separate owning threads."""

    def __init__(self, *, live_allowed: bool, sdk_root: Path, channel: str = "can0@1000000") -> None:
        self.live_allowed = live_allowed
        self.sdk_root = Path(sdk_root)
        self.channel = channel
        self.park_file = Path(__file__).resolve().parents[1] / "b601_park.toml"
        self.hands = Hands()
        self.hands.presentation = True
        self.preview = B601Preview(scale=0.5)
        self.motion = B601HandMotion(scale=0.5)
        self.mode = "off"
        self.error: str | None = None
        self.driver: B601Motor | None = None
        self.anchor_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.grip_open = False
        self.grip_latched = False
        self.server = None

    def start(self, port: int = 8765) -> None:
        self.server = make_server(self.hands, "0.0.0.0", port)
        threading.Thread(target=self.server.serve_forever, name="b601-hands", daemon=True).start()
        link = usb_tunnel(port)
        print(f"B601 Spectacles bridge: {link or f'ws://<Mac IP>:{port}'}", flush=True)
        self._report()

    def _command(self, message: dict) -> None:
        action = message.get("command")
        enabled = message.get("enabled") is not False
        self.error = None
        if action == "b601":
            if self.driver is not None:
                self.error = "Park B601 with NEUTRAL before preview"
            else:
                self.mode = "preview" if enabled else "off"
                self.preview = B601Preview(scale=0.5)
        elif action == "b601_live":
            if not enabled:
                if self.driver is not None:
                    self.driver.hold()
                    self.mode = "hold"
                return
            if not self.live_allowed:
                self.error = "LIVE is locked: restart B601 bridge with --live"
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
            self.motion = B601HandMotion(scale=0.5)
            self.anchor_pose = None
            self.mode = "live"
            print("B601 LIVE enabled: seven present-position holds; 4 deg/s joint cap", flush=True)
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
            self.error = "Only B601 is connected; use the B601 buttons"

    def _grip(self, packet: dict | None, age: float) -> None:
        if self.driver is None or age > 0.3 or not isinstance(packet, dict):
            self.grip_latched = False
            return
        hand = packet.get("right")
        if not isinstance(hand, dict) or not hand.get("tracked") or hand.get("pinch"):
            self.grip_latched = False
            return
        try:
            thumb = np.asarray(hand["thumb"], dtype=float)
            pinky = np.asarray(hand["pinkyTip"], dtype=float)
            gap = float(np.linalg.norm(thumb - pinky))
        except (KeyError, TypeError, ValueError):
            return
        if not math.isfinite(gap):
            return
        if gap > 5.5:
            self.grip_latched = False
        elif gap < 2.5 and not self.grip_latched:
            self.grip_latched = True
            self.grip_open = not self.grip_open
        self.driver.set_grip(self.grip_open)

    def tick(self) -> None:
        for command in self.hands.take_commands():
            self._command(command)
        packet, age = self.hands.playout()
        if self.mode == "preview":
            self.preview.update(packet, age)
        elif self.mode == "live" and self.driver is not None:
            if not self.hands.connected:
                self.driver.hold()
                self.motion = B601HandMotion(scale=0.5)
                self.anchor_pose = None
            else:
                previous_anchor = self.motion.anchor
                displacement = self.motion.update(packet, age)
                if displacement is None:
                    self.driver.hold()
                    self.anchor_pose = None
                    self._grip(packet, age)
                else:
                    if previous_anchor is None:
                        self.anchor_pose = self.driver.tool_pose()
                    if self.anchor_pose is not None:
                        delta, rotation = displacement
                        target_position = self.anchor_pose[0] + delta
                        target_rotation = rotation @ self.anchor_pose[1]
                        self.driver.target_tool(target_position, target_rotation)
        if self.driver is not None:
            try:
                parked = self.driver.step()
            except Exception as exc:
                self.error = f"B601 CAN/control fault: {exc}"
                self.driver.close()
                self.driver = None
                self.mode = "off"
            else:
                if parked:
                    self.driver = None
                    self.mode = "off"
                    self.motion = B601HandMotion(scale=0.5)
                    print("B601 parked; all motors disabled", flush=True)
        self._report()

    def _report(self) -> None:
        if self.driver is not None:
            state = self.driver.state
            joints = [round(math.degrees(angle), 1) for angle in self.driver.feedback]
            guide = {"mode": "moving" if self.mode in ("live", "parking") else "holding",
                     "state": state, "joints": joints, "jaw": 60 if self.grip_open else 0,
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
                  "standaloneB601": True, "videoAvailable": False,
                  "b601Holding": self.mode == "hold", "b601Parking": self.mode == "parking",
                  "auto": False, "busy": "b601" if self.driver is not None else None,
                  "controlError": self.error, "emptyPhotographed": True,
                  "hands": {"right": guide}, "arms": {"b601": guide}}
        self.hands.status = state
        self.hands.report = json.dumps(report, separators=(",", ":"))

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.driver is not None:
            self.driver.close()
            self.driver = None


def main() -> int:
    parser = argparse.ArgumentParser(description="B601-RS Spectacles bridge (SO-101 disconnected)")
    parser.add_argument("--sdk-root", type=Path, default=Path("/private/tmp/trashdrop-rebot-sdk"))
    parser.add_argument("--channel", default="can0@1000000")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--live", action="store_true", help="allow the separate B601 LIVE button to enable motors")
    args = parser.parse_args()
    bridge = B601GlassesBridge(live_allowed=args.live, sdk_root=args.sdk_root, channel=args.channel)
    bridge.start(args.port)
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
