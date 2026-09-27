"""The camera-only jury page and B601 share one Lens socket and one CAN owner."""

from __future__ import annotations

import json
from pathlib import Path

from ..b601_glasses import B601GlassesBridge
from ..station import repository_root
from .manual import _LatestFrameCapture


class _NoOpticalStream:
    def latest(self):
        return None, 0


class B601WebControl:
    """Adapt the B601 control loop to the unified page's Spectacles routes."""

    def __init__(self, cell, *, live_allowed: bool, sdk_root: Path) -> None:
        self.cell = cell
        self.folder = repository_root() / "out" / "spectacles"
        self.spectator = _NoOpticalStream()
        self.network_error: str | None = None
        self.bridge = B601GlassesBridge(live_allowed=live_allowed, sdk_root=sdk_root, video_enabled=True)
        self.bridge.web_mode = True
        self.bridge.hands.presentation = False
        self.bridge.hands.command_handler = self._lens_command

    @property
    def active(self) -> bool:
        return self.bridge.mode != "off"

    @property
    def starting(self) -> bool:
        return False

    def start_network(self) -> None:
        self.bridge.video_capture = _LatestFrameCapture(self.cell.camera)
        self.bridge.tape_pixels = self.cell.rig.tape_pixels
        self.bridge.reference_size = tuple(self.cell.scene()["size"])
        self.bridge.start_network(background=True)

    def _lens_command(self, message: dict) -> bool:
        if message.get("command") != "presentation":
            return False
        error = self.set_presentation(message.get("enabled") is not False)
        if error:
            self.bridge.error = error
            self.bridge._report()
        return True

    def configure(self, *, mode: str = "pinch", scale: float = 1.0, facing: str = "same",
                  target: str = "b601") -> str | None:
        if target != "b601":
            return "SO-101 arms are unplugged; select B601"
        if mode != "pinch":
            return "B601 uses pinch control"
        return None

    def set_presentation(self, enabled: bool) -> str | None:
        if enabled and not self.bridge.hands.connected:
            return "the glasses are not connected"
        if not enabled and self.bridge.driver is not None:
            return "Park B601 with NEUTRAL before leaving the glasses UI"
        self.bridge.hands.presentation = enabled
        self.cell.version += 1
        return None

    def start(self, *, mode: str = "pinch", scale: float = 1.0, facing: str = "same",
              target: str = "b601") -> str | None:
        error = self.configure(mode=mode, scale=scale, facing=facing, target=target)
        if error:
            return error
        if not self.bridge.hands.connected:
            return "the glasses are not connected"
        self.bridge.hands.receive({"command": "b601", "enabled": True})
        return None

    def stop(self) -> str:
        self.bridge.hands.receive({"command": "hold"})
        return "B601 holding its current pose; NEUTRAL parks and disables motors"

    def command(self, message: dict) -> str | None:
        action = message.get("command")
        # Motor enabling is only available at the glasses, not from a web
        # request that another device on venue Wi-Fi might make.
        if action not in ("neutral", "hold", "stop"):
            return "The glasses' B601 button enables motors; web can only hold or park"
        self.bridge.hands.receive(message)
        return None

    def request_snapshot(self) -> str | None:
        return "optical snapshots are paused for Lens stability"

    def state(self) -> dict:
        try:
            report = json.loads(self.bridge.hands.report or "{}")
        except ValueError:
            report = {}
        return {
            "available": self.network_error is None,
            "network_error": self.network_error,
            "active": self.active,
            "starting": False,
            "connected": self.bridge.hands.connected,
            "mode": "pinch",
            "target": "b601",
            "scale": 1.0,
            "dry_run": not self.bridge.live_allowed,
            "error": self.bridge.error,
            "status": report.get("status", self.bridge.hands.status),
            "hands": report.get("hands", {}),
            "arms": report.get("arms", {}),
            "stopped": self.bridge.mode == "hold",
            "presentation": bool(self.bridge.hands.presentation),
            "optical_stream": False,
            "view_sequence": 0,
            "urls": {"hands": "ws://127.0.0.1:8765", "video": "ws://127.0.0.1:8766"},
            "snapshots": [],
        }

    def close(self) -> None:
        self.bridge.close()
