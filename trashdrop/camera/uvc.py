"""Talk to a UVC webcam directly over USB: focus, exposure, white balance.

Why this exists: on macOS, OpenCV reaches the camera through AVFoundation,
which accepts and then silently ignores every property -- ``trashdrop
camcheck`` reports "controllable: nothing" for a C920. The camera itself
speaks UVC, the USB class standard every webcam implements, and UVC has
requests for exactly these controls. Sending them needs libusb, which ships as
a pip wheel (``libusb-package``), so nothing is installed system-wide.

The catch, measured on this project's MacBook: macOS lets any process read the
camera's descriptors, but refuses control requests unless the process runs as
root, because its own UVC driver holds the device. So applying and tuning run
under ``sudo``; see docs/CAMERA.md for the exact commands.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

VIDEO_CLASS = 0x0E
VIDEO_CONTROL_SUBCLASS = 0x01
CS_INTERFACE = 0x24
VC_INPUT_TERMINAL = 0x02
VC_PROCESSING_UNIT = 0x05
ITT_CAMERA = 0x0201

SET_CUR, GET_CUR, GET_MIN, GET_MAX, GET_RES, GET_INFO, GET_DEF = (
    0x01, 0x81, 0x82, 0x83, 0x84, 0x86, 0x87,
)
TO_INTERFACE, FROM_INTERFACE = 0x21, 0xA1

# Auto-exposure mode bitmap, UVC 1.5 section 4.2.2.1.2.
AE_MANUAL = 0x01
AE_AUTO = 0x02
AE_APERTURE_PRIORITY = 0x08

# On/off controls. The UVC spec makes GET_MIN/GET_MAX optional for these, and a
# C920 stalls on them, so their range is known rather than asked for.
BOOLEAN = frozenset({"exposure_auto", "exposure_priority", "focus_auto", "white_balance_auto"})


@dataclass(frozen=True)
class UvcControl:
    name: str
    unit: str  # "camera" terminal or "processing" unit
    selector: int
    size: int
    signed: bool = False


# Named for humans; the selector numbers are from the UVC 1.5 specification.
CONTROLS: dict[str, UvcControl] = {
    control.name: control
    for control in (
        UvcControl("exposure_auto", "camera", 0x02, 1),
        UvcControl("exposure_priority", "camera", 0x03, 1),
        UvcControl("exposure", "camera", 0x04, 4),
        UvcControl("focus", "camera", 0x06, 2),
        UvcControl("focus_auto", "camera", 0x08, 1),
        UvcControl("zoom", "camera", 0x0B, 2),
        UvcControl("backlight_compensation", "processing", 0x01, 2),
        UvcControl("brightness", "processing", 0x02, 2, signed=True),
        UvcControl("contrast", "processing", 0x03, 2),
        UvcControl("gain", "processing", 0x04, 2),
        UvcControl("power_line_frequency", "processing", 0x05, 1),
        UvcControl("saturation", "processing", 0x07, 2),
        UvcControl("sharpness", "processing", 0x08, 2),
        UvcControl("white_balance", "processing", 0x0A, 2),
        UvcControl("white_balance_auto", "processing", 0x0B, 1),
    )
}


@dataclass
class ControlRange:
    minimum: int
    maximum: int
    step: int
    default: int


class CameraPermissionError(RuntimeError):
    """The OS refused the control request. On macOS: not running as root."""


def permission_hint() -> str:
    return (
        "macOS only lets root send settings to a webcam (its own driver holds the\n"
        "device). Run this one command with sudo, from the project folder:\n"
        "    sudo .venv/bin/python -B -m trashdrop camera apply\n"
        "It only talks to the camera; -B stops it leaving root-owned files behind."
    )


def parse_video_control(extra: bytes) -> tuple[int | None, int | None]:
    """Camera-terminal and processing-unit ids from class-specific descriptors."""

    camera_id = processing_id = None
    position = 0
    while position + 3 <= len(extra):
        length, kind, subtype = extra[position], extra[position + 1], extra[position + 2]
        if length < 3:
            break
        if kind == CS_INTERFACE and subtype == VC_INPUT_TERMINAL and position + 6 <= len(extra):
            if struct.unpack_from("<H", extra, position + 4)[0] == ITT_CAMERA:
                camera_id = extra[position + 3]
        if kind == CS_INTERFACE and subtype == VC_PROCESSING_UNIT:
            processing_id = extra[position + 3]
        position += length
    return camera_id, processing_id


def _decode(data, size: int, signed: bool) -> int:
    return int.from_bytes(bytes(data[:size]), "little", signed=signed)


def _encode(value: int, size: int, signed: bool) -> bytes:
    return int(value).to_bytes(size, "little", signed=signed)


class UvcCamera:
    """One UVC webcam, found on the USB bus."""

    backend_name = "uvc"

    def __init__(self, device, interface: int, camera_id: int, processing_id: int) -> None:
        self.device = device
        self.interface = interface
        self.entities = {"camera": camera_id, "processing": processing_id}
        self.usb_id = f"{device.idVendor:04x}:{device.idProduct:04x}"
        self._ranges: dict[str, ControlRange] | None = None
        # The raw mode that means "automatic" on this camera; learnt in ranges().
        self._ae_auto_mode = AE_APERTURE_PRIORITY

    # --- discovery ---------------------------------------------------------

    @classmethod
    def find(cls, usb_id: str | None = None) -> "UvcCamera":
        """First UVC device on the bus, or the one matching ``vvvv:pppp``.

        On an Apple-silicon MacBook the built-in camera is not a USB device,
        so the first UVC device found is the external webcam.
        """

        try:
            import libusb_package
            import usb.core
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("camera control needs the rig extra: uv sync --extra rig") from error

        backend = libusb_package.get_libusb1_backend()
        wanted = None
        if usb_id:
            vendor, product = (int(part, 16) for part in usb_id.split(":"))
            wanted = (vendor, product)

        for device in usb.core.find(find_all=True, backend=backend):
            if wanted and (device.idVendor, device.idProduct) != wanted:
                continue
            for configuration in device:
                for interface in configuration:
                    if (
                        interface.bInterfaceClass == VIDEO_CLASS
                        and interface.bInterfaceSubClass == VIDEO_CONTROL_SUBCLASS
                    ):
                        camera_id, processing_id = parse_video_control(
                            bytes(interface.extra_descriptors)
                        )
                        if camera_id is not None and processing_id is not None:
                            return cls(device, interface.bInterfaceNumber, camera_id, processing_id)
        where = f" matching {usb_id}" if usb_id else ""
        raise RuntimeError(f"no UVC webcam found on USB{where} -- is it plugged in?")

    # --- raw requests ------------------------------------------------------

    def _request(self, request: int, control: UvcControl, payload=None):
        import usb.core

        value = control.selector << 8
        index = (self.entities[control.unit] << 8) | self.interface
        try:
            if request == SET_CUR:
                return self.device.ctrl_transfer(TO_INTERFACE, SET_CUR, value, index, payload, 1000)
            return self.device.ctrl_transfer(FROM_INTERFACE, request, value, index, control.size, 1000)
        except usb.core.USBError as error:
            if getattr(error, "errno", None) == 13 or "Access denied" in str(error):
                raise CameraPermissionError(permission_hint()) from error
            raise

    def supports(self, name: str) -> bool:
        import usb.core

        control = CONTROLS[name]
        try:
            info = self._request(GET_INFO, UvcControl(name, control.unit, control.selector, 1))
        except usb.core.USBError:
            return False
        return bool(info) and bool(info[0] & 0x01) and bool(info[0] & 0x02)

    # --- public ------------------------------------------------------------

    def _read(self, request: int, control: UvcControl) -> int:
        return _decode(self._request(request, control), control.size, control.signed)

    def ranges(self) -> dict[str, ControlRange]:
        """Every control this camera supports, with its limits."""

        import usb.core

        if self._ranges is not None:
            return self._ranges

        found: dict[str, ControlRange] = {}
        for name, control in CONTROLS.items():
            if not self.supports(name):
                continue
            try:
                default = self._read(GET_DEF, control)
            except usb.core.USBError:
                continue

            if name == "exposure_auto":
                # GET_RES on the AE mode control is the bitmap of modes the
                # camera offers; prefer aperture priority, which is what a
                # C920 calls automatic.
                try:
                    modes = self._read(GET_RES, control)
                except usb.core.USBError:
                    modes = default
                if modes & AE_APERTURE_PRIORITY:
                    self._ae_auto_mode = AE_APERTURE_PRIORITY
                elif modes & AE_AUTO:
                    self._ae_auto_mode = AE_AUTO
                found[name] = ControlRange(0, 1, 1, int(default != AE_MANUAL))
                continue
            if name in BOOLEAN:
                found[name] = ControlRange(0, 1, 1, int(bool(default)))
                continue

            try:
                found[name] = ControlRange(
                    minimum=self._read(GET_MIN, control),
                    maximum=self._read(GET_MAX, control),
                    step=max(1, self._read(GET_RES, control)),
                    default=default,
                )
            except usb.core.USBError:
                continue

        self._ranges = found
        return found

    def get(self, name: str) -> int:
        control = CONTROLS[name]
        raw = _decode(self._request(GET_CUR, control), control.size, control.signed)
        if name == "exposure_auto":
            return int(raw != AE_MANUAL)
        return raw

    def set(self, name: str, value: int) -> None:
        control = CONTROLS[name]
        if name == "exposure_auto":
            self.ranges()  # learns which raw mode means "automatic"
            value = self._ae_auto_mode if value else AE_MANUAL
        self._request(SET_CUR, control, _encode(int(value), control.size, control.signed))

    def describe(self) -> str:
        return f"UVC webcam {self.usb_id}"
