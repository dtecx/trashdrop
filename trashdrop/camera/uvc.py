"""Talk to a UVC webcam directly over USB: focus, exposure, white balance.

Why this exists: on macOS, OpenCV reaches the camera through AVFoundation,
which accepts and then silently ignores every property -- ``trashdrop
camcheck`` reports "controllable: nothing" for a C920. The camera itself
speaks UVC, the USB class standard every webcam implements, and UVC has
requests for exactly these controls.

How the requests reach the camera depends on the OS:

* **macOS** -- a small IOKit helper (``iokit.py``). Measured on this project's
  MacBook with a C920: works as a normal user, no sudo, even while another app
  is streaming. libusb was tried first and refused even as root; the reason is
  in ``uvc_iokit.c``.
* **Linux** -- libusb (``pip install pyusb libusb-package``), with the usual
  udev permission on the device.

Everything above the transport -- control names, ranges, the auto-exposure
mapping -- is the same on both.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass

from .iokit import TransportError

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
    """The OS refused the control request."""


def permission_hint(error: Exception | None = None) -> str:
    detail = f" ({error})" if error else ""
    if sys.platform == "darwin":
        return (
            f"macOS refused the camera request{detail}. Unplug and replug the camera, "
            "then check with: uv run trashdrop camera probe"
        )
    return (
        f"the OS refused the camera request{detail}. On Linux, give your user access "
        "to the device (a udev rule for the camera's vendor id), or run as root."
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


def parse_configuration(config: bytes) -> tuple[int, int, int] | None:
    """(VideoControl interface number, camera id, processing id) from a whole
    configuration descriptor, or None if there is no usable VideoControl."""

    position = 0
    while position + 2 <= len(config):
        length, kind = config[position], config[position + 1]
        if length < 2:
            break
        if kind == 0x04 and length >= 9 and position + 7 <= len(config):
            number, klass, subclass = config[position + 2], config[position + 5], config[position + 6]
            if klass == VIDEO_CLASS and subclass == VIDEO_CONTROL_SUBCLASS:
                # The class-specific descriptors follow until the next interface.
                start = end = position + length
                while end + 2 <= len(config) and config[end] >= 2 and config[end + 1] != 0x04:
                    end += config[end]
                camera_id, processing_id = parse_video_control(config[start:end])
                if camera_id is not None and processing_id is not None:
                    return number, camera_id, processing_id
        position += length
    return None


def _decode(data, size: int, signed: bool) -> int:
    return int.from_bytes(bytes(data[:size]), "little", signed=signed)


def _encode(value: int, size: int, signed: bool) -> bytes:
    return int(value).to_bytes(size, "little", signed=signed)


class _LibusbDevice:
    """pyusb device with its errors translated to TransportError."""

    def __init__(self, device) -> None:
        self._device = device
        self.idVendor, self.idProduct = device.idVendor, device.idProduct

    def ctrl_transfer(self, *args):
        import usb.core

        try:
            return self._device.ctrl_transfer(*args)
        except usb.core.USBError as error:
            number = getattr(error, "errno", None)
            raise TransportError(
                number or -1, permission=number == 13, stall=number == 32, detail=str(error)
            ) from error


def _parse_usb_id(usb_id: str | None) -> tuple[int, int] | None:
    if not usb_id:
        return None
    vendor, product = (int(part, 16) for part in usb_id.split(":"))
    return vendor, product


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
        """The UVC camera matching ``vvvv:pppp``, or the only one plugged in.

        On an Apple-silicon MacBook the built-in camera is not a USB device,
        so a lone USB webcam needs no id. Several need one.
        """

        wanted = _parse_usb_id(usb_id)
        if sys.platform == "darwin":
            return cls._find_iokit(wanted)
        return cls._find_libusb(wanted)

    @staticmethod
    def _only(ids: list[tuple[int, int]]) -> tuple[int, int]:
        """The one camera there is. With several, refuse rather than guess.

        With arm cameras plugged in next to the webcam, "the first one found"
        is whichever the OS lists first -- and tuning the wrong camera has
        already cost this project an afternoon.
        """

        if not ids:
            raise RuntimeError("no UVC webcam found on USB -- is it plugged in?")
        if len(ids) > 1:
            listed = ", ".join(f"{vendor:04x}:{product:04x}" for vendor, product in ids)
            raise RuntimeError(
                f"several UVC cameras are plugged in ({listed}); say which one with --usb-id, "
                "or name the overhead camera in rig.toml"
            )
        return ids[0]

    @classmethod
    def _find_iokit(cls, wanted: tuple[int, int] | None) -> "UvcCamera":
        from .iokit import IOKitDevice, list_devices

        device = IOKitDevice(*(wanted or cls._only(list_devices())))
        found = parse_configuration(device.configuration_descriptor())
        if found is None:
            device.close()
            raise RuntimeError(f"{device.idVendor:04x}:{device.idProduct:04x} has no UVC VideoControl interface")
        return cls(device, *found)

    @classmethod
    def _find_libusb(cls, wanted: tuple[int, int] | None) -> "UvcCamera":
        try:
            import libusb_package
            import usb.core
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("camera control on this OS needs: uv sync --extra rig") from error

        backend = libusb_package.get_libusb1_backend()
        found = []
        for device in usb.core.find(find_all=True, backend=backend):
            if wanted and (device.idVendor, device.idProduct) != wanted:
                continue
            for configuration in device:
                for interface in configuration:
                    if (
                        interface.bInterfaceClass == VIDEO_CLASS
                        and interface.bInterfaceSubClass == VIDEO_CONTROL_SUBCLASS
                    ):
                        camera_id, processing_id = parse_video_control(bytes(interface.extra_descriptors))
                        if camera_id is not None and processing_id is not None:
                            found.append((device, interface.bInterfaceNumber, camera_id, processing_id))
        if not wanted:
            cls._only([(device.idVendor, device.idProduct) for device, *_ in found])
        if not found:
            raise RuntimeError("no UVC webcam found on USB -- is it plugged in?")
        device, interface, camera_id, processing_id = found[0]
        return cls(_LibusbDevice(device), interface, camera_id, processing_id)

    # --- raw requests ------------------------------------------------------

    def _request(self, request: int, control: UvcControl, payload=None):
        value = control.selector << 8
        index = (self.entities[control.unit] << 8) | self.interface
        try:
            if request == SET_CUR:
                return self.device.ctrl_transfer(TO_INTERFACE, SET_CUR, value, index, payload, 1000)
            return self.device.ctrl_transfer(FROM_INTERFACE, request, value, index, control.size, 1000)
        except TransportError as error:
            if error.permission:
                raise CameraPermissionError(permission_hint(error)) from error
            raise

    def _read(self, request: int, control: UvcControl) -> int:
        return _decode(self._request(request, control), control.size, control.signed)

    def supports(self, name: str) -> bool:
        control = CONTROLS[name]
        try:
            info = self._request(GET_INFO, UvcControl(name, control.unit, control.selector, 1))
        except TransportError:
            return False
        return bool(info) and bool(info[0] & 0x01) and bool(info[0] & 0x02)

    # --- public ------------------------------------------------------------

    def ranges(self) -> dict[str, ControlRange]:
        """Every control this camera supports, with its limits."""

        if self._ranges is not None:
            return self._ranges

        found: dict[str, ControlRange] = {}
        for name, control in CONTROLS.items():
            if not self.supports(name):
                continue
            try:
                default = self._read(GET_DEF, control)
            except TransportError:
                continue

            if name == "exposure_auto":
                # GET_RES on the AE mode control is the bitmap of modes the
                # camera offers; prefer aperture priority, which is what a
                # C920 calls automatic.
                try:
                    modes = self._read(GET_RES, control)
                except TransportError:
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
            except TransportError:
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


def probe_access(usb_id: str | None = None, log=print) -> str:
    """Can this machine read and write the camera's settings? "ok" or "blocked".

    Only ``focus_auto`` is touched, and it is written back unchanged.
    """

    camera = UvcCamera.find(usb_id)
    log(
        f"found {camera.describe()}: VideoControl interface {camera.interface}, "
        f"camera terminal {camera.entities['camera']}, processing unit {camera.entities['processing']}"
    )
    control = CONTROLS["focus_auto"]
    try:
        value = _decode(camera._request(GET_CUR, control), 1, False)
        camera._request(SET_CUR, control, _encode(value, 1, False))
    except (TransportError, CameraPermissionError) as error:
        log(f"read/write: failed -- {error}")
        return "blocked"
    log(f"read/write: ok (focus_auto = {value}, written back unchanged)")
    return "ok"
