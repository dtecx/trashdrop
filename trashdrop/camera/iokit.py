"""macOS transport for UVC control requests: a small IOKit helper via ctypes.

The helper is ``uvc_iokit.c`` next to this file, compiled into a dylib the
first time it is needed (Xcode Command Line Tools provide the compiler) and
rebuilt whenever the source changes. It lives in ``_build/`` inside the
package, so nothing is installed anywhere else and deleting the project
removes it.

Measured on this project's MacBook with a C920: as a normal user, no sudo, the
device opens and every control can be read and written -- while libusb, on the
same machine, was refused even as root. See uvc_iokit.c for why.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("uvc_iokit.c")
BUILD_DIR = Path(__file__).with_name("_build")
LIBRARY = BUILD_DIR / "libtrashdrop_uvc.dylib"

# IOReturn codes worth naming. Everything else is reported as hex.
IORETURN_NAMES = {
    0xE00002C0: "no device",
    0xE00002C1: "not privileged",
    0xE00002C5: "exclusive access (another client holds the device)",
    0xE00002CD: "device not open",
    0xE00002D6: "timeout",
    0xE00002E2: "not permitted",
    0xE000404F: "pipe stalled (the camera does not support this request)",
}
PERMISSION_CODES = {0xE00002C1, 0xE00002C5, 0xE00002E2}
STALL_CODES = {0xE000404F}

_library: ctypes.CDLL | None = None


class TransportError(RuntimeError):
    """A control request failed at the USB level."""

    def __init__(self, code: int, *, permission: bool = False, stall: bool = False, detail: str = "") -> None:
        self.code = code
        self.permission = permission
        self.stall = stall
        name = detail or IORETURN_NAMES.get(code, "")
        super().__init__(f"USB request failed: {code:#x}{f' ({name})' if name else ''}")

    @classmethod
    def from_ioreturn(cls, code: int) -> "TransportError":
        code &= 0xFFFFFFFF
        return cls(code, permission=code in PERMISSION_CODES, stall=code in STALL_CODES)


def _compiler() -> list[str]:
    if shutil.which("xcrun"):
        return ["xcrun", "clang"]
    if shutil.which("clang"):
        return ["clang"]
    raise RuntimeError(
        "camera control on macOS compiles a small helper and needs a C compiler. "
        "Install the Xcode Command Line Tools: xcode-select --install"
    )


def build(force: bool = False) -> Path:
    """Compile the helper if it is missing or older than its source."""

    if not force and LIBRARY.is_file() and LIBRARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIBRARY
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    command = _compiler() + [
        "-dynamiclib", "-O2", "-Wall",
        "-framework", "IOKit", "-framework", "CoreFoundation",
        "-o", str(LIBRARY), str(SOURCE),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"building the camera helper failed:\n{result.stderr.strip()}")
    return LIBRARY


def library() -> ctypes.CDLL:
    global _library
    if _library is None:
        lib = ctypes.CDLL(str(build()))
        u8, u16, u32 = ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32
        lib.trashdrop_uvc_list.argtypes = [ctypes.POINTER(u32), ctypes.c_int]
        lib.trashdrop_uvc_list.restype = ctypes.c_int
        lib.trashdrop_uvc_open.argtypes = [u16, u16]
        lib.trashdrop_uvc_open.restype = ctypes.c_int
        lib.trashdrop_uvc_close.argtypes = []
        lib.trashdrop_uvc_close.restype = None
        lib.trashdrop_uvc_config_descriptor.argtypes = [ctypes.POINTER(u8), u32, ctypes.POINTER(u32)]
        lib.trashdrop_uvc_config_descriptor.restype = ctypes.c_int
        lib.trashdrop_uvc_request.argtypes = [u8, u8, u16, u16, ctypes.POINTER(u8), u16, ctypes.POINTER(u32)]
        lib.trashdrop_uvc_request.restype = ctypes.c_int
        _library = lib
    return _library


def list_devices() -> list[tuple[int, int]]:
    """(vendor, product) of every USB device with a UVC VideoControl interface."""

    ids = (ctypes.c_uint32 * 32)()
    count = library().trashdrop_uvc_list(ids, 32)
    return [(value >> 16, value & 0xFFFF) for value in ids[:count]]


class IOKitDevice:
    """One webcam opened through IOKit, with a pyusb-shaped ``ctrl_transfer``."""

    def __init__(self, vendor: int, product: int) -> None:
        self._lib = library()
        code = self._lib.trashdrop_uvc_open(vendor, product)
        if code == -1:
            raise RuntimeError(f"no webcam {vendor:04x}:{product:04x} on USB -- is it plugged in?")
        # A refused open is not fatal by itself: requests on the default pipe
        # may still go through, and the first one will say.
        self.open_status = code & 0xFFFFFFFF
        self.idVendor, self.idProduct = vendor, product

    def configuration_descriptor(self) -> bytes:
        buffer = (ctypes.c_uint8 * 4096)()
        length = ctypes.c_uint32(0)
        code = self._lib.trashdrop_uvc_config_descriptor(buffer, 4096, ctypes.byref(length))
        if code != 0:
            raise TransportError.from_ioreturn(code)
        return bytes(buffer[: length.value])

    def ctrl_transfer(self, request_type, request, value, index, data_or_length, timeout=1000):
        incoming = bool(request_type & 0x80)
        if incoming:
            size = int(data_or_length)
            buffer = (ctypes.c_uint8 * max(1, size))()
        else:
            payload = bytes(data_or_length or b"")
            size = len(payload)
            buffer = (ctypes.c_uint8 * max(1, size)).from_buffer_copy(payload or b"\0")
        done = ctypes.c_uint32(0)
        code = self._lib.trashdrop_uvc_request(
            request_type, request, value, index, buffer, size, ctypes.byref(done)
        )
        if code != 0:
            raise TransportError.from_ioreturn(code)
        return bytes(buffer[: done.value]) if incoming else done.value

    def close(self) -> None:
        self._lib.trashdrop_uvc_close()
