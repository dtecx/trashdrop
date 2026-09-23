"""Camera control: the UVC protocol, camera.toml, the tune, the marker sheet.

The real webcam needs root on macOS, so the protocol layer is exercised against
a fake USB device that answers the way a C920 does -- including stalling on
GET_MIN/GET_MAX for on/off controls, which the UVC spec allows and a real C920
does.
"""

from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

import pytest

usb_core = pytest.importorskip("usb.core", reason="needs the camera extra")
cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
import numpy as np  # noqa: E402

from trashdrop.camera import config as camera_config  # noqa: E402
from trashdrop.camera.config import CameraSettings, apply, load, render, save  # noqa: E402
from trashdrop.camera.markers import MARKER_WORLD, marker_page_positions, render_sheet, write_pdf  # noqa: E402
from trashdrop.camera.tune import TuneError, focus_chart, sharpness, tune  # noqa: E402
from trashdrop.camera.uvc import (  # noqa: E402
    AE_APERTURE_PRIORITY,
    AE_MANUAL,
    CONTROLS,
    CameraPermissionError,
    ControlRange,
    UvcCamera,
    parse_video_control,
)
from trashdrop.station import PICK_ZONE  # noqa: E402

GET_CUR, GET_MIN, GET_MAX, GET_RES, GET_INFO, GET_DEF, SET_CUR = 0x81, 0x82, 0x83, 0x84, 0x86, 0x87, 0x01


def video_control_descriptors() -> bytes:
    """Class-specific VC descriptors shaped like a C920's."""

    header = bytes([13, 0x24, 0x01, 0x00, 0x01, 0x6B, 0x00, 0x00, 0x6C, 0xDC, 0x02, 0x01, 0x01])
    camera_terminal = bytes([18, 0x24, 0x02, 0x01, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x2A, 0x00, 0x02])
    processing_unit = bytes([11, 0x24, 0x05, 0x03, 0x01, 0x00, 0x40, 0x02, 0x5B, 0x17, 0x00])
    output_terminal = bytes([9, 0x24, 0x03, 0x04, 0x01, 0x01, 0x00, 0x03, 0x00])
    return header + camera_terminal + processing_unit + output_terminal


class FakeC920:
    """Answers UVC control requests like a C920, from a table of controls."""

    idVendor, idProduct = 0x046D, 0x08E5

    # name -> (min, max, step, default); booleans stall on MIN/MAX/RES.
    SPEC = {
        "exposure_auto": (None, None, 0x09, AE_APERTURE_PRIORITY),
        "exposure_priority": (None, None, None, 1),
        "exposure": (3, 2047, 1, 250),
        "focus_auto": (None, None, None, 1),
        "focus": (0, 250, 5, 0),
        "zoom": (100, 500, 1, 100),
        "gain": (0, 255, 1, 0),
        "white_balance_auto": (None, None, None, 1),
        "white_balance": (2000, 6500, 1, 4000),
        "power_line_frequency": (0, 2, 1, 2),
        "brightness": (0, 255, 1, 128),
    }

    def __init__(self, deny: bool = False) -> None:
        self.deny = deny
        self.by_key = {
            ({"camera": 1, "processing": 3}[c.unit], c.selector): c for c in CONTROLS.values() if c.name in self.SPEC
        }
        self.current = {name: spec[3] for name, spec in self.SPEC.items()}
        self.sets: list[tuple[str, int]] = []

    def ctrl_transfer(self, request_type, request, value, index, data, timeout=None):
        if self.deny:
            raise usb_core.USBError("Access denied (insufficient permissions)", errno=13)
        control = self.by_key.get((index >> 8, value >> 8))
        if control is None:
            raise usb_core.USBError("Pipe error", errno=32)
        minimum, maximum, step, default = self.SPEC[control.name]
        if request == SET_CUR:
            number = int.from_bytes(bytes(data), "little", signed=control.signed)
            self.current[control.name] = number
            self.sets.append((control.name, number))
            return len(data)
        answer = {
            GET_INFO: 0x03,
            GET_CUR: self.current[control.name],
            GET_DEF: default,
            GET_MIN: minimum,
            GET_MAX: maximum,
            GET_RES: step,
        }[request]
        if answer is None:
            raise usb_core.USBError("Pipe error", errno=32)  # a real C920 stalls here
        size = 1 if request == GET_INFO else control.size
        return list(int(answer).to_bytes(size, "little", signed=control.signed))


def fake_camera(deny: bool = False) -> tuple[UvcCamera, FakeC920]:
    device = FakeC920(deny=deny)
    return UvcCamera(device, interface=0, camera_id=1, processing_id=3), device


class DescriptorTests(unittest.TestCase):
    def test_finds_the_camera_terminal_and_processing_unit(self) -> None:
        self.assertEqual(parse_video_control(video_control_descriptors()), (1, 3))

    def test_truncated_descriptors_do_not_crash(self) -> None:
        blob = video_control_descriptors()
        # Cut inside the camera terminal, before its type field: nothing usable.
        self.assertEqual(parse_video_control(blob[:16]), (None, None))
        # Cut after the terminal's id and type but before the processing unit:
        # take what is there, invent nothing.
        self.assertEqual(parse_video_control(blob[:20]), (1, None))
        self.assertEqual(parse_video_control(b""), (None, None))


class UvcTests(unittest.TestCase):
    def test_ranges_survive_stalls_on_boolean_controls(self) -> None:
        camera, _ = fake_camera()
        ranges = camera.ranges()
        self.assertEqual(ranges["focus"], ControlRange(0, 250, 5, 0))
        self.assertEqual(ranges["focus_auto"], ControlRange(0, 1, 1, 1))
        self.assertEqual(ranges["exposure_auto"].default, 1)

    def test_exposure_auto_speaks_in_booleans(self) -> None:
        camera, device = fake_camera()
        camera.set("exposure_auto", 0)
        self.assertEqual(device.current["exposure_auto"], AE_MANUAL)
        self.assertEqual(camera.get("exposure_auto"), 0)
        camera.set("exposure_auto", 1)
        self.assertEqual(device.current["exposure_auto"], AE_APERTURE_PRIORITY)
        self.assertEqual(camera.get("exposure_auto"), 1)

    def test_permission_errors_explain_sudo(self) -> None:
        camera, _ = fake_camera(deny=True)
        with self.assertRaises(CameraPermissionError) as caught:
            camera.get("focus")
        self.assertIn("sudo", str(caught.exception))


class ConfigTests(unittest.TestCase):
    def settings(self) -> CameraSettings:
        return CameraSettings(
            backend="uvc",
            device="046d:08e5",
            controls={"focus_auto": 0, "focus": 45, "exposure_auto": 0, "exposure": 156,
                      "white_balance_auto": 0, "white_balance": 4200, "power_line_frequency": 1},
            tuned_at="2026-09-23 18:00",
            note="home rig, 70 cm",
        )

    def test_render_is_valid_toml_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.toml"
            path.write_text(render(self.settings()), encoding="utf-8")
            self.assertEqual(load(path).controls, self.settings().controls)
            parsed = tomllib.loads(path.read_text(encoding="utf-8"))
            self.assertIs(parsed["controls"]["focus_auto"], False)  # a human reads true/false, not 0/1

    def test_every_value_is_explained_in_the_file(self) -> None:
        text = render(self.settings(), fake_camera()[0].ranges())
        self.assertIn("Autofocus hunts", text)
        self.assertIn("camera range 0..250, step 5", text)

    def test_a_typo_is_refused_not_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.toml"
            path.write_text("[controls]\nfocuss = 40\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load(path)
            self.assertIn("focuss", str(caught.exception))

    def test_modes_go_before_their_values(self) -> None:
        camera, device = fake_camera()
        apply(camera, self.settings(), pause=lambda _s: None)
        order = [name for name, _ in device.sets]
        self.assertLess(order.index("focus_auto"), order.index("focus"))
        self.assertLess(order.index("exposure_auto"), order.index("exposure"))
        self.assertLess(order.index("white_balance_auto"), order.index("white_balance"))

    def test_values_are_fitted_to_the_camera(self) -> None:
        camera, device = fake_camera()
        settings = self.settings()
        settings.controls["focus"] = 47  # C920 focus moves in steps of 5
        results = {r.name: r for r in apply(camera, settings, pause=lambda _s: None)}
        self.assertEqual(device.current["focus"], 45)
        self.assertIn("adjusted", results["focus"].note)
        self.assertTrue(all(r.ok for r in results.values()))

    def test_save_keeps_a_backup_of_the_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.toml"
            save(self.settings(), path)
            save(self.settings(), path)
            self.assertEqual(len(list(Path(directory).glob("camera.*.toml.bak"))), 1)


class FocusScene:
    """A camera whose image is sharpest at one lens position, like a real one."""

    def __init__(self, camera: UvcCamera, device: FakeC920, true_focus: int = 45, textured: bool = True):
        self.camera, self.device, self.true_focus = camera, device, true_focus
        rng = np.random.default_rng(3)
        if textured:
            tile = (rng.random((45, 80)) > 0.5).astype(np.uint8) * 255
            self.base = cv2.resize(tile, (640, 360), interpolation=cv2.INTER_NEAREST)
        else:
            self.base = np.full((360, 640), 170, np.uint8)
        self.base = cv2.cvtColor(self.base, cv2.COLOR_GRAY2BGR)

    def read(self):
        error = abs(self.device.current["focus"] - self.true_focus)
        radius = 1 + error // 10
        frame = cv2.GaussianBlur(self.base, (2 * radius + 1, 2 * radius + 1), 0) if error else self.base.copy()
        return True, frame


class TuneTests(unittest.TestCase):
    def test_the_sweep_finds_the_sharp_position(self) -> None:
        camera, device = fake_camera()
        result = tune(FocusScene(camera, device, true_focus=45), camera,
                      converge_frames=3, settle_frames=1, log=lambda *_: None)
        self.assertEqual(result.controls["focus"], 45)
        self.assertEqual(result.controls["focus_auto"], 0)
        self.assertEqual(device.current["focus"], 45, "the camera must be left at the chosen focus")

    def test_auto_exposure_choice_is_frozen_not_invented(self) -> None:
        camera, device = fake_camera()
        device.current["exposure"] = 312  # what "auto" converged to
        device.current["white_balance"] = 3900
        result = tune(FocusScene(camera, device), camera,
                      converge_frames=3, settle_frames=1, log=lambda *_: None)
        self.assertEqual(result.controls["exposure"], 312)
        self.assertEqual(result.controls["white_balance"], 3900)
        self.assertEqual(result.controls["exposure_auto"], 0)
        self.assertEqual(result.controls["white_balance_auto"], 0)
        self.assertEqual(result.controls["power_line_frequency"], 1)  # 50 Hz, Europe

    def test_a_blank_scene_is_refused(self) -> None:
        camera, device = fake_camera()
        with self.assertRaises(TuneError) as caught:
            tune(FocusScene(camera, device, textured=False), camera,
                 converge_frames=1, settle_frames=1, log=lambda *_: None)
        self.assertIn("printed page", str(caught.exception))

    def test_chart_marks_the_choice(self) -> None:
        self.assertIn("<- chosen", focus_chart({0: 1.0, 5: 3.0, 10: 2.0}))


class MarkerSheetTests(unittest.TestCase):
    def test_markers_are_detectable_where_the_zone_corners_are(self) -> None:
        page = render_sheet()
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters()
        )
        corners, ids, _ = detector.detectMarkers(page)
        self.assertIsNotNone(ids)
        found = {int(i): c.reshape(4, 2).mean(axis=0) for i, c in zip(ids.flatten(), corners)}
        self.assertEqual(set(found), set(MARKER_WORLD))
        for marker_id, (x, y) in marker_page_positions().items():
            self.assertAlmostEqual(found[marker_id][0], x, delta=2)
            self.assertAlmostEqual(found[marker_id][1], y, delta=2)

    def test_nothing_is_drawn_over_a_marker(self) -> None:
        # The detector can still cope with a clean render when a line or a
        # caption crosses a marker; a printed sheet under a webcam cannot. So
        # every marker and its quiet zone must be exactly what ArUco generated.
        from trashdrop.camera.markers import DICTIONARY, MARKER_MM, _px

        page = render_sheet()
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
        side, pad = _px(MARKER_MM), _px(4)
        for marker_id, (x, y) in marker_page_positions().items():
            expected = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
            expected[pad : pad + side, pad : pad + side] = cv2.aruco.generateImageMarker(
                dictionary, marker_id, side
            )
            top, left = y - side // 2 - pad, x - side // 2 - pad
            actual = page[top : top + side + 2 * pad, left : left + side + 2 * pad]
            self.assertTrue(np.array_equal(actual, expected), f"something is drawn over marker {marker_id}")

    def test_marker_centres_are_exactly_the_zone_corners(self) -> None:
        self.assertEqual(set(MARKER_WORLD.values()), set(PICK_ZONE.corners()))

    def test_markers_span_20_by_15_cm_at_300_dpi(self) -> None:
        positions = marker_page_positions()
        mm_per_px = 25.4 / 300
        self.assertAlmostEqual((positions[1][0] - positions[0][0]) * mm_per_px, 200.0, delta=0.2)
        self.assertAlmostEqual((positions[3][1] - positions[0][1]) * mm_per_px, 150.0, delta=0.2)

    def test_pdf_page_is_a4_landscape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_pdf(np.full((100, 141), 255, np.uint8), Path(directory) / "sheet.pdf")
            data = path.read_bytes()
        self.assertTrue(data.startswith(b"%PDF-1.4"))
        self.assertIn(b"/MediaBox [0 0 841.89 595.28]", data)
        self.assertTrue(data.rstrip().endswith(b"%%EOF"))

    def test_sheet_has_enough_texture_to_focus_on(self) -> None:
        page = cv2.cvtColor(cv2.resize(render_sheet(), (1280, 905)), cv2.COLOR_GRAY2BGR)
        blank = np.full_like(page, 255)
        self.assertGreater(sharpness(page), 10 * (sharpness(blank) + 1.0))


class HandBackTests(unittest.TestCase):
    def test_hand_back_is_a_no_op_without_sudo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.toml"
            path.write_text("x", encoding="utf-8")
            camera_config.hand_back(path)  # must not raise as a normal user
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
