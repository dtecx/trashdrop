"""A printable A4 sheet that is both a focus target and the calibration.

Four ArUco markers sit exactly on the corners of the pick zone (20 x 15 cm), so
the printed sheet *is* the zone: lay it where the items will land, detect the
markers, and the homography from pixels to table coordinates falls out. A
Siemens star in the middle gives ``camera tune`` something sharp to focus on.

Written as a PDF with an exact physical page size, not a PNG. An image has no
reliable print size -- print dialogs guess, often "fit to page" -- and a sheet
printed at 97 % makes every coordinate 3 % wrong, which the arm then misses by.
The scale bar on the sheet is there to check: it must measure 100 mm.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from ..station import PICK_ZONE

DICTIONARY = "DICT_4X4_50"
DPI = 300
PAGE_MM = (297.0, 210.0)  # A4 landscape
MARKER_MM = 35.0

# Marker id -> table coordinate of its centre. Ids run clockwise from the
# corner nearest the FRONT arm on the -x side, matching the way the sheet is
# laid: top edge of the page toward the front arm.
_min_x, _max_x, _min_y, _max_y = PICK_ZONE.bounds
MARKER_WORLD: dict[int, tuple[float, float]] = {
    0: (_min_x, _max_y),
    1: (_max_x, _max_y),
    2: (_max_x, _min_y),
    3: (_min_x, _min_y),
}


# The same centres in the sheet's own frame, in cm: origin at the middle of the
# zone, x to the right of the printed page, y towards its top edge.
MARKER_SHEET_CM: dict[int, tuple[float, float]] = {
    marker_id: (round((x - PICK_ZONE.center_x) * 100.0, 2), round((y - PICK_ZONE.center_y) * 100.0, 2))
    for marker_id, (x, y) in MARKER_WORLD.items()
}


def _px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def marker_page_positions() -> dict[int, tuple[int, int]]:
    """Pixel centre of each marker on the rendered page."""

    width, height = _px(PAGE_MM[0]), _px(PAGE_MM[1])
    cx, cy = width / 2, height / 2
    positions = {}
    for marker_id, (x, y) in MARKER_WORLD.items():
        # Page x grows with table x; page y grows DOWN while table y grows
        # toward the front arm, i.e. up the page.
        dx_mm = (x - PICK_ZONE.center_x) * 1000.0
        dy_mm = (y - PICK_ZONE.center_y) * 1000.0
        positions[marker_id] = (int(round(cx + _px(dx_mm))), int(round(cy - _px(dy_mm))))
    return positions


def render_sheet() -> np.ndarray:
    """The sheet as a greyscale image at 300 dpi."""

    import cv2

    width, height = _px(PAGE_MM[0]), _px(PAGE_MM[1])
    page = np.full((height, width), 255, np.uint8)
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, DICTIONARY))
    side = _px(MARKER_MM)

    # Pick-zone outline, drawn FIRST: it runs through the marker centres, and
    # anything crossing a marker afterwards corrupts its bits so the detector
    # silently stops finding it.
    positions = marker_page_positions()
    corners = np.array([positions[i] for i in (0, 1, 2, 3)], np.int32)
    cv2.polylines(page, [corners], True, 170, 3, cv2.LINE_AA)

    for marker_id, (x, y) in positions.items():
        marker = aruco.generateImageMarker(dictionary, marker_id, side)
        top, left = y - side // 2, x - side // 2
        # A white quiet zone around each marker, so the outline never touches it.
        pad = _px(4)
        page[top - pad : top + side + pad, left - pad : left + side + pad] = 255
        page[top : top + side, left : left + side] = marker
        cv2.putText(page, f"{marker_id}", (left, top - _px(6)), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 0, 5, cv2.LINE_AA)

    # Siemens star: sharp radial edges at every scale, ideal for focusing.
    cx, cy, radius = width // 2, height // 2, _px(38)
    for spoke in range(36):
        if spoke % 2:
            continue
        a0, a1 = np.deg2rad(spoke * 10), np.deg2rad((spoke + 1) * 10)
        wedge = np.array(
            [[cx, cy], [cx + radius * np.cos(a0), cy + radius * np.sin(a0)],
             [cx + radius * np.cos(a1), cy + radius * np.sin(a1)]],
            np.int32,
        )
        cv2.fillPoly(page, [wedge], 0, cv2.LINE_AA)

    # Text and the scale bar go in the free bands between the markers, never
    # over them: anything drawn across a marker can stop it being detected.
    def centred(text: str, baseline: int, scale: float, thickness: int, shade: int) -> None:
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(page, text, ((width - text_width) // 2, baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, shade, thickness, cv2.LINE_AA)

    centred("TOP EDGE -> toward the FRONT arm", _px(10), 1.6, 4, 0)
    centred(
        f"TrashDrop pick zone {PICK_ZONE.half_x * 200:.0f} x {PICK_ZONE.half_y * 200:.0f} cm - "
        f"ArUco {DICTIONARY} ids 0-3",
        _px(17), 1.05, 2, 90,
    )

    # A 100 mm scale bar, to check the print came out at actual size.
    bar_y = height - _px(13)
    bar_x0 = (width - _px(100)) // 2
    bar_x1 = bar_x0 + _px(100)
    cv2.line(page, (bar_x0, bar_y), (bar_x1, bar_y), 0, 6)
    for tick in range(11):
        x = bar_x0 + _px(10 * tick)
        cv2.line(page, (x, bar_y - _px(3 if tick % 5 else 5)), (x, bar_y), 0, 4)
    centred("must measure 100 mm - print at ACTUAL SIZE / 100 %", bar_y - _px(7), 1.05, 2, 0)
    return page


def write_pdf(image: np.ndarray, path: Path, page_mm: tuple[float, float] = PAGE_MM) -> Path:
    """Minimal one-page PDF with the image filling the page at physical size.

    Hand-written rather than pulling in a PDF library: it is one image on one
    page, and lossless (Flate) so marker edges stay crisp.
    """

    height, width = image.shape[:2]
    points = (page_mm[0] / 25.4 * 72.0, page_mm[1] / 25.4 * 72.0)
    pixels = zlib.compress(np.ascontiguousarray(image, np.uint8).tobytes(), 9)
    draw = f"q {points[0]:.2f} 0 0 {points[1]:.2f} 0 0 cm /Im0 Do Q".encode()

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {points[0]:.2f} {points[1]:.2f}] "
            "/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ).encode(),
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(pixels)} >>\nstream\n"
        ).encode()
        + pixels
        + b"\nendstream",
        f"<< /Length {len(draw)} >>\nstream\n".encode() + draw + b"\nendstream",
    ]

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(output))
    return path


def write_sheet(out_dir: Path) -> tuple[Path, Path]:
    import cv2

    page = render_sheet()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = write_pdf(page, out_dir / "trashdrop_markers_A4.pdf")
    png = out_dir / "trashdrop_markers_A4.png"
    cv2.imwrite(str(png), page)
    return pdf, png
