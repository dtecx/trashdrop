"""One-handed, manual dataset capture on a calibrated tabletop.

One Q/SPACE press stores exactly one image. The ArUco calibration selects the
physical pick area; background subtraction only gives live feedback and never
presses the shutter on the operator's behalf.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..station import SORT_CATEGORIES
from .manifest import ManifestWriter, read_manifest
from .shutter import AutoShutter, ShutterView
from .zone import DEFAULT_CALIBRATION, DetectionZone, load_zone, save_zone

WINDOW = "trashdrop capture"
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 900
VIDEO_TOP = 118
VIDEO_BOTTOM = 670
STRIP_TOP = 680
THUMB_COUNT = 5
UNDO_BUTTON = (1030, STRIP_TOP + 2, 1260, STRIP_TOP + 31)
LIGHTING_PRESETS = ("default", "daylight", "side_light")
CAPTURE_CATEGORIES = SORT_CATEGORIES
CLASS_KEYS = {ord(str(index + 1)): category for index, category in enumerate(CAPTURE_CATEGORIES)}
CLASS_HINTS = "    ".join(f"{index} {category.upper()}" for index, category in enumerate(CAPTURE_CATEGORIES, 1))
WHITE = (240, 240, 240)
MUTED = (175, 175, 175)
CYAN = (255, 210, 0)
GREEN = (95, 225, 95)
RED = (80, 90, 245)
RUSSIAN_KEYBOARD = dict(zip("йцукеач", "qwertfx"))


def _capture_key(key: int) -> int:
    """Accept QWERTY shortcuts with either Latin or Russian keyboard layout."""

    if key < 0:
        return key
    try:
        letter = chr(key).lower()
    except ValueError:
        return key
    return ord(RUSSIAN_KEYBOARD.get(letter, letter))


@dataclass
class CaptureConfig:
    session: str
    root: Path = Path("data")
    source: str | int = 0
    width: int = 1920
    height: int = 1080
    burst: int = 1  # Kept for older callers; the UI always takes one frame.
    burst_interval: float = 0.0
    lighting: str = "default"
    auto: bool = False  # Kept for older callers; automatic shooting is disabled.
    zone_calibration: Path = DEFAULT_CALIBRATION


def open_camera(source: str | int, width: int, height: int):
    import cv2

    capture = cv2.VideoCapture(source)
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {source!r}. On macOS, run this from Terminal "
            "with Camera access in System Settings > Privacy & Security > Camera. "
            "The rootless UVC helper controls settings but cannot grant video access. "
            "Run `uv run trashdrop cameras` if the stream index is uncertain."
        )
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(f"Camera {source!r} opened but returned no frame")
    return capture


def next_object_id(raw_root: Path, category: str, current: str | None = None) -> str:
    pattern = re.compile(rf"^{re.escape(category)}_(\d+)$")
    used = [0]
    folder = raw_root / category
    if folder.is_dir():
        for child in folder.iterdir():
            match = pattern.match(child.name)
            if child.is_dir() and match:
                used.append(int(match.group(1)))
    if current:
        match = pattern.match(current)
        if match:
            used.append(int(match.group(1)))
    return f"{category}_{max(used) + 1:02d}"


def _clean_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())[:48]


def _lighting_base(value: str) -> str:
    return re.sub(r"_v\d+$", "", value)


def _prompt(message: str, current: str, options: list[str] | None = None) -> str:
    print(f"\n  {message} (type in this terminal)")
    if options:
        print(f"  options: {', '.join(options)}")
    answer = input(f"  new value, enter keeps {current!r}: ").strip()
    if not answer:
        return current
    if options and answer not in options:
        return current
    return answer


class _OpenCvUi:
    def __init__(self) -> None:
        import cv2

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self.fullscreen = True
        self._clicked_key: int | None = None

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        import cv2

        x0, y0, x1, y1 = UNDO_BUTTON
        if event == cv2.EVENT_LBUTTONDOWN and x0 <= x <= x1 and y0 <= y <= y1:
            self._clicked_key = ord("w")

    def show(self, image) -> None:
        import cv2

        cv2.imshow(WINDOW, image)

    def key(self) -> int:
        import cv2

        # A one-millisecond event window misses short key presses when image
        # analysis takes longer than the key is held down.
        key = cv2.waitKeyEx(20)
        if self._clicked_key is not None:
            key, self._clicked_key = self._clicked_key, None
        return key

    def toggle_fullscreen(self) -> None:
        import cv2

        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(
            WINDOW, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
        )

    def close(self) -> None:
        import cv2

        cv2.destroyWindow(WINDOW)


def _text(canvas, value: str, x: int, y: int, colour=WHITE, scale: float = 0.75,
          thickness: int = 1) -> None:
    import cv2

    cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, thickness, cv2.LINE_AA)


class _Session:
    def __init__(self, config: CaptureConfig, category: str, object_id: str | None, clock) -> None:
        self.config = config
        self.clock = clock
        self.raw_root = config.root / "raw" / config.session
        self.bg_root = config.root / "bg" / config.session
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.bg_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.raw_root / "manifest.csv"
        self.rows = read_manifest(manifest_path) if manifest_path.is_file() else []
        self.zone_path = self.raw_root / "zone.json"
        session_zone = load_zone(self.zone_path)
        current_zone = load_zone(config.zone_calibration)
        if session_zone is not None and current_zone is not None and session_zone != current_zone:
            if self.rows:
                raise ValueError(
                    f"Capture zone changed since session {config.session!r} began. "
                    "Start a new --session to keep its photos and autolabel calibration consistent."
                )
            # An empty session has no photos tied to its original polygon.
            save_zone(self.zone_path, current_zone)
            print(f"Updated empty session {config.session!r} to the current camera zone")
            self.zone = current_zone
        elif session_zone is not None:
            self.zone = session_zone
        else:
            self.zone = current_zone
            if self.zone is not None:
                save_zone(self.zone_path, self.zone)
        self.manifest = ManifestWriter(manifest_path)
        self.category = category
        self.object_id = object_id or next_object_id(self.raw_root, category)
        self.lighting = self.latest_lighting(config.lighting)
        self.background = None
        self.background_wall_time: float | None = None
        self.recent = self.rows[-THUMB_COUNT:]
        self.total = 0
        self.message = ""
        self.message_until = 0.0
        self.flash_until = 0.0
        self.thumbnail_cache: dict[str, np.ndarray] = {}

    @property
    def object_dir(self) -> Path:
        return self.raw_root / self.category / self.object_id

    def frames_of_current_object(self) -> int:
        return sum(row.category == self.category and row.object_id == self.object_id
                   for row in self.rows)

    def _next_frame_path(self) -> Path:
        from .review import read_rejected_frames

        folder = self.object_dir
        folder.mkdir(parents=True, exist_ok=True)
        undone = self.raw_root / "_undone" / self.category / self.object_id
        indices = []
        for parent in (folder, undone):
            for path in parent.glob("frame_*.jpg"):
                try:
                    indices.append(int(path.stem.split("_")[-1]))
                except ValueError:
                    pass
        prefix = f"raw/{self.config.session}/{self.category}/{self.object_id}/"
        for image in read_rejected_frames(self.config.root, self.config.session):
            if image.startswith(prefix):
                try:
                    indices.append(int(Path(image).stem.split("_")[-1]))
                except ValueError:
                    pass
        return folder / f"frame_{max(indices, default=-1) + 1:04d}.jpg"

    def save(self, frame) -> Path:
        import cv2

        path = self._next_frame_path()
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Could not write {path}")
        self.manifest.append(
            image=str(path.relative_to(self.config.root)),
            category=self.category, object_id=self.object_id,
            lighting=self.lighting, session=self.config.session,
        )
        self.rows = read_manifest(self.manifest.path)
        self.recent = self.rows[-THUMB_COUNT:]
        self.total += 1
        return path

    def undo(self) -> Path | None:
        rows = self.rows
        if not rows:
            return None
        row = rows[-1]
        path = self.config.root / row.image
        if not path.is_file():
            raise FileNotFoundError(f"Last captured photo is missing: {path}")
        quarantine = self.raw_root / "_undone" / row.category / row.object_id / path.name
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            raise FileExistsError(f"Undo destination already exists: {quarantine}")
        # Rejection also removes derived labels/crops if autolabel was run while
        # the capture window was open. The sidecar keeps this path excluded.
        from .review import reject_frames

        reject_frames(self.config.session, [row.image], self.config.root)
        path.rename(quarantine)
        try:
            self.manifest.pop_last(row.image)
        except Exception:
            quarantine.rename(path)
            raise
        self.rows = read_manifest(self.manifest.path)
        self.recent = self.rows[-THUMB_COUNT:]
        self.thumbnail_cache.pop(row.image, None)
        self.total = max(0, self.total - 1)
        return path

    def load_background(self):
        import cv2

        path = self.bg_root / f"{self.lighting}.jpg"
        self.background = cv2.imread(str(path)) if path.is_file() else None
        self.background_wall_time = path.stat().st_mtime if self.background is not None else None

    def latest_lighting(self, base: str) -> str:
        """Select the newest reference for a lighting preset when resuming."""

        if re.search(r"_v\d+$", base):
            return base
        choices = [(0, base)]
        pattern = re.compile(rf"^{re.escape(base)}_v(\d+)$")
        for path in self.bg_root.glob(f"{base}*.jpg"):
            match = pattern.fullmatch(path.stem)
            if match:
                choices.append((int(match.group(1)), path.stem))
        return max(choices)[1]

    def save_background(self, frame) -> Path:
        import cv2

        path = self.bg_root / f"{self.lighting}.jpg"
        if path.is_file() and any(row.lighting == self.lighting for row in self.rows):
            # Every manifest row names its exact reference by lighting key.
            # Never overwrite that reference after photos have used it.
            base = _lighting_base(self.lighting)
            latest = self.latest_lighting(base)
            version = int(latest.rsplit("_v", 1)[1]) + 1 if latest != base else 1
            self.lighting = f"{base}_v{version:02d}"
            path = self.bg_root / f"{self.lighting}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Could not write {path}")
        self.background = frame.copy()
        self.background_wall_time = time.time()
        return path

    def say(self, message: str, seconds: float = 2.2) -> None:
        self.message = message
        self.message_until = self.clock() + seconds
        print(f"  {message}")

    def _thumbnail(self, row):
        import cv2

        if row.image not in self.thumbnail_cache:
            image = cv2.imread(str(self.config.root / row.image))
            if image is not None:
                if self.zone is not None and self.zone.matches(image):
                    image = self.zone.crop(image)
                self.thumbnail_cache[row.image] = image
        return self.thumbnail_cache.get(row.image)

    def render(self, frame, view: ShutterView):
        import cv2

        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (31, 31, 34), np.uint8)
        video_height = VIDEO_BOTTOM - VIDEO_TOP
        scale = min(CANVAS_WIDTH / frame.shape[1], video_height / frame.shape[0])
        width = round(frame.shape[1] * scale)
        height = round(frame.shape[0] * scale)
        x0, y0 = (CANVAS_WIDTH - width) // 2, VIDEO_TOP + (video_height - height) // 2
        preview = cv2.resize(frame, (width, height))
        if self.zone is not None and self.zone.matches(frame):
            polygon = self.zone.preview_corners(preview.shape)
            mask = np.zeros(preview.shape[:2], np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            dimmed = cv2.convertScaleAbs(preview, alpha=0.38)
            dimmed[mask != 0] = preview[mask != 0]
            preview = dimmed
            cv2.polylines(preview, [polygon], True, CYAN, 3, cv2.LINE_AA)
        if view.box is not None:
            bx, by, bw, bh = view.box
            cv2.rectangle(preview, (round(bx * scale), round(by * scale)),
                          (round((bx + bw) * scale), round((by + bh) * scale)),
                          RED if view.tone == "problem" else GREEN, 2)
        canvas[y0 : y0 + height, x0 : x0 + width] = preview

        _text(canvas, f"[{self.category.upper()}]  {self.object_id}", 24, 39, WHITE, 0.95, 2)
        _text(canvas, f"Object: {self.frames_of_current_object()}    Session: {len(self.rows)}",
              24, 76, MUTED, 0.72)
        _text(canvas, f"Light: {self.lighting}", 990, 76, MUTED, 0.62)
        if self.background_wall_time is None:
            bg_text, bg_colour = "BACKGROUND MISSING - clear zone, press R", RED
        else:
            age = (time.time() - self.background_wall_time) / 60
            bg_text = f"Background: {age:.0f} min old"
            bg_colour = (0, 190, 255) if age > 20 else MUTED
        _text(canvas, bg_text, 620, 39, bg_colour, 0.73)
        status = self.message if self.clock() < self.message_until else view.status
        _text(canvas, status[:90], 24, 105, GREEN if self.clock() < self.message_until else MUTED, 0.65)

        _text(canvas, "RECENT PHOTOS  /  W UNDO LAST", 24, STRIP_TOP + 25, MUTED, 0.65)
        ux0, uy0, ux1, uy1 = UNDO_BUTTON
        cv2.rectangle(canvas, (ux0, uy0), (ux1, uy1), (75, 75, 90), -1)
        cv2.rectangle(canvas, (ux0, uy0), (ux1, uy1), WHITE, 1)
        _text(canvas, "UNDO LAST  [W]", ux0 + 16, uy1 - 8, WHITE, 0.58)
        slot_w, thumb_w, thumb_h = 245, 218, 126
        for index, row in enumerate(self.recent[-THUMB_COUNT:]):
            x = 24 + index * slot_w
            image = self._thumbnail(row)
            if image is None:
                continue
            thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            canvas[STRIP_TOP + 32 : STRIP_TOP + 32 + thumb_h, x : x + thumb_w] = thumb
            border = GREEN if index == len(self.recent[-THUMB_COUNT:]) - 1 else MUTED
            cv2.rectangle(canvas, (x, STRIP_TOP + 32), (x + thumb_w, STRIP_TOP + 32 + thumb_h), border, 2)
            cv2.rectangle(canvas, (x + 2, STRIP_TOP + 32 + thumb_h - 25),
                          (x + thumb_w - 2, STRIP_TOP + 32 + thumb_h - 2), (20, 20, 20), -1)
            _text(canvas, f"{row.category.upper()}  {row.object_id}",
                  x + 6, STRIP_TOP + 32 + thumb_h - 7, WHITE, 0.54)
            _text(canvas, Path(row.image).stem, x, STRIP_TOP + 174, MUTED, 0.45)
        _text(canvas, CLASS_HINTS, 24, 871, WHITE, 0.73)
        _text(canvas, "Q/SPACE SHOT    W UNDO    E NEXT    R BACKGROUND    T LIGHT    F FULLSCREEN    ESC EXIT",
              24, 893, WHITE, 0.58)
        if self.clock() < self.flash_until:
            cv2.rectangle(canvas, (1, 1), (CANVAS_WIDTH - 2, CANVAS_HEIGHT - 2), WHITE, 8)
        return canvas


def run_capture(config: CaptureConfig, category: str = "plastic", object_id: str | None = None,
                *, capture=None, ui=None, clock=time.monotonic, prompt=_prompt) -> Path:
    if category not in CAPTURE_CATEGORIES:
        raise ValueError(f"{category!r} is not a capture category: {CAPTURE_CATEGORIES}")
    session = _Session(config, category, _clean_id(object_id) if object_id else None, clock)
    reapply_at_frame = None
    try:
        if capture is None:
            capture = open_camera(config.source, config.width, config.height)
            from ..camera import apply_saved_settings

            print(apply_saved_settings())
            reapply_at_frame = 30
            if isinstance(config.source, int):
                from ..__main__ import _require_same_camera, _webcam

                _require_same_camera(capture, config.source, _webcam(), refuse=True)
        ui = ui or _OpenCvUi()
    except Exception:
        if capture is not None:
            capture.release()
        session.manifest.close()
        raise
    shutter = AutoShutter(None, clock=clock)
    session.load_background()

    def set_analysis_background() -> None:
        if session.zone is not None and session.background is not None and session.zone.matches(session.background):
            shutter.set_background(session.zone.crop(session.background))
        else:
            shutter.set_background(None)

    set_analysis_background()
    print(f"\n{CLASS_HINTS} | Q/SPACE shoot one | W undo | E next object | R empty background | T light | F fullscreen | ESC exit")
    if session.zone is None:
        print("ZONE MISSING: place the printed sheet in view and run `uv run trashdrop camera zone` first.")
    print(f"  now shooting [{session.category}] {session.object_id}")

    frames_seen = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                if _capture_key(ui.key()) in (27, ord("x")):
                    break
                continue
            frames_seen += 1
            if reapply_at_frame is not None and frames_seen == reapply_at_frame:
                from ..camera import apply_saved_settings

                print(apply_saved_settings())

            if session.zone is None:
                view = ShutterView(False, "NO ZONE - run: uv run trashdrop camera zone", "problem")
            elif not session.zone.matches(frame):
                view = ShutterView(False, "ZONE RESOLUTION MISMATCH - recalibrate at this resolution", "problem")
            elif session.background is None or not session.zone.matches(session.background):
                view = ShutterView(False, "CLEAR THE ZONE AND PRESS R FOR BACKGROUND", "problem")
            else:
                detected = shutter.update(session.zone.crop(frame), session.zone.analysis_mask())
                status = detected.status.replace("press b", "press R").replace("(b)", "(R)")
                if detected.capture:
                    status = "READY - press Q/SPACE to save one photo"
                elif status.startswith("got it"):
                    status = "SAVED - move item to a new pose, then press Q"
                view = ShutterView(False, status, detected.tone,
                                   detected.box, detected.empty)
                if view.box is not None:
                    x, y, w, h = view.box
                    view.box = (x + session.zone.x, y + session.zone.y, w, h)
            ui.show(session.render(frame, view))
            key = _capture_key(ui.key())
            if key in (27, ord("x")):
                break
            if key in (ord("q"), ord(" ")):
                if session.zone is None or not session.zone.matches(frame):
                    session.say("No valid calibrated zone; run `trashdrop camera zone`")
                elif session.background is None or not session.zone.matches(session.background):
                    session.say("Clear the zone and press R for the background first")
                else:
                    path = session.save(frame)
                    shutter.mark_captured()
                    session.flash_until = clock() + 0.2
                    session.say(f"Saved {path.name}  [{session.category}] {session.object_id}")
            elif key == ord("w"):
                path = session.undo()
                shutter.forget_last()
                session.say(f"Undid {path.name}" if path else "Nothing to undo")
            elif key == ord("e"):
                session.object_id = next_object_id(session.raw_root, session.category, session.object_id)
                shutter.forget_last()
                session.say(f"Next item: [{session.category}] {session.object_id}")
            elif key in CLASS_KEYS:
                selected = CLASS_KEYS[key]
                if selected != session.category:
                    session.category = selected
                    session.object_id = next_object_id(session.raw_root, selected)
                    shutter.forget_last()
                    session.say(f"Class: [{selected}] {session.object_id}")
            elif key == ord("r"):
                path = session.save_background(frame)
                set_analysis_background()
                session.say(f"Background saved: {path.name}")
            elif key == ord("t"):
                base = _lighting_base(session.lighting)
                current = LIGHTING_PRESETS.index(base) if base in LIGHTING_PRESETS else -1
                session.lighting = session.latest_lighting(
                    LIGHTING_PRESETS[(current + 1) % len(LIGHTING_PRESETS)])
                session.load_background()
                set_analysis_background()
                session.say(f"Light: {session.lighting} - clear zone and press R if background is missing")
            elif key == ord("f") and hasattr(ui, "toggle_fullscreen"):
                ui.toggle_fullscreen()
    finally:
        capture.release()
        ui.close()
        session.manifest.close()
        from ..camera.config import hand_back

        hand_back(session.raw_root)
        hand_back(session.bg_root)
    print(f"\n{session.total} net new frames -> {session.raw_root}")
    print(f"next: uv run trashdrop autolabel --session {config.session}")
    return session.raw_root
