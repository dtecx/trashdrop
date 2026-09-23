"""Shoot one item at a time on the rig, ideally without touching the keyboard.

The workflow this supports is the one that makes labelling nearly free:
shoot a reference photo of the empty table, then put ONE item down, step back,
and let the auto-shutter take the frame once the hand is out and nothing moves.
Reposition, step back, repeat. ``autolabel`` recovers the mask and box later by
differencing against the reference, and the class comes from the folder, so
nobody draws a box and nobody types a name.

Camera source can be a device index (``0``) or a URL. The URL form is what to
use for an Android phone -- any "IP webcam" app, e.g.
``http://192.168.1.5:8080/video``.

Layout written:

    data/raw/<session>/<class>/<object_id>/frame_0000.jpg
    data/bg/<session>/<lighting>.jpg
    data/raw/<session>/manifest.csv

Keys in the live window:

    1-5    class: bio, paper, plastic, metal, mixed   (starts the next object)
    n      next object of the same class
    a      auto-shutter on/off                        SPACE  shoot a burst by hand
    b      (re)shoot the background -- table EMPTY    l      change lighting label
    r      rename the object (type in the terminal)   q      finish
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..station import ALL_CATEGORIES
from .manifest import ManifestWriter
from .shutter import AutoShutter, ShutterView

WINDOW = "trashdrop capture"
PREVIEW_WIDTH = 960
# A reference older than this has probably drifted from the room light.
BACKGROUND_STALE_SECONDS = 20 * 60
CLASS_KEYS = {ord(str(index + 1)): category for index, category in enumerate(ALL_CATEGORIES)}
TONES = {  # BGR
    "idle": (190, 190, 190),
    "wait": (0, 200, 255),
    "ready": (80, 220, 80),
    "problem": (70, 70, 245),
}


@dataclass
class CaptureConfig:
    session: str
    root: Path = Path("data")
    source: str | int = 0
    width: int = 1920
    height: int = 1080
    burst: int = 3
    burst_interval: float = 0.12
    lighting: str = "default"
    auto: bool = True


def open_camera(source: str | int, width: int, height: int):
    """Open a camera and verify it actually delivers a frame."""

    import cv2

    capture = cv2.VideoCapture(source)
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {source!r}.\n"
            "  - macOS: the first run only ASKS for camera permission. Click Allow,\n"
            "    then run the command again. Check System Settings > Privacy &\n"
            "    Security > Camera if nothing was asked.\n"
            "  - wrong index? run: uv run trashdrop cameras\n"
            "  - a phone: pass the full stream URL, e.g. http://192.168.1.5:8080/video"
        )
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(f"Camera {source!r} opened but returned no frame")
    return capture


def next_object_id(raw_root: Path, category: str, current: str | None = None) -> str:
    """The next free ``<class>_NN`` id, so nobody has to type a name."""

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


def _prompt(message: str, current: str, options: list[str] | None = None) -> str:
    """Ask on the terminal; the OpenCV window cannot take text input."""

    print(f"\n  {message} (type in THIS terminal)")
    if options:
        print(f"  options: {', '.join(options)}")
    answer = input(f"  new value, enter keeps {current!r}: ").strip()
    if not answer:
        return current
    if options and answer not in options:
        print(f"  {answer!r} is not one of the options; keeping {current!r}")
        return current
    return answer


class _OpenCvUi:
    def show(self, image) -> None:
        import cv2

        cv2.imshow(WINDOW, image)

    def key(self) -> int:
        import cv2

        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        import cv2

        cv2.destroyAllWindows()


def _text(canvas, text: str, origin: tuple[int, int], colour, scale: float = 0.6) -> None:
    import cv2

    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


class _Session:
    """Everything the capture loop keeps track of between frames."""

    def __init__(self, config: CaptureConfig, category: str, object_id: str | None, clock) -> None:
        self.config = config
        self.clock = clock
        self.raw_root = config.root / "raw" / config.session
        self.bg_root = config.root / "bg" / config.session
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.bg_root.mkdir(parents=True, exist_ok=True)
        self.manifest = ManifestWriter(self.raw_root / "manifest.csv")

        self.category = category
        self.object_id = object_id or next_object_id(self.raw_root, category)
        self.lighting = config.lighting
        self.auto = config.auto
        self.total = 0
        self.message = ""
        self.message_until = 0.0
        self.flash_until = 0.0
        self.background: np.ndarray | None = None
        self.background_wall_time: float | None = None

    # --- files -------------------------------------------------------------

    @property
    def object_dir(self) -> Path:
        return self.raw_root / self.category / self.object_id

    def frames_of_current_object(self) -> int:
        folder = self.object_dir
        return len(list(folder.glob("frame_*.jpg"))) if folder.is_dir() else 0

    def save(self, frame) -> Path:
        import cv2

        folder = self.object_dir
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"frame_{self.frames_of_current_object():04d}.jpg"
        cv2.imwrite(str(path), frame)
        self.manifest.append(
            image=str(path.relative_to(self.config.root)),
            category=self.category,
            object_id=self.object_id,
            lighting=self.lighting,
            session=self.config.session,
        )
        self.total += 1
        return path

    def load_background(self) -> np.ndarray | None:
        import cv2

        path = self.bg_root / f"{self.lighting}.jpg"
        self.background = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() else None
        self.background_wall_time = path.stat().st_mtime if self.background is not None else None
        return self.background

    def save_background(self, frame) -> Path:
        import cv2

        path = self.bg_root / f"{self.lighting}.jpg"
        cv2.imwrite(str(path), frame)
        self.background = frame.copy()
        self.background_wall_time = time.time()
        return path

    # --- feedback ----------------------------------------------------------

    def say(self, text: str, seconds: float = 1.8) -> None:
        self.message = text
        self.message_until = self.clock() + seconds
        print(f"  {text}")

    def background_age_text(self) -> tuple[str, tuple[int, int, int]]:
        if self.background_wall_time is None:
            return "MISSING - clear the table, press b", TONES["problem"]
        minutes = (time.time() - self.background_wall_time) / 60.0
        colour = TONES["wait"] if minutes * 60 > BACKGROUND_STALE_SECONDS else TONES["idle"]
        suffix = " - re-shoot (b)" if minutes * 60 > BACKGROUND_STALE_SECONDS else ""
        return f"{minutes:.0f} min old{suffix}", colour

    def render(self, frame, view: ShutterView):
        import cv2

        scale = PREVIEW_WIDTH / frame.shape[1]
        preview = cv2.resize(frame, (PREVIEW_WIDTH, int(round(frame.shape[0] * scale))))
        tone = TONES.get(view.tone, TONES["idle"])
        if view.box is not None:
            x, y, w, h = (int(round(v * scale)) for v in view.box)
            cv2.rectangle(preview, (x, y), (x + w, y + h), tone, 2)

        mode = "AUTO" if self.auto else "MANUAL"
        _text(preview, f"[{self.category}] {self.object_id}   light={self.lighting}   {mode}",
              (12, 28), (255, 255, 255))
        age, age_colour = self.background_age_text()
        _text(preview, f"this object {self.frames_of_current_object()}   session {self.total}",
              (12, 54), (255, 255, 255))
        _text(preview, f"background {age}", (12, 80), age_colour)
        _text(preview, view.status, (12, 110), tone, 0.7)

        if self.clock() < self.message_until:
            _text(preview, self.message, (12, preview.shape[0] // 2), (255, 255, 255), 0.8)
        _text(
            preview,
            "1-5 class  n next  a auto  SPACE shoot  b background  l light  r rename  q quit",
            (12, preview.shape[0] - 14),
            (220, 220, 220),
            0.45,
        )
        if self.clock() < self.flash_until:
            cv2.rectangle(preview, (0, 0), (preview.shape[1] - 1, preview.shape[0] - 1), (255, 255, 255), 10)
        return preview


def run_capture(
    config: CaptureConfig,
    category: str = "plastic",
    object_id: str | None = None,
    *,
    capture=None,
    ui=None,
    clock=time.monotonic,
    prompt=_prompt,
) -> Path:
    """Interactive capture loop. Returns the session's raw folder.

    ``capture``, ``ui``, ``clock`` and ``prompt`` exist so the loop can be
    driven by a test with scripted frames and key presses.
    """

    if category not in ALL_CATEGORIES:
        raise ValueError(f"{category!r} is not a station category: {ALL_CATEGORIES}")

    session = _Session(config, category, _clean_id(object_id) if object_id else None, clock)
    ui = ui or _OpenCvUi()
    if capture is None:
        capture = open_camera(config.source, config.width, config.height)
        # Push camera.toml (fixed focus, exposure, white balance) once the
        # stream is open, in case opening it reset anything.
        from ..camera import apply_saved_settings

        print(apply_saved_settings())
    shutter = AutoShutter(None, clock=clock)
    if session.load_background() is not None:
        shutter.set_background(session.background)

    print(
        "\n1-5 class | n next object | a auto | SPACE shoot | b background | "
        "l light | r rename | q quit\n"
        "Clear the table and press b FIRST. Then: drop an item, step back, wait for the flash.\n"
    )
    print(f"  now shooting [{session.category}] {session.object_id}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                if ui.key() == ord("q"):
                    break
                continue

            if session.auto:
                view = shutter.update(frame)
            else:
                view = ShutterView(False, "manual - SPACE to shoot", "idle")
            if view.capture:
                session.save(frame)
                shutter.mark_captured()
                session.flash_until = clock() + 0.25

            ui.show(session.render(frame, view))
            key = ui.key()

            if key == ord("q"):
                break
            if key == ord("b"):
                path = session.save_background(frame)
                shutter.set_background(frame)
                session.say(f"background for {session.lighting!r} saved -> {path.name}")
            elif key == ord(" "):
                session.save(frame)
                for _ in range(max(0, config.burst - 1)):
                    if config.burst_interval:
                        time.sleep(config.burst_interval)
                    ok, extra = capture.read()
                    if ok and extra is not None:
                        session.save(extra)
                shutter.mark_captured()
                session.flash_until = clock() + 0.25
                session.say(f"shot {config.burst} by hand")
            elif key == ord("a"):
                session.auto = not session.auto
                session.say(f"auto-shutter {'ON' if session.auto else 'OFF'}")
            elif key in CLASS_KEYS:
                session.category = CLASS_KEYS[key]
                session.object_id = next_object_id(session.raw_root, session.category)
                shutter.forget_last()
                session.say(f"now shooting [{session.category}] {session.object_id}")
            elif key == ord("n"):
                session.object_id = next_object_id(
                    session.raw_root, session.category, session.object_id
                )
                shutter.forget_last()
                session.say(f"now shooting [{session.category}] {session.object_id}")
            elif key == ord("r"):
                renamed = _clean_id(prompt("object id", session.object_id))
                if renamed:
                    session.object_id = renamed
                    shutter.forget_last()
                    session.say(f"now shooting [{session.category}] {session.object_id}")
            elif key == ord("l"):
                session.lighting = _clean_id(prompt("lighting label", session.lighting)) or "default"
                shutter.set_background(session.load_background())
                if session.background is None:
                    session.say(f"no background for {session.lighting!r} yet - clear the table, press b")
                else:
                    session.say(f"lighting {session.lighting!r}: background loaded")
    finally:
        capture.release()
        ui.close()
        session.manifest.close()
        # If someone ran capture under sudo anyway, hand the frames back to
        # the real user rather than leaving a root-owned dataset.
        from ..camera.config import hand_back

        hand_back(session.raw_root)
        hand_back(session.bg_root)

    print(f"\n{session.total} frames this run -> {session.raw_root}")
    print(f"manifest: {session.manifest.path}")
    print(f"next: uv run trashdrop autolabel --session {config.session}")
    return session.raw_root
