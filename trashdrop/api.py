"""Open intake API: how other teams' robots hand us trash.

Alien Bazaar judges proactive collaboration between teams, and TrashDrop is
where the house's trash ends up, so this is a first-class part of the project
rather than a side feature. Another team's robot announces a delivery, we sort
it, and it can ask afterwards what went where.

Deliberately built on the standard library. No team should have to install
anything to talk to us, this runs identically on the macOS and Windows laptops
in the team, and there is nothing to go wrong on venue wifi at 2 a.m.

Run it:

    uv run trashdrop serve --mock          # before the arms exist
    uv run trashdrop serve                 # wired to a real cell

``--mock`` sorts deliveries on a timer with plausible results, so another team
can integrate against a working endpoint days before our hardware exists.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .station import (
    ALL_CATEGORIES,
    BINS,
    MAX_GRASP_WIDTH,
    MAX_PAYLOAD_KG,
    MIXED_CATEGORY,
    PICK_ZONE,
    SORT_CATEGORIES,
)

DEFAULT_PORT = 8742
MAX_BODY_BYTES = 64 * 1024
MAX_ITEMS_PER_DELIVERY = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SortedItem:
    """One item the cell dealt with."""

    item_id: str
    category: str
    bin: str
    confidence: float
    arm: str
    sorted_at: str
    note: str = ""


@dataclass
class Delivery:
    """A batch of trash another robot left in the pick zone."""

    delivery_id: str
    robot: str
    announced_items: int
    state: str  # queued | sorting | done | rejected
    created_at: str
    updated_at: str
    note: str = ""
    items: list[SortedItem] = field(default_factory=list)


class SortingLog:
    """Shared state between the HTTP layer and whatever does the sorting.

    Thread-safe because the HTTP server is threaded and the cell runs in its
    own thread. Kept in memory, with an optional JSON snapshot so a crash
    mid-demo does not lose the tally.
    """

    def __init__(self, snapshot: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._deliveries: dict[str, Delivery] = {}
        self._order: list[str] = []
        self.snapshot = Path(snapshot) if snapshot else None
        self.started_at = _now()

    def announce(self, robot: str, items: int, note: str = "") -> Delivery:
        with self._lock:
            delivery = Delivery(
                delivery_id=uuid.uuid4().hex[:12],
                robot=robot,
                announced_items=items,
                state="queued",
                created_at=_now(),
                updated_at=_now(),
                note=note,
            )
            self._deliveries[delivery.delivery_id] = delivery
            self._order.append(delivery.delivery_id)
        self._persist()
        return delivery

    def get(self, delivery_id: str) -> Delivery | None:
        with self._lock:
            return self._deliveries.get(delivery_id)

    def recent(self, limit: int = 20) -> list[Delivery]:
        with self._lock:
            return [self._deliveries[i] for i in self._order[-limit:]][::-1]

    def set_state(self, delivery_id: str, state: str, note: str = "") -> None:
        with self._lock:
            delivery = self._deliveries.get(delivery_id)
            if delivery is None:
                return
            delivery.state = state
            delivery.updated_at = _now()
            if note:
                delivery.note = note
        self._persist()

    def record(self, delivery_id: str, item: SortedItem) -> None:
        with self._lock:
            delivery = self._deliveries.get(delivery_id)
            if delivery is None:
                return
            delivery.items.append(item)
            delivery.updated_at = _now()
        self._persist()

    def totals(self) -> dict:
        with self._lock:
            per_bin = {category: 0 for category in ALL_CATEGORIES}
            sorted_total = 0
            for delivery in self._deliveries.values():
                for item in delivery.items:
                    per_bin[item.category] = per_bin.get(item.category, 0) + 1
                    sorted_total += 1
            return {
                "since": self.started_at,
                "deliveries": len(self._deliveries),
                "items_sorted": sorted_total,
                "per_bin": per_bin,
            }

    def _persist(self) -> None:
        if self.snapshot is None:
            return
        with self._lock:
            payload = {
                "started_at": self.started_at,
                "deliveries": [asdict(self._deliveries[i]) for i in self._order],
            }
        self.snapshot.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def capabilities() -> dict:
    """What we accept and where to put it. This is the contract other teams read."""

    min_x, max_x, min_y, max_y = PICK_ZONE.bounds
    return {
        "service": "TrashDrop",
        "summary": "Two SO-101 arms sorting plastic, paper, and metal waste, with a mixed fallback.",
        "target_categories": list(SORT_CATEGORIES),
        "sorts_into": list(ALL_CATEGORIES),
        "fallback_bin": MIXED_CATEGORY,
        "fallback_policy": (
            "Anything we cannot classify confidently, or cannot physically "
            "handle, goes to the mixed bin. We flag it rather than guess."
        ),
        "drop_zone": {
            "frame": "trashdrop_table",
            "units": "metres",
            "description": (
                "Leave items anywhere inside this rectangle, on the flat "
                "surface. Approach from the +x side: the two arms occupy the "
                "-y and +y sides and will not move while a delivery is in "
                "progress."
            ),
            "bounds": {
                "min_x": round(min_x, 3),
                "max_x": round(max_x, 3),
                "min_y": round(min_y, 3),
                "max_y": round(max_y, 3),
            },
            "size_m": [round(max_x - min_x, 3), round(max_y - min_y, 3)],
        },
        "item_limits": {
            "max_grasp_width_m": MAX_GRASP_WIDTH,
            "max_mass_kg": MAX_PAYLOAD_KG,
            "note": (
                "An empty drink can is fine. Glass bottles are too heavy for "
                "an SO-101 and will be routed to mixed, not lifted."
            ),
        },
        "bins": [
            {"category": spec.category, "key": key, "x": round(spec.x, 3), "y": round(spec.y, 3)}
            for key, spec in BINS.items()
        ],
    }


INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>TrashDrop intake API</title>
<style>
 body{{font:15px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;max-width:52rem;
      margin:2rem auto;padding:0 1rem;background:#12131a;color:#d7dae3}}
 h1{{font-size:1.4rem;color:#b48ce8}} h2{{font-size:1rem;margin-top:2rem;color:#8fd3a0}}
 code,pre{{background:#1c1e28;border-radius:4px}} code{{padding:.1rem .3rem}}
 pre{{padding:.8rem;overflow-x:auto;border:1px solid #2a2d3a}}
 td{{padding:.15rem .8rem .15rem 0;vertical-align:top}}
 .m{{color:#e8c48c}} a{{color:#7fb5ff}}
</style></head><body>
<h1>TrashDrop &mdash; open intake API</h1>
<p>Bring us your trash. We sort it into
{bins} and tell you where every item went.
Anything we are not sure about goes to <code>mixed</code> rather than being guessed.</p>
<p>Status: <strong>{state}</strong> &middot; sorted so far: <strong>{sorted_total}</strong>
&middot; deliveries: <strong>{deliveries}</strong></p>

<h2>Announce a delivery</h2>
<pre>curl -X POST {base}/deliveries \\
  -H 'content-type: application/json' \\
  -d '{{"robot": "your-robot-name", "items": 3}}'</pre>
<p>Returns a <code>delivery_id</code>. Drop the items in the zone first, then call this.</p>

<h2>Check what happened to it</h2>
<pre>curl {base}/deliveries/&lt;delivery_id&gt;</pre>

<h2>Everything else</h2>
<table>
<tr><td><span class="m">GET</span> /capabilities</td><td>what we sort, the drop zone, size and weight limits</td></tr>
<tr><td><span class="m">GET</span> /deliveries</td><td>recent deliveries</td></tr>
<tr><td><span class="m">GET</span> /stats</td><td>running totals per bin</td></tr>
<tr><td><span class="m">GET</span> /health</td><td>liveness</td></tr>
</table>
<p>CORS is open, so you can call this straight from a browser page.</p>
</body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "TrashDrop/1.0"
    log: SortingLog
    sorter = None
    mode = "live"

    # --- plumbing ----------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        print(f"  {self.address_string()} {fmt % args}")

    def _send(self, status: int, payload: dict | list, content_type="application/json") -> None:
        body = json.dumps(payload, indent=2).encode() if content_type == "application/json" else payload
        if isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("content-type", f"{content_type}; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type")
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"error": message})

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            return None

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, {})

    # --- routes ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        if path == "/":
            totals = self.log.totals()
            base = f"http://{self.headers.get('host', 'localhost')}"
            page = INDEX_HTML.format(
                bins=", ".join(f"<code>{c}</code>" for c in ALL_CATEGORIES),
                state=self.mode,
                sorted_total=totals["items_sorted"],
                deliveries=totals["deliveries"],
                base=base,
            )
            self._send(200, page, content_type="text/html")
            return

        if path == "/health":
            self._send(200, {"ok": True, "mode": self.mode})
            return
        if path == "/capabilities":
            self._send(200, capabilities())
            return
        if path == "/stats":
            self._send(200, self.log.totals())
            return
        if path == "/deliveries":
            self._send(200, [asdict(d) for d in self.log.recent()])
            return
        if path.startswith("/deliveries/"):
            delivery = self.log.get(path.rsplit("/", 1)[-1])
            if delivery is None:
                self._error(404, "no such delivery")
                return
            self._send(200, asdict(delivery))
            return

        self._error(404, f"no route for {path}; see / for the endpoint list")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path != "/deliveries":
            self._error(404, f"no route for {path}; see / for the endpoint list")
            return

        payload = self._read_json()
        if payload is None or not isinstance(payload, dict):
            self._error(400, "send a JSON object body")
            return

        robot = str(payload.get("robot", "")).strip()[:64]
        if not robot:
            self._error(400, "'robot' is required: tell us who is delivering")
            return
        try:
            items = int(payload.get("items", 1))
        except (TypeError, ValueError):
            self._error(400, "'items' must be a whole number")
            return
        if not 1 <= items <= MAX_ITEMS_PER_DELIVERY:
            self._error(400, f"'items' must be between 1 and {MAX_ITEMS_PER_DELIVERY}")
            return

        delivery = self.log.announce(robot, items, str(payload.get("note", ""))[:200])
        if self.sorter is not None:
            self.sorter(delivery.delivery_id)
        self._send(
            202,
            {
                "delivery_id": delivery.delivery_id,
                "state": delivery.state,
                "poll": f"/deliveries/{delivery.delivery_id}",
            },
        )


def mock_sorter(log: SortingLog, seconds_per_item: float = 2.5):
    """Sort deliveries on a timer with plausible results.

    Lets another team integrate against a live endpoint before our arms exist.
    It is honest about being a mock: /health reports mode "mock".
    """

    import random

    def start(delivery_id: str) -> None:
        def run() -> None:
            delivery = log.get(delivery_id)
            if delivery is None:
                return
            log.set_state(delivery_id, "sorting")
            for index in range(delivery.announced_items):
                time.sleep(seconds_per_item)
                confidence = random.uniform(0.35, 0.99)
                category = random.choice(SORT_CATEGORIES)
                # Same rule as the real dispatcher: unsure goes to mixed.
                routed = category if confidence >= 0.55 else MIXED_CATEGORY
                log.record(
                    delivery_id,
                    SortedItem(
                        item_id=f"{delivery_id}-{index}",
                        category=routed,
                        bin=routed,
                        confidence=round(confidence, 2),
                        arm=random.choice(["front", "back"]),
                        sorted_at=_now(),
                        note="" if routed == category else f"low confidence for {category}",
                    ),
                )
            log.set_state(delivery_id, "done")

        threading.Thread(target=run, daemon=True).start()

    return start


def serve(
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    log: SortingLog | None = None,
    sorter=None,
    mode: str = "live",
) -> None:
    """Block, serving the intake API."""

    log = log or SortingLog()
    handler = type(
        "BoundHandler",
        (_Handler,),
        {
            "log": log,
            # staticmethod matters: a plain function in a class attribute
            # becomes a bound method and would be handed `self` as its first
            # argument the moment a request arrives.
            "sorter": staticmethod(sorter) if sorter is not None else None,
            "mode": mode,
        },
    )
    server = ThreadingHTTPServer((host, port), handler)

    print(f"TrashDrop intake API on http://{host}:{port}  (mode: {mode})")
    print(f"  share this with other teams: http://<your-ip>:{port}/")
    print("  ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
