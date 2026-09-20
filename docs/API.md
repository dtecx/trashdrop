# Open intake API

**Bring us your trash.** TrashDrop is where the house's waste ends up, so the
way another team's robot hands it over is part of the project rather than an
afterthought. Alien Bazaar judges collaboration between teams, and this is the
surface that makes it possible.

Share this page and the running URL with anyone who wants to integrate.

## Running it

```bash
uv run trashdrop serve --mock      # sorts on a timer; integrate before our arms exist
uv run trashdrop serve             # wired to the real cell
```

No dependencies — it is standard library only, so nothing needs installing on
either side and it behaves the same on macOS and Windows. It binds `0.0.0.0`,
so on the venue wifi other teams reach it at `http://<your-ip>:8742/`.

Open `/` in a browser: the page documents itself, shows live totals and carries
copy-pasteable examples.

## The three calls that matter

**1. Find out what we take**

```bash
curl http://<host>:8742/capabilities
```

Returns the bin list, the drop-zone rectangle in metres, and the size and mass
limits. Worth reading once: an empty drink can is fine, a glass bottle is
heavier than an SO-101 should lift and will be routed to `mixed` rather than
picked up.

**2. Announce a delivery** — after the items are in the zone, not before

```bash
curl -X POST http://<host>:8742/deliveries \
  -H 'content-type: application/json' \
  -d '{"robot": "cleaner-1", "items": 3}'
```

```json
{"delivery_id": "94a9ed997dbd", "state": "queued", "poll": "/deliveries/94a9ed997dbd"}
```

`robot` is required — with several teams delivering, we need to tell the
batches apart. `items` is how many things you left, 1 to 50.

**3. Ask what happened to it**

```bash
curl http://<host>:8742/deliveries/94a9ed997dbd
```

```json
{
  "delivery_id": "94a9ed997dbd",
  "robot": "cleaner-1",
  "state": "done",
  "items": [
    {"item_id": "94a9ed997dbd-0", "category": "plastic", "bin": "plastic",
     "confidence": 0.7, "arm": "back", "sorted_at": "2026-09-20T16:33:41+00:00"},
    {"item_id": "94a9ed997dbd-1", "category": "mixed", "bin": "mixed",
     "confidence": 0.42, "arm": "back", "note": "low confidence for metal"}
  ]
}
```

`state` goes `queued` → `sorting` → `done`. Poll it; there is no callback yet.

## Everything else

| | |
|---|---|
| `GET /` | self-documenting page with live status |
| `GET /health` | liveness, and whether we are in `mock` or `live` mode |
| `GET /deliveries` | recent deliveries |
| `GET /stats` | running totals per bin — this is what a dashboard reads |

CORS is open, so a browser page can call it directly.

## What we promise

We sort into `bio`, `paper`, `plastic`, `metal` and `mixed`.

**Anything we are not sure about goes to `mixed`, and we say so.** A low
classifier score, an unknown material, an item too wide for the jaws or too
heavy for the arm — all of it is flagged in the `note` field rather than
guessed into a material bin. If you get an item back as `mixed` with a note,
that is the system working, not failing.

## Approaching the cell

The two arms sit on opposite sides of the drop zone, facing each other, and
they share that volume. **Approach from the +x side** — the arms occupy the
±y sides. While a delivery is being sorted the arms are working in the zone,
so announce the delivery *after* you have withdrawn.

## Not built yet

Webhooks instead of polling, and a reservation call so a delivering robot can
claim the zone before it moves in. Both are easy to add — ask if you need one
and we will wire it during the event.
