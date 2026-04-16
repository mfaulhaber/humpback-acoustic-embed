# Classify Review: Event Badges, Palette "None" Indicator, and Added-Event Region Fix

**Date:** 2026-04-16
**Scope:** Three combined changes to the Call Parsing Classify Review workspace.

---

## Motivation

While reviewing classification job `728d2e47-1916-4761-bdfa-84020156731c`, three issues surfaced:

1. **Added events do not render.** User-added events from boundary corrections (event_id prefix `add-`) navigate correctly but their spectrogram region does not render. The detail panel shows the event's metadata while the workspace falls back to the "No events to display" placeholder.
2. **Unclear unlabeled state in the palette.** When an event has no above-threshold prediction and no saved correction, the palette does not communicate that state — no button highlights and the user is left unsure whether their correction cleared.
3. **No at-a-glance event type on the spectrogram.** Event bars show correction state (adjust/add/delete) but not the classifier's predicted type, so the user must select each event individually to see its label.

This design addresses all three as a single change set because they share the same workspace, components, and data flow.

---

## Root cause — Issue 1 (added events don't render)

The frontend locates `currentRegion` via `regions.find(r => r.region_id === currentEvent.regionId)`. When `regionId` is empty, no region matches and the spectrogram block falls through to the "No events to display" branch.

`currentEvent.regionId` comes from `GET /call-parsing/classification-jobs/{id}/typed-events`. That endpoint builds its `event_id → region_id` map from the segmentation job's `events.parquet` alone:

```
events_path = segmentation_job_dir(...) / "events.parquet"
events = read_events(events_path)
event_region_map = {e.event_id: e.region_id for e in events}
...
"region_id": event_region_map.get(te.event_id, "")
```

User-added events from `event_boundary_corrections` are not in `events.parquet` — they exist only as correction rows. The endpoint returns empty `region_id` for every `add-` event, which the frontend cannot resolve.

Verification on the example job: `typed_events.parquet` contains 47 `add-`-prefixed event_ids, all of which are missing from `events.parquet`. Event 14 in the navigable list is one such add-event.

The classify worker uses `load_corrected_events()` ([extraction.py:89](../../src/humpback/call_parsing/segmentation/extraction.py:89)), which merges `events.parquet` with `event_boundary_corrections` and synthesizes proper `Event` rows (with `region_id`) for adds. The endpoint should reuse that function.

---

## Changes

### 1. Backend — correct `region_id` for added events

In `src/humpback/api/routers/call_parsing.py` at `get_classification_typed_events` (line 704):

Replace the manual `events.parquet` read that builds `event_region_map` with a call to `load_corrected_events(session, job.event_segmentation_job_id, settings.storage_root)`. Build `event_region_map` from the resulting `Event` list. No other behavior changes — typed events are still read from `typed_events.parquet`; the endpoint still returns the same JSON shape sorted by `(start_sec, type_name)`.

The replacement inherits `load_corrected_events`'s existing "no corrections exist" fast path, so jobs without corrections keep today's behavior.

### 2. Frontend — "None" status indicator in palette

In `frontend/src/components/call-parsing/TypePalette.tsx`:

Add a non-interactive "None" chip as the leftmost element in the palette (before the existing `(Negative)` chip). Visible only when `activeType === null`; when `activeType` is anything else (a type name, or `""` for negative), the chip stays in the DOM but becomes invisible via `visibility: hidden` so layout does not shift.

Visual style: disabled-looking chip — muted foreground color, dashed muted border, no hover state, no click handler, no focus ring. Label text: `None`.

No new props. `activeType` already flows in from the workspace.

### 3. Frontend — two-letter type badge on event bars

In `frontend/src/components/call-parsing/EventBarOverlay.tsx` and `ClassifyReviewWorkspace.tsx`:

**Data shape.** Extend `EffectiveEvent` with:

- `effectiveType: string | null` — the type name that should drive the badge, or `null` for unlabeled.
- `typeSource: "inference" | "correction" | "negative" | null` — which path set `effectiveType`.

Build these fields in the `regionEffectiveEvents` memo in `ClassifyReviewWorkspace.tsx`, where both `AggregatedEvent.predictedType` / `correctedType` and the boundary-overlay merge already exist. Logic:

- If `correctedType === null` (negative correction): `effectiveType = null`, `typeSource = "negative"`.
- Else if `correctedType` is a string: `effectiveType = correctedType`, `typeSource = "correction"`.
- Else if `predictedType` is a string: `effectiveType = predictedType`, `typeSource = "inference"`.
- Else: `effectiveType = null`, `typeSource = null`.

Added events without an above-threshold prediction simply have `typeSource = null` and render with no badge, same as any other unlabeled event.

**Rendering.** Inside each non-deleted bar, at the top-left corner of the bar (`left: 0; top: 0`), render a small `<div>` sized ~20×14 px with `pointer-events: none; overflow: visible`. The 2-letter code is the first two characters of `effectiveType`, uppercased (e.g., "Ascending Moan" → "AS", "Moan" → "MO"). Color comes from `typeColor(effectiveType)`.

Style by `typeSource`:

- `"inference"` — transparent/dark background, 1px `typeColor` border, `typeColor` text.
- `"correction"` — solid `typeColor` background, white text, 1px `typeColor` border.
- `"negative"` — solid `hsl(0, 70%, 50%)` background, white text, label `—` (em dash).
- `null` — badge is not rendered.

Badge overflows the right edge when the bar is narrower than the badge — acceptable because the bar's left edge still anchors it. Drag-preview bars get the same badge, pinned to the rendered left edge so it tracks during edge-resize. Deleted bars have no badge (they already display a red strikethrough strip).

**Selected bar ring.** The existing `ring-2 ring-white/80` selection ring sits on the bar itself. The badge is a child of the bar, so it appears above the ring without further z-index work.

---

## Non-goals

- No changes to the backend classify worker, classifier training, or typed_events storage format.
- No changes to the palette's existing `(Negative)`, type, or `Add Type` buttons beyond adding the "None" chip as a sibling.
- No new API endpoints or database migrations.
- No changes to the detail panel.

---

## Testing

### Backend

- Integration test on the classification typed-events endpoint: seed a segmentation job with both `events.parquet` rows and an "add" boundary correction, then an event classification job that references it. Write a `typed_events.parquet` whose rows include the added event_id. Call the endpoint and assert the added event_id's returned `region_id` equals the correction row's `region_id`.
- Regression test: a job with no boundary corrections still returns the original `region_id` values unchanged.

### Frontend

- Playwright test: load a classification review workspace with seeded data containing an add-event; navigate to that event; assert the spectrogram canvas is present (no "No events to display" placeholder).
- Playwright test: navigate to an event with no above-threshold prediction and no correction; assert the "None" chip is visible and has the disabled/dashed style. Click a type button; assert "None" is no longer visible (but still occupies space). Click `(Negative)`; assert "None" stays hidden and `(Negative)` is highlighted.
- Playwright test: palette layout does not reflow when the "None" chip toggles. Record the bounding rect of the `(Negative)` chip with and without the "None" chip visible; assert identical `left`.
- Playwright test for badges:
  - Inference-only event: badge exists with `data-source="inference"` and has border style (no solid fill).
  - After clicking a palette type, the badge flips to solid fill (`data-source="correction"`).
  - After clicking `(Negative)`, the badge becomes solid red with `—` (`data-source="negative"`).
  - An event with no prediction above threshold and no correction: no badge rendered.

### Manual smoke

On job `728d2e47-1916-4761-bdfa-84020156731c`:
- Event 14 (`add-e9921545…`) renders its spectrogram.
- Events with below-threshold top scores (e.g., event 23) show the "None" chip in the palette.
- Every classified event shows a 2-letter badge; applied corrections flip border style to solid fill.
