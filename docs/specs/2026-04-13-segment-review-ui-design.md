# Segment Review UI — Design Spec

**Date:** 2026-04-13
**Status:** Approved

## Goal

Refactor the Call Parsing / Segment page to support human-in-the-loop feedback for Pass 2 (event segmentation). Users review detected events within regions on an interactive spectrogram, adjust event boundaries by dragging visual markers, add missed events, delete false positives, and save corrections for future model retraining.

## Context

The existing Segment page is purely job management — create segmentation jobs, monitor status, view event counts. There is no way to review or correct the model's output. The backend already has full correction infrastructure (`event_boundary_corrections` table, batch POST/GET/DELETE endpoints, service layer) but no frontend consumes it.

The Vocalization/Labeling page serves as the UI reference: accumulated edits before save, interactive spectrogram with overlays, audio playback, and visual state indicators for pending changes.

## Scope

### In scope
- Backend: spectrogram tile endpoint for region detection jobs
- Frontend: two-tab layout (Jobs + Review) on the Segment page
- Interactive spectrogram viewer with draggable event boundary markers
- Event add/delete interactions
- Accumulated edits with Save/Cancel
- Per-event audio playback
- Placeholder Retrain button (not wired)

### Out of scope
- Retrain workflow (separate session)
- Segment Training page updates
- Pass 1 region corrections
- Pass 3 type corrections on this page
- Confidence trace overlay, keyboard navigation, bulk operations

## Design

### Page Structure: Two Tabs

The Segment page gains a tab wrapper:

- **Jobs tab**: existing job management UI (form + active/previous job tables), unchanged except for a "Review" action link on completed jobs
- **Review tab**: new correction workspace

### Backend: Tile Endpoint

New endpoint reusing existing tile generation logic:

```
GET /call-parsing/region-jobs/{job_id}/tile?zoom_level=5m&tile_index={n}
```

Region detection jobs already have the audio source reference (hydrophone_id + time range or file path) needed to fetch audio and render PCEN spectrograms. The endpoint supports all existing zoom levels but the frontend fixes to `5m`.

No new tables or migrations — `event_boundary_corrections` and its API already exist.

### Review Tab Layout

Top to bottom, left to right:

1. **Job selector** — dropdown of completed segmentation jobs with metadata (source, model, event count, region count)
2. **Region sidebar** (left) — scrollable list of regions from the selected job. Each shows time range, event count, correction progress indicator (pending/partial/reviewed)
3. **Toolbar** (above spectrogram) — region summary, Play (region audio), + Add (enter add mode), Save, Cancel buttons, unsaved change count
4. **Spectrogram panel** (main area) — pannable, fixed 5-min zoom, event bars overlaid with draggable edges
5. **Event detail panel** (below spectrogram) — selected event metadata, Play Slice, Delete Event, original vs. adjusted values
6. **Footer** — Retrain placeholder button + unsaved change count

### Components

| Component | Responsibility |
|-----------|---------------|
| `SegmentReviewWorkspace` | Top-level orchestrator. Job selection, data fetching (regions + events + existing corrections), owns accumulated edits state (Map keyed by event_id). Save/Cancel handlers. |
| `RegionSidebar` | Scrollable region list with correction progress indicators. Click to select active region. |
| `RegionSpectrogramViewer` | Pannable spectrogram scoped to selected region's padded bounds. Fixed 5-min zoom. Reuses `TileCanvas`. Renders `EventBarOverlay`. Handles pan gestures and click-to-select. |
| `EventBarOverlay` | Renders event bars on spectrogram. Selectable, draggable left/right edges. Visual states: original (solid), adjusted (dashed outline + solid fill), added (green outline), pending delete (red strikethrough, 30% opacity). |
| `EventDetailPanel` | Selected event info: timestamps, duration, confidence, correction status. Play Slice and Delete Event buttons. Shows original vs. adjusted values. |
| `ReviewToolbar` | Region summary, Play/Add/Save/Cancel actions. Unsaved change count badge. Retrain placeholder. |

### Interaction Model

**Selecting:** Click event bar to select (detail panel populates). Click empty space to deselect. One selection at a time.

**Adjusting boundaries:** Hover near left/right edge of selected event → `col-resize` cursor. Drag to adjust start/end. Snap to 0.1s. Clamped: edges cannot cross each other or overlap adjacent events. Ghost preview during drag. Records `adjust` correction on drag-end.

**Adding events:** Click `+ Add` → crosshair cursor. Click spectrogram to place new event (default 1s width centered on click). Drag edges to refine. Records `add` correction. Escape or re-click `+ Add` to exit add mode.

**Deleting events:** Select event → click "Delete Event" in detail panel. Bar goes 30% opacity with red strikethrough. Records `delete` correction. Click deleted event to undo (removes pending correction).

**Save:** Serializes pending edits into `BoundaryCorrectionRequest`, POSTs to `/segmentation-jobs/{job_id}/corrections`. Clears pending state, refetches corrections.

**Cancel:** Confirms if dirty ("Discard N unsaved changes?"), clears pending state.

**Region switching with unsaved changes:** Prompts to save or discard.

### State Management

```
pendingCorrections: Map<string, BoundaryCorrection>
```

Keyed by `event_id` for adjust/delete. Keyed by `add-{uuid}` for adds. Each entry: `{ event_id, region_id, correction_type, start_sec, end_sec }`.

Computed via `useMemo`:
- `isDirty` — `pendingCorrections.size > 0`
- `pendingChangeCount` — number of entries
- `effectiveEvents` — merges original events with pending corrections for rendering

Existing saved corrections fetched on load via `GET /segmentation-jobs/{job_id}/corrections`, displayed as already-applied (visually distinct from originals but not "pending"). New edits layer on top.

Corrections scoped to the job, not per-region. Pending map accumulates across regions within the selected job. Only Save or Cancel clears.

### Audio Playback

- **Play Slice** (detail panel): plays selected event's audio range (adjusted range if boundary was modified)
- **Play** (toolbar): plays region audio from current pan position
