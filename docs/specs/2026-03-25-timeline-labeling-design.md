# Timeline Labeling Design

**Date:** 2026-03-25
**Status:** Approved

## Overview

Add interactive species labeling to the timeline viewer. Users can add, move, delete, and change labels directly on the spectrogram. A batch save persists edits to the detection job row store. As a prerequisite, remove the TSV sync layer so the Parquet row store is the sole source of truth.

## Goals

- Allow users to visually label detection windows on the timeline spectrogram
- Support add (new 5s windows), move, delete, and change-type operations
- Enforce single-label-per-row (one of: humpback, orca, ship, background)
- Prevent overlapping labeled windows
- Remove TSV as an intermediate data representation
- Provide color-coded visual distinction between whale (warm) and non-whale (cool) labels

## Non-Goals

- Resizing existing label windows
- Undo/redo stack (user can discard unsaved changes)
- Labeling at zoom levels wider than 5m
- Changes to the existing LabelingTab or vocalization label system

---

## 1. Interaction Model

### Entering Label Mode

A "Label" button appears in the playback controls toolbar when audio is paused and zoom level is 1m or 5m. Clicking it activates label mode. At wider zoom levels the button is disabled with a tooltip: "Zoom to 5m or closer to edit labels."

Entering label mode pauses playback. Starting playback exits label mode (with a dirty-state confirmation if unsaved edits exist). Panning the spectrogram remains functional in label mode.

### Sub-Modes

Label mode has two sub-modes toggled by buttons in the label toolbar:

**Select mode** (default on entry): Click a label bar to select it. Once selected:
- **Change type** — click a different radio button
- **Move** — click and drag the label bar left/right (snaps to 0.5s grid, clamped at neighbor edges to prevent overlap)
- **Delete** — click Delete button or press Delete key

**Add mode**: Click on the spectrogram to place a new 5-second label centered on the click position, snapped to 0.5s grid. The label type comes from the currently selected radio button. Placement is blocked if it would overlap an existing labeled row. User stays in Add mode for rapid sequential labeling.

### Label Toolbar

Appears below the spectrogram when label mode is active:

```
[Select | Add]  [● humpback ○ orca ○ ship ○ background]  [Delete]  [Save]  [Extract Labels]  [Cancel]
```

An unsaved-changes indicator appears next to Save when edits exist. Attempting to exit label mode or navigate away with unsaved changes shows a confirmation prompt.

---

## 2. Visual Design

### Label Bar Colors

| Label | Fill Color | Opacity | Family |
|-------|-----------|---------|--------|
| humpback | Amber/warm yellow `rgb(234, 179, 8)` | 0.45 | Whale (warm) |
| orca | Orange `rgb(249, 115, 22)` | 0.45 | Whale (warm) |
| ship | Steel blue `rgb(100, 149, 237)` | 0.35 | Non-whale (cool) |
| background | Gray `rgb(156, 163, 175)` | 0.30 | Non-whale (cool) |

Unlabeled detection rows retain their current confidence-scaled cyan appearance and are not interactive in label mode.

### Selection State

Selected label gets a 2px solid white border and subtle outer glow. Cursor changes to `grab` on hover over a label bar, `grabbing` during drag.

### Add Mode Ghost Preview

Crosshair cursor over the spectrogram. A semi-transparent 5s bar in the selected label color follows the mouse horizontally, snapping to the 0.5s grid.

- **Overlaps a labeled row** — ghost turns red (placement blocked)
- **Overlaps unlabeled row(s)** — ghost displays normally; overlapped unlabeled rows dim/fade to signal they will be replaced
- **No overlap** — ghost displays normally

The same dimming effect applies during drag-move of a labeled row over unlabeled rows.

### Manual vs. Machine Labels

Manually-placed labels (confidence = null) display a thin dashed top-border to distinguish them from classifier-generated labels. Subtle and non-distracting.

---

## 3. Data Model & Changeset

### Single-Label Enforcement

Each detection row carries at most one species label. The existing `humpback`, `orca`, `ship`, `background` columns remain but are mutually exclusive — setting one clears the others. Enforced in the batch save endpoint and in the frontend changeset logic.

### Frontend Changeset (`useLabelEdits` reducer)

```typescript
type LabelType = "humpback" | "orca" | "ship" | "background";

type LabelEdit =
  | { action: "add"; id: string; start_sec: number; end_sec: number; label: LabelType }
  | { action: "move"; row_id: string; new_start_sec: number; new_end_sec: number }
  | { action: "delete"; row_id: string }
  | { action: "change_type"; row_id: string; label: LabelType };
```

The reducer maintains a flat list of edits. Multiple edits to the same row collapse (e.g., move then change-type becomes a single entry with both changes).

### New Rows from Add

New rows receive a temporary client-side UUID as `id`. The backend assigns the real `row_id` on save. Confidence fields (`avg_confidence`, `peak_confidence`) are `null`. `start_sec`/`end_sec` are job-relative seconds.

### Type Updates for Null Confidence

`DetectionRow` in `frontend/src/api/types.ts` must update `avg_confidence` and `peak_confidence` from `number` to `number | null`. The corresponding Pydantic response schema must also allow `Optional[float]` for these fields.

### Null Confidence Audit

Before implementation, validate that `null` confidence is non-breaking across:
- `DetectionRow` TypeScript type (`avg_confidence`, `peak_confidence` → `number | null`)
- Pydantic detection row response schema (→ `Optional[float]`)
- Confidence strip rendering (skip nulls — already handled)
- Detection row sorting (null sorts last)
- Extraction positive-selection logic (null rows use label only, no score-based windowing)
- Minimap heatmap (skip nulls)
- Any division/comparison operations on confidence values

---

## 4. Backend API

### Prerequisite: Remove TSV Sync

Remove the TSV as an intermediate representation. The Parquet row store becomes the sole source of truth for detection rows.

- Detection worker stops writing `detections.tsv` alongside the row store
- Remove all `sync_detection_tsv()` calls (found in `classifier.py`, `detection_rows.py`, `extractor.py`)
- Update `ensure_detection_row_store()` to remove its TSV-path fallback parameter
- The download endpoint (`GET /classifier/detection-jobs/{job_id}/download`) generates TSV on-the-fly from the Parquet row store
- DB migration drops `output_tsv_path` from `detection_jobs` table
- Remove `output_tsv_path` from Pydantic schemas and API responses
- Running jobs write directly to a streaming Parquet row store (replacing the current "running jobs write TSV, completed jobs build Parquet" split)
- The `/content` endpoint reads from Parquet for all job statuses

### New Endpoint: Batch Save Label Edits

```
PATCH /classifier/detection-jobs/{job_id}/labels
```

**Request body:**
```json
{
  "edits": [
    { "action": "add", "start_sec": 142.5, "end_sec": 147.5, "label": "humpback" },
    { "action": "move", "row_id": "abc123", "new_start_sec": 200.0, "new_end_sec": 205.0 },
    { "action": "delete", "row_id": "def456" },
    { "action": "change_type", "row_id": "ghi789", "label": "ship" }
  ]
}
```

**Response:** `200` with the full updated detection row list, so the frontend can refresh its state.

**Validation (applied before any writes):**
- Job status must be `complete`, `paused`, or `canceled`
- No overlapping windows in the resulting state (compute final positions of all rows after all edits, reject the entire batch if any overlap)
- Each label must be one of the four valid types
- Move targets must be within `[0, job_duration]` where job duration is `end_timestamp - start_timestamp` for hydrophone jobs, or `max(row.end_sec)` across existing rows for local jobs
- All referenced `row_id`s must exist in the row store
- Single-label enforcement on all modified rows

**Write path:** Read current Parquet row store, apply edits, validate final state, write updated row store atomically (temp file + rename). No TSV sync.

### Unlabeled Row Replacement

When an "add" edit places a new labeled row that overlaps one or more unlabeled detection rows, those unlabeled rows are deleted from the row store as part of the batch save. The new labeled row replaces them.

---

## 5. Frontend Component Architecture

### New Components

**`LabelEditor.tsx`** — Sibling to `DetectionOverlay` inside `SpectrogramViewport`. Renders interactive label bars when label mode is active. Handles click-to-select, click-to-place (with ghost preview), drag-to-move, overlap detection and clamping, coordinate transforms using the same `pxPerSec`/`centerTimestamp` math as `DetectionOverlay`.

**`LabelToolbar.tsx`** — Rendered by `TimelineViewer` below the spectrogram when label mode is active. Contains mode toggle, radio buttons, Delete, Save, Extract Labels, Cancel. Receives changeset state for dirty indicator and button enable/disable.

**`ExtractDialog.tsx`** — Reuse the existing extraction dialog from the classifier tab.

### New Hook

**`useLabelEdits.ts`** — `useReducer`-based hook managing the changeset. Actions: `add`, `move`, `delete`, `change_type`, `clear`. Exposes:
- `edits` — current changeset array
- `isDirty` — boolean
- `dispatch` — reducer dispatch
- `mergedRows` — computed: applies pending edits to the original detection rows, producing the preview state that `LabelEditor` renders

### State Flow

```
TimelineViewer
├── labelMode: boolean (toggled by Label button)
├── useLabelEdits(detectionRows) → { edits, isDirty, dispatch, mergedRows }
├── DetectionOverlay (visible when !labelMode && showLabels)
├── LabelEditor (visible when labelMode, receives mergedRows + dispatch)
├── LabelToolbar (visible when labelMode)
└── PlaybackControls (Label button disabled when playing or zoom > 5m)
```

---

## 6. Overlap Prevention & Edge Cases

### Overlap Detection

Two windows overlap if `candidate.start < existing.end && candidate.end > existing.start`. Only labeled rows participate in overlap blocking; unlabeled rows are replaceable.

### Placement (Add Mode)

- Candidate = click position minus 2.5s (centered), snapped to 0.5s grid
- Overlaps labeled row → ghost red, click is no-op
- Overlaps unlabeled row(s) → ghost normal, unlabeled rows dim (will be replaced)
- Extends before job start or past job end → clamp to boundary

### Movement (Drag)

- Candidate position computed continuously during drag, snapped to 0.5s grid
- Overlaps labeled neighbor → hard-clamp to neighbor's edge
- Squeezed between two labeled neighbors with no valid position → label stays at original position
- Overlaps unlabeled row(s) → unlabeled rows dim (will be replaced on save)

### Deletion

Deleting a row removes it from the row store only. VocalizationLabels and LabelingAnnotations referencing the deleted `row_id` become orphaned. Acceptable for this phase; cleanup can be added later.

### Boundary Clamping

Labels clamp to `[0, job_duration]`. A 5s window placed near the end of the job is clamped so `end_sec` does not exceed job duration.

---

## 7. Testing Strategy

### Backend

- Unit tests for `PATCH /labels`: add, move, delete, change-type, overlap rejection, boundary clamping, single-label enforcement, unlabeled row replacement
- Unit tests for on-the-fly TSV generation from Parquet
- Unit tests for null-confidence handling across confidence strip, minimap, sorting, extraction
- Integration test: create detection job with rows, apply batch edit, verify Parquet row store updated correctly

### Frontend

- Playwright tests for:
  - Label mode entry/exit (button state, toolbar visibility, zoom restriction)
  - Add a label, verify bar appears with correct color
  - Select and delete a label
  - Save and verify changes persist on reload
  - Dirty-state warning on navigate-away attempt
  - Ghost preview and overlap-blocked (red) state

### Null Confidence Audit

Dedicated test creating detection rows with `null` confidence values, exercising every code path: rendering, sorting, extraction positive-selection, minimap, confidence strip.
