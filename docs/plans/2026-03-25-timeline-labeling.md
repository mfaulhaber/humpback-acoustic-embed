# Timeline Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add interactive species labeling to the timeline viewer, with batch save and extraction support, while removing the TSV sync layer so Parquet is the sole source of truth.

**Architecture:** New `LabelEditor` component renders interactive label bars as a sibling to the existing `DetectionOverlay`. A `useLabelEdits` reducer manages the client-side changeset. A new `PATCH /labels` endpoint applies batch edits atomically to the Parquet row store. TSV removal is a preparatory step that simplifies all write paths.

**Tech Stack:** React 18, TypeScript, Tailwind, FastAPI, PyArrow Parquet, Alembic (SQLite)

**Spec:** `docs/specs/2026-03-25-timeline-labeling-design.md`

---

## File Map

### Backend — New Files
| File | Responsibility |
|------|---------------|
| `alembic/versions/027_drop_output_tsv_path.py` | Migration to drop `output_tsv_path` column |
| `tests/unit/test_label_batch_endpoint.py` | Unit tests for the batch label edit endpoint |

### Backend — Modified Files
| File | Changes |
|------|---------|
| `src/humpback/classifier/detection_rows.py` | Remove `sync_detection_tsv()`, `write_tsv_rows()`, `read_tsv_rows()` exports from row store path; refactor `ensure_detection_row_store()` to remove TSV fallback; add `apply_label_edits()` function |
| `src/humpback/classifier/detector.py` | Keep `append_detections_tsv()` and `write_detections_tsv()` for running-job streaming (they stay until Task 2 replaces them); remove after Task 2 |
| `src/humpback/api/routers/classifier.py` | Remove `sync_detection_tsv` calls from `save_detection_labels()` and `save_detection_row_state()`; refactor `/content` endpoint to read Parquet for all statuses; add new `PATCH /labels` batch endpoint |
| `src/humpback/classifier/extractor.py` | Remove `sync_detection_tsv` call from `_write_detection_rows()` |
| `src/humpback/models/classifier.py` | Drop `output_tsv_path` field from `DetectionJob` model |
| `src/humpback/schemas/classifier.py` | Remove `output_tsv_path` from `DetectionJobOut`; add `LabelEditRequest`/`LabelEdit` schemas |
| `src/humpback/workers/classifier_worker.py` | Write running-job detections to Parquet row store instead of TSV |
| `src/humpback/storage.py` | Keep `detection_tsv_path()` for on-the-fly download generation (no change needed) |

### Frontend — New Files
| File | Responsibility |
|------|---------------|
| `frontend/src/components/timeline/LabelEditor.tsx` | Interactive label bars with select, drag-move, add ghost preview |
| `frontend/src/components/timeline/LabelToolbar.tsx` | Mode toggle, radio buttons, Save/Delete/Extract/Cancel buttons |
| `frontend/src/hooks/queries/useLabelEdits.ts` | `useReducer`-based changeset manager |
| `frontend/e2e/timeline-labeling.spec.ts` | Playwright tests for labeling interactions |

### Frontend — Modified Files
| File | Changes |
|------|---------|
| `frontend/src/components/timeline/TimelineViewer.tsx` | Add `labelMode` state, integrate `useLabelEdits`, render `LabelEditor` and `LabelToolbar`, gate playback on label mode |
| `frontend/src/components/timeline/DetectionOverlay.tsx` | Update colors to new warm/cool palette; hide when `labelMode` active |
| `frontend/src/components/timeline/DetectionPopover.tsx` | Guard `.toFixed()` calls against null confidence |
| `frontend/src/components/timeline/PlaybackControls.tsx` | Add Label button (disabled when playing or zoom > 5m) |
| `frontend/src/components/timeline/SpectrogramViewport.tsx` | Pass label mode props through to `LabelEditor`; adjust click handling when label mode active |
| `frontend/src/components/timeline/constants.ts` | Add label color constants for new palette |
| `frontend/src/api/types.ts` | Update `avg_confidence`/`peak_confidence` to `number \| null`; add `LabelEdit` types |
| `frontend/src/api/client.ts` | Add `patchDetectionLabels()` function |
| `frontend/src/hooks/queries/useTimeline.ts` | Add `useSaveLabels` mutation hook |

---

## Task 1: Remove TSV Sync from Existing Write Paths

Remove all `sync_detection_tsv()` calls from label-save and extraction endpoints. The download endpoint already generates TSV on-the-fly from the row store, so no change needed there.

**Files:**
- Modify: `src/humpback/classifier/detection_rows.py:676-687` (remove `sync_detection_tsv` body, keep for now as no-op)
- Modify: `src/humpback/api/routers/classifier.py:1034` (remove sync call)
- Modify: `src/humpback/api/routers/classifier.py:1135` (remove sync call)
- Modify: `src/humpback/classifier/extractor.py:201` (remove sync call)
- Modify: `src/humpback/classifier/detection_rows.py:1144` (remove sync call in `ensure_detection_row_store`)
- Test: `tests/unit/test_detection_rows.py`

- [ ] **Step 1: Remove `sync_detection_tsv` call from `save_detection_labels()`**

In `src/humpback/api/routers/classifier.py`, line 1034, remove:
```python
sync_detection_tsv(tsv, updated_rows, fieldnames)
```
And remove `tsv = detection_tsv_path(settings.storage_root, job.id)` at line 997 (now unused).

- [ ] **Step 2: Remove `sync_detection_tsv` call from `save_detection_row_state()`**

In `src/humpback/api/routers/classifier.py`, line 1135, remove:
```python
sync_detection_tsv(tsv, existing_rows, fieldnames)
```
And remove `tsv = detection_tsv_path(settings.storage_root, job.id)` at line 1061 (now unused).

- [ ] **Step 3: Remove `sync_detection_tsv` call from `_write_detection_rows()` in extractor**

In `src/humpback/classifier/extractor.py`, line 201, change the row-store branch to:
```python
if using_row_store and row_store_path is not None:
    write_detection_row_store(row_store_path, rows)
    return
```
Remove the `sync_detection_tsv` import from the file's imports.

- [ ] **Step 4: Remove `sync_detection_tsv` call from `ensure_detection_row_store()`**

In `src/humpback/classifier/detection_rows.py`, line 1144, remove:
```python
sync_detection_tsv(tsv_path, store_rows)
```

- [ ] **Step 5: Mark `sync_detection_tsv` as deprecated / delete**

In `src/humpback/classifier/detection_rows.py`, delete the `sync_detection_tsv()` function (lines 676-687) and `write_tsv_rows()` (lines 610-628). Also remove the `ROW_STORE_TSV_FIELDNAMES` constant (lines 96-98) if no longer referenced.

Keep `iter_detection_rows_as_tsv()` (lines 705-721) — it's used by the download endpoint for on-the-fly generation.

- [ ] **Step 6: Remove unused TSV imports from classifier router**

In `src/humpback/api/routers/classifier.py`, remove `sync_detection_tsv` and `detection_tsv_path` from imports (check for remaining usage in `/content` endpoint first — that's Task 2).

- [ ] **Step 7: Run tests to verify no regressions**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass. Some tests may need TSV-related assertions updated.

- [ ] **Step 8: Fix any failing tests**

Update test assertions that check for TSV file creation alongside row store writes. Tests in `tests/unit/test_detection_rows.py`, `tests/unit/test_extractor.py`, and `tests/integration/test_classifier_api.py` may assert TSV sync behavior — remove those assertions.

- [ ] **Step 9: Commit**

```bash
git add -u
git commit -m "Remove sync_detection_tsv from all write paths

Parquet row store is now the sole write target for label saves
and extraction updates. TSV download is generated on-the-fly."
```

---

## Task 2: Convert Running-Job Detection Output to Parquet

Replace TSV append during running detection jobs with incremental Parquet writes. The `/content` endpoint reads Parquet for all job statuses.

**Files:**
- Modify: `src/humpback/workers/classifier_worker.py` (lines 761-863 local, 932-1270 hydrophone)
- Modify: `src/humpback/api/routers/classifier.py:904-948` (`/content` endpoint)
- Modify: `src/humpback/classifier/detection_rows.py` (refactor `ensure_detection_row_store` to drop TSV param)
- Test: `tests/unit/test_classifier_worker.py`

- [ ] **Step 1: Add `append_detection_row_store()` function**

In `src/humpback/classifier/detection_rows.py`, add a function that appends rows to an existing Parquet file (read existing + append + atomic write):

```python
def append_detection_row_store(
    path: Path,
    new_rows: list[dict[str, str]],
) -> None:
    """Append rows to an existing Parquet row store, creating if needed."""
    existing_rows: list[dict[str, str]] = []
    if path.is_file():
        _, existing_rows = read_detection_row_store(path)
    write_detection_row_store(path, existing_rows + new_rows)
```

- [ ] **Step 2: Write failing test for `append_detection_row_store`**

In `tests/unit/test_detection_rows.py`, add:
```python
def test_append_detection_row_store(tmp_path):
    path = tmp_path / "rows.parquet"
    row1 = {"row_id": "a", "filename": "f1", "start_sec": "0", "end_sec": "5"}
    row2 = {"row_id": "b", "filename": "f2", "start_sec": "5", "end_sec": "10"}
    append_detection_row_store(path, [row1])
    _, rows = read_detection_row_store(path)
    assert len(rows) == 1
    append_detection_row_store(path, [row2])
    _, rows = read_detection_row_store(path)
    assert len(rows) == 2
    assert rows[1]["row_id"] == "b"
```

- [ ] **Step 3: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_detection_rows.py::test_append_detection_row_store -v`

- [ ] **Step 4: Replace TSV writes in local detection worker**

In `src/humpback/workers/classifier_worker.py`, in the local detection callback (`on_file_complete`, around line 793):
- Replace `append_detections_tsv(file_detections, tsv_path)` with building row-store-format dicts and calling `append_detection_row_store(rs_path, new_rows)`
- Remove the final `write_detections_tsv(detections, tsv_path)` around line 840
- Change `tsv_path` variable to `rs_path` using `detection_row_store_path()`

The `ensure_detection_row_store()` call at completion (line 858) can be simplified since the row store is already being written incrementally.

- [ ] **Step 5: Replace TSV writes in hydrophone detection worker**

In `src/humpback/workers/classifier_worker.py`, in the hydrophone detection callback (`on_chunk_complete`, around line 1017):
- Replace `append_detections_tsv(chunk_detections, tsv_path, ...)` with `append_detection_row_store(rs_path, new_rows)`
- Remove TSV resume reads (line 942-948) — use `read_detection_row_store()` instead
- Update all hydrophone completion paths (lines 1234-1277) to use row store directly

- [ ] **Step 6: Update `/content` endpoint to read Parquet for all statuses**

In `src/humpback/api/routers/classifier.py`, lines 920-925, replace the running-job TSV branch:
```python
# Before (remove):
if job.status == "running":
    if not tsv.is_file():
        raise HTTPException(404, "TSV file not found on disk")
    _fieldnames, raw_rows = read_tsv_rows(tsv)

# After:
rs_path = detection_row_store_path(settings.storage_root, job.id)
if job.status == "running":
    if not rs_path.is_file():
        return []  # No detections yet
    _fieldnames, raw_rows = read_detection_row_store(rs_path)
```

- [ ] **Step 7: Refactor `ensure_detection_row_store()` to drop TSV parameter**

Remove the `tsv_path` parameter. If the Parquet row store doesn't exist, return empty rows instead of falling back to TSV. Update all call sites.

- [ ] **Step 8: Remove `_ensure_detection_row_store_for_job()` TSV references**

In `src/humpback/api/routers/classifier.py`, the helper at lines 849-874 no longer needs `tsv` or `detection_tsv_path`. Simplify to just read Parquet.

- [ ] **Step 9: Run tests, fix any failures**

Run: `uv run pytest tests/ -x -q`
Update test fixtures that create TSV files — they should create Parquet row stores instead.

- [ ] **Step 10: Commit**

```bash
git add -u src/ tests/
git commit -m "Replace TSV streaming with incremental Parquet row store

Running detection jobs now append to Parquet directly.
The /content endpoint reads Parquet for all job statuses."
```

---

## Task 3: Drop `output_tsv_path` Column and Clean Up

Database migration and schema cleanup.

**Files:**
- Create: `alembic/versions/027_drop_output_tsv_path.py`
- Modify: `src/humpback/models/classifier.py:53` (remove field)
- Modify: `src/humpback/schemas/classifier.py:88` (remove field)
- Modify: `src/humpback/api/routers/classifier.py` (remove field from response construction)

- [ ] **Step 1: Create Alembic migration**

Create `alembic/versions/027_drop_output_tsv_path.py`:
```python
"""Drop output_tsv_path from detection_jobs."""

revision = "027"
down_revision = "026"

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("output_tsv_path")


def downgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(sa.Column("output_tsv_path", sa.String(), nullable=True))
```

- [ ] **Step 2: Run the migration**

Run: `uv run alembic upgrade head`

- [ ] **Step 3: Remove `output_tsv_path` from SQLAlchemy model**

In `src/humpback/models/classifier.py`, remove:
```python
output_tsv_path: Mapped[Optional[str]] = mapped_column(default=None)
```

- [ ] **Step 4: Remove `output_tsv_path` from Pydantic schema**

In `src/humpback/schemas/classifier.py`, remove `output_tsv_path: Optional[str] = None` from `DetectionJobOut`.

- [ ] **Step 5: Remove from API response construction**

Search for `output_tsv_path` in `src/humpback/api/routers/classifier.py` and remove any explicit field mapping in response construction.

- [ ] **Step 6: Clean up remaining TSV references**

Remove `detection_tsv_path` import from `classifier_worker.py` if no longer used. Remove `read_tsv_rows` from detection_rows.py if no longer called. Keep `iter_detection_rows_as_tsv` for the download endpoint.

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Run: `uv run pyright src/humpback/`

- [ ] **Step 8: Commit**

```bash
git add alembic/ src/ tests/
git commit -m "Drop output_tsv_path column from detection_jobs

Migration 027 removes the column. Parquet row store is the
sole detection data representation."
```

---

## Task 4: Null Confidence Audit and Type Updates

Make confidence fields nullable across the stack before adding manual labels.

**Files:**
- Modify: `frontend/src/api/types.ts:466-504` (DetectionRow interface)
- Modify: `frontend/src/components/timeline/DetectionPopover.tsx:103,115` (guard `.toFixed()`)
- Modify: `frontend/src/components/timeline/DetectionOverlay.tsx` (guard confidence reads)
- Modify: `frontend/src/components/timeline/SpectrogramViewport.tsx` (guard confidence strip)
- Test: `tests/unit/test_label_batch_endpoint.py` (null confidence test)

- [ ] **Step 1: Update `DetectionRow` TypeScript type**

In `frontend/src/api/types.ts`, change:
```typescript
avg_confidence: number;
peak_confidence: number;
```
to:
```typescript
avg_confidence: number | null;
peak_confidence: number | null;
```

- [ ] **Step 2: Fix `DetectionPopover.tsx` null crash**

In `frontend/src/components/timeline/DetectionPopover.tsx`, lines 103 and 115, guard the `.toFixed()` calls:
```typescript
// Line 103
{row.avg_confidence != null ? row.avg_confidence.toFixed(3) : "—"}
// Line 115
{row.peak_confidence != null ? row.peak_confidence.toFixed(3) : "—"}
```

- [ ] **Step 3: Guard confidence reads in `DetectionOverlay.tsx`**

In `frontend/src/components/timeline/DetectionOverlay.tsx`, the `confidenceColor()` function (line 29) receives `row.avg_confidence` — guard against null:
```typescript
const alpha = row.avg_confidence != null
  ? 0.08 + row.avg_confidence * 0.17
  : 0.08;
```

- [ ] **Step 4: Guard confidence tooltip in `DetectionOverlay.tsx`**

In the tooltip rendering, guard the confidence display values:
```typescript
avgConfidence: row.avg_confidence ?? 0,
peakConfidence: row.peak_confidence ?? 0,
```

- [ ] **Step 5: Run TypeScript type check**

Run: `cd frontend && npx tsc --noEmit`
Fix any remaining type errors where `number | null` causes issues.

- [ ] **Step 6: Update `normalize_detection_row()` for null confidence**

In `src/humpback/classifier/detection_rows.py`, the `normalize_detection_row()` function currently uses `safe_float(row.get("avg_confidence"), 0.0)` and `safe_float(row.get("peak_confidence"), 0.0)` which return `0.0` for empty strings. Change these to use `safe_optional_float()` so empty/missing values become `None`:

```python
# Before:
"avg_confidence": safe_float(row.get("avg_confidence"), 0.0),
"peak_confidence": safe_float(row.get("peak_confidence"), 0.0),

# After:
"avg_confidence": safe_optional_float(row.get("avg_confidence")),
"peak_confidence": safe_optional_float(row.get("peak_confidence")),
```

This ensures manually-placed labels (which have empty confidence in the row store) return `null` in the API response.

- [ ] **Step 7: Write backend test for null confidence row**

In `tests/unit/test_label_batch_endpoint.py`, add a test that creates a detection row with null/empty confidence and verifies it normalizes to `None`:
```python
def test_normalize_null_confidence_row():
    row = {
        "filename": "test.wav",
        "start_sec": "0.0",
        "end_sec": "5.0",
        "avg_confidence": "",
        "peak_confidence": "",
        "humpback": "1",
    }
    result = normalize_detection_row(
        row, is_hydrophone=False, window_size_seconds=5.0
    )
    assert result["avg_confidence"] is None
    assert result["peak_confidence"] is None
```

- [ ] **Step 8: Run test**

Run: `uv run pytest tests/unit/test_label_batch_endpoint.py::test_normalize_null_confidence_row -v`

- [ ] **Step 8: Commit**

```bash
git add frontend/src/ tests/
git commit -m "Make confidence fields nullable for manual labels

DetectionRow.avg_confidence and peak_confidence are now
number | null. Guards added in popover, overlay, and tooltip."
```

---

## Task 5: Add Label Color Constants and Update DetectionOverlay Palette

Update the label bar colors to the new warm/cool scheme.

**Files:**
- Modify: `frontend/src/components/timeline/constants.ts`
- Modify: `frontend/src/components/timeline/DetectionOverlay.tsx:18-26`

- [ ] **Step 1: Add label color constants**

In `frontend/src/components/timeline/constants.ts`, add after the existing `COLORS` object:

```typescript
export const LABEL_COLORS = {
  humpback: { fill: "rgba(234, 179, 8, 0.45)", hover: "rgba(234, 179, 8, 0.65)", border: "rgb(234, 179, 8)" },
  orca: { fill: "rgba(249, 115, 22, 0.45)", hover: "rgba(249, 115, 22, 0.65)", border: "rgb(249, 115, 22)" },
  ship: { fill: "rgba(100, 149, 237, 0.35)", hover: "rgba(100, 149, 237, 0.55)", border: "rgb(100, 149, 237)" },
  background: { fill: "rgba(156, 163, 175, 0.30)", hover: "rgba(156, 163, 175, 0.50)", border: "rgb(156, 163, 175)" },
} as const;

export type LabelType = keyof typeof LABEL_COLORS;
```

- [ ] **Step 2: Update `DetectionOverlay.tsx` to use new colors**

Replace the `POSITIVE_COLORS` and `POSITIVE_HOVER_COLORS` constants (lines 18-26) with imports from `constants.ts`. Update the color selection logic to use `LABEL_COLORS[label].fill` and `LABEL_COLORS[label].hover`. Also render ship and background labels (currently skipped) so all four label types are visible on the timeline.

- [ ] **Step 3: Verify visually**

Start the dev server (`cd frontend && npm run dev`) and check a detection job timeline to confirm the new colors render correctly. Humpback = amber, orca = orange, ship = blue, background = gray.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/timeline/
git commit -m "Update timeline label colors to warm/cool palette

Humpback=amber, orca=orange (whale/warm), ship=blue,
background=gray (non-whale/cool). All label types now visible."
```

---

## Task 6: Add `useLabelEdits` Reducer Hook

The core state management for label editing.

**Files:**
- Create: `frontend/src/hooks/queries/useLabelEdits.ts`
- Test: (tested via Playwright in Task 12)

- [ ] **Step 1: Create the reducer hook**

Create `frontend/src/hooks/queries/useLabelEdits.ts`:

```typescript
import { useReducer, useMemo, useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import type { DetectionRow } from "@/api/types";
import type { LabelType } from "@/components/timeline/constants";

export interface LabelEdit {
  action: "add" | "move" | "delete" | "change_type";
  id?: string;          // for "add" — temp client ID
  row_id?: string;      // for move/delete/change_type — existing row
  start_sec?: number;   // for "add"
  end_sec?: number;     // for "add"
  new_start_sec?: number; // for "move"
  new_end_sec?: number;   // for "move"
  label?: LabelType;    // for "add" and "change_type"
}

interface State {
  edits: LabelEdit[];
  selectedId: string | null;  // row_id or temp id of selected label
}

export type Action =
  | { type: "add"; start_sec: number; end_sec: number; label: LabelType }
  | { type: "move"; row_id: string; new_start_sec: number; new_end_sec: number }
  | { type: "delete"; row_id: string }
  | { type: "change_type"; row_id: string; label: LabelType }
  | { type: "select"; id: string | null }
  | { type: "clear" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "add": {
      const id = uuidv4();
      return {
        ...state,
        edits: [
          ...state.edits,
          { action: "add", id, start_sec: action.start_sec, end_sec: action.end_sec, label: action.label },
        ],
        selectedId: id,
      };
    }
    case "move": {
      // Collapse: if already moved, update the existing move; if it's an add, update the add
      const existing = state.edits.findIndex(
        (e) => (e.action === "move" && e.row_id === action.row_id) ||
               (e.action === "add" && e.id === action.row_id)
      );
      if (existing >= 0) {
        const edits = [...state.edits];
        const e = { ...edits[existing] };
        if (e.action === "add") {
          e.start_sec = action.new_start_sec;
          e.end_sec = action.new_end_sec;
        } else {
          e.new_start_sec = action.new_start_sec;
          e.new_end_sec = action.new_end_sec;
        }
        edits[existing] = e;
        return { ...state, edits };
      }
      return {
        ...state,
        edits: [...state.edits, { action: "move", row_id: action.row_id, new_start_sec: action.new_start_sec, new_end_sec: action.new_end_sec }],
      };
    }
    case "delete": {
      // If deleting an "add" edit, just remove it
      const addIdx = state.edits.findIndex((e) => e.action === "add" && e.id === action.row_id);
      if (addIdx >= 0) {
        return {
          ...state,
          edits: state.edits.filter((_, i) => i !== addIdx),
          selectedId: state.selectedId === action.row_id ? null : state.selectedId,
        };
      }
      // Remove any prior edits for this row, add delete
      return {
        ...state,
        edits: [
          ...state.edits.filter((e) => e.row_id !== action.row_id),
          { action: "delete", row_id: action.row_id },
        ],
        selectedId: state.selectedId === action.row_id ? null : state.selectedId,
      };
    }
    case "change_type": {
      // Collapse: if it's an add, update in place; if already changed, update
      const addIdx = state.edits.findIndex((e) => e.action === "add" && e.id === action.row_id);
      if (addIdx >= 0) {
        const edits = [...state.edits];
        edits[addIdx] = { ...edits[addIdx], label: action.label };
        return { ...state, edits };
      }
      const changeIdx = state.edits.findIndex((e) => e.action === "change_type" && e.row_id === action.row_id);
      if (changeIdx >= 0) {
        const edits = [...state.edits];
        edits[changeIdx] = { ...edits[changeIdx], label: action.label };
        return { ...state, edits };
      }
      return {
        ...state,
        edits: [...state.edits, { action: "change_type", row_id: action.row_id, label: action.label }],
      };
    }
    case "select":
      return { ...state, selectedId: action.id };
    case "clear":
      return { edits: [], selectedId: null };
    default:
      return state;
  }
}

export function useLabelEdits(originalRows: DetectionRow[]) {
  const [state, dispatch] = useReducer(reducer, { edits: [], selectedId: null });

  const mergedRows = useMemo(() => {
    const rows = originalRows.map((r) => ({ ...r }));
    const deletedIds = new Set(
      state.edits.filter((e) => e.action === "delete").map((e) => e.row_id)
    );

    // Apply moves and type changes
    for (const edit of state.edits) {
      if (edit.action === "move") {
        const row = rows.find((r) => r.row_id === edit.row_id);
        if (row && edit.new_start_sec != null && edit.new_end_sec != null) {
          const duration = row.end_sec - row.start_sec;
          row.start_sec = edit.new_start_sec;
          row.end_sec = edit.new_end_sec;
        }
      }
      if (edit.action === "change_type") {
        const row = rows.find((r) => r.row_id === edit.row_id);
        if (row && edit.label) {
          row.humpback = edit.label === "humpback" ? 1 : 0;
          row.orca = edit.label === "orca" ? 1 : 0;
          row.ship = edit.label === "ship" ? 1 : 0;
          row.background = edit.label === "background" ? 1 : 0;
        }
      }
    }

    // Add new rows
    for (const edit of state.edits) {
      if (edit.action === "add" && edit.start_sec != null && edit.end_sec != null && edit.label) {
        rows.push({
          row_id: edit.id ?? null,
          filename: "",
          start_sec: edit.start_sec,
          end_sec: edit.end_sec,
          avg_confidence: null,
          peak_confidence: null,
          n_windows: null,
          humpback: edit.label === "humpback" ? 1 : 0,
          orca: edit.label === "orca" ? 1 : 0,
          ship: edit.label === "ship" ? 1 : 0,
          background: edit.label === "background" ? 1 : 0,
        } as DetectionRow);
      }
    }

    // Remove deleted rows and unlabeled rows that overlap with added rows
    const addEdits = state.edits.filter((e) => e.action === "add");
    return rows.filter((r) => {
      if (deletedIds.has(r.row_id ?? "")) return false;
      // Check if unlabeled row overlaps an added label
      const isLabeled = r.humpback === 1 || r.orca === 1 || r.ship === 1 || r.background === 1;
      if (!isLabeled) {
        for (const add of addEdits) {
          if (add.start_sec != null && add.end_sec != null &&
              r.start_sec < add.end_sec && r.end_sec > add.start_sec) {
            return false; // Replaced by new labeled row
          }
        }
      }
      return true;
    });
  }, [originalRows, state.edits]);

  const isDirty = state.edits.length > 0;

  return { state, dispatch, mergedRows, isDirty, selectedId: state.selectedId };
}
```

- [ ] **Step 2: Add uuid dependency if not present**

Run: `cd frontend && npm ls uuid` — check if uuid is installed. If not:
Run: `cd frontend && npm install uuid && npm install -D @types/uuid`

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/queries/useLabelEdits.ts frontend/package.json frontend/package-lock.json
git commit -m "Add useLabelEdits reducer hook for timeline label editing

Manages add/move/delete/change_type changeset with edit
collapsing and merged row computation."
```

---

## Task 7: Add Backend Batch Label Edit Endpoint

The `PATCH /classifier/detection-jobs/{job_id}/labels` endpoint.

**Files:**
- Modify: `src/humpback/schemas/classifier.py` (add request schema)
- Modify: `src/humpback/classifier/detection_rows.py` (add `apply_label_edits()`)
- Modify: `src/humpback/api/routers/classifier.py` (add endpoint)
- Create: `tests/unit/test_label_batch_endpoint.py`

- [ ] **Step 1: Write failing test for batch edit — add**

Create `tests/unit/test_label_batch_endpoint.py`:

```python
import pytest
from pathlib import Path
from humpback.classifier.detection_rows import (
    apply_label_edits,
    read_detection_row_store,
    write_detection_row_store,
    ROW_STORE_FIELDNAMES,
)


def _make_row(row_id: str, start: float, end: float, label: str | None = None) -> dict[str, str]:
    row = {f: "" for f in ROW_STORE_FIELDNAMES}
    row["row_id"] = row_id
    row["start_sec"] = str(start)
    row["end_sec"] = str(end)
    if label:
        row[label] = "1"
    return row


def test_apply_label_edits_add(tmp_path: Path):
    rows = [_make_row("r1", 0.0, 5.0, "humpback")]
    edits = [
        {"action": "add", "start_sec": 10.0, "end_sec": 15.0, "label": "orca"},
    ]
    result = apply_label_edits(rows, edits, job_duration=100.0, window_size_seconds=5.0)
    assert len(result) == 2
    new_row = [r for r in result if r["row_id"] != "r1"][0]
    assert new_row["orca"] == "1"
    assert new_row["humpback"] == ""
    assert float(new_row["start_sec"]) == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_label_batch_endpoint.py::test_apply_label_edits_add -v`
Expected: FAIL — `apply_label_edits` not yet defined.

- [ ] **Step 3: Write failing tests for move, delete, change_type, overlap rejection**

Add to `tests/unit/test_label_batch_endpoint.py`:

```python
def test_apply_label_edits_move(tmp_path: Path):
    rows = [
        _make_row("r1", 0.0, 5.0, "humpback"),
        _make_row("r2", 10.0, 15.0, "orca"),
    ]
    edits = [{"action": "move", "row_id": "r1", "new_start_sec": 5.0, "new_end_sec": 10.0}]
    result = apply_label_edits(rows, edits, job_duration=100.0, window_size_seconds=5.0)
    moved = [r for r in result if r["row_id"] == "r1"][0]
    assert float(moved["start_sec"]) == 5.0
    assert float(moved["end_sec"]) == 10.0


def test_apply_label_edits_delete(tmp_path: Path):
    rows = [_make_row("r1", 0.0, 5.0, "humpback")]
    edits = [{"action": "delete", "row_id": "r1"}]
    result = apply_label_edits(rows, edits, job_duration=100.0, window_size_seconds=5.0)
    assert len(result) == 0


def test_apply_label_edits_change_type(tmp_path: Path):
    rows = [_make_row("r1", 0.0, 5.0, "humpback")]
    edits = [{"action": "change_type", "row_id": "r1", "label": "ship"}]
    result = apply_label_edits(rows, edits, job_duration=100.0, window_size_seconds=5.0)
    assert result[0]["ship"] == "1"
    assert result[0]["humpback"] == ""


def test_apply_label_edits_overlap_rejected():
    rows = [_make_row("r1", 0.0, 5.0, "humpback")]
    edits = [{"action": "add", "start_sec": 3.0, "end_sec": 8.0, "label": "orca"}]
    with pytest.raises(ValueError, match="overlap"):
        apply_label_edits(rows, edits, job_duration=100.0, window_size_seconds=5.0)


def test_apply_label_edits_move_out_of_bounds():
    rows = [_make_row("r1", 0.0, 5.0, "humpback")]
    edits = [{"action": "move", "row_id": "r1", "new_start_sec": -1.0, "new_end_sec": 4.0}]
    with pytest.raises(ValueError, match="bounds"):
        apply_label_edits(rows, edits, job_duration=100.0, window_size_seconds=5.0)


def test_apply_label_edits_unlabeled_replacement():
    """Adding a label over an unlabeled row should replace it."""
    rows = [
        _make_row("r1", 0.0, 5.0),  # unlabeled
        _make_row("r2", 10.0, 15.0, "humpback"),
    ]
    edits = [{"action": "add", "start_sec": 0.0, "end_sec": 5.0, "label": "orca"}]
    result = apply_label_edits(rows, edits, job_duration=100.0, window_size_seconds=5.0)
    assert len(result) == 2  # r1 replaced by new, r2 kept
    new_row = [r for r in result if r.get("orca") == "1"][0]
    assert float(new_row["start_sec"]) == 0.0


def test_apply_label_edits_single_label_enforcement():
    """Setting a label must clear all other labels."""
    rows = [_make_row("r1", 0.0, 5.0, "humpback")]
    edits = [{"action": "change_type", "row_id": "r1", "label": "orca"}]
    result = apply_label_edits(rows, edits, job_duration=100.0, window_size_seconds=5.0)
    r = result[0]
    assert r["orca"] == "1"
    assert r["humpback"] == ""
    assert r["ship"] == ""
    assert r["background"] == ""
```

- [ ] **Step 4: Implement `apply_label_edits()` in detection_rows.py**

Add to `src/humpback/classifier/detection_rows.py`:

```python
LABEL_FIELDS = ["humpback", "orca", "ship", "background"]


def apply_label_edits(
    rows: list[dict[str, str]],
    edits: list[dict],
    *,
    job_duration: float,
    window_size_seconds: float,
) -> list[dict[str, str]]:
    """Apply a batch of label edits to detection rows. Returns updated rows.

    Raises ValueError if the resulting state has overlapping labeled windows
    or if edits reference invalid row_ids / out-of-bounds positions.
    """
    result = [dict(row) for row in rows]
    row_by_id = {r["row_id"]: r for r in result if r.get("row_id")}

    # Collect deletes
    delete_ids = {e["row_id"] for e in edits if e["action"] == "delete"}

    # Apply moves
    for edit in edits:
        if edit["action"] == "move":
            row = row_by_id.get(edit["row_id"])
            if row is None:
                raise ValueError(f"Row {edit['row_id']} not found")
            new_start = edit["new_start_sec"]
            new_end = edit["new_end_sec"]
            if new_start < 0 or new_end > job_duration:
                raise ValueError(f"Move target out of bounds: [{new_start}, {new_end}]")
            row["start_sec"] = f"{new_start:.6f}"
            row["end_sec"] = f"{new_end:.6f}"

    # Apply type changes
    for edit in edits:
        if edit["action"] == "change_type":
            row = row_by_id.get(edit["row_id"])
            if row is None:
                raise ValueError(f"Row {edit['row_id']} not found")
            for field in LABEL_FIELDS:
                row[field] = "1" if field == edit["label"] else ""

    # Process adds: check for unlabeled replacements
    add_edits = [e for e in edits if e["action"] == "add"]
    replaced_ids: set[str] = set()
    for add in add_edits:
        start, end = add["start_sec"], add["end_sec"]
        if start < 0 or end > job_duration:
            raise ValueError(f"Add position out of bounds: [{start}, {end}]")
        for row in result:
            if row["row_id"] in delete_ids or row["row_id"] in replaced_ids:
                continue
            row_start = float(row["start_sec"] or 0)
            row_end = float(row["end_sec"] or 0)
            if row_start < end and row_end > start:
                # Check if labeled
                is_labeled = any(row.get(f, "").strip() == "1" for f in LABEL_FIELDS)
                if is_labeled:
                    raise ValueError(f"Add overlaps labeled row {row['row_id']}")
                else:
                    replaced_ids.add(row["row_id"])

    # Remove deleted and replaced rows
    result = [r for r in result if r["row_id"] not in delete_ids and r["row_id"] not in replaced_ids]

    # Create new rows for adds
    for add in add_edits:
        new_row = {f: "" for f in ROW_STORE_FIELDNAMES}
        new_row["row_id"] = build_detection_row_id({
            "filename": "",
            "detection_filename": "",
            "start_sec": add["start_sec"],
            "end_sec": add["end_sec"],
        })
        new_row["start_sec"] = f"{add['start_sec']:.6f}"
        new_row["end_sec"] = f"{add['end_sec']:.6f}"
        for field in LABEL_FIELDS:
            new_row[field] = "1" if field == add["label"] else ""
        result.append(new_row)

    # Validate: no overlapping labeled windows in final state
    labeled = [
        r for r in result
        if any(r.get(f, "").strip() == "1" for f in LABEL_FIELDS)
    ]
    labeled.sort(key=lambda r: float(r["start_sec"] or 0))
    for i in range(len(labeled) - 1):
        a_end = float(labeled[i]["end_sec"] or 0)
        b_start = float(labeled[i + 1]["start_sec"] or 0)
        if a_end > b_start:
            raise ValueError(
                f"Overlapping labeled windows: [{labeled[i]['start_sec']}, {labeled[i]['end_sec']}] "
                f"and [{labeled[i+1]['start_sec']}, {labeled[i+1]['end_sec']}]"
            )

    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_label_batch_endpoint.py -v`
Expected: All 7 tests pass.

- [ ] **Step 6: Add Pydantic request schema**

In `src/humpback/schemas/classifier.py`, add:

```python
from typing import Literal


class LabelEditItem(BaseModel):
    action: Literal["add", "move", "delete", "change_type"]
    row_id: Optional[str] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    new_start_sec: Optional[float] = None
    new_end_sec: Optional[float] = None
    label: Optional[Literal["humpback", "orca", "ship", "background"]] = None


class LabelEditRequest(BaseModel):
    edits: list[LabelEditItem]
```

- [ ] **Step 7: Add PATCH endpoint**

In `src/humpback/api/routers/classifier.py`, add:

```python
@router.patch("/detection-jobs/{job_id}/labels")
async def batch_edit_labels(
    job_id: str,
    body: LabelEditRequest,
    session: SessionDep,
    settings: SettingsDep,
) -> list[dict]:
    """Apply a batch of label edits (add/move/delete/change_type)."""
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    if job.status not in ("paused", "complete", "canceled"):
        raise HTTPException(400, "Detection job not complete or no output available")
    _require_windowed_detection_job(job, operation="batch edit labels")

    rs_path = detection_row_store_path(settings.storage_root, job.id)
    window_size_seconds = await _get_classifier_window_size(
        session, job.classifier_model_id
    )
    fieldnames, existing_rows = await _ensure_detection_row_store_for_job(
        session, job, settings=settings, window_size_seconds=window_size_seconds,
    )

    is_hydrophone = job.hydrophone_id is not None
    if is_hydrophone:
        job_duration = (job.end_timestamp or 0) - (job.start_timestamp or 0)
    else:
        job_duration = max(
            (float(r.get("end_sec") or 0) for r in existing_rows), default=0.0
        )

    try:
        updated_rows = apply_label_edits(
            existing_rows,
            [e.model_dump() for e in body.edits],
            job_duration=job_duration,
            window_size_seconds=window_size_seconds,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    # Re-apply positive selection for modified rows
    for row in updated_rows:
        apply_effective_positive_selection(row, window_size_seconds=window_size_seconds)

    write_detection_row_store(rs_path, updated_rows)

    has_positive = any(
        row.get("humpback") == "1" or row.get("orca") == "1" for row in updated_rows
    )
    job.has_positive_labels = has_positive
    await session.commit()

    # Return normalized rows
    return [
        normalize_detection_row(
            row, is_hydrophone=is_hydrophone, window_size_seconds=window_size_seconds
        )
        for row in updated_rows
    ]
```

- [ ] **Step 8: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Run: `uv run pyright src/humpback/`

- [ ] **Step 9: Commit**

```bash
git add src/ tests/
git commit -m "Add PATCH /detection-jobs/{job_id}/labels batch edit endpoint

Supports add, move, delete, and change_type operations with
overlap validation and single-label enforcement."
```

---

## Task 8: Add Frontend API Client and Mutation Hook

Wire the batch save endpoint to the frontend.

**Files:**
- Modify: `frontend/src/api/types.ts` (add LabelEdit type)
- Modify: `frontend/src/api/client.ts` (add `patchDetectionLabels()`)
- Modify: `frontend/src/hooks/queries/useTimeline.ts` (add `useSaveLabels()`)

- [ ] **Step 1: Add types**

In `frontend/src/api/types.ts`, add:

```typescript
export interface LabelEditItem {
  action: "add" | "move" | "delete" | "change_type";
  row_id?: string;
  start_sec?: number;
  end_sec?: number;
  new_start_sec?: number;
  new_end_sec?: number;
  label?: "humpback" | "orca" | "ship" | "background";
}

export interface LabelEditRequest {
  edits: LabelEditItem[];
}
```

- [ ] **Step 2: Add API client function**

In `frontend/src/api/client.ts`, add:

```typescript
export const patchDetectionLabels = (jobId: string, edits: LabelEditItem[]) =>
  patch<DetectionRow[]>(`/classifier/detection-jobs/${jobId}/labels`, { edits });
```

If no `patch()` helper exists yet, add one alongside the existing `put()` helper:
```typescript
async function patch<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

- [ ] **Step 3: Add mutation hook**

In `frontend/src/hooks/queries/useTimeline.ts`, add:

```typescript
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { patchDetectionLabels } from "@/api/client";
import type { LabelEditItem } from "@/api/types";

export function useSaveLabels(jobId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (edits: LabelEditItem[]) => patchDetectionLabels(jobId, edits),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["timelineDetections", jobId] });
    },
  });
}
```

- [ ] **Step 4: TypeScript check**

Run: `cd frontend && npx tsc --noEmit`

- [ ] **Step 5: Commit**

```bash
git add frontend/src/
git commit -m "Add frontend API client and hook for batch label edits"
```

---

## Task 9: Add LabelToolbar Component

The control bar shown during label mode.

**Files:**
- Create: `frontend/src/components/timeline/LabelToolbar.tsx`

- [ ] **Step 1: Create LabelToolbar**

Create `frontend/src/components/timeline/LabelToolbar.tsx`:

```typescript
import { Tag, Trash2, Save, Download, X } from "lucide-react";
import { COLORS, LABEL_COLORS, type LabelType } from "./constants";

const LABEL_OPTIONS: LabelType[] = ["humpback", "orca", "ship", "background"];

export interface LabelToolbarProps {
  mode: "select" | "add";
  onModeChange: (mode: "select" | "add") => void;
  selectedLabel: LabelType;
  onLabelChange: (label: LabelType) => void;
  onDelete: () => void;
  onSave: () => void;
  onExtract: () => void;
  onCancel: () => void;
  isDirty: boolean;
  isSaving: boolean;
  hasSelection: boolean;
}

export function LabelToolbar({
  mode,
  onModeChange,
  selectedLabel,
  onLabelChange,
  onDelete,
  onSave,
  onExtract,
  onCancel,
  isDirty,
  isSaving,
  hasSelection,
}: LabelToolbarProps) {
  return (
    <div
      className="flex items-center gap-4 px-4 py-2"
      style={{
        background: COLORS.headerBg,
        borderTop: `1px solid ${COLORS.border}`,
      }}
    >
      {/* Mode toggle */}
      <div className="flex items-center gap-1 rounded overflow-hidden"
           style={{ border: `1px solid ${COLORS.border}` }}>
        <button
          className="px-2.5 py-1 text-[10px] font-medium"
          style={{
            background: mode === "select" ? COLORS.accent : "transparent",
            color: mode === "select" ? COLORS.bgDark : COLORS.textMuted,
          }}
          onClick={() => onModeChange("select")}
        >
          Select
        </button>
        <button
          className="px-2.5 py-1 text-[10px] font-medium"
          style={{
            background: mode === "add" ? COLORS.accent : "transparent",
            color: mode === "add" ? COLORS.bgDark : COLORS.textMuted,
          }}
          onClick={() => onModeChange("add")}
        >
          Add
        </button>
      </div>

      {/* Label radio buttons */}
      <div className="flex items-center gap-3">
        {LABEL_OPTIONS.map((label) => (
          <label
            key={label}
            className="flex items-center gap-1.5 cursor-pointer text-[10px]"
            style={{ color: selectedLabel === label ? LABEL_COLORS[label].border : COLORS.textMuted }}
          >
            <input
              type="radio"
              name="label-type"
              checked={selectedLabel === label}
              onChange={() => onLabelChange(label)}
              className="w-3 h-3"
              style={{ accentColor: LABEL_COLORS[label].border }}
            />
            {label}
          </label>
        ))}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Actions */}
      <div className="flex items-center gap-2">
        <button
          className="flex items-center gap-1 px-2 py-1 rounded text-[10px]"
          style={{
            color: hasSelection ? "#ef4444" : COLORS.textMuted,
            opacity: hasSelection ? 1 : 0.4,
          }}
          onClick={onDelete}
          disabled={!hasSelection}
          title="Delete selected label"
        >
          <Trash2 size={12} /> Delete
        </button>

        <button
          className="flex items-center gap-1 px-2.5 py-1 rounded text-[10px] font-medium"
          style={{
            background: isDirty ? COLORS.accent : "transparent",
            color: isDirty ? COLORS.bgDark : COLORS.textMuted,
            opacity: isDirty ? 1 : 0.4,
          }}
          onClick={onSave}
          disabled={!isDirty || isSaving}
          title="Save label changes"
        >
          <Save size={12} /> {isSaving ? "Saving…" : "Save"}
          {isDirty && !isSaving && (
            <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 ml-1" />
          )}
        </button>

        <button
          className="flex items-center gap-1 px-2 py-1 rounded text-[10px]"
          style={{ color: COLORS.textMuted }}
          onClick={onExtract}
          title="Extract labeled audio"
        >
          <Download size={12} /> Extract
        </button>

        <button
          className="flex items-center gap-1 px-2 py-1 rounded text-[10px]"
          style={{ color: COLORS.textMuted }}
          onClick={onCancel}
          title="Exit label mode"
        >
          <X size={12} /> Cancel
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/timeline/LabelToolbar.tsx
git commit -m "Add LabelToolbar component for timeline label mode"
```

---

## Task 10: Add LabelEditor Component

The interactive label bar rendering with select, drag-move, and add-with-ghost.

**Files:**
- Create: `frontend/src/components/timeline/LabelEditor.tsx`

- [ ] **Step 1: Create LabelEditor**

Create `frontend/src/components/timeline/LabelEditor.tsx`. This is the largest new component. Key behaviors:

1. Renders all labeled rows from `mergedRows` as colored bars (same coordinate math as `DetectionOverlay`)
2. In **select mode**: clicking a bar selects it (white border + glow). Dragging a selected bar moves it horizontally (0.5s snap grid, overlap clamping).
3. In **add mode**: mouse-tracking ghost preview follows cursor (5s width, label color). Click to place.
4. Unlabeled rows that would be replaced by add/move dim with reduced opacity.
5. Manual labels (null confidence) show dashed top border.

The component receives: `mergedRows`, `mode`, `selectedLabel`, `selectedId`, `dispatch`, `jobStart`, `jobDuration`, `centerTimestamp`, `zoomLevel`, `width`, `height`.

Core implementation structure:

```typescript
import { useCallback, useRef, useState, useMemo } from "react";
import type { DetectionRow } from "@/api/types";
import { LABEL_COLORS, TILE_WIDTH_PX, TILE_DURATION, type LabelType } from "./constants";
import type { ZoomLevel } from "./constants";
import type { Action } from "@/hooks/queries/useLabelEdits";
// Note: Action type is exported from useLabelEdits.ts

const SNAP_GRID = 0.5; // seconds
const WINDOW_DURATION = 5.0; // seconds

function snapToGrid(sec: number): number {
  return Math.round(sec / SNAP_GRID) * SNAP_GRID;
}

function getRowLabel(row: DetectionRow): LabelType | null {
  if (row.humpback === 1) return "humpback";
  if (row.orca === 1) return "orca";
  if (row.ship === 1) return "ship";
  if (row.background === 1) return "background";
  return null;
}

function isLabeled(row: DetectionRow): boolean {
  return getRowLabel(row) !== null;
}

// ... (full component implementation with drag handling, ghost preview,
//      overlap detection, and coordinate transforms)
```

The full component handles:
- `onMouseDown` on a bar → start drag (select mode) or no-op (add mode)
- `onMouseMove` on container → update drag position or ghost position
- `onMouseUp` → commit move via dispatch, snapped to grid
- `onClick` on container (add mode) → dispatch add if no overlap with labeled row
- Overlap checking: iterate `mergedRows` labeled rows, check `candidate.start < existing.end && candidate.end > existing.start`
- Clamping: during drag, if overlap detected with neighbor, clamp to neighbor edge

- [ ] **Step 2: Implement coordinate transforms**

Use the same math as `DetectionOverlay`:
```typescript
const pxPerSec = width / VIEWPORT_SPAN[zoomLevel];
// Convert row seconds to pixel x:
const x = (jobStart + row.start_sec - centerTimestamp) * pxPerSec + width / 2;
const w = (row.end_sec - row.start_sec) * pxPerSec;
// Convert pixel x to seconds:
const sec = (x - width / 2) / pxPerSec + centerTimestamp - jobStart;
```

Import `VIEWPORT_SPAN` from constants.

- [ ] **Step 3: Implement drag-move with overlap clamping**

During drag, compute candidate position, check labeled neighbors, clamp if needed:
```typescript
const clampedStart = Math.max(0, Math.min(jobDuration - duration, candidateStart));
// Check labeled neighbors
for (const other of labeledRows) {
  if (other.row_id === dragRowId) continue;
  if (clampedStart < other.end_sec && clampedStart + duration > other.start_sec) {
    // Clamp to neighbor edge
    if (dragDirection > 0) clampedStart = other.start_sec - duration;
    else clampedStart = other.end_sec;
  }
}
```

- [ ] **Step 4: Implement ghost preview for add mode**

Track mouse position, show a semi-transparent bar at the snapped position:
```typescript
const ghostStart = snapToGrid(mouseSec - WINDOW_DURATION / 2);
const ghostEnd = ghostStart + WINDOW_DURATION;
const overlapsLabeled = labeledRows.some(
  r => r.start_sec < ghostEnd && r.end_sec > ghostStart
);
const overlapsUnlabeled = mergedRows.some(
  r => !isLabeled(r) && r.start_sec < ghostEnd && r.end_sec > ghostStart
);
// Render ghost: red if overlapsLabeled, normal + dim unlabeled if overlapsUnlabeled
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/timeline/LabelEditor.tsx
git commit -m "Add LabelEditor component with select, drag-move, and add mode"
```

---

## Task 11: Integrate Label Mode into TimelineViewer

Wire everything together in the main component.

**Files:**
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`
- Modify: `frontend/src/components/timeline/PlaybackControls.tsx`
- Modify: `frontend/src/components/timeline/SpectrogramViewport.tsx`

- [ ] **Step 1: Add Label button to PlaybackControls**

In `frontend/src/components/timeline/PlaybackControls.tsx`, add new props:
```typescript
onLabelMode?: () => void;
labelModeEnabled: boolean;  // false when playing or zoom > 5m
```

Add a Label button after the zoom controls:
```typescript
<button
  className="flex items-center gap-1 px-2.5 py-1 rounded text-[10px] font-medium ml-4"
  style={{
    background: "transparent",
    color: labelModeEnabled ? COLORS.accent : COLORS.textMuted,
    opacity: labelModeEnabled ? 1 : 0.3,
    border: `1px solid ${labelModeEnabled ? COLORS.accent : COLORS.border}`,
  }}
  onClick={onLabelMode}
  disabled={!labelModeEnabled}
  title={labelModeEnabled ? "Enter label mode" : "Zoom to 5m or closer to edit labels"}
>
  <Tag size={12} /> Label
</button>
```

- [ ] **Step 2: Add label mode state to TimelineViewer**

In `frontend/src/components/timeline/TimelineViewer.tsx`, add:
```typescript
const [labelMode, setLabelMode] = useState(false);
const [labelSubMode, setLabelSubMode] = useState<"select" | "add">("select");
const [selectedLabel, setSelectedLabel] = useState<LabelType>("humpback");
const [extractOpen, setExtractOpen] = useState(false);

const { state: labelState, dispatch: labelDispatch, mergedRows, isDirty, selectedId } =
  useLabelEdits(detections ?? []);
const saveMutation = useSaveLabels(jobId ?? "");
```

- [ ] **Step 3: Gate playback on label mode**

When entering label mode, pause. When starting playback, exit label mode:
```typescript
const enterLabelMode = useCallback(() => {
  setIsPlaying(false);
  setLabelMode(true);
}, []);

const togglePlay = useCallback(() => {
  if (labelMode) {
    if (isDirty && !confirm("Discard unsaved label changes?")) return;
    setLabelMode(false);
    labelDispatch({ type: "clear" });
  }
  setIsPlaying((p) => !p);
}, [labelMode, isDirty, labelDispatch]);
```

- [ ] **Step 4: Wire LabelToolbar and LabelEditor into JSX**

Below the `SpectrogramViewport` and above `PlaybackControls`:
```typescript
{labelMode && (
  <LabelToolbar
    mode={labelSubMode}
    onModeChange={setLabelSubMode}
    selectedLabel={selectedLabel}
    onLabelChange={setSelectedLabel}
    onDelete={() => selectedId && labelDispatch({ type: "delete", row_id: selectedId })}
    onSave={() => {
      const items = labelState.edits.map((e) => ({
        action: e.action,
        row_id: e.row_id,
        start_sec: e.start_sec,
        end_sec: e.end_sec,
        new_start_sec: e.new_start_sec,
        new_end_sec: e.new_end_sec,
        label: e.label,
      }));
      saveMutation.mutate(items, {
        onSuccess: () => labelDispatch({ type: "clear" }),
      });
    }}
    onExtract={() => setExtractOpen(true)}
    onCancel={() => {
      if (isDirty && !confirm("Discard unsaved label changes?")) return;
      setLabelMode(false);
      labelDispatch({ type: "clear" });
    }}
    isDirty={isDirty}
    isSaving={saveMutation.isPending}
    hasSelection={selectedId !== null}
  />
)}
```

Inside `SpectrogramViewport`, toggle between `DetectionOverlay` and `LabelEditor`:
```typescript
{!labelMode && showLabels && <DetectionOverlay ... />}
{labelMode && <LabelEditor mergedRows={mergedRows} ... />}
```

- [ ] **Step 5: Compute `labelModeEnabled` flag**

```typescript
const labelModeEnabled = !isPlaying && (zoomLevel === "1m" || zoomLevel === "5m");
```

Pass to `PlaybackControls`.

- [ ] **Step 6: Add ExtractDialog integration**

Import `ExtractDialog` from the classifier components:
```typescript
{extractOpen && (
  <ExtractDialog
    open={extractOpen}
    onOpenChange={setExtractOpen}
    selectedIds={new Set([jobId ?? ""])}
    extractMutation={extractMutation}
    onSuccess={() => setExtractOpen(false)}
  />
)}
```

Wire up the extract mutation from `useExtractLabeledSamples()`.

- [ ] **Step 7: Handle dirty state on navigation**

Add a `beforeunload` handler when dirty:
```typescript
useEffect(() => {
  if (!isDirty) return;
  const handler = (e: BeforeUnloadEvent) => {
    e.preventDefault();
    e.returnValue = "";
  };
  window.addEventListener("beforeunload", handler);
  return () => window.removeEventListener("beforeunload", handler);
}, [isDirty]);
```

- [ ] **Step 8: TypeScript check + visual test**

Run: `cd frontend && npx tsc --noEmit`
Start dev server and verify label mode toggle, toolbar display, and basic interactions work.

- [ ] **Step 9: Commit**

```bash
git add frontend/src/components/timeline/
git commit -m "Integrate label mode into TimelineViewer

Label button in playback controls, mode gating, toolbar,
editor component, extract dialog, and dirty state warning."
```

---

## Task 12: Playwright Tests

End-to-end tests for the labeling workflow.

**Files:**
- Create: `frontend/e2e/timeline-labeling.spec.ts`

- [ ] **Step 1: Create Playwright test file**

Create `frontend/e2e/timeline-labeling.spec.ts`:

```typescript
import { test, expect } from "@playwright/test";

test.describe("Timeline Labeling", () => {
  // Find a completed detection job with timeline
  test.beforeEach(async ({ page }) => {
    // Navigate to classifier tab and find a job with timeline
    await page.goto("http://localhost:5173/app/classifier/hydrophone");
  });

  test("label button disabled at wide zoom", async ({ page }) => {
    // Navigate to a timeline viewer
    // Verify Label button is disabled when zoom > 5m
    // Zoom to 5m, verify button becomes enabled
  });

  test("enter and exit label mode", async ({ page }) => {
    // Click Label button
    // Verify toolbar appears with Select/Add modes and radio buttons
    // Click Cancel
    // Verify toolbar disappears
  });

  test("add a label in add mode", async ({ page }) => {
    // Enter label mode
    // Switch to Add mode
    // Select humpback radio
    // Click on spectrogram
    // Verify a colored bar appears
    // Verify dirty indicator shown
  });

  test("save persists labels", async ({ page }) => {
    // Add a label
    // Click Save
    // Wait for save to complete
    // Reload page
    // Verify label still present
  });

  test("dirty state warning on cancel", async ({ page }) => {
    // Add a label
    // Click Cancel
    // Verify confirm dialog appears
  });
});
```

- [ ] **Step 2: Run tests to verify they execute**

Run: `cd frontend && npx playwright test e2e/timeline-labeling.spec.ts --headed`
Tests will skip gracefully if no completed detection job is available.

- [ ] **Step 3: Commit**

```bash
git add frontend/e2e/timeline-labeling.spec.ts
git commit -m "Add Playwright tests for timeline labeling workflow"
```

---

## Task 13: Verification and Documentation

Run all verification gates and update documentation.

**Files:**
- Modify: `CLAUDE.md` (update §9.1 capabilities, §8.5 storage, §3.7 frontend structure)
- Run: all verification commands

- [ ] **Step 1: Run backend verification**

```bash
uv run ruff format --check src/humpback/ tests/
uv run ruff check src/humpback/ tests/
uv run pyright src/humpback/ scripts/ tests/
uv run pytest tests/ -x -q
```

- [ ] **Step 2: Run frontend verification**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npx playwright test
```

- [ ] **Step 3: Fix any failures**

Address lint, type, or test failures before proceeding.

- [ ] **Step 4: Update CLAUDE.md**

In §9.1 (Implemented Capabilities), update the timeline viewer bullet:
```
- Timeline viewer: zoomable spectrogram with background tile pre-caching, interactive species labeling (add/move/delete/change-type with batch save), positive-only detection label bars with hover tooltips, audio-authoritative playhead sync, gapless double-buffered MP3 playback
```

In §8.5 (Storage Layout), remove the `/detections/{job_id}/detections.tsv` line.

In §3.7 (Frontend File Structure), add under `timeline/`:
```
│   ├── timeline/            (TimelineViewer, Minimap, SpectrogramViewport, TileCanvas, LabelEditor, LabelToolbar, etc.)
```

In §4.6 or relevant behavioral section, add:
```
- Timeline label editing enforces single-label-per-row (mutual exclusivity of humpback/orca/ship/background)
- TSV is no longer persisted; Parquet row store is the sole detection data representation
- Download endpoint generates TSV on-the-fly from Parquet
```

- [ ] **Step 5: Commit documentation**

```bash
git add CLAUDE.md
git commit -m "Update docs for timeline labeling and TSV removal"
```

- [ ] **Step 6: Final full verification**

```bash
uv run pytest tests/ -x -q
cd frontend && npx tsc --noEmit
```
