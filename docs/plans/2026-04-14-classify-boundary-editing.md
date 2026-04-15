# Classify Boundary Editing Implementation Plan

**Goal:** Enable boundary editing (adjust, add, delete) in the classify review UI, with corrected boundaries flowing to both classification inference and classifier feedback training.
**Spec:** [docs/specs/2026-04-14-classify-boundary-editing-design.md](../specs/2026-04-14-classify-boundary-editing-design.md)

---

### Task 1: Shared `load_corrected_events()` utility

**Files:**
- Modify: `src/humpback/call_parsing/segmentation/extraction.py`
- Modify: `tests/unit/test_segmentation_feedback_worker.py`

**Acceptance criteria:**
- [ ] New async function `load_corrected_events(session, segmentation_job_id, storage_root)` added
- [ ] Reads `events.parquet` via `read_events()` from `call_parsing/storage.py`
- [ ] Queries `EventBoundaryCorrection` rows for the given segmentation job
- [ ] Calls existing `apply_corrections()` to merge
- [ ] Returns `list[Event]` (not the raw dicts from `apply_corrections()` — convert back to `Event` namedtuples)
- [ ] Returns original events unchanged when no corrections exist

**Tests needed:**
- Unit test: no corrections returns original events
- Unit test: corrections are applied (adjust, add, delete) and result is `list[Event]`
- Unit test: mixed correction types in a single call

---

### Task 2: Classification inference worker uses corrected events

**Files:**
- Modify: `src/humpback/workers/event_classification_worker.py`

**Acceptance criteria:**
- [ ] Replace `read_events(events_path)` call (line ~183) with `load_corrected_events(session, seg_job_id, storage_root)`
- [ ] Worker passes corrected events to `_run_classification_pipeline`
- [ ] Added events are classified; deleted events are excluded; adjusted events use corrected boundaries

**Tests needed:**
- Unit test: worker with boundary corrections produces typed_events reflecting corrected boundaries
- Unit test: worker with no corrections behaves identically to before

---

### Task 3: Classifier feedback worker uses corrected events

**Files:**
- Modify: `src/humpback/workers/event_classifier_feedback_worker.py`

**Acceptance criteria:**
- [ ] Worker calls `load_corrected_events()` for the upstream segmentation job when assembling training samples
- [ ] Corrected event boundaries are used for audio cropping instead of raw `typed_events.parquet` boundaries
- [ ] Boundary-deleted events are excluded before type resolution
- [ ] Added events with no type correction are excluded from training (no label = no training signal)
- [ ] Added events with a type correction are included in training

**Tests needed:**
- Unit test: feedback worker uses corrected boundaries for audio crop coordinates
- Unit test: deleted events are excluded from training data
- Unit test: added events with type corrections are included; without type corrections are excluded

---

### Task 4: Frontend — Passive boundary editing on selected event

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] `EventBarOverlay` receives a real `onAdjust` handler instead of `() => {}` (line ~601)
- [ ] Drag handles appear only on the currently selected event (pass prop or configure `EventBarOverlay`)
- [ ] Dragging a handle updates a `pendingBoundaryCorrections` map (new state, parallel to existing `pendingTypeCorrections`)
- [ ] Spectrogram event bar and detail panel reflect adjusted boundaries immediately
- [ ] Unsaved change counter includes both type and boundary pending corrections

**Tests needed:**
- Playwright: select event, drag handle adjusts boundary, detail panel updates
- Playwright: unsaved counter increments for boundary adjustment

---

### Task 5: Frontend — Right-click add event and Delete key

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`
- Modify: `frontend/src/components/call-parsing/EventBarOverlay.tsx` (if context menu support needed)

**Acceptance criteria:**
- [ ] Right-click on empty space in spectrogram/event bar area opens a context menu with "Add event"
- [ ] Clicking "Add event" creates a new event at the click position with ~1 second default duration
- [ ] New event is selected with drag handles visible; no type assigned (shows unlabeled in palette and detail panel)
- [ ] New event added to `pendingBoundaryCorrections` with `correction_type: "add"`
- [ ] Delete key on selected event creates a boundary correction with `correction_type: "delete"`
- [ ] Deleted event is removed from display; navigation advances to next event
- [ ] Reference implementation: `SegmentReviewWorkspace.tsx` lines 230-272

**Tests needed:**
- Playwright: right-click adds event, appears in event list, no type assigned
- Playwright: Delete key removes selected event from display

---

### Task 6: Frontend — Dual-save for type and boundary corrections

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Single Save button persists both correction types in parallel
- [ ] Type corrections: `POST /classification-jobs/{id}/corrections` (existing)
- [ ] Boundary corrections: `POST /segmentation-jobs/{segJobId}/corrections` (existing, uses hooks from `useCallParsing.ts`)
- [ ] `segJobId` resolved from the selected classification job's `event_segmentation_job_id`
- [ ] Save failure on one type rolls back UI state for that type only
- [ ] Cancel clears both pending maps

**Tests needed:**
- Playwright: save with mixed corrections persists both types
- Playwright: verify corrections survive page reload

---

### Task 7: Backend — Extend correction-counts endpoint for "update available"

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `src/humpback/schemas/call_parsing.py` (response schema)
- Modify: `src/humpback/services/call_parsing.py` (query logic)

**Acceptance criteria:**
- [ ] `GET /call-parsing/segmentation-jobs/with-correction-counts` response includes `has_new_corrections: bool` and `latest_correction_at: Optional[datetime]`
- [ ] `has_new_corrections` is true when corrections are newer than the latest training dataset that consumed samples from that job
- [ ] When no training dataset exists but corrections do exist, `has_new_corrections` is true

**Tests needed:**
- Integration test: endpoint returns `has_new_corrections=false` with no corrections
- Integration test: endpoint returns `has_new_corrections=true` after new correction
- Integration test: endpoint returns `has_new_corrections=false` after dataset extraction consumes corrections

---

### Task 8: Frontend — "Update available" indicator on segment training table

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentTrainingPage.tsx` (or the relevant dataset table component)

**Acceptance criteria:**
- [ ] Segment training dataset table shows a badge/indicator when `has_new_corrections` is true for related segmentation jobs
- [ ] Indicator text: "New corrections available" or similar
- [ ] Indicator links to or pre-populates the "create dataset from corrections" action

**Tests needed:**
- Playwright: indicator appears when corrections exist without a newer dataset
- Playwright: indicator disappears after dataset is created

---

### Task 9: ADR-054 — Read-time correction overlay pattern

**Files:**
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] ADR-054 appended to DECISIONS.md
- [ ] Documents: corrections applied as read-time overlays via shared `load_corrected_events()` utility
- [ ] Documents: parquet artifacts remain immutable inference snapshots
- [ ] Documents: all downstream consumers (inference + training) must use shared loader
- [ ] References ADR-053 as the foundation (correction table design)

**Tests needed:**
- None (documentation only)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/call_parsing/segmentation/extraction.py src/humpback/workers/event_classification_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py`
2. `uv run ruff check src/humpback/call_parsing/segmentation/extraction.py src/humpback/workers/event_classification_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py`
3. `uv run pyright src/humpback/call_parsing/segmentation/extraction.py src/humpback/workers/event_classification_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
