# Event-Level Vocalization Corrections Implementation Plan

**Goal:** Replace per-window and per-event correction tables with a unified
`vocalization_corrections` table, update both review UIs to use event-level corrections,
and add event boundary navigation to the Window Classify review.

**Spec:** [docs/specs/2026-04-23-event-level-vocalization-corrections-design.md](../specs/2026-04-23-event-level-vocalization-corrections-design.md)

---

### Task 1: Alembic Migration and Backend Model

Create the `VocalizationCorrection` SQLAlchemy model and Alembic migration that creates
the new table and drops the two retired tables.

**Files:**
- Create: `alembic/versions/054_vocalization_corrections.py`
- Modify: `src/humpback/models/call_parsing.py` — add `VocalizationCorrection`, remove `WindowScoreCorrection`
- Modify: `src/humpback/models/feedback_training.py` — remove `EventTypeCorrection` (lines 59-73)

**Acceptance criteria:**
- [ ] `VocalizationCorrection` model has columns: id (UUID PK), region_detection_job_id, start_sec (Float), end_sec (Float), type_name, correction_type, created_at, updated_at
- [ ] Unique constraint on `(region_detection_job_id, start_sec, end_sec, type_name)`
- [ ] Index on `region_detection_job_id`
- [ ] Migration creates `vocalization_corrections` table
- [ ] Migration drops `window_score_corrections` table
- [ ] Migration drops `event_type_corrections` table
- [ ] Migration uses `op.batch_alter_table()` for SQLite compatibility
- [ ] `WindowScoreCorrection` class removed from `models/call_parsing.py`
- [ ] `EventTypeCorrection` class removed from `models/feedback_training.py`
- [ ] `uv run alembic upgrade head` succeeds against the production DB

**Tests needed:**
- Migration upgrade and downgrade execute without error

---

### Task 2: Pydantic Schemas

Add request/response schemas for the new vocalization corrections API. Remove the old
window score and event type correction schemas.

**Files:**
- Modify: `src/humpback/schemas/call_parsing.py` — add new schemas, remove `WindowScoreCorrectionItem` (611-617), `WindowScoreCorrectionRequest` (620-623), `WindowScoreCorrectionResponse` (626-638), `TypeCorrection` (476-480), `TypeCorrectionRequest` (483-486), `TypeCorrectionResponse` (489-499)

**Acceptance criteria:**
- [ ] `VocalizationCorrectionItem` schema with fields: start_sec, end_sec, type_name, correction_type (validated as "add"/"remove")
- [ ] `VocalizationCorrectionRequest` schema wrapping region_detection_job_id + list of `VocalizationCorrectionItem`
- [ ] `VocalizationCorrectionResponse` schema mirroring the full DB row (id, region_detection_job_id, start_sec, end_sec, type_name, correction_type, created_at, updated_at)
- [ ] All six old correction schemas removed
- [ ] No remaining imports of removed schemas elsewhere in the codebase

**Tests needed:**
- Schema validation rejects invalid correction_type values

---

### Task 3: Service Layer

Add service functions for unified vocalization corrections. Remove old window score and
event type correction functions. Update the window classification job delete cascade.

**Files:**
- Modify: `src/humpback/services/call_parsing.py` — add `upsert_vocalization_corrections`, `list_vocalization_corrections`, `clear_vocalization_corrections`; remove `upsert_window_score_corrections` (1406-1447), `list_window_score_corrections` (1450-1459), `clear_window_score_corrections` (1462-1473), `upsert_type_corrections` (912-955), `list_type_corrections` (958-970), `clear_type_corrections` (973-987); update `delete_window_classification_job` cascade (1338-1341)

**Acceptance criteria:**
- [ ] `upsert_vocalization_corrections(session, region_detection_job_id, corrections)` validates detection job exists, upserts rows keyed by `(region_detection_job_id, start_sec, end_sec, type_name)`
- [ ] `list_vocalization_corrections(session, region_detection_job_id)` returns all corrections for a detection job, ordered by created_at
- [ ] `clear_vocalization_corrections(session, region_detection_job_id)` deletes all corrections for a detection job
- [ ] Deleting a region detection job cascades to delete its vocalization corrections
- [ ] Deleting a window classification job no longer cascades to window_score_corrections (table gone)
- [ ] All six old service functions removed
- [ ] No remaining calls to removed functions elsewhere in the codebase

**Tests needed:**
- Upsert creates new correction rows
- Upsert on same key updates correction_type
- List returns corrections filtered by detection job
- Clear removes all corrections for a detection job
- Upsert with invalid detection job raises CallParsingFKError

---

### Task 4: API Endpoints

Add new vocalization correction routes. Remove old window score and event type
correction routes.

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py` — add POST/GET/DELETE `/vocalization-corrections`; remove window score correction routes (1088-1119) and type correction routes (897-922)

**Acceptance criteria:**
- [ ] POST `/call-parsing/vocalization-corrections` accepts `VocalizationCorrectionRequest`, returns list of `VocalizationCorrectionResponse`
- [ ] GET `/call-parsing/vocalization-corrections?region_detection_job_id={id}` returns list of corrections
- [ ] DELETE `/call-parsing/vocalization-corrections?region_detection_job_id={id}` returns 204
- [ ] POST returns 404 if detection job not found, 409 if state error
- [ ] All six old correction routes removed
- [ ] No remaining references to removed route handlers

**Tests needed:**
- POST creates corrections and returns them
- POST upserts existing correction
- GET returns corrections filtered by detection job
- DELETE clears corrections
- POST with missing detection job returns 404

---

### Task 5: Frontend API, Types, and Hooks

Add frontend API client functions, TypeScript types, and TanStack Query hooks for
vocalization corrections. Remove old correction code.

**Files:**
- Modify: `frontend/src/api/types.ts` — add `VocalizationCorrection`, `VocalizationCorrectionItem` types; remove `WindowScoreCorrection`, `WindowScoreCorrectionItem`, `TypeCorrectionItem`, `TypeCorrectionResponse`
- Modify: `frontend/src/api/client.ts` — add `upsertVocalizationCorrections`, `fetchVocalizationCorrections`, `clearVocalizationCorrections`; remove old correction functions (lines ~1070-1182)
- Modify: `frontend/src/hooks/queries/useCallParsing.ts` — add `useVocalizationCorrections`, `useUpsertVocalizationCorrections`, `useClearVocalizationCorrections`; remove `useWindowScoreCorrections`, `useUpsertWindowScoreCorrections`, `useClearWindowScoreCorrections`, `useTypeCorrections`, `useUpsertTypeCorrections`, `useClearTypeCorrections`

**Acceptance criteria:**
- [ ] `VocalizationCorrection` type mirrors the API response shape
- [ ] `VocalizationCorrectionItem` type for request payloads with start_sec, end_sec, type_name, correction_type
- [ ] `fetchVocalizationCorrections(regionDetectionJobId)` calls GET endpoint
- [ ] `upsertVocalizationCorrections(regionDetectionJobId, corrections)` calls POST endpoint
- [ ] `clearVocalizationCorrections(regionDetectionJobId)` calls DELETE endpoint
- [ ] Query hooks use `["vocalizationCorrections", regionDetectionJobId]` query key
- [ ] Upsert mutation invalidates the correction query on success
- [ ] All old correction types, functions, and hooks removed
- [ ] `npx tsc --noEmit` passes

**Tests needed:**
- TypeScript compilation validates type correctness

---

### Task 6: Window Classify Review — Event Overlay and Corrections

Overhaul the `WindowClassifyReviewWorkspace` to show Pass 2 event boundaries, support
event-level corrections, add region/event navigation, and bind playback to the selected
event. Remove per-window correction interaction.

**Files:**
- Modify: `frontend/src/components/call-parsing/WindowClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Prerequisite gate: if no completed Pass 2 event segmentation job exists for the detection job, show a message instead of the review workspace
- [ ] Fetch Pass 2 events via existing `fetchSegmentationJobEvents` for the relevant segmentation job
- [ ] Event boundaries rendered as overlays on the per-window score timeline
- [ ] Exactly one event is always selected (highlighted); first event of first region on initial load
- [ ] Selected event shows a correction panel with type badges displaying max score from overlapping windows
- [ ] Badge toggle creates add/remove corrections in pending local state
- [ ] Dirty indicator shown when pending corrections exist
- [ ] Save button posts to `/call-parsing/vocalization-corrections` with the event's start_sec/end_sec
- [ ] Cancel with confirmation when dirty
- [ ] `beforeunload` warning when dirty
- [ ] Keyboard: `→`/`D` next event, `←`/`A` previous event (within region)
- [ ] Keyboard: `Shift+→`/`Shift+D` next region (first event), `Shift+←`/`Shift+A` previous region (last event)
- [ ] `Space` plays/pauses audio for the selected event's time range
- [ ] Timeline viewport centers on the selected event's region
- [ ] Per-window score detail preserved underneath event overlays
- [ ] Old per-window badge toggle interaction removed
- [ ] Old `useWindowScoreCorrections`/`useUpsertWindowScoreCorrections` imports removed

**Tests needed:**
- Playwright: prerequisite gate displays when no segmentation job exists
- Playwright: event boundaries visible on the timeline
- Playwright: keyboard navigation cycles through events and regions
- Playwright: correction badge toggle and save round-trips to API

---

### Task 7: Classify Review — Switch to Unified Corrections

Update the `ClassifyReviewWorkspace` to read/write vocalization corrections instead of
event type corrections.

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Correction read uses `useVocalizationCorrections(regionDetectionJobId)` instead of `useTypeCorrections(jobId)`
- [ ] Correction save uses `useUpsertVocalizationCorrections` with the event's `(start_sec, end_sec)` as time anchor
- [ ] Corrections made in Window Classify review appear in Classify review (shared query key)
- [ ] Resolve the region_detection_job_id from the event classification job's upstream chain (event classification → segmentation → region detection)
- [ ] Old type correction hook imports removed
- [ ] No navigation or layout changes
- [ ] `npx tsc --noEmit` passes

**Tests needed:**
- Playwright: corrections saved in Classify review appear when re-opening the workspace
- Playwright: corrections saved in Window Classify review are visible in Classify review

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/models/call_parsing.py src/humpback/models/feedback_training.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py`
2. `uv run ruff check src/humpback/models/call_parsing.py src/humpback/models/feedback_training.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py`
3. `uv run pyright src/humpback/models/call_parsing.py src/humpback/models/feedback_training.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
