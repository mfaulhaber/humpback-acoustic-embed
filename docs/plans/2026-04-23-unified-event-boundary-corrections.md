# Unified Event Boundary Corrections Implementation Plan

**Goal:** Reanchor event boundary corrections from `event_segmentation_job_id` to `region_detection_job_id`, unifying the table across Pass 2, Pass 3, and Window Classify review surfaces, and adding full boundary editing to Window Classify review.

**Spec:** [docs/specs/2026-04-23-unified-event-boundary-corrections-design.md](../specs/2026-04-23-unified-event-boundary-corrections-design.md)

---

### Task 1: Alembic Migration — Drop and Recreate `event_boundary_corrections`

**Files:**
- Create: `alembic/versions/055_unified_event_boundary_corrections.py`

**Acceptance criteria:**
- [ ] Drop existing `event_boundary_corrections` table
- [ ] Create new `event_boundary_corrections` table with columns: `id` (PK), `region_detection_job_id` (NOT NULL), `region_id` (NOT NULL), `correction_type` (NOT NULL), `original_start_sec` (nullable), `original_end_sec` (nullable), `corrected_start_sec` (nullable), `corrected_end_sec` (nullable), `created_at`, `updated_at`
- [ ] Index on `region_detection_job_id` named `ix_event_boundary_corrections_detection_job`
- [ ] Uses `op.batch_alter_table()` for SQLite compatibility
- [ ] Migration runs successfully: `uv run alembic upgrade head`

**Tests needed:**
- Migration upgrade/downgrade round-trips cleanly

---

### Task 2: New SQLAlchemy Model and Pydantic Schemas

**Files:**
- Modify: `src/humpback/models/call_parsing.py` (add new `EventBoundaryCorrection` class after `VocalizationCorrection`)
- Modify: `src/humpback/models/feedback_training.py` (remove old `EventBoundaryCorrection` class)
- Modify: `src/humpback/schemas/call_parsing.py` (replace old `BoundaryCorrection`/`BoundaryCorrectionRequest`/`BoundaryCorrectionResponse` with new schemas using `region_detection_job_id` and explicit original/corrected time pairs)

**Acceptance criteria:**
- [ ] New `EventBoundaryCorrection` model in `models/call_parsing.py` with `region_detection_job_id`, `region_id`, `correction_type`, `original_start_sec`, `original_end_sec`, `corrected_start_sec`, `corrected_end_sec`
- [ ] Old model removed from `models/feedback_training.py`
- [ ] New Pydantic schemas: `EventBoundaryCorrectionItem` (input), `EventBoundaryCorrectionRequest` (batch with `region_detection_job_id`), `EventBoundaryCorrectionResponse` (output with `id`, timestamps)
- [ ] Old Pydantic schemas removed
- [ ] All imports updated across the codebase (grep for old model/schema references)

**Tests needed:**
- Schema validation: correction_type enum, nullable field rules per correction_type

---

### Task 3: Service Layer — Unified CRUD Functions

**Files:**
- Modify: `src/humpback/services/call_parsing.py` (replace old `upsert_boundary_corrections`/`list_boundary_corrections`/`clear_boundary_corrections` with new versions keyed on `region_detection_job_id`)

**Acceptance criteria:**
- [ ] `upsert_event_boundary_corrections(session, region_detection_job_id, corrections)` — validates detection job exists, upserts by `(region_detection_job_id, region_id, original_start_sec, original_end_sec)` for adjust/delete, by `(region_detection_job_id, region_id, corrected_start_sec, corrected_end_sec)` for add
- [ ] `list_event_boundary_corrections(session, region_detection_job_id)` — returns all corrections for a detection job ordered by `created_at`
- [ ] `clear_event_boundary_corrections(session, region_detection_job_id)` — deletes all corrections for a detection job
- [ ] Old functions removed

**Tests needed:**
- Upsert deduplication for each correction_type (adjust, add, delete)
- Upsert update-on-conflict behavior (e.g., adjust then re-adjust same event)
- List returns correct results
- Clear removes all corrections for a job

---

### Task 4: API Endpoints — Unified Routes

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py` (replace old `/segmentation-jobs/{job_id}/corrections` endpoints with new `/event-boundary-corrections` endpoints)

**Acceptance criteria:**
- [ ] `POST /call-parsing/event-boundary-corrections` — accepts `EventBoundaryCorrectionRequest`, returns upserted corrections
- [ ] `GET /call-parsing/event-boundary-corrections?region_detection_job_id={id}` — returns `list[EventBoundaryCorrectionResponse]`
- [ ] `DELETE /call-parsing/event-boundary-corrections?region_detection_job_id={id}` — clears corrections, returns 204
- [ ] Old three endpoints removed
- [ ] Endpoint pattern matches vocalization corrections endpoints

**Tests needed:**
- Integration tests for all three endpoints
- Validation error handling (missing region_detection_job_id, invalid correction_type)

---

### Task 5: Update Overlay Logic — `extraction.py`

**Files:**
- Modify: `src/humpback/call_parsing/segmentation/extraction.py` (update `load_corrected_events`, `apply_corrections`, `collect_corrected_samples`)

**Acceptance criteria:**
- [ ] `load_corrected_events(session, region_detection_job_id, segmentation_job_id, storage_root)` — queries new table by `region_detection_job_id`, reads parquet by `segmentation_job_id`
- [ ] `apply_corrections()` updated to match by `(region_id, original_start_sec, original_end_sec)` instead of `event_id`, and uses explicit original/corrected time pairs
- [ ] `collect_corrected_samples()` updated to query by `region_detection_job_id` instead of `event_segmentation_job_id`
- [ ] All three correction types (adjust, add, delete) work correctly with time-range matching

**Tests needed:**
- Overlay with each correction type applied individually
- Multiple corrections on the same region
- Mixed correction types in a single overlay pass
- `collect_corrected_samples` returns correct training data with new query key

---

### Task 6: Update Downstream Workers

**Files:**
- Modify: `src/humpback/workers/event_classification_worker.py` (pass `region_detection_job_id` to `load_corrected_events`)
- Modify: `src/humpback/workers/event_classifier_feedback_worker.py` (pass `region_detection_job_id` to `load_corrected_events`)

**Acceptance criteria:**
- [ ] Classification worker resolves `region_detection_job_id` from the job's parent chain and passes it to `load_corrected_events`
- [ ] Feedback training worker resolves `region_detection_job_id` and passes it to `load_corrected_events`
- [ ] Both workers continue to function correctly with corrected event boundaries

**Tests needed:**
- Existing worker tests updated to pass new argument

---

### Task 7: Frontend — New Hooks and API Client

**Files:**
- Modify: `frontend/src/api/types.ts` (replace old boundary correction types with new ones using `region_detection_job_id` and explicit time pairs)
- Modify: `frontend/src/api/client.ts` (replace old boundary correction API functions with new ones hitting unified endpoint)
- Modify: `frontend/src/hooks/queries/useCallParsing.ts` (replace old hooks with `useEventBoundaryCorrections`, `useUpsertEventBoundaryCorrections`, `useClearEventBoundaryCorrections`)

**Acceptance criteria:**
- [ ] New TypeScript types match the new Pydantic schemas
- [ ] API client functions target `/call-parsing/event-boundary-corrections` with `region_detection_job_id` query param
- [ ] New React Query hooks use consistent query keys
- [ ] Old types, client functions, and hooks removed

**Tests needed:**
- Type-check passes (`npx tsc --noEmit`)

---

### Task 8: Frontend — Update Pass 2 SegmentReviewWorkspace

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Resolves `region_detection_job_id` from the segmentation job's parent chain (fetch or prop)
- [ ] Uses new `useEventBoundaryCorrections(regionDetectionJobId)` hook instead of old `useBoundaryCorrections(segJobId)`
- [ ] Save handler calls `useUpsertEventBoundaryCorrections` with `region_detection_job_id` and corrections in new schema shape (explicit original/corrected time pairs)
- [ ] Pending correction state maps to new schema (original + corrected time pairs instead of event_id + single time pair)
- [ ] All editing operations (adjust, add, delete) work as before

**Tests needed:**
- Playwright: adjust, add, delete events in Segment Review still function correctly

---

### Task 9: Frontend — Update Pass 3 ClassifyReviewWorkspace

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Uses new `useEventBoundaryCorrections(regionDetectionJobId)` hook (already has `region_detection_job_id` available)
- [ ] Save handler sends boundary corrections via new unified endpoint
- [ ] Pending boundary correction state uses new schema shape
- [ ] Dual-correction display (boundary + vocalization) continues to work

**Tests needed:**
- Playwright: boundary editing in Classify Review still functions correctly

---

### Task 10: Frontend — Add Boundary Editing to WindowClassifyReviewWorkspace

**Files:**
- Modify: `frontend/src/components/call-parsing/WindowClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Import and render `EventBarOverlay` component with draggable handles
- [ ] Add pending boundary correction state (`Map<correctionKey, BoundaryCorrection>`)
- [ ] Implement adjust (drag handles), add (add mode click-drag), and delete (delete key) operations
- [ ] Save handler sends both boundary corrections and vocalization corrections in parallel POSTs (same pattern as ClassifyReviewWorkspace)
- [ ] Boundary corrections display merged with saved corrections from the unified table
- [ ] Event overlays reflect both pending and saved boundary corrections

**Tests needed:**
- Playwright: adjust, add, delete events in Window Classify Review
- Playwright: saving boundary + vocalization corrections together

---

### Task 11: Update Existing Tests and Dead Code Cleanup

**Files:**
- Modify: `tests/unit/test_feedback_training_schemas_service.py` (update or replace boundary correction tests)
- Modify: `tests/integration/test_call_parsing_router.py` (update endpoint tests)
- Modify: `tests/integration/test_dataset_from_corrections.py` (update for new query key)
- Modify: any other test files referencing old table/schemas

**Acceptance criteria:**
- [ ] All existing boundary correction tests updated to use new table, schemas, and endpoints
- [ ] No references to old `event_segmentation_job_id`-keyed boundary corrections remain in test code
- [ ] No dead imports or unused fixtures remain

**Tests needed:**
- Full test suite passes

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/ tests/`
2. `uv run ruff check src/humpback/ tests/`
3. `uv run pyright src/humpback/ tests/`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
