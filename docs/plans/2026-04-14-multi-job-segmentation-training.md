# Multi-Job Segmentation Training Implementation Plan

**Goal:** Enable training segmentation models from boundary corrections across multiple segmentation jobs, consolidating on a single canonical training path.
**Spec:** `docs/specs/2026-04-14-multi-job-segmentation-training-design.md`

---

### Task 1: Alembic migration — drop `event_segmentation_training_jobs` table

**Files:**
- Create: `alembic/versions/047_drop_event_segmentation_training_jobs.py`

**Acceptance criteria:**
- [x] Migration drops `event_segmentation_training_jobs` table
- [x] Uses `op.drop_table()` (batch not needed for full table drop)
- [x] Downgrade recreates the table with original schema
- [x] `uv run alembic upgrade head` succeeds against the production DB

**Tests needed:**
- Migration unit test verifying upgrade drops table and downgrade recreates it

---

### Task 2: Remove feedback training backend code

**Files:**
- Modify: `src/humpback/models/feedback_training.py` — remove `EventSegmentationTrainingJob` class
- Modify: `src/humpback/models/__init__.py` — remove `EventSegmentationTrainingJob` from imports/exports
- Delete: `src/humpback/workers/event_segmentation_feedback_worker.py`
- Modify: `src/humpback/workers/runner.py` — remove the feedback training job dispatch branch (~lines 340-358)
- Modify: `src/humpback/workers/queue.py` — remove `claim_segmentation_feedback_training_job()` function and stale-job recovery for `EventSegmentationTrainingJob` (~lines 317-335, 845-859)
- Modify: `src/humpback/services/call_parsing.py` — remove `create_segmentation_feedback_training_job()`, `list_segmentation_feedback_training_jobs()`, `get_segmentation_feedback_training_job()`, `delete_segmentation_feedback_training_job()`
- Modify: `src/humpback/api/routers/call_parsing.py` — remove 4 feedback training endpoints (POST/GET/GET-by-id/DELETE at `/segmentation-feedback-training-jobs`)
- Modify: `src/humpback/schemas/call_parsing.py` — remove `CreateSegmentationFeedbackTrainingJobRequest`, `SegmentationFeedbackTrainingJobResponse`

**Acceptance criteria:**
- [x] No references to `EventSegmentationTrainingJob` remain in `src/humpback/` (except the migration files)
- [x] No references to `event_segmentation_feedback_worker` remain
- [x] No references to `segmentation-feedback-training-jobs` endpoints remain
- [x] `EventBoundaryCorrection` and `EventClassifierTrainingJob` in `feedback_training.py` are untouched
- [x] Pyright passes on all modified files
- [x] Existing tests updated — removed segmentation feedback tests from `test_feedback_training_schemas_service.py`, `test_call_parsing_router.py`, `test_call_parsing_workers.py`

**Tests needed:**
- Verify removed endpoints return 404/405 (or just remove the tests that call them)
- Remaining feedback training tests (Pass 3 classifier) still pass

---

### Task 3: Refactor `create_dataset_from_corrections()` for multi-job support

**Files:**
- Modify: `src/humpback/schemas/call_parsing.py` — change `CreateDatasetFromCorrectionsRequest` to accept `segmentation_job_ids: list[str]` (min_length=1) instead of single `segmentation_job_id`
- Modify: `src/humpback/services/call_parsing.py` — refactor `create_dataset_from_corrections()` to loop over multiple job IDs, skip jobs with no corrections, raise if total samples is zero
- Modify: `src/humpback/api/routers/call_parsing.py` — update endpoint to pass new field name

**Acceptance criteria:**
- [x] Endpoint accepts `segmentation_job_ids` list
- [x] Each job validated as existing and complete
- [x] Jobs with zero corrections silently skipped
- [x] Raises 400 if no samples collected across all jobs
- [x] Each sample's `source_ref` records its originating job ID
- [x] Default name is `corrections-{N}jobs-{first_id[:8]}` for multi-job, `corrections-{id[:8]}` for single
- [x] Existing single-job usage still works (list of one)

**Tests needed:**
- Unit test: multi-job dataset creation with corrections from 2+ jobs produces samples with correct `source_ref` values
- Unit test: jobs with no corrections are skipped, others still collected
- Unit test: all jobs having no corrections raises ValueError
- Integration test: endpoint accepts list, returns correct sample count
- Update existing tests in `tests/integration/test_dataset_from_corrections.py` for new request shape

---

### Task 4: Add correction-counts endpoint

**Files:**
- Modify: `src/humpback/services/call_parsing.py` — add `list_segmentation_jobs_with_correction_counts()` service function
- Modify: `src/humpback/schemas/call_parsing.py` — add response schema with `EventSegmentationJobSummary` fields plus `correction_count: int`
- Modify: `src/humpback/api/routers/call_parsing.py` — add `GET /segmentation-jobs/with-correction-counts` endpoint

**Acceptance criteria:**
- [x] Returns all completed segmentation jobs with their correction count
- [x] Jobs with zero corrections return `correction_count: 0`
- [x] Uses a single query with LEFT JOIN subquery (no N+1)
- [x] Response includes all fields from `EventSegmentationJobSummary` plus `correction_count`, `hydrophone_id`, `start_timestamp`, `end_timestamp`

**Tests needed:**
- Unit test: service function returns correct counts for jobs with/without corrections
- Integration test: endpoint returns expected shape and counts

---

### Task 5: Add quick-retrain endpoint and service

**Files:**
- Modify: `src/humpback/services/call_parsing.py` — add `create_dataset_and_train()` convenience function
- Modify: `src/humpback/schemas/call_parsing.py` — add `QuickRetrainRequest` and `QuickRetrainResponse` schemas
- Modify: `src/humpback/api/routers/call_parsing.py` — add `POST /segmentation-training/quick-retrain` endpoint

**Acceptance criteria:**
- [x] Creates a single-job dataset from corrections and queues a `SegmentationTrainingJob` in one call
- [x] Returns dataset ID, training job ID, and sample count
- [x] Uses `create_dataset_from_corrections()` internally (single-element list)
- [x] Raises 400 if job has no corrections, 404 if job not found

**Tests needed:**
- Unit test: convenience function creates both dataset and training job
- Integration test: endpoint returns expected IDs and triggers training job creation

---

### Task 6: Remove frontend feedback training code

**Files:**
- Delete: `frontend/src/components/call-parsing/FeedbackTrainingJobTable.tsx`
- Modify: `frontend/src/api/types.ts` — remove `SegmentationFeedbackTrainingJob` type
- Modify: `frontend/src/api/client.ts` — remove feedback training API functions
- Modify: `frontend/src/hooks/queries/useCallParsing.ts` — remove `useSegmentationFeedbackTrainingJobs`, `useCreateSegmentationFeedbackTrainingJob`, `useDeleteSegmentationFeedbackTrainingJob`
- Modify: `frontend/src/components/call-parsing/SegmentTrainingPage.tsx` — remove `FeedbackTrainingJobTable` import and usage

**Acceptance criteria:**
- [ ] No references to `FeedbackTrainingJob` in frontend code
- [ ] No references to `segmentation-feedback-training-jobs` API path in frontend
- [ ] `SegmentTrainingPage` still renders (just the models table for now)
- [ ] TypeScript compiles cleanly

**Tests needed:**
- Verify `npx tsc --noEmit` passes

---

### Task 7: Add frontend API client, types, and hooks for new endpoints

**Files:**
- Modify: `frontend/src/api/types.ts` — add types for correction-count response, dataset creation request/response, quick-retrain request/response, training dataset summary
- Modify: `frontend/src/api/client.ts` — add API functions for correction-counts, create-dataset, list-datasets, create-training-job, quick-retrain
- Modify: `frontend/src/hooks/queries/useCallParsing.ts` — add `useSegmentationJobsWithCorrectionCounts()`, `useCreateSegmentationTrainingDataset()`, `useSegmentationTrainingDatasets()`, `useCreateSegmentationTrainingJob()`, `useQuickRetrain()`

**Acceptance criteria:**
- [ ] All new API functions match endpoint signatures
- [ ] Hooks use TanStack Query with appropriate cache keys and polling intervals
- [ ] TypeScript compiles cleanly

**Tests needed:**
- Verify `npx tsc --noEmit` passes

---

### Task 8: Rework SegmentTrainingPage with job picker and dataset table

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentTrainingPage.tsx` — build three-section layout
- Create: `frontend/src/components/call-parsing/SegmentationJobPicker.tsx` — paginated multi-select table with hydrophone short-id, date range, correction count; name/description inputs; "Create Training Dataset" button
- Create: `frontend/src/components/call-parsing/TrainingDatasetTable.tsx` — dataset list with name, sample count, source job count, created date; "Train" button per row

**Acceptance criteria:**
- [ ] Job picker shows paginated table of completed segmentation jobs with prev/next navigation
- [ ] Hydrophone column shows the short-id moniker consistent with the review page
- [ ] Correction count column populated from the new endpoint
- [ ] Checkbox multi-select enables "Create Training Dataset" button when at least one job selected
- [ ] Dataset table lists existing datasets with "Train" button that queues a `SegmentationTrainingJob`
- [ ] Models table (existing `SegmentModelTable`) remains at the bottom
- [ ] TypeScript compiles cleanly

**Tests needed:**
- Verify `npx tsc --noEmit` passes
- Manual browser test: select multiple jobs, create dataset, train from it

---

### Task 9: Update SegmentReviewWorkspace retrain button

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx` — replace `createFeedbackJob` with `useQuickRetrain` hook

**Acceptance criteria:**
- [ ] Retrain button calls quick-retrain endpoint instead of feedback training endpoint
- [ ] Same UX: one click, confirm dialog, toast on success/error, poll for completion
- [ ] Re-segment button still works (uses the model from the new training job)
- [ ] TypeScript compiles cleanly

**Tests needed:**
- Verify `npx tsc --noEmit` passes
- Manual browser test: retrain from review workspace, verify model appears in models table

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/models/feedback_training.py src/humpback/models/__init__.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/workers/runner.py src/humpback/workers/queue.py`
2. `uv run ruff check src/humpback/models/feedback_training.py src/humpback/models/__init__.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/workers/runner.py src/humpback/workers/queue.py`
3. `uv run pyright src/humpback/models/feedback_training.py src/humpback/models/__init__.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/workers/runner.py src/humpback/workers/queue.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
