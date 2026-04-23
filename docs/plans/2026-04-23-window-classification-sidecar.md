# Window Classification Sidecar — Implementation Plan

**Goal:** Add a standalone sidecar enrichment that scores cached Perch embeddings from Pass 1 regions through existing multi-label vocalization classifiers, producing dense per-window probability vectors, with a review UI for inspection and correction.

**Spec:** `docs/specs/2026-04-23-window-classification-sidecar-design.md`

---

### Task 1: Pass 1 Embedding Cache

Modify the region detection worker to persist the full 1536-d Perch embedding vectors alongside the existing scalar trace. The embeddings are already computed in `_run_window_pipeline` (stored in `state.embeddings`) but discarded by `score_audio_windows`.

**Files:**
- Modify: `src/humpback/classifier/detector.py`
- Modify: `src/humpback/workers/region_detection_worker.py`
- Modify: `src/humpback/call_parsing/storage.py`
- Modify: `src/humpback/call_parsing/types.py`

**Acceptance criteria:**
- [ ] `score_audio_windows` returns both scalar records and embedding vectors
- [ ] `_load_file_trace` and `_load_hydrophone_trace` propagate embeddings through to the caller
- [ ] `run_region_detection_job` writes `embeddings.parquet` to the region job directory alongside `trace.parquet`
- [ ] `embeddings.parquet` has schema: `time_sec` (float64), `embedding` (list_[float32, 1536])
- [ ] Storage helpers added: `write_embeddings` / `read_embeddings` in `call_parsing/storage.py`
- [ ] Hydrophone chunked path accumulates embeddings per-chunk and deduplicates by `time_sec` (same as trace dedup)

**Tests needed:**
- Unit test that `score_audio_windows` returns embeddings array matching window count
- Unit test for `write_embeddings` / `read_embeddings` round-trip
- Integration test that a completed region detection job produces `embeddings.parquet` with correct shape

---

### Task 2: Database Model & Migration

Add the `WindowClassificationJob` and `WindowScoreCorrection` SQLAlchemy models plus an Alembic migration creating both tables.

**Files:**
- Modify: `src/humpback/models/call_parsing.py`
- Create: `alembic/versions/053_window_classification_tables.py`

**Acceptance criteria:**
- [ ] `WindowClassificationJob` model with all columns from spec (status, region_detection_job_id, vocalization_model_id, config_json, window_count, vocabulary_snapshot, error_message, started_at, completed_at)
- [ ] `WindowScoreCorrection` model with all columns from spec (window_classification_job_id, time_sec, region_id, correction_type, type_name)
- [ ] Migration creates both tables with `op.batch_alter_table()` for SQLite compatibility
- [ ] `uv run alembic upgrade head` succeeds against the production DB

**Tests needed:**
- Migration up/down test (standard pattern)

---

### Task 3: Pydantic Schemas

Add request/response schemas for window classification jobs, scores, and corrections.

**Files:**
- Modify: `src/humpback/schemas/call_parsing.py`

**Acceptance criteria:**
- [ ] `CreateWindowClassificationJobRequest` with `region_detection_job_id` and `vocalization_model_id` fields
- [ ] `WindowClassificationJobSummary` response model matching the DB columns
- [ ] `WindowScoreRow` model for the `/scores` endpoint response (time_sec, region_id, scores dict)
- [ ] `WindowScoreCorrectionRequest` / `WindowScoreCorrectionResponse` for batch upsert/list

**Tests needed:**
- Pydantic model validation tests for required fields and defaults

---

### Task 4: Service Layer

Add service functions for CRUD operations on window classification jobs and corrections, plus the scores read logic with filtering.

**Files:**
- Modify: `src/humpback/services/call_parsing.py`

**Acceptance criteria:**
- [ ] `create_window_classification_job` validates region job exists and is complete, validates vocalization model exists and is sklearn_perch_embedding family
- [ ] `list_window_classification_jobs`, `get_window_classification_job`, `delete_window_classification_job` (deletes artifacts + DB row)
- [ ] `read_window_scores` loads `window_scores.parquet`, applies optional filters (region_id, min_score, type_name), returns list of score dicts
- [ ] `upsert_window_score_corrections`, `list_window_score_corrections`, `clear_window_score_corrections`

**Tests needed:**
- Service creation validates region job status and model family
- Service deletion removes both DB row and storage directory
- Score reading with and without filters

---

### Task 5: Worker

Implement the window classification worker following the existing worker pattern.

**Files:**
- Create: `src/humpback/workers/window_classification_worker.py`
- Modify: `src/humpback/workers/queue.py`
- Modify: `src/humpback/workers/runner.py`

**Acceptance criteria:**
- [ ] `claim_window_classification_job` added to queue.py (same pattern as other claim functions)
- [ ] Worker registered in runner.py iteration loop
- [ ] Worker loads upstream `regions.parquet` + `embeddings.parquet` from region job directory
- [ ] Worker loads vocalization model via `load_vocalization_model()` from `vocalization_inference.py`
- [ ] Window inclusion: selects embeddings where `time_sec + 2.5` falls within `[padded_start_sec, padded_end_sec]`
- [ ] Scores selected embeddings through `score_embeddings()` and writes `window_scores.parquet` (wide format: time_sec, region_id, plus one float64 column per type)
- [ ] Writes atomically via temp-file rename
- [ ] On success: updates job with status=complete, window_count, vocabulary_snapshot
- [ ] On failure: cleans up partial artifacts, sets status=failed + error_message
- [ ] Storage helper `window_classification_job_dir` added to `call_parsing/storage.py`

**Tests needed:**
- Unit test: window inclusion logic selects correct windows per region
- Unit test: wide-format parquet is written with correct columns
- Integration test: end-to-end worker run with mock embeddings and model

---

### Task 6: API Endpoints

Add window classification endpoints to the call parsing router.

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`

**Acceptance criteria:**
- [ ] `POST /window-classification-jobs` creates job and returns summary
- [ ] `GET /window-classification-jobs` lists all jobs
- [ ] `GET /window-classification-jobs/{id}` returns single job
- [ ] `DELETE /window-classification-jobs/{id}` deletes job + artifacts
- [ ] `GET /window-classification-jobs/{id}/scores` reads parquet, supports `region_id`, `min_score`, `type_name` query params
- [ ] `POST /window-classification-jobs/{id}/corrections` batch upserts corrections
- [ ] `GET /window-classification-jobs/{id}/corrections` lists corrections
- [ ] `DELETE /window-classification-jobs/{id}/corrections` clears corrections

**Tests needed:**
- API integration tests for create, list, get, delete
- Score endpoint returns correct JSON shape with and without filters
- Correction CRUD round-trip

---

### Task 7: Frontend — API Client & Hooks

Add TypeScript types, API client functions, and TanStack Query hooks for window classification.

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useCallParsing.ts`

**Acceptance criteria:**
- [ ] TypeScript types for `WindowClassificationJob`, `WindowScoreRow`, `WindowScoreCorrection`
- [ ] API client functions: create/list/get/delete jobs, fetch scores, upsert/list/clear corrections
- [ ] TanStack Query hooks: `useWindowClassificationJobs`, `useCreateWindowClassificationJob`, `useDeleteWindowClassificationJob`, `useWindowScores`, `useWindowScoreCorrections`, `useUpsertWindowScoreCorrections`, `useClearWindowScoreCorrections`

**Tests needed:**
- TypeScript compiles without errors (`npx tsc --noEmit`)

---

### Task 8: Frontend — Jobs Tab (WindowClassifyPage)

Create the Window Classify page with Jobs and Review tabs. Implement the Jobs tab with job form and tables.

**Files:**
- Create: `frontend/src/components/call-parsing/WindowClassifyPage.tsx`
- Create: `frontend/src/components/call-parsing/WindowClassifyJobForm.tsx`
- Create: `frontend/src/components/call-parsing/WindowClassifyJobTable.tsx`
- Modify: `frontend/src/App.tsx` (add route)
- Modify: `frontend/src/components/layout/SideNav.tsx` (add nav entry)
- Modify: `frontend/src/components/layout/Breadcrumbs.tsx` (add breadcrumb)

**Acceptance criteria:**
- [ ] Route at `/app/call-parsing/window-classify` with Jobs/Review tabs (URL-synced via searchParams)
- [ ] "Window Classify" appears at the bottom of Call Parsing sub-nav
- [ ] Job form: completed region detection job selector + vocalization model selector (sklearn_perch_embedding only) + Create button
- [ ] Active Jobs table: source, region job, model, status (polls at 3s)
- [ ] Previous Jobs table: source, region job, model, window count, status, Review button
- [ ] Review button switches to Review tab with jobId in searchParams
- [ ] Breadcrumb entry for the new route

**Tests needed:**
- TypeScript compiles without errors

---

### Task 9: Frontend — Review Workspace

Implement the review workspace with spectrogram, confidence strip, window selection, and correction UI.

**Files:**
- Create: `frontend/src/components/call-parsing/WindowClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Job selector dropdown showing completed window classification jobs (labeled with source + short ID + window count)
- [ ] Region navigator: prev/next stepping through regions, shows "Region N of M" + time range
- [ ] Spectrogram via TimelineProvider + Spectrogram + RegionBoundaryMarkers from upstream region detection job
- [ ] Selected window highlighted with vertical band overlay on spectrogram
- [ ] Click on spectrogram selects the window at that position (snaps to nearest window by time_sec)
- [ ] Single ConfidenceStrip below spectrogram showing per-window scores
- [ ] Type selector dropdown: "All types (max)" or individual type name
- [ ] Threshold input: numeric value, defaults from model's per-type thresholds
- [ ] ZoomSelector component below strip
- [ ] Detail panel when window selected: time range, region ID, play button
- [ ] Vocalization badges in detail panel (labeling workspace pattern): above-threshold colored badges with scores, below-threshold dimmed, pending corrections with ring highlight, Plus popover for adding types, click to toggle removal
- [ ] Corrections accumulated locally as pending adds/removes, batch-saved with Save/Cancel toolbar
- [ ] Dirty indicator + beforeunload warning
- [ ] Keyboard shortcuts: `[`/`A` prev region, `]`/`D` next region, Space play, `+`/`-` zoom, arrows pan

**Tests needed:**
- TypeScript compiles without errors

---

### Task 10: Documentation Updates

Update project documentation to reflect the new capability.

**Files:**
- Modify: `CLAUDE.md` (§9.1 capabilities, §9.2 schema)
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/storage-layout.md`

**Acceptance criteria:**
- [ ] §9.1 mentions window classification sidecar
- [ ] §9.2 lists new tables and updates migration count
- [ ] Data model reference includes both new tables
- [ ] Storage layout includes `window_classification/<job_id>/` and `embeddings.parquet` in regions dir
- [ ] UI nav list updated in §9.1

**Tests needed:**
- None (documentation only)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/models/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/workers/window_classification_worker.py src/humpback/workers/queue.py src/humpback/workers/runner.py src/humpback/call_parsing/storage.py src/humpback/call_parsing/types.py src/humpback/classifier/detector.py src/humpback/workers/region_detection_worker.py`
2. `uv run ruff check src/humpback/models/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/workers/window_classification_worker.py src/humpback/workers/queue.py src/humpback/workers/runner.py src/humpback/call_parsing/storage.py src/humpback/call_parsing/types.py src/humpback/classifier/detector.py src/humpback/workers/region_detection_worker.py`
3. `uv run pyright src/humpback/models/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/workers/window_classification_worker.py src/humpback/workers/queue.py src/humpback/workers/runner.py src/humpback/call_parsing/storage.py src/humpback/call_parsing/types.py src/humpback/classifier/detector.py src/humpback/workers/region_detection_worker.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
