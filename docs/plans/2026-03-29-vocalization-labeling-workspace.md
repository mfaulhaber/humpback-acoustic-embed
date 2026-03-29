# Vocalization Labeling Workspace Implementation Plan

**Goal:** Replace the inference-only Vocalization/Labeling tab with a progressive pipeline page supporting manual labeling with inference assistance and one-click retrain.
**Spec:** `docs/specs/2026-03-29-vocalization-labeling-workspace-design.md`

---

### Task 1: Backend — Detection Embedding Generation Infrastructure

**Files:**
- Create: `src/humpback/models/detection_embedding_job.py`
- Create: `src/humpback/workers/detection_embedding_worker.py`
- Create: `alembic/versions/031_detection_embedding_jobs.py`
- Modify: `src/humpback/database.py` (import new model)
- Modify: `src/humpback/workers/queue.py` (add claim/complete/fail/recover functions)
- Modify: `src/humpback/workers/runner.py` (add to priority loop after detection, before vocalization training)

**Acceptance criteria:**
- [ ] `DetectionEmbeddingJob` model with fields: id, status, detection_job_id, progress_current, progress_total, error_message, created_at, updated_at
- [ ] Queue functions: claim, complete, fail following existing patterns
- [ ] Stale job recovery added to `recover_stale_jobs()`
- [ ] Worker task reads detection row store, slices audio windows, runs through classifier model embedding layer, writes `detection_embeddings.parquet`
- [ ] Worker reports progress (progress_current/progress_total) during generation
- [ ] Alembic migration creates table with SQLite-compatible ops
- [ ] Runner inserts the new job type after detection jobs, before vocalization training

**Tests needed:**
- Unit test for the embedding generation logic (mock model, synthetic row store)
- Integration test for the job lifecycle (queue -> claim -> complete)

---

### Task 2: Backend — Embedding Status and Generation API Endpoints

**Files:**
- Modify: `src/humpback/api/routers/classifier.py` (add two endpoints)
- Modify: `src/humpback/schemas/classifier.py` (add response schemas)

**Acceptance criteria:**
- [ ] `GET /classifier/detection-jobs/{id}/embedding-status` returns `{ has_embeddings: bool, count: int | null }`; checks `detection_embeddings.parquet` existence and row count
- [ ] `POST /classifier/detection-jobs/{id}/generate-embeddings` validates detection job exists and is complete, creates `DetectionEmbeddingJob`, returns 202 with job status
- [ ] `GET /classifier/detection-jobs/{id}/embedding-generation-status` returns current generation job status + progress for polling
- [ ] Returns 404 if detection job not found, 409 if generation already in progress

**Tests needed:**
- Integration test: check embedding status for job with/without embeddings
- Integration test: queue generation, verify job created, poll status

---

### Task 3: Backend — Vocalization Model Training Source Endpoint

**Files:**
- Modify: `src/humpback/api/routers/vocalization.py` (add endpoint)
- Modify: `src/humpback/schemas/vocalization.py` (add response schema)

**Acceptance criteria:**
- [ ] `GET /vocalization/models/{id}/training-source` returns `{ source_config, parameters }` from the training job that produced this model (joined via `vocalization_model_id`)
- [ ] Returns 404 if model not found, null source_config if training job not found (e.g., imported model)

**Tests needed:**
- Integration test: create training job + model, verify training-source returns correct config

---

### Task 4: Frontend — API Client, Types, and Query Hooks

**Files:**
- Modify: `frontend/src/api/types.ts` (add new interfaces)
- Modify: `frontend/src/api/client.ts` (add new fetch functions)
- Modify: `frontend/src/hooks/queries/useVocalization.ts` (add new hooks)

**Acceptance criteria:**
- [ ] Types: `EmbeddingStatus`, `EmbeddingGenerationJob`, `VocalizationTrainingSource`
- [ ] Client functions: `fetchEmbeddingStatus(jobId)`, `generateEmbeddings(jobId)`, `fetchEmbeddingGenerationStatus(jobId)`, `fetchVocModelTrainingSource(modelId)`
- [ ] Query hooks: `useEmbeddingStatus(jobId)`, `useGenerateEmbeddings()` (mutation), `useEmbeddingGenerationStatus(jobId)`, `useVocModelTrainingSource(modelId)`
- [ ] Hooks use appropriate staleTime and refetchInterval for polling states

**Tests needed:**
- Type correctness verified by `npx tsc --noEmit`

---

### Task 5: Frontend — DetectionJobPicker Component

**Files:**
- Create: `frontend/src/components/vocalization/DetectionJobPicker.tsx`

**Acceptance criteria:**
- [ ] Shows all completed detection jobs (no binary label or extraction filter)
- [ ] Hydrophone jobs display: `{hydrophone_name}    {start_timestamp UTC} — {end_timestamp UTC}`
- [ ] Local jobs display: `{audio_folder}    {N} windows` (window count from result_summary)
- [ ] Jobs grouped into Hydrophone and Local optgroups
- [ ] Emits `onSelect(detectionJobId)` callback
- [ ] Card style with "Source" header, consistent with page layout

**Tests needed:**
- TypeScript type-check passes

---

### Task 6: Frontend — EmbeddingStatusPanel Component

**Files:**
- Create: `frontend/src/components/vocalization/EmbeddingStatusPanel.tsx`

**Acceptance criteria:**
- [ ] Fetches embedding status when detection job is selected
- [ ] Shows collapsed `✓ Ready (N vectors)` when embeddings exist
- [ ] Shows expanded panel with `[Generate Embeddings]` button when missing
- [ ] Button triggers generation mutation; panel shows progress spinner with current/total while running
- [ ] Shows error + retry on failure
- [ ] Emits `onReady()` callback when embeddings are available
- [ ] Collapsible card style

**Tests needed:**
- TypeScript type-check passes

---

### Task 7: Frontend — InferencePanel Component

**Files:**
- Create: `frontend/src/components/vocalization/InferencePanel.tsx`

**Acceptance criteria:**
- [ ] Disabled until embeddings are ready (receives `embeddingsReady` prop)
- [ ] Auto-detects existing completed inference job for the selected detection job
- [ ] If found: auto-selects and collapses, shows model name + scored count
- [ ] If not found: shows model selector (defaults to active) + `[Run Inference]` button
- [ ] While running: shows spinner, polls status
- [ ] Offers `[Rescore]` if active model differs from inference job's model
- [ ] Emits `onInferenceReady(inferenceJobId)` callback
- [ ] Collapsible card style

**Tests needed:**
- TypeScript type-check passes

---

### Task 8: Frontend — LabelingWorkspace Component

**Files:**
- Create: `frontend/src/components/vocalization/LabelingWorkspace.tsx`

**Acceptance criteria:**
- [ ] Receives inference job ID; fetches paginated results via existing `useVocClassifierInferenceResults`
- [ ] Each row shows: spectrogram thumbnail, audio play button, inference score, UTC time range
- [ ] Binary label displayed as read-only badge (humpback/orca/ship/background/—); sourced from detection row store binary label if present
- [ ] Vocalization labels shown as removable chips; `[+ add]` dropdown populated from vocabulary
- [ ] Add label calls `POST /labeling/vocalization-labels/{detection_job_id}`; remove calls `DELETE /labeling/vocalization-labels/{label_id}`; optimistic UI updates
- [ ] Default sort: uncertainty (score closest to threshold midpoint); toggleable to score descending or chronological
- [ ] Pagination: 50 rows per page with page navigation
- [ ] Tracks label additions/removals count for retrain footer
- [ ] Emits `onLabelCountChange(count)` for the retrain footer

**Tests needed:**
- TypeScript type-check passes

---

### Task 9: Frontend — RetrainFooter Component

**Files:**
- Create: `frontend/src/components/vocalization/RetrainFooter.tsx`

**Acceptance criteria:**
- [ ] Sticky bar at bottom of page
- [ ] Shows label count: `N new labels since last training` or `N labels (not yet used in training)`
- [ ] `[Retrain Model]` button fetches active model's training source, extends source_config with current detection job, creates training job
- [ ] Shows inline status while training runs (queued/running/complete)
- [ ] On completion, shows `[Activate new model]` button
- [ ] Disabled when no labels exist or no active model

**Tests needed:**
- TypeScript type-check passes

---

### Task 10: Frontend — Rewire VocalizationLabelingTab

**Files:**
- Modify: `frontend/src/components/vocalization/VocalizationLabelingTab.tsx`

**Acceptance criteria:**
- [ ] Orchestrates the full pipeline: DetectionJobPicker -> EmbeddingStatusPanel -> InferencePanel -> LabelingWorkspace + RetrainFooter
- [ ] State flows: selectedJobId drives EmbeddingStatusPanel; embeddingsReady drives InferencePanel; inferenceJobId drives LabelingWorkspace
- [ ] Changing the selected detection job resets downstream state
- [ ] Old VocalizationInferenceForm and VocalizationResultsBrowser imports removed (files kept for reference if needed)

**Tests needed:**
- TypeScript type-check passes
- Playwright test: select a detection job, verify pipeline panels render, verify label add/remove interaction

---

### Task 11: Backend and Frontend Tests

**Files:**
- Create: `tests/integration/test_detection_embedding_api.py`
- Create: `frontend/e2e/vocalization-labeling.spec.ts`
- Modify: `tests/integration/test_vocalization_api.py` (add training-source endpoint test)

**Acceptance criteria:**
- [ ] Backend integration tests for embedding-status, generate-embeddings, embedding-generation-status endpoints
- [ ] Backend integration test for vocalization model training-source endpoint
- [ ] Playwright test: navigate to Vocalization/Labeling, verify detection job picker renders with correct display format
- [ ] Playwright test: verify labeling workspace interaction (add/remove vocalization label)
- [ ] All existing tests still pass

**Tests needed:**
- This task IS the tests

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/api/routers/classifier.py src/humpback/api/routers/vocalization.py src/humpback/schemas/classifier.py src/humpback/schemas/vocalization.py src/humpback/models/detection_embedding_job.py src/humpback/workers/detection_embedding_worker.py src/humpback/workers/queue.py src/humpback/workers/runner.py src/humpback/database.py`
2. `uv run ruff check src/humpback/ tests/`
3. `uv run pyright src/humpback/ tests/`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test e2e/vocalization-labeling.spec.ts`
