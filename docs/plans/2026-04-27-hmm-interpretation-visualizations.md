# HMM Interpretation Visualizations Implementation Plan

**Goal:** Add PCA/UMAP overlay scatter, state-to-label distribution bar chart, and state exemplar gallery to completed HMM sequence jobs — the interpretation layer that makes latent states explorable.
**Spec:** [docs/specs/2026-04-27-sequence-models-design.md](../specs/2026-04-27-sequence-models-design.md) — PR 3

---

### Task 1: Storage helpers for interpretation artifacts

**Files:**
- Modify: `src/humpback/storage.py`

**Acceptance criteria:**
- [ ] `hmm_sequence_overlay_path(storage_root, job_id)` → `hmm_sequences/{job_id}/pca_overlay.parquet`
- [ ] `hmm_sequence_label_distribution_path(storage_root, job_id)` → `hmm_sequences/{job_id}/label_distribution.json`
- [ ] `hmm_sequence_exemplars_dir(storage_root, job_id)` → `hmm_sequences/{job_id}/exemplars/`
- [ ] `hmm_sequence_exemplars_path(storage_root, job_id)` → `hmm_sequences/{job_id}/exemplars/exemplars.json`
- [ ] All helpers follow the existing `hmm_sequence_*` naming convention

**Tests needed:**
- Path helpers are exercised by subsequent tasks; no dedicated tests

---

### Task 2: PCA/UMAP overlay computation module

**Files:**
- Create: `src/humpback/sequence_models/overlay.py`

**Acceptance criteria:**
- [ ] `compute_overlay()` function that accepts: PCA model (joblib), raw embedding sequences (list of ndarrays), Viterbi state arrays, metadata columns (merged_span_id, window_index_in_span, start_time_sec, end_time_sec), and optional UMAP parameters (n_neighbors, min_dist, random_state)
- [ ] Applies PCA model to full-dimensional embeddings and takes the first 2 components → `pca_x`, `pca_y`
- [ ] Computes UMAP 2D reduction on PCA-reduced embeddings (using the already-fitted PCA at configured `pca_dims`, then UMAP on that) → `umap_x`, `umap_y`
- [ ] Returns a PyArrow Table with columns: `merged_span_id` (int32), `window_index_in_span` (int32), `start_time_sec` (float64), `end_time_sec` (float64), `pca_x` (float32), `pca_y` (float32), `umap_x` (float32), `umap_y` (float32), `viterbi_state` (int16), `max_state_probability` (float32)
- [ ] Deterministic given a fixed random_state for UMAP
- [ ] L2-normalization of raw embeddings uses the same `l2_normalize` flag as the HMM job
- [ ] Sort order: `(merged_span_id, window_index_in_span)`

**Tests needed:**
- `tests/sequence_models/test_overlay.py`: shape correctness for PCA and UMAP columns; Viterbi state assignment matches input; determinism on fixed seed; handles single-span and multi-span inputs

---

### Task 3: State-to-label distribution computation module

**Files:**
- Create: `src/humpback/sequence_models/label_distribution.py`

**Acceptance criteria:**
- [ ] `compute_label_distribution()` function that accepts: states rows (list of dicts or PyArrow table with start_time_sec, end_time_sec, viterbi_state), detection windows (list of dicts with row_id, start_utc, end_utc per the detection row store), labels (list of VocalizationLabel-like dicts with row_id, label)
- [ ] Join semantics per spec §5.4: an HMM window inherits labels from any detection window whose 5s extent contains the HMM window's center timestamp (`(start + end) / 2`), scoped to the same audio source
- [ ] Multi-label: a single HMM window can inherit multiple labels from the same or different detection windows
- [ ] Unlabeled HMM windows (no matching detection window has labels) contribute to an `"unlabeled"` bucket per state
- [ ] Returns a dict: `{ states: { "0": { "label_a": count, "label_b": count, "unlabeled": count }, ... }, n_states: int, total_windows: int }`
- [ ] Pure function — no database access; callers provide the label data

**Tests needed:**
- `tests/sequence_models/test_label_distribution.py`: single label inheritance; multi-label window; unlabeled windows; no detection window overlap; center-time boundary precision; empty inputs

---

### Task 4: State exemplar selection module

**Files:**
- Create: `src/humpback/sequence_models/exemplars.py`

**Acceptance criteria:**
- [ ] `select_exemplars()` function that accepts: PCA-reduced embeddings (concatenated ndarray), Viterbi states (ndarray), metadata rows (merged_span_id, window_index_in_span, audio_file_id, start_time_sec, end_time_sec, max_state_probability), n_exemplars_per_type (default 3)
- [ ] For each state, selects three exemplar categories:
  - `high_confidence`: windows with highest `max_state_probability` for that state
  - `mean_nearest`: windows closest to the PCA centroid of that state (L2 distance)
  - `boundary`: windows with lowest `max_state_probability` for that state (state-boundary ambiguity)
- [ ] Each exemplar record contains: `merged_span_id`, `window_index_in_span`, `audio_file_id`, `start_time_sec`, `end_time_sec`, `max_state_probability`, `exemplar_type` ("high_confidence" | "mean_nearest" | "boundary")
- [ ] Returns a dict: `{ states: { "0": [exemplar_records...], "1": [...], ... }, n_states: int }`
- [ ] Handles states with fewer windows than `n_exemplars_per_type` gracefully (returns what's available)

**Tests needed:**
- `tests/sequence_models/test_exemplars.py`: correct selection for each category; tie-breaking consistency; state with few windows; determinism

---

### Task 5: Interpretation generation service and worker extension

**Files:**
- Modify: `src/humpback/services/hmm_sequence_service.py`
- Modify: `src/humpback/workers/hmm_sequence_worker.py`

**Acceptance criteria:**
- [ ] `generate_interpretations()` service function that, given a completed HMM job ID and a DB session: loads embeddings.parquet, pca_model.joblib, states.parquet; calls `compute_overlay()` and `select_exemplars()`; atomically writes `pca_overlay.parquet` and `exemplars/exemplars.json`
- [ ] `generate_label_distribution()` service function that, given a completed HMM job ID and a DB session: traces HMMSequenceJob → ContinuousEmbeddingJob → RegionDetectionJob to get the detection_job_id; reads the detection row store parquet to get `(row_id → start_utc, end_utc)` per detection window; queries `vocalization_labels` from the DB for that detection_job_id; reads states.parquet for HMM window timestamps and states; calls `compute_label_distribution()`; atomically writes `label_distribution.json`
- [ ] Worker extension: after successful HMM training (after existing artifact writes, before marking job complete), call `generate_interpretations()` to produce overlay and exemplars automatically — label distribution is NOT generated by the worker (depends on external labeling state)
- [ ] Both functions are idempotent (overwrite existing artifacts safely via atomic rename)
- [ ] Overlay + exemplars generation failure does not fail the HMM job — log a warning and continue to complete

**Tests needed:**
- `tests/services/test_hmm_sequence_service.py`: extend with tests for `generate_interpretations()` and `generate_label_distribution()` — verify artifact files produced, verify source-job validation
- `tests/workers/test_hmm_sequence_worker.py`: extend to verify overlay + exemplars artifacts appear after job completion

---

### Task 6: Pydantic schemas and API endpoints

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/api/routers/sequence_models.py`

**Acceptance criteria:**
- [ ] `OverlayPoint` schema: `merged_span_id`, `window_index_in_span`, `start_time_sec`, `end_time_sec`, `pca_x`, `pca_y`, `umap_x`, `umap_y`, `viterbi_state`, `max_state_probability`
- [ ] `OverlayResponse` schema: `total: int`, `items: list[OverlayPoint]`
- [ ] `LabelDistributionResponse` schema: `n_states: int`, `total_windows: int`, `states: dict[str, dict[str, int]]`
- [ ] `ExemplarRecord` schema: `merged_span_id`, `window_index_in_span`, `audio_file_id`, `start_time_sec`, `end_time_sec`, `max_state_probability`, `exemplar_type`
- [ ] `ExemplarsResponse` schema: `n_states: int`, `states: dict[str, list[ExemplarRecord]]`
- [ ] `GET /sequence-models/hmm-sequences/{id}/overlay` — reads `pca_overlay.parquet`, returns paginated `OverlayResponse`; 404 if job or artifact not found
- [ ] `GET /sequence-models/hmm-sequences/{id}/label-distribution` — reads `label_distribution.json` if cached, or computes on-demand via `generate_label_distribution()`; returns `LabelDistributionResponse`; 404 if job not found
- [ ] `GET /sequence-models/hmm-sequences/{id}/exemplars` — reads `exemplars/exemplars.json`; returns `ExemplarsResponse`; 404 if job or artifact not found
- [ ] `POST /sequence-models/hmm-sequences/{id}/generate-interpretations` — regenerates all three artifacts for an existing completed job; returns 200 with summary; 400 if job not complete; 404 if not found

**Tests needed:**
- Endpoint tests covered via integration tests in Task 5 service tests and E2E tests in Task 8

---

### Task 7: Frontend interpretation visualizations

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`

**Acceptance criteria:**
- [ ] TanStack Query hooks: `useHMMOverlay(jobId, enabled)`, `useHMMLabelDistribution(jobId, enabled)`, `useHMMExemplars(jobId, enabled)`, `useGenerateInterpretations()`
- [ ] **PCA/UMAP Scatter** component: Plotly scattergl with `pca_x`/`pca_y` as default axes, toggle to `umap_x`/`umap_y`, points colored by `viterbi_state` using existing `STATE_COLORS`, hover showing state + probability + time; placed after the State Timeline card in the detail page
- [ ] **Label Distribution** component: per-state stacked bar chart using Plotly; each bar is one state, stacked segments are label types (plus "unlabeled"); x-axis = state, y-axis = count; placed after transition matrix card
- [ ] A "Generate Label Distribution" button that triggers `POST /generate-interpretations` re-computation — shown when label_distribution.json doesn't exist or user wants to refresh
- [ ] **Exemplar Gallery** component: per-state collapsible section showing exemplar cards grouped by type (high_confidence, mean_nearest, boundary); each card shows audio_file_id, time range, max_state_probability; placed after dwell histograms card
- [ ] All new chart containers have `data-testid` attributes: `hmm-pca-umap-scatter`, `hmm-label-distribution`, `hmm-exemplar-gallery`
- [ ] New components only render when job status is `complete` and data is loaded (match existing conditional pattern)
- [ ] PCA/UMAP axis toggle defaults to PCA view

**Tests needed:**
- Playwright E2E tests in Task 8

---

### Task 8: Backend unit tests

**Files:**
- Create: `tests/sequence_models/test_overlay.py`
- Create: `tests/sequence_models/test_label_distribution.py`
- Create: `tests/sequence_models/test_exemplars.py`

**Acceptance criteria:**
- [ ] `test_overlay.py`: PCA projection shape (N, 2); UMAP projection shape (N, 2); Viterbi states match input; determinism on fixed seed; multi-span input; single-window edge case
- [ ] `test_label_distribution.py`: single detection window with one label → correct state-label count; multi-label window → both labels counted; unlabeled HMM windows → "unlabeled" bucket; center-time boundary precision (window at exact edge); no detection windows → all unlabeled; empty states → empty result
- [ ] `test_exemplars.py`: high_confidence picks have highest max_state_probability per state; mean_nearest picks are closest to PCA centroid; boundary picks have lowest max_state_probability; n_exemplars_per_type respected; state with 1 window returns that window for all categories; deterministic output
- [ ] All tests use synthetic data (no real model files or audio)

**Tests needed:**
- These ARE the tests

---

### Task 9: Playwright E2E tests and documentation

**Files:**
- Modify: `frontend/e2e/sequence-models/hmm-sequence.spec.ts`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `CLAUDE.md` (§9.1 only)

**Acceptance criteria:**
- [ ] E2E test: detail page for complete job renders `hmm-pca-umap-scatter` container
- [ ] E2E test: detail page for complete job renders `hmm-label-distribution` container
- [ ] E2E test: detail page for complete job renders `hmm-exemplar-gallery` container
- [ ] E2E mocks: add route handlers for `/overlay`, `/label-distribution`, `/exemplars` endpoints with synthetic response data
- [ ] `sequence-models-api.md`: document all four new endpoints (GET overlay, GET label-distribution, GET exemplars, POST generate-interpretations) with request/response schemas
- [ ] `storage-layout.md`: add `pca_overlay.parquet`, `label_distribution.json`, `exemplars/` under `hmm_sequences/{job_id}/`
- [ ] `CLAUDE.md` §9.1: add interpretation visualizations to Sequence Models capability description

**Tests needed:**
- These ARE the tests

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/sequence_models/overlay.py src/humpback/sequence_models/label_distribution.py src/humpback/sequence_models/exemplars.py src/humpback/storage.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/services/hmm_sequence_service.py src/humpback/workers/hmm_sequence_worker.py`
2. `uv run ruff check src/humpback/sequence_models/overlay.py src/humpback/sequence_models/label_distribution.py src/humpback/sequence_models/exemplars.py src/humpback/storage.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/services/hmm_sequence_service.py src/humpback/workers/hmm_sequence_worker.py`
3. `uv run pyright src/humpback/sequence_models/overlay.py src/humpback/sequence_models/label_distribution.py src/humpback/sequence_models/exemplars.py src/humpback/storage.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/services/hmm_sequence_service.py src/humpback/workers/hmm_sequence_worker.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test e2e/sequence-models/hmm-sequence.spec.ts`
