# HMM Sequence Jobs Implementation Plan

**Goal:** Add HMM training + Viterbi decode with minimum-viable visualization — the vertical-slice MVP where an operator can look at a real run and judge "is this learning anything?"
**Spec:** [docs/specs/2026-04-27-sequence-models-design.md](../specs/2026-04-27-sequence-models-design.md) — PR 2

---

### Task 1: Add `hmmlearn` dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (via `uv lock`)

**Acceptance criteria:**
- [ ] `hmmlearn>=0.3.2` added to the main dependency list in `pyproject.toml`
- [ ] `uv lock` succeeds without conflicts
- [ ] `uv sync --group dev --extra tf-macos` installs cleanly
- [ ] `uv run python -c "import hmmlearn; print(hmmlearn.__version__)"` succeeds

**Tests needed:**
- No dedicated tests — dependency verified by import in subsequent tasks

---

### Task 2: Alembic migration `058_hmm_sequence_jobs.py`

**Files:**
- Create: `alembic/versions/058_hmm_sequence_jobs.py`

**Acceptance criteria:**
- [ ] **Database backup**: read production DB path from `HUMPBACK_DATABASE_URL` in `.env`, copy to `<path>.YYYY-MM-DD-HH:mm.bak` (UTC), confirm backup exists with non-zero size — before any migration runs
- [ ] Migration creates `hmm_sequence_jobs` table with all columns from spec §5.2: `id`, `status`, `continuous_embedding_job_id` (FK → `continuous_embedding_jobs(id)`), `n_states`, `pca_dims`, `pca_whiten`, `l2_normalize`, `covariance_type`, `n_iter`, `random_seed`, `min_sequence_length_frames`, `tol`, `library`, `train_log_likelihood`, `n_train_sequences`, `n_train_frames`, `n_decoded_sequences`, `artifact_dir`, `error_message`, `created_at`, `updated_at`
- [ ] Uses `op.batch_alter_table()` pattern for SQLite compatibility in downgrade
- [ ] Index on `status` column
- [ ] `down_revision = "057"`
- [ ] `uv run alembic upgrade head` succeeds against the production database

**Tests needed:**
- Migration verified by running against production DB; existing test suite exercises the schema via SQLAlchemy model

---

### Task 3: SQLAlchemy ORM model `HMMSequenceJob`

**Files:**
- Modify: `src/humpback/models/sequence_models.py`

**Acceptance criteria:**
- [ ] `HMMSequenceJob` class with all columns matching migration 058
- [ ] Uses `UUIDMixin`, `TimestampMixin`, `Base` consistent with `ContinuousEmbeddingJob`
- [ ] Uses `JobStatus` enum from `humpback.models.processing`
- [ ] Exported from module `__all__`

**Tests needed:**
- Model is exercised by service and worker tests in subsequent tasks

---

### Task 4: Pydantic schemas for HMM sequence jobs

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`

**Acceptance criteria:**
- [ ] `HMMSequenceJobCreate` with fields: `continuous_embedding_job_id`, `n_states` (required), `pca_dims` (default 50), `pca_whiten` (default false), `l2_normalize` (default true), `covariance_type` (default "diag", validated to "diag"/"full"), `n_iter` (default 100), `random_seed` (default 42), `min_sequence_length_frames` (default 10), `tol` (default 1e-4)
- [ ] `HMMSequenceJobOut` exposing all DB row fields
- [ ] `HMMSequenceJobDetail` combining job + summary stats from `state_summary.json`
- [ ] `HMMStateSummary` for the per-state summary payload
- [ ] `TransitionMatrixResponse` for the transition matrix endpoint
- [ ] `DwellHistogramResponse` for the dwell-time histogram endpoint

**Tests needed:**
- Schema validation edge cases tested in service tests

---

### Task 5: Storage helpers for HMM sequence artifacts

**Files:**
- Modify: `src/humpback/storage.py`

**Acceptance criteria:**
- [ ] `hmm_sequence_dir(storage_root, job_id)` → `storage_root / "hmm_sequences" / job_id`
- [ ] `hmm_sequence_states_path(storage_root, job_id)` → `…/states.parquet`
- [ ] `hmm_sequence_pca_model_path(storage_root, job_id)` → `…/pca_model.joblib`
- [ ] `hmm_sequence_hmm_model_path(storage_root, job_id)` → `…/hmm_model.joblib`
- [ ] `hmm_sequence_transition_matrix_path(storage_root, job_id)` → `…/transition_matrix.npy`
- [ ] `hmm_sequence_summary_path(storage_root, job_id)` → `…/state_summary.json`
- [ ] `hmm_sequence_training_log_path(storage_root, job_id)` → `…/training_log.json`

**Tests needed:**
- Helpers are pure path functions; exercised by worker tests

---

### Task 6: `sequence_models/` package — PCA pipeline

**Files:**
- Create: `src/humpback/sequence_models/__init__.py`
- Create: `src/humpback/sequence_models/pca_pipeline.py`

**Acceptance criteria:**
- [ ] `fit_pca(sequences, pca_dims, whiten, random_state)` → fitted `PCA` object
- [ ] `transform_sequences(pca, sequences)` → list of PCA-transformed arrays
- [ ] Optional L2 normalization applied before PCA when `l2_normalize=True`
- [ ] Deterministic output given fixed `random_state`

**Tests needed:**
- `tests/sequence_models/test_pca_pipeline.py`: L2-norm + PCA fit/transform; with/without whiten; deterministic given seed; shape correctness

---

### Task 7: `sequence_models/` package — HMM trainer

**Files:**
- Create: `src/humpback/sequence_models/hmm_trainer.py`

**Acceptance criteria:**
- [ ] `fit_hmm(sequences, n_states, covariance_type, n_iter, tol, random_state)` → fitted `GaussianHMM`
- [ ] Accepts concatenated sequences with per-sequence lengths (the `hmmlearn` multi-sequence API)
- [ ] `min_sequence_length_frames` filtering applied before training but all sequences still available for decoding
- [ ] Returns training metadata: `train_log_likelihood`, `n_train_sequences`, `n_train_frames`
- [ ] Deterministic given fixed `random_state`

**Tests needed:**
- `tests/sequence_models/test_hmm_trainer.py`: recovers planted state structure on synthetic sequences (Hungarian-aligned state accuracy >= 0.85); transition matrix recovery within tolerance; min_sequence_length_frames filter; determinism on fixed seed

---

### Task 8: `sequence_models/` package — HMM decoder

**Files:**
- Create: `src/humpback/sequence_models/hmm_decoder.py`

**Acceptance criteria:**
- [ ] `decode_sequences(hmm_model, sequences)` → list of `(viterbi_states, posteriors)` per sequence
- [ ] `max_state_probability` derived as argmax of posterior per window
- [ ] Handles sequences that were below `min_sequence_length_frames` (they get decoded but marked `was_used_for_training=False`)

**Tests needed:**
- `tests/sequence_models/test_hmm_decoder.py`: Viterbi + posterior shape correctness; max_state_probability matches argmax(posterior)

---

### Task 9: `sequence_models/` package — summary statistics

**Files:**
- Create: `src/humpback/sequence_models/summary.py`

**Acceptance criteria:**
- [ ] `compute_summary(viterbi_states_per_sequence, n_states, posteriors_per_sequence)` → dict with per-state occupancy fractions, dwell-time histograms, transition matrix
- [ ] Transition matrix: row-normalized; shape `(n_states, n_states)`
- [ ] Dwell-time histograms: per-state list of run-lengths in frames
- [ ] Occupancy: fraction of total frames assigned to each state

**Tests needed:**
- `tests/sequence_models/test_summary.py`: dwell-time histogram bins; occupancy fractions sum to 1.0; transition matrix shape and row-normalization

---

### Task 10: Synthetic sequence test fixture

**Files:**
- Create: `tests/fixtures/sequence_models/synthetic_sequences.py`

**Acceptance criteria:**
- [ ] `generate_synthetic_sequences(n_states, n_sequences, min_length, max_length, vector_dim, transition_matrix, seed)` → tuple of (embeddings list, ground_truth_states list)
- [ ] Planted state structure: sequences follow the given transition matrix with per-state Gaussian emission clusters
- [ ] Reproducible given fixed seed

**Tests needed:**
- Fixture used by `test_hmm_trainer.py`, `test_hmm_decoder.py`, `test_summary.py`

---

### Task 11: HMM sequence service

**Files:**
- Create: `src/humpback/services/hmm_sequence_service.py`

**Acceptance criteria:**
- [ ] `create_hmm_sequence_job(session, body)` → validates source `ContinuousEmbeddingJob` is `complete`, creates `HMMSequenceJob` row with status `queued`
- [ ] `list_hmm_sequence_jobs(session, status, continuous_embedding_job_id)` → filtered list
- [ ] `get_hmm_sequence_job(session, job_id)` → job or None
- [ ] `cancel_hmm_sequence_job(session, job_id)` → cancels `queued`/`running` jobs; 409 for terminal states
- [ ] Validation: rejects non-existent or non-complete `continuous_embedding_job_id`

**Tests needed:**
- `tests/services/test_hmm_sequence_service.py`: source-job-must-be-complete validation; create/list/get/cancel lifecycle

---

### Task 12: HMM sequence worker

**Files:**
- Create: `src/humpback/workers/hmm_sequence_worker.py`

**Acceptance criteria:**
- [ ] `run_hmm_sequence_job(session, job, settings)` implements the full runtime flow from spec §4.3: reads embeddings parquet, groups by `merged_span_id`, applies L2 normalization, fits PCA, filters short sequences for training, fits HMM, decodes all sequences, computes summary, persists artifacts atomically
- [ ] Outputs all six artifact files: `pca_model.joblib`, `hmm_model.joblib`, `states.parquet`, `transition_matrix.npy`, `state_summary.json`, `training_log.json`
- [ ] `states.parquet` schema: all columns from `embeddings.parquet` minus `embedding`, plus `viterbi_state`, `state_posterior`, `max_state_probability`, `was_used_for_training`
- [ ] Atomic writes: temp files then `os.rename` into final paths
- [ ] Cancellation check between major phases (PCA fit, HMM fit, decode)
- [ ] `random_seed` flows through both PCA and HMM
- [ ] On failure, sets job status to `failed` with `error_message`
- [ ] `run_one_iteration` claim-and-run function matching existing worker pattern

**Tests needed:**
- `tests/workers/test_hmm_sequence_worker.py`: end-to-end producing all expected artifacts from synthetic data; failure mode; cancellation between phases

---

### Task 13: Worker queue claim function + runner.py integration

**Files:**
- Modify: `src/humpback/workers/queue.py`
- Modify: `src/humpback/workers/runner.py`

**Acceptance criteria:**
- [ ] `claim_hmm_sequence_job(session)` added to `queue.py` following the existing pattern (3 retries, claims `queued` → `running`)
- [ ] `runner.py` main loop dispatches to `run_hmm_sequence_job` after the continuous embedding worker block
- [ ] Also wire in `claim_continuous_embedding_job` / `run_continuous_embedding_job` dispatch in `runner.py` (currently missing from main loop)

**Tests needed:**
- Claim function exercised by worker integration tests

---

### Task 14: API endpoints for HMM sequence jobs

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`

**Acceptance criteria:**
- [ ] `POST /sequence-models/hmm-sequences` — create job, 201 on success, 400 on validation error
- [ ] `GET /sequence-models/hmm-sequences` — list jobs with optional `?status=` and `?continuous_embedding_job_id=` filters
- [ ] `GET /sequence-models/hmm-sequences/{id}` — job detail + state_summary.json sidecar
- [ ] `GET /sequence-models/hmm-sequences/{id}/states` — paginated states.parquet rows as JSON
- [ ] `GET /sequence-models/hmm-sequences/{id}/transitions` — transition matrix as nested list
- [ ] `GET /sequence-models/hmm-sequences/{id}/dwell` — dwell-time histograms as JSON
- [ ] `POST /sequence-models/hmm-sequences/{id}/cancel` — cancel with 409 for terminal states

**Tests needed:**
- `tests/integration/test_sequence_models_api.py`: extend existing test file with HMM endpoint tests — create with valid/invalid source, list, get detail, cancel lifecycle

---

### Task 15: Frontend — TanStack Query hooks for HMM

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`

**Acceptance criteria:**
- [ ] TypeScript interfaces: `HMMSequenceJob`, `HMMSequenceJobDetail`, `HMMStateSummary`, `CreateHMMSequenceJobRequest`, `TransitionMatrix`, `DwellHistograms`
- [ ] API functions: `fetchHMMSequenceJobs`, `fetchHMMSequenceJob`, `createHMMSequenceJob`, `cancelHMMSequenceJob`, `fetchHMMTransitions`, `fetchHMMDwell`
- [ ] TanStack Query hooks: `useHMMSequenceJobs`, `useHMMSequenceJob`, `useCreateHMMSequenceJob`, `useCancelHMMSequenceJob`, `useHMMTransitions`, `useHMMDwell`
- [ ] Polling at 3s for active jobs

**Tests needed:**
- Exercised by Playwright tests

---

### Task 16: Frontend — HMM Sequence jobs page + create form

**Files:**
- Create: `frontend/src/components/sequence-models/HMMSequenceJobsPage.tsx`
- Create: `frontend/src/components/sequence-models/HMMSequenceCreateForm.tsx`
- Create: `frontend/src/components/sequence-models/HMMSequenceJobCard.tsx`

**Acceptance criteria:**
- [ ] Jobs page with active/previous split, job cards showing status + summary stats
- [ ] Create form: source selector (dropdown of completed `ContinuousEmbeddingJob`s), inputs for `n_states`, `pca_dims`, `covariance_type` (diag/full select), `n_iter`, `random_seed`, `min_sequence_length_frames`
- [ ] Create form defaults match schema defaults
- [ ] Navigates to detail page on successful creation

**Tests needed:**
- Playwright tests (Task 19)

---

### Task 17: Frontend — HMM Sequence detail page with three Plotly charts

**Files:**
- Create: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`

**Acceptance criteria:**
- [ ] Job summary section: status badge, all hyperparameters, training stats (log-likelihood, sequence counts, frame counts)
- [ ] State timeline chart (Plotly): per merged span, horizontal bars colored by `viterbi_state`, with span selector dropdown when multiple spans exist
- [ ] Transition matrix heatmap (Plotly): n_states × n_states with annotated probabilities
- [ ] Dwell-time histograms grid (Plotly): one histogram per state showing dwell durations
- [ ] Error display if job failed; loading states while running
- [ ] Cancel button for active jobs

**Tests needed:**
- Playwright tests (Task 19)

---

### Task 18: Frontend — routing and navigation

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/layout/SideNav.tsx`

**Acceptance criteria:**
- [ ] Route `/app/sequence-models/hmm-sequence` → HMMSequenceJobsPage
- [ ] Route `/app/sequence-models/hmm-sequence/:jobId` → HMMSequenceDetailPage
- [ ] "HMM Sequence" sub-nav item under "Sequence Models" section in SideNav
- [ ] `/app/sequence-models` redirect unchanged (still goes to continuous-embedding)

**Tests needed:**
- Playwright navigation test (Task 19)

---

### Task 19: Playwright tests for HMM sequence pages

**Files:**
- Create: `frontend/tests/sequence-models/hmm-sequence.spec.ts`

**Acceptance criteria:**
- [ ] Navigation: HMM Sequence nav item loads the jobs page
- [ ] Create form: constrained to completed continuous-embedding jobs
- [ ] Detail page: renders all three chart containers on a complete job
- [ ] Span selector: switches between merged spans in the state timeline

**Tests needed:**
- This task IS the test task

---

### Task 20: Documentation updates

**Files:**
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/data-model.md`
- Modify: `CLAUDE.md` (§9.1 capabilities, §9.2 schema)

**Acceptance criteria:**
- [ ] `sequence-models-api.md` extended with all HMM endpoint documentation
- [ ] `storage-layout.md` extended with `hmm_sequences/{job_id}/` tree
- [ ] `data-model.md` extended with `hmm_sequence_jobs` table reference
- [ ] CLAUDE.md §9.1 updated to mention HMM training + Viterbi decode + visualization
- [ ] CLAUDE.md §9.2 updated: latest migration → `058_hmm_sequence_jobs.py`, `hmm_sequence_jobs` added to table list

**Tests needed:**
- No code tests; doc correctness verified by review

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/sequence_models/ src/humpback/services/hmm_sequence_service.py src/humpback/workers/hmm_sequence_worker.py src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/storage.py src/humpback/api/routers/sequence_models.py src/humpback/workers/queue.py src/humpback/workers/runner.py tests/sequence_models/ tests/services/test_hmm_sequence_service.py tests/workers/test_hmm_sequence_worker.py tests/integration/test_sequence_models_api.py tests/fixtures/sequence_models/`
2. `uv run ruff check src/humpback/sequence_models/ src/humpback/services/hmm_sequence_service.py src/humpback/workers/hmm_sequence_worker.py src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/storage.py src/humpback/api/routers/sequence_models.py src/humpback/workers/queue.py src/humpback/workers/runner.py tests/sequence_models/ tests/services/test_hmm_sequence_service.py tests/workers/test_hmm_sequence_worker.py tests/integration/test_sequence_models_api.py tests/fixtures/sequence_models/`
3. `uv run pyright src/humpback/sequence_models/ src/humpback/services/hmm_sequence_service.py src/humpback/workers/hmm_sequence_worker.py src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/storage.py src/humpback/api/routers/sequence_models.py src/humpback/workers/queue.py src/humpback/workers/runner.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
