# Perch v2 Classifier Tuning Implementation Plan

**Goal:** Make `perch_v2.tflite` a first-class embedding model across tuning, training, and detection by extending existing tables, workers, and UI surgically (Approach A) rather than introducing a unified labeled-manifest artifact.
**Spec:** [docs/specs/2026-04-17-perch-v2-classifier-tuning-design.md](../specs/2026-04-17-perch-v2-classifier-tuning-design.md)

---

### Task 1: Storage path helper becomes model-version aware

**Files:**
- Modify: `src/humpback/storage.py`

**Acceptance criteria:**
- [ ] `detection_embeddings_path` accepts a required `model_version: str` argument and includes it in the returned path.
- [ ] Every call site in the codebase is updated to pass `model_version`.
- [ ] No lingering default that would hide a missing value (explicit parameter only).

**Tests needed:**
- Unit test in `tests/test_storage.py` asserting the emitted path includes the model version and differs between two model versions for the same detection job id.

---

### Task 2: Alembic migration 049 — `detection_embedding_jobs` gains model version and progress fields

**Files:**
- Create: `alembic/versions/049_detection_embedding_jobs_model_version.py`
- Modify: `src/humpback/models/detection_embedding_job.py`

**Acceptance criteria:**
- [ ] Adds `model_version: str` to `detection_embedding_jobs` (nullable in one batch, backfilled from the detection job's source classifier `model_version`, then altered to `NOT NULL`).
- [ ] Adds `rows_processed: int NOT NULL DEFAULT 0` and `rows_total: int NULL`.
- [ ] Replaces any existing single-column uniqueness on `detection_job_id` with a composite unique `(detection_job_id, model_version)`.
- [ ] Physically relocates any existing detection embedding parquet to the new model-versioned path during `upgrade()`; `downgrade()` reverses the move.
- [ ] Uses `op.batch_alter_table()` for SQLite compatibility.
- [ ] Runs cleanly against the dev DB via `uv run alembic upgrade head`.

**Tests needed:**
- Migration test in `tests/migrations/test_049_detection_embedding_jobs_model_version.py` that seeds a pre-migration schema, applies the migration, and asserts column presence, backfilled values, composite uniqueness, and file relocation.

---

### Task 3: Alembic migration 050 — `hyperparameter_manifests.embedding_model_version`

**Files:**
- Create: `alembic/versions/050_hyperparameter_manifests_embedding_model_version.py`
- Modify: `src/humpback/models/hyperparameter.py`

**Acceptance criteria:**
- [ ] Adds `embedding_model_version: str` (nullable in one batch, backfilled from the first resolved source of each manifest row, then altered to `NOT NULL`).
- [ ] Backfill logic handles manifests whose sources are training jobs (pull `model_version` from the training job) and manifests whose sources are detection jobs (pull from the source classifier's `model_version`).
- [ ] Runs cleanly via `uv run alembic upgrade head`.

**Tests needed:**
- Migration test in `tests/migrations/test_050_hyperparameter_manifests_embedding_model_version.py` covering both training-job-sourced and detection-job-sourced backfill cases.

---

### Task 4: Alembic migration 051 — perch_v2 `ModelConfig` seed

**Files:**
- Create: `alembic/versions/051_perch_v2_model_config_seed.py`

**Acceptance criteria:**
- [ ] Idempotent insert of a `ModelConfig` row with `name=perch_v2`, `display_name="Perch v2 (TFLite)"`, `path="models/perch_v2.tflite"`, `model_type=tflite`, `input_format=waveform`, `vector_dim=1536`, `is_default=False`.
- [ ] Skips insert if a row with that `name` already exists.
- [ ] `downgrade()` removes only the row it inserted.

**Tests needed:**
- Migration test in `tests/migrations/test_051_perch_v2_model_config_seed.py` that asserts the row exists after upgrade, idempotent on re-run, and is removed on downgrade.

---

### Task 5: Re-embedding worker accepts target model version

**Files:**
- Modify: `src/humpback/workers/detection_embedding_worker.py`
- Modify: `src/humpback/services/detection_embedding_service.py` (if present; otherwise the equivalent service module)

**Acceptance criteria:**
- [ ] Worker resolves its embedding model via `get_model_by_version(job.model_version)` rather than by chasing the source classifier's `model_version`.
- [ ] Service exposes `create_reembedding_job(detection_job_id, model_version)` returning the existing row if one is already `queued`, `running`, or `complete`; otherwise inserts a new row.
- [ ] Worker sets `rows_total` after decoding the row store and increments `rows_processed` per batch committed.
- [ ] Failed runs populate `error_message` and leave `status="failed"`; a subsequent `create_reembedding_job` call with the same key re-enqueues by resetting `status` and clearing `error_message`.
- [ ] Parquet is written via the updated `detection_embeddings_path(storage_root, detection_job_id, model_version)` helper and keyed by `row_id`.

**Tests needed:**
- `tests/workers/test_detection_reembedding_worker.py` using a fake waveform TFLite model. Cover: end-to-end success path, idempotency (second enqueue is a no-op when complete), progress field updates, failure path writes `error_message`, retry after failure clears the error.
- `tests/services/test_detection_embedding_service.py` covering the idempotent enqueue contract.

---

### Task 6: Read endpoint for re-embedding status

**Files:**
- Create or modify: `src/humpback/api/routers/classifier/detection_embedding_jobs.py`
- Modify: `src/humpback/schemas/detection_embedding_job.py` (or create, to define the response payload)

**Acceptance criteria:**
- [ ] `GET /detection-embedding-jobs?detection_job_ids=...&model_version=...` returns a list of rows matching the requested pairs, one entry per requested `detection_job_id` (including rows that do not yet exist, with `status="not_started"`).
- [ ] Response includes `detection_job_id`, `model_version`, `status`, `rows_processed`, `rows_total`, `error_message`, `created_at`, `updated_at`.
- [ ] Endpoint is wired into the existing classifier router group.

**Tests needed:**
- API test in `tests/api/test_detection_embedding_jobs_router.py` covering: mixed existing + non-existing pairs, rejection of empty query param, correct progress fields for a running job.

---

### Task 7: Manifest builder requires explicit embedding model version

**Files:**
- Modify: `src/humpback/services/hyperparameter_service/manifest.py`
- Modify: `src/humpback/workers/hyperparameter_worker.py`

**Acceptance criteria:**
- [ ] Manifest generation requires an `embedding_model_version` input and persists it on the `hyperparameter_manifests` row.
- [ ] For every `training_job_id` source, the builder verifies `ClassifierTrainingJob.model_version == embedding_model_version`; mismatch raises a descriptive error that surfaces in the manifest row's `error_message` and marks status `failed`.
- [ ] For every `detection_job_id` source, the builder reads embeddings from `detection_embeddings_path(storage_root, detection_job_id, embedding_model_version)` and fails loudly if that parquet does not exist.
- [ ] For perch_v2 detection sources specifically, the label join uses only the binary row-store labels (`humpback`, `background`, `ship`, `orca`) and does not hit `vocalization_labels`.
- [ ] TF2 manifest generation paths retain existing vocalization-label behavior (no regressions).

**Tests needed:**
- Unit tests in `tests/services/test_hyperparameter_manifest_builder.py` covering: mixed-model rejection, missing-embedding rejection, binary-only label path for perch_v2, unchanged TF2 behavior, persisted `embedding_model_version` on the row.

---

### Task 8: Training service gains detection-manifest source mode

**Files:**
- Modify: `src/humpback/services/classifier_service/training.py`
- Modify: `src/humpback/schemas/classifier_training.py` (or equivalent request schema module)
- Modify: `src/humpback/api/routers/classifier/training.py`

**Acceptance criteria:**
- [ ] Request schema accepts either (positive/negative `embedding_set_ids`) **or** (`detection_job_ids` + `embedding_model_version`), never both.
- [ ] Service rejects mixed input with a clear 4xx error.
- [ ] For detection-manifest input, the service builds an internal manifest (via the updated manifest builder) and creates a `classifier_training_job` with `source_mode="detection_manifest"`, populating `source_detection_job_ids`, `model_version`, `window_size_seconds`, `target_sample_rate`, and `manifest_path`.
- [ ] Validates at least one positive and one negative binary-labeled row across the selected detection jobs; rejects otherwise.
- [ ] Existing embedding-set submission path is unchanged.

**Tests needed:**
- Unit tests in `tests/services/test_classifier_service_validation.py` covering: mixed-source rejection, mismatched model-version rejection, missing-labels rejection, successful detection-manifest submission.
- API test in `tests/api/test_classifier_training_router.py` extending existing coverage with the detection-manifest payload shape.

---

### Task 9: Training worker runs the detection-manifest path

**Files:**
- Modify: `src/humpback/workers/classifier_worker/training.py`

**Acceptance criteria:**
- [ ] Worker branches on `source_mode="detection_manifest"` and runs the manifest-driven training path currently used by autoresearch promotion, without coupling to autoresearch-specific metadata.
- [ ] Produced `ClassifierModel` row carries `model_version = manifest.embedding_model_version`, correct `vector_dim` (from the embeddings), and a `training_summary` that describes the detection-job source (ids + embedding model + split summary).
- [ ] Written artifact (`model.joblib`) is compatible with the detection worker's existing load path.
- [ ] Existing autoresearch promotion and embedding-set training paths continue to work unchanged.

**Tests needed:**
- `tests/workers/test_classifier_training_detection_manifest.py` end-to-end with a fake embedding model and synthetic binary-labeled detection rows. Asserts `ClassifierModel` fields, `vector_dim`, and artifact loadability.
- Regression-guard test exercising the existing embedding-set path.

---

### Task 10: Shared `<DetectionSourcePicker>` frontend component

**Files:**
- Create: `frontend/src/components/classifier/DetectionSourcePicker.tsx`
- Create: `frontend/src/components/classifier/ReembeddingStatusTable.tsx`
- Create: `frontend/src/api/detectionEmbeddingJobs.ts` (if not present) — thin TanStack Query hooks for listing status and enqueuing jobs.

**Acceptance criteria:**
- [ ] Component exposes: detection-job multi-select, embedding-model selector, inline re-embedding status table, Re-embed-now action, and a boolean `isReady` signal indicating every selected pair is `Complete`.
- [ ] Status table renders Not started / Queued / Running (with `rows_processed`/`rows_total` + percentage) / Complete / Failed (with popover for `error_message` and Retry action).
- [ ] Polls status every ~2 s while any pair is `queued` or `running`; stops polling otherwise.
- [ ] Default embedding-model selection is inferred from the first selected detection job's source classifier's `model_version`; the user may override.
- [ ] Component is purely presentational over the provided TanStack Query hooks — no direct `fetch` calls inside the component body.

**Tests needed:**
- Playwright test in `frontend/tests/e2e/classifier-detection-source-picker.spec.ts` with API stubs covering: missing embeddings show the status table, polling transitions Running → Complete enable the downstream action, Failed state exposes error and Retry.

---

### Task 11: TuningTab integrates the shared picker

**Files:**
- Modify: `frontend/src/components/classifier/TuningTab.tsx`
- Modify: `frontend/src/components/classifier/ManifestsSection.tsx`

**Acceptance criteria:**
- [ ] ManifestsSection replaces its ad-hoc detection-job picker with `<DetectionSourcePicker>`.
- [ ] The Create Manifest button is disabled until `<DetectionSourcePicker>.isReady` is `true`.
- [ ] Training-job sources are validated against the selected embedding model; mismatches show a blocking inline error.
- [ ] Submitted request carries `embedding_model_version` explicitly.

**Tests needed:**
- Playwright test in `frontend/tests/e2e/classifier-tuning-reembed.spec.ts` covering: select detection jobs with missing perch_v2 embeddings → status table appears → Create Manifest disabled → re-embed completes → button enables → manifest is created with `embedding_model_version=perch_v2`.

---

### Task 12: TrainingTab gains detection-jobs source mode

**Files:**
- Modify: `frontend/src/components/classifier/TrainingTab.tsx`

**Acceptance criteria:**
- [ ] Introduces a Source mode radio: *Embedding sets* (today) vs *Detection jobs* (new).
- [ ] Detection-jobs mode renders `<DetectionSourcePicker>` in place of positive/negative embedding-set selectors; classifier-head advanced options are preserved.
- [ ] Submit payload uses the schema branch from Task 8 (either embedding sets or detection jobs + embedding model version, never both).
- [ ] Embedding-sets mode is fully preserved.

**Tests needed:**
- Playwright test in `frontend/tests/e2e/classifier-training-detection-mode.spec.ts` exercising the detection-jobs submission end-to-end and verifying a subsequent `ClassifierModel` row appears in the classifier list.

---

### Task 13: ADR-055 and documentation updates

**Files:**
- Modify: `DECISIONS.md`
- Modify: `CLAUDE.md` (§9.1 capability list, §9.2 latest-migration note)
- Modify: `README.md`
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/classifier-api.md`
- Modify: `docs/reference/storage-layout.md`

**Acceptance criteria:**
- [ ] Append ADR-055 following the existing DECISIONS.md conventions; captures Approach A decision, B/C alternatives, label-preservation rationale, and the future-work note about hardcoded spectrogram feature params in `detector.py`.
- [ ] CLAUDE.md §9.1 lists perch_v2 as a registered embedding model family and references deployable perch_v2 classifier support; §9.2 updates the latest-migration marker.
- [ ] README.md feature list mentions perch_v2 classifier support.
- [ ] `docs/reference/data-model.md` reflects the new `detection_embedding_jobs` shape (composite key + progress fields).
- [ ] `docs/reference/classifier-api.md` documents the new `GET /detection-embedding-jobs` endpoint and the `detection_manifest` training source mode.
- [ ] `docs/reference/storage-layout.md` reflects the model-versioned detection-embeddings path.

**Tests needed:**
- Documentation review only; no automated tests required for this task.

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/ tests/`
2. `uv run ruff check src/ tests/`
3. `uv run pyright`
4. `uv run alembic upgrade head`
5. `uv run pytest tests/`
6. `cd frontend && npx tsc --noEmit`
7. `cd frontend && npx playwright test`
