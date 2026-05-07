# CRNN Event Encoder Implementation Plan

**Goal:** Add a retained Sequence Models Event Encoder job that tokenizes Pass 2 events from existing CRNN Continuous Embedding chunks and produces reportable token sequences.
**Spec:** [docs/specs/2026-05-07-crnn-event-encoder-design.md](../specs/2026-05-07-crnn-event-encoder-design.md)
**Primary domain:** sequence-models
**Neighbor domains:** call-parsing, core-platform, frontend-shell

---

### Task 1: Add Event Encoder Job Schema, Storage, And Idempotent Service

**Files:**
- Create: `alembic/versions/076_event_encoder_jobs.py`
- Create: `tests/unit/test_migration_076_event_encoder_jobs.py`
- Create: `src/humpback/services/event_encoder_service.py`
- Create: `tests/services/test_event_encoder_service.py`
- Modify: `src/humpback/models/sequence_models.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/storage.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`
- Modify: `tests/unit/test_storage.py`

**Acceptance criteria:**
- [x] Before creating or running the Alembic migration, complete the CLAUDE.md database backup step from `.env`: run `DATABASE_URL=$(grep -E '^DATABASE_URL=' .env | cut -d= -f2-)`, run `DB_PATH=${DATABASE_URL#sqlite+aiosqlite:///}` and if unchanged run `DB_PATH=${DATABASE_URL#sqlite:///}`, run `BACKUP_PATH="$DB_PATH.$(date -u +%Y%m%dT%H%M%SZ).bak"`, run `cp "$DB_PATH" "$BACKUP_PATH"`, and run `test -s "$BACKUP_PATH"`.
- [x] Migration creates `event_encoder_jobs` with a unique `tokenization_signature` and all provenance, config, counter, artifact-path, status, error, and timestamp columns from the spec.
- [x] ORM model and Pydantic schemas expose create, list/detail, manifest, report, and job-output shapes without weakening existing Continuous Embedding schemas.
- [x] Storage helpers return paths under `storage_root/event_encoders/{job_id}/` for manifest, report, vectors parquet, tokens parquet, sequences parquet, preprocessing model, and k-means models.
- [x] Service validates completed Pass 2 segmentation input and completed matching `region_crnn` Continuous Embedding input.
- [x] Service computes `tokenization_signature`, includes effective-event correction revision when requested, reuses queued/running/complete rows, and resets failed/canceled rows to `queued`.

**Tests needed:**
- Migration test for upgrade/downgrade behavior and unique signature creation.
- Schema validation tests for defaults, invalid k values, invalid PCA dimension, raw/effective source mode, and invalid source combinations.
- Storage helper tests proving all event encoder paths live under `storage_root`.
- Service tests for source validation, idempotency reuse/reset, and effective-mode correction revision changes.

---

### Task 2: Implement Event Vector And Tokenization Utilities

**Files:**
- Create: `src/humpback/sequence_models/event_encoder.py`
- Create: `src/humpback/sequence_models/event_tokenization.py`
- Create: `tests/sequence_models/test_event_encoder.py`
- Create: `tests/sequence_models/test_event_tokenization.py`
- Modify: `src/humpback/sequence_models/__init__.py`

**Acceptance criteria:**
- [x] Event/chunk overlap is recomputed against the selected raw or effective event set and does not rely on upstream `nearest_event_id`.
- [x] Pooling emits stable fixed-width `mean_pool`, `top_k_pool`, `start_pool`, `middle_pool`, and `end_pool` blocks, including fallback behavior for short events.
- [x] Acoustic descriptors are computed for duration, log energy, peak frequency, spectral centroid, bandwidth, spectral entropy, frequency slope, and gap to previous event.
- [x] Preprocessing supports optional per-pool L2 normalization, PCA dimension clamping, robust descriptor scaling, and embedding/descriptor feature weights.
- [x] K-means fitting supports one or more valid k values, skips impossible k values with report metadata, remaps labels deterministically, and computes token distance/confidence fields.
- [x] Pure utilities do not import FastAPI, SQLAlchemy sessions, or worker lifecycle code.

**Tests needed:**
- Unit tests for weighted pooling, short-event fallbacks, effective-event overlap, descriptor calculations on deterministic audio, PCA dimension clamping, robust z-score scaling, deterministic token labels, and confidence edge cases.

---

### Task 3: Add Event Encoder Worker And Queue Integration

**Files:**
- Create: `src/humpback/workers/event_encoder_worker.py`
- Create: `tests/workers/test_event_encoder_worker.py`
- Modify: `src/humpback/workers/queue.py`
- Modify: `src/humpback/workers/runner.py`
- Modify: `tests/unit/test_queue.py`

**Acceptance criteria:**
- [x] Queue claim uses the standard atomic queued-to-running status transition for `EventEncoderJob`.
- [x] Worker loads selected events, CRNN embedding chunks, and source audio through the Pass 2 to Pass 1 source chain without mutating Call Parsing artifacts.
- [x] Worker writes `manifest.json`, `report.json`, `event_vectors.parquet`, `event_tokens.parquet`, `token_sequences.parquet`, preprocessing model, and one k-means model per valid k with atomic replace semantics.
- [x] Worker records total, encoded, skipped, vector-dimension, artifact path, and error state on the job row.
- [x] Worker cooperatively honors cancellation after input loading, between region/vector batches, and between k-means fits.
- [x] Worker failure leaves a clear `error_message` and does not expose partial artifact paths as completed outputs.

**Tests needed:**
- Worker tests for successful artifact generation with stubbed chunks/audio, cancellation cleanup, failed no-encodable-events case, invalid k skip reporting, and queue claim behavior.

---

### Task 4: Add Event Encoder API Endpoints

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `tests/integration/test_sequence_models_api.py`

**Acceptance criteria:**
- [x] `POST /sequence-models/event-encoders` creates or reuses jobs and returns 201 for new rows and 200 for reused/reset rows.
- [x] `GET /sequence-models/event-encoders` lists jobs newest-first with optional status filter.
- [x] `GET /sequence-models/event-encoders/{job_id}` returns job plus nullable manifest and report payloads.
- [x] `POST /sequence-models/event-encoders/{job_id}/cancel` matches Continuous Embedding terminal-state behavior.
- [x] `DELETE /sequence-models/event-encoders/{job_id}` deletes the row and event encoder artifact directory.
- [x] Existing Continuous Embedding endpoint behavior and tests remain unchanged.

**Tests needed:**
- Integration tests for create/list/detail/cancel/delete, validation errors, reused-job status code, terminal cancel conflict, missing job 404s, and Continuous Embedding regression coverage.

---

### Task 5: Add Sequence Models Event Encoder Frontend

**Files:**
- Create: `frontend/src/components/sequence-models/EventEncoderCreateForm.tsx`
- Create: `frontend/src/components/sequence-models/EventEncoderDetailPage.tsx`
- Create: `frontend/src/components/sequence-models/EventEncoderJobTable.tsx`
- Create: `frontend/src/components/sequence-models/EventEncoderJobsPage.tsx`
- Create: `frontend/e2e/sequence-models/event-encoder.spec.ts`
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/layout/SideNav.tsx`
- Modify: `frontend/src/components/layout/Breadcrumbs.tsx`

**Acceptance criteria:**
- [x] Sequence Models navigation includes Continuous Embedding and Event Encoder.
- [x] `/app/sequence-models/event-encoder` shows create form, active jobs, and previous jobs using existing Sequence Models UI patterns.
- [x] Create form filters completed `region_crnn` Continuous Embedding jobs to the selected completed segmentation job.
- [x] Create form supports raw/effective source selection, PCA dimension 64/128, k values 50/100/200, and advanced pooling/weight/seed controls.
- [x] Detail page polls active jobs, stops polling terminal jobs, and renders source provenance, count summaries, token distributions, sequence preview, descriptor summaries, exemplar ids, artifact paths, and errors.
- [x] Route, breadcrumb, and API hook changes remain consistent with the existing frontend shell.

**Tests needed:**
- Frontend TypeScript compile.
- Playwright test covering nav route, create-form filtering, create submission payload, active/terminal table rendering, detail report rendering, and error rendering.

---

### Task 6: Update Reference Docs And Agent Context

**Files:**
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `docs/reference/frontend.md`
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/invariants.md`
- Modify: `docs/agent-context/domains/sequence-models/references.md`
- Modify: `docs/agent-context/domains/sequence-models/tests.md`

**Acceptance criteria:**
- [x] Data model reference documents `EventEncoderJob`.
- [x] Storage layout documents `event_encoders/{job_id}/` artifacts.
- [x] Sequence Models API reference documents Event Encoder endpoints, create defaults, source validation, and artifact schemas.
- [x] Behavioral constraints document Event Encoder idempotency and raw/effective event semantics.
- [x] Sequence Models domain capsule lists the new paths, artifact root, invariants, references, and targeted tests.
- [x] Docs do not revive retired HMM, Masked Transformer, or motif-extraction runtime surfaces.

**Tests needed:**
- Search docs for stale Sequence Models claims and run `git diff --check`.

---

### Verification

Run in order after all tasks:

1. `git diff --check`
2. `uv run ruff format --check src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/services/event_encoder_service.py src/humpback/sequence_models/event_encoder.py src/humpback/sequence_models/event_tokenization.py src/humpback/workers/event_encoder_worker.py src/humpback/workers/queue.py src/humpback/workers/runner.py src/humpback/api/routers/sequence_models.py src/humpback/storage.py tests/unit/test_migration_076_event_encoder_jobs.py tests/unit/test_sequence_models_schemas.py tests/unit/test_storage.py tests/services/test_event_encoder_service.py tests/sequence_models/test_event_encoder.py tests/sequence_models/test_event_tokenization.py tests/workers/test_event_encoder_worker.py tests/integration/test_sequence_models_api.py`
3. `uv run ruff check src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/services/event_encoder_service.py src/humpback/sequence_models/event_encoder.py src/humpback/sequence_models/event_tokenization.py src/humpback/workers/event_encoder_worker.py src/humpback/workers/queue.py src/humpback/workers/runner.py src/humpback/api/routers/sequence_models.py src/humpback/storage.py tests/unit/test_migration_076_event_encoder_jobs.py tests/unit/test_sequence_models_schemas.py tests/unit/test_storage.py tests/services/test_event_encoder_service.py tests/sequence_models/test_event_encoder.py tests/sequence_models/test_event_tokenization.py tests/workers/test_event_encoder_worker.py tests/integration/test_sequence_models_api.py`
4. `uv run pyright`
5. `uv run pytest tests/unit/test_migration_076_event_encoder_jobs.py tests/unit/test_sequence_models_schemas.py tests/unit/test_storage.py tests/services/test_event_encoder_service.py tests/sequence_models/test_event_encoder.py tests/sequence_models/test_event_tokenization.py tests/workers/test_event_encoder_worker.py tests/integration/test_sequence_models_api.py -q`
6. `uv run pytest tests/sequence_models tests/unit/test_sequence_models_schemas.py tests/services/test_continuous_embedding_service.py tests/workers/test_continuous_embedding_worker.py tests/integration/test_sequence_models_api.py -q`
7. `cd frontend && npx tsc --noEmit`
8. `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts e2e/sequence-models/continuous-embedding.spec.ts`
9. `uv run pytest tests/`
