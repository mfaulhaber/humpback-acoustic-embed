# HMM Motif Extraction Implementation Plan

**Goal:** Add first-class motif extraction jobs for completed HMM Sequence jobs, including symbolic state collapse, n-gram motif mining, ranked motif artifacts, API surfaces, and an HMM detail-page Motifs panel.
**Spec:** [docs/specs/2026-04-30-hmm-motif-extraction-design.md](../specs/2026-04-30-hmm-motif-extraction-design.md)
**Branch:** `feature/hmm-motif-extraction`

---

### Task 1: Motif Extraction Core

**Files:**
- Create: `src/humpback/sequence_models/motifs.py`
- Create: `tests/sequence_models/test_motifs.py`

**Acceptance criteria:**
- [ ] `MotifExtractionConfig` captures n-gram bounds, minimum occurrences, minimum event sources, rank weights, and optional `call_probability_weight`
- [ ] Decoded state rows are grouped by SurfPerch `merged_span_id` / `window_index_in_span` and CRNN `region_id` / `chunk_index_in_region`
- [ ] Consecutive repeated states collapse into symbolic tokens while preserving raw row ranges, timestamps, event-source keys, event-core duration, background duration, and nullable call probability
- [ ] N-gram candidates are extracted from collapsed tokens for all configured lengths
- [ ] Motifs are filtered by `minimum_occurrences` and `minimum_event_sources`
- [ ] Motifs are ranked from frequency, event-source recurrence, event-core association, low background occupancy, and optional call probability
- [ ] Event midpoint anchors are selected when event ids are available, with documented fallback strategies
- [ ] Pure artifact serialization helpers can write/read manifest, motifs parquet, and occurrences parquet without DB access

**Tests needed:**
- Unit coverage for repeated-state collapse, n-gram extraction, minimum occurrence filtering, minimum event-source filtering, rank-weight ordering, SurfPerch `is_in_pad` metrics, CRNN `tier` metrics, nullable call probability, event-midpoint anchoring, fallback anchoring, and stable config signatures.

---

### Task 2: Motif Extraction Data Model and Migration

**Files:**
- Create: `alembic/versions/062_motif_extraction_jobs.py`
- Modify: `src/humpback/models/sequence_models.py`
- Modify: `src/humpback/database.py`
- Create: `tests/db/test_migration_062.py`

**Acceptance criteria:**
- [ ] **MANDATORY DB BACKUP (CLAUDE.md section 3.5):** Before running this migration against the production database, read `HUMPBACK_DATABASE_URL` from `.env`, copy the database file with a UTC timestamp, and verify the backup is non-zero size using these commands before `uv run alembic upgrade head`: `DB_URL=$(grep '^HUMPBACK_DATABASE_URL=' .env | cut -d= -f2-)`, `DB_PATH=${DB_URL#sqlite+aiosqlite:///}`, `BACKUP="$DB_PATH.$(date -u +%Y-%m-%d-%H:%M).bak"`, `cp "$DB_PATH" "$BACKUP"`, `test -s "$BACKUP"`. If any backup command fails or is skipped, stop and do not apply the migration.
- [ ] Migration creates `motif_extraction_jobs` with status, `hmm_sequence_job_id`, source kind, extraction config, rank weights, `config_signature`, counters, `artifact_dir`, `error_message`, and timestamps
- [ ] Migration adds indexes on `status`, `hmm_sequence_job_id`, and `config_signature`
- [ ] SQLAlchemy model `MotifExtractionJob` is added and exported from `src/humpback/models/sequence_models.py`
- [ ] Downgrade removes the table and indexes
- [ ] `uv run alembic upgrade head` succeeds against the backed-up production DB

**Tests needed:**
- Migration round-trip test on SQLite fixture verifies table columns, defaults, nullability, indexes, upgrade, and downgrade.

---

### Task 3: Storage, Service, and Queue Integration

**Files:**
- Modify: `src/humpback/storage.py`
- Create: `src/humpback/services/motif_extraction_service.py`
- Modify: `src/humpback/workers/queue.py`
- Modify: `src/humpback/workers/runner.py`
- Modify: `tests/unit/test_queue.py`
- Create: `tests/services/test_motif_extraction_service.py`

**Acceptance criteria:**
- [ ] Storage helpers return `motif_extractions/{job_id}/manifest.json`, `motifs.parquet`, and `occurrences.parquet`
- [ ] Service creates motif extraction jobs only for completed HMM Sequence jobs
- [ ] Service computes deterministic `config_signature` from HMM job id and extraction config
- [ ] Service returns an existing queued, running, or complete job for an identical config signature
- [ ] Failed and canceled jobs do not block retries with the same config
- [ ] Service supports list, fetch, cancel, and delete operations
- [ ] Delete removes motif extraction artifacts and DB row
- [ ] Queue can atomically claim motif extraction jobs and recover stale running jobs
- [ ] Worker runner polls motif extraction jobs after HMM Sequence jobs

**Tests needed:**
- Service tests for validation, idempotency, retry after failed/canceled, cancel terminal behavior, delete cleanup, and list filters. Queue tests for claim and stale-job recovery.

---

### Task 4: Motif Extraction Worker

**Files:**
- Create: `src/humpback/workers/motif_extraction_worker.py`
- Create: `tests/workers/test_motif_extraction_worker.py`

**Acceptance criteria:**
- [ ] Worker claims a queued motif extraction job and validates the source HMM job remains complete
- [ ] Worker resolves parent Continuous Embedding job and source kind
- [ ] Worker reads `states.parquet` from the source HMM job
- [ ] Worker reads parent CEJ `embeddings.parquet` for CRNN nearest-event ids and optional call probabilities
- [ ] Worker reads parent segmentation `events.parquet` for event midpoint anchors when available
- [ ] Worker writes manifest, motifs parquet, and occurrences parquet atomically
- [ ] Worker updates motif counters, `artifact_dir`, status, and error message
- [ ] Worker supports cooperative cancellation before final artifact rename
- [ ] Worker marks failures with useful error messages and leaves no final partial artifacts

**Tests needed:**
- Worker tests for completed SurfPerch fixture, completed CRNN fixture, cancellation before artifact rename, missing source artifacts, and failure status/error handling.

---

### Task 5: API Schemas and Router Endpoints

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Create: `tests/integration/test_motif_extraction_api.py`

**Acceptance criteria:**
- [ ] Pydantic schemas cover motif extraction create, job output, detail, manifest, motif summary, motif response, occurrence row, and occurrence response
- [ ] `POST /sequence-models/motif-extractions` creates or reuses motif extraction jobs
- [ ] `GET /sequence-models/motif-extractions` lists jobs with optional `status` and `hmm_sequence_job_id` filters
- [ ] `GET /sequence-models/motif-extractions/{id}` returns DB row plus manifest when available
- [ ] `POST /sequence-models/motif-extractions/{id}/cancel` cancels queued/running jobs and rejects terminal jobs
- [ ] `DELETE /sequence-models/motif-extractions/{id}` deletes DB row and artifacts
- [ ] `GET /sequence-models/motif-extractions/{id}/motifs` returns paginated ranked motif rows
- [ ] `GET /sequence-models/motif-extractions/{id}/motifs/{motif_key}/occurrences` returns paginated occurrence rows for one motif
- [ ] API returns clear 404, 409, and 422 errors for missing jobs, terminal cancellation, invalid configs, non-complete HMM jobs, and missing artifacts

**Tests needed:**
- Integration tests for create/list/detail/cancel/delete, idempotent config reuse, completed artifact reads, motif pagination, occurrence pagination, `minimum_event_sources` filtering, CRNN grouping, and error cases.

---

### Task 6: Frontend API Client and Types

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`

**Acceptance criteria:**
- [ ] TypeScript interfaces mirror motif extraction job, manifest, motif summary, occurrence, and response schemas
- [ ] API helpers cover create, list, fetch, cancel, delete, motif list, and occurrence list
- [ ] React Query hooks invalidate motif extraction lists and detail data after create, cancel, and delete
- [ ] Hooks support filtering motif jobs by HMM Sequence job id
- [ ] Existing Sequence Models API hooks continue to typecheck without changes to callers

**Tests needed:**
- TypeScript verification via `cd frontend && npx tsc --noEmit`.

---

### Task 7: HMM Detail Motifs Panel

**Files:**
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`
- Create: `frontend/src/components/sequence-models/MotifExtractionPanel.tsx`
- Create: `frontend/src/components/sequence-models/MotifExampleAlignment.tsx`
- Modify: `frontend/e2e/sequence-models/hmm-sequence.spec.ts`

**Acceptance criteria:**
- [ ] Completed HMM jobs show a Motifs panel after the HMM State Timeline Viewer
- [ ] Panel shows a create form when no motif extraction job exists for the HMM job
- [ ] Create form includes n-gram range, minimum occurrences, and minimum event sources
- [ ] Advanced section exposes frequency, event-source, event-core, low-background, and optional call-probability weights
- [ ] Queued/running motif jobs show status and cancel action
- [ ] Failed motif jobs show error and rerun controls
- [ ] Complete motif jobs show ranked motif table with state sequence, occurrence count, event-source count, event-core fraction, background fraction, optional call probability, mean duration, and rank score
- [ ] Selecting a motif loads and renders aligned occurrence examples around event midpoint
- [ ] Jump-to-timeline action updates existing event/region selection and scroll target
- [ ] The panel works for both SurfPerch and CRNN source kinds

**Tests needed:**
- Playwright coverage for empty/create, advanced weights, running status, complete motif table, aligned examples with zero marker, rerun config, cancel action, and jump-to-timeline behavior.

---

### Task 8: Reference Documentation and ADR

**Files:**
- Modify: `DECISIONS.md`
- Modify: `CLAUDE.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/frontend.md`
- Modify: `README.md`

**Acceptance criteria:**
- [ ] `DECISIONS.md` adds ADR-058 for first-class motif extraction jobs and why Approach C was selected
- [ ] `CLAUDE.md` Sequence Models section mentions motif extraction jobs and migration 062 where appropriate
- [ ] Storage reference documents `motif_extractions/{job_id}` artifacts and parquet schemas
- [ ] API reference documents motif extraction endpoints, request fields, response fields, status behavior, and error cases
- [ ] Data model reference documents `motif_extraction_jobs`
- [ ] Frontend reference documents the HMM detail Motifs panel
- [ ] README feature list mentions motif extraction only after implementation is complete

**Tests needed:**
- Documentation review only.

---

## Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/motifs.py src/humpback/services/motif_extraction_service.py src/humpback/workers/motif_extraction_worker.py src/humpback/workers/queue.py src/humpback/workers/runner.py src/humpback/storage.py src/humpback/models/sequence_models.py src/humpback/database.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py alembic/versions/062_motif_extraction_jobs.py tests/sequence_models/test_motifs.py tests/services/test_motif_extraction_service.py tests/workers/test_motif_extraction_worker.py tests/integration/test_motif_extraction_api.py tests/db/test_migration_062.py tests/unit/test_queue.py`
2. `uv run ruff check src/humpback/sequence_models/motifs.py src/humpback/services/motif_extraction_service.py src/humpback/workers/motif_extraction_worker.py src/humpback/workers/queue.py src/humpback/workers/runner.py src/humpback/storage.py src/humpback/models/sequence_models.py src/humpback/database.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py alembic/versions/062_motif_extraction_jobs.py tests/sequence_models/test_motifs.py tests/services/test_motif_extraction_service.py tests/workers/test_motif_extraction_worker.py tests/integration/test_motif_extraction_api.py tests/db/test_migration_062.py tests/unit/test_queue.py`
3. `uv run pyright`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test frontend/e2e/sequence-models/hmm-sequence.spec.ts`

Backup gate (CLAUDE.md section 3.5) is covered by Task 2 acceptance criterion #1; no migration runs without it.
