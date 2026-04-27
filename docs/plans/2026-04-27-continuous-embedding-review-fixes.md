# Continuous Embedding Review Fixes Implementation Plan

**Goal:** Fix the PR 145 review issues so continuous embedding jobs preserve correct window geometry and provenance, execute successfully in production, and enforce idempotency under concurrent submission.
**Spec:** [docs/specs/2026-04-27-sequence-models-design.md](../specs/2026-04-27-sequence-models-design.md)

---

### Task 1: Make continuous embedding storage and geometry faithful to Pass 1 regions

**Files:**
- Modify: `src/humpback/workers/continuous_embedding_worker.py`
- Modify: `src/humpback/processing/region_windowing.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `docs/reference/sequence-models-api.md`

**Acceptance criteria:**
- [ ] The worker derives merge and window-membership geometry from the unpadded Pass 1 region bounds, applying `pad_seconds` exactly once.
- [ ] `is_in_pad` reflects whether the window center falls inside the original unpadded source region extent, not the padded extent.
- [ ] `source_region_ids` are preserved in parquet rows using the project’s real UUID string identifiers rather than silently dropping them.
- [ ] The manifest and API documentation remain accurate for the updated parquet schema and provenance fields.

**Tests needed:**
- Extend processing and worker tests to cover regions with distinct raw and padded extents, asserting correct merged span bounds, correct `is_in_pad` classification, and preserved string `source_region_ids`.

---

### Task 2: Wire the worker to a real production embedder path

**Files:**
- Modify: `src/humpback/workers/continuous_embedding_worker.py`
- Modify: `src/humpback/services/continuous_embedding_service.py`
- Modify: `src/humpback/config.py` or the relevant model/audio integration modules if needed
- Modify: `docs/reference/sequence-models-api.md`

**Acceptance criteria:**
- [ ] `run_one_iteration()` can execute a real continuous embedding job without requiring a test-only injected embedder.
- [ ] The default embedder resolves the SurfPerch model through existing runtime infrastructure and produces one embedding per expected window for supported hydrophone jobs.
- [ ] Unsupported source shapes are rejected clearly before queueing or execution if the producer cannot safely process them.
- [ ] Worker failure handling remains atomic and cancellation-safe after production wiring is added.

**Tests needed:**
- Add or extend worker tests to exercise the production path via monkeypatched model/audio dependencies rather than the injected stub only.
- Add service or API coverage for any new validation introduced around supported source types.

---

### Task 3: Make idempotent creation concurrency-safe

**Files:**
- Modify: `alembic/versions/057_continuous_embedding_jobs.py`
- Modify: `src/humpback/services/continuous_embedding_service.py`
- Modify: `tests/services/test_continuous_embedding_service.py`
- Modify: `tests/integration/test_sequence_models_api.py`

**Acceptance criteria:**
- [ ] **Pre-migration backup of production DB performed.** Read `HUMPBACK_DATABASE_URL` from `.env`, copy the SQLite file to `<original_path>.YYYY-MM-DD-HH:mm.bak` using a UTC timestamp, and confirm the backup file exists with non-zero size before running any migration command.
- [ ] The database enforces the intended one-row-per-signature invariant for active or completed continuous embedding jobs.
- [ ] Service-level create logic remains idempotent when two matching requests arrive concurrently, returning the canonical row instead of creating duplicates.
- [ ] Existing completed job reuse and in-flight job reuse behavior remains intact.

**Tests needed:**
- Add service coverage that simulates a duplicate-insert race or integrity error and verifies the service returns the canonical row.
- Extend integration coverage for duplicate create behavior after the persistence change.

---

### Task 4: Verify the repaired producer end to end

**Files:**
- Modify: `tests/workers/test_continuous_embedding_worker.py`
- Modify: `tests/processing/test_region_windowing.py`
- Modify: `tests/integration/test_sequence_models_api.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`

**Acceptance criteria:**
- [ ] The updated tests cover the four reviewed regressions directly.
- [ ] Any schema or manifest changes are reflected in unit and integration assertions.
- [ ] The verification commands for the touched backend and frontend code pass.

**Tests needed:**
- Run the repo verification commands listed below, including frontend type-checking if frontend files change.

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/workers/continuous_embedding_worker.py src/humpback/processing/region_windowing.py src/humpback/schemas/sequence_models.py src/humpback/services/continuous_embedding_service.py alembic/versions/057_continuous_embedding_jobs.py tests/workers/test_continuous_embedding_worker.py tests/processing/test_region_windowing.py tests/services/test_continuous_embedding_service.py tests/integration/test_sequence_models_api.py tests/unit/test_sequence_models_schemas.py docs/reference/sequence-models-api.md`
2. `uv run ruff check src/humpback/workers/continuous_embedding_worker.py src/humpback/processing/region_windowing.py src/humpback/schemas/sequence_models.py src/humpback/services/continuous_embedding_service.py alembic/versions/057_continuous_embedding_jobs.py tests/workers/test_continuous_embedding_worker.py tests/processing/test_region_windowing.py tests/services/test_continuous_embedding_service.py tests/integration/test_sequence_models_api.py tests/unit/test_sequence_models_schemas.py`
3. `uv run pyright src/humpback/workers/continuous_embedding_worker.py src/humpback/processing/region_windowing.py src/humpback/schemas/sequence_models.py src/humpback/services/continuous_embedding_service.py`
4. `uv run pytest tests/processing/test_region_windowing.py tests/services/test_continuous_embedding_service.py tests/unit/test_sequence_models_schemas.py tests/integration/test_sequence_models_api.py tests/workers/test_continuous_embedding_worker.py`
5. `uv run pytest tests/`
6. `cd frontend && npx tsc --noEmit`
