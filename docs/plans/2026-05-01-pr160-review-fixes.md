# PR 160 Review Fixes Implementation Plan

**Goal:** Fix the masked-transformer worker recovery, per-k artifact retry, and motif k-scoping issues found during review of PR #160.
**Spec:** Bug fix from review findings; no design spec required.

---

### Task 1: Recover Stale Masked-Transformer Jobs

**Files:**
- Modify: `src/humpback/workers/queue.py`
- Modify: `tests/workers/test_masked_transformer_worker.py` or relevant queue recovery test file

**Acceptance criteria:**
- [x] `recover_stale_jobs()` imports `MaskedTransformerJob`.
- [x] Stale `masked_transformer_jobs.status == "running"` rows older than the stale timeout are reset to `queued`.
- [x] Recovery count includes masked-transformer rows.
- [x] A regression test covers stale masked-transformer recovery.

**Tests needed:**
- Targeted pytest coverage for `recover_stale_jobs()` with a running masked-transformer job older than the stale timeout.

---

### Task 2: Treat Per-k Bundles as Done Only When All Required Artifacts Exist

**Files:**
- Modify: `src/humpback/workers/masked_transformer_worker.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`

**Acceptance criteria:**
- [x] `_previously_done_k()` requires all six per-k artifacts before skipping a k value.
- [x] Requeued jobs regenerate missing interpretation artifacts when only `decoded.parquet`, `kmeans.joblib`, and `run_lengths.json` exist.
- [x] Regression tests cover partial per-k artifact recovery.

**Tests needed:**
- Targeted worker pytest that creates a completed tokenization-only k directory, requeues the job, and asserts missing overlay, exemplars, and label distribution are generated.

---

### Task 3: Scope Masked-Transformer Motif Jobs by Selected k

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `src/humpback/services/motif_extraction_service.py`
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/MotifExtractionPanel.tsx`
- Modify: relevant backend and frontend tests

**Acceptance criteria:**
- [x] Motif extraction list filtering accepts `k`.
- [x] Backend service filters masked-transformer motif jobs by `k` when provided.
- [x] Frontend motif API includes `k` in list query parameters.
- [x] `MotifExtractionPanel` passes the selected masked-transformer `k` when listing motif jobs.
- [x] Regression tests cover same masked-transformer job with different k values.

**Tests needed:**
- Backend service or integration pytest for `k` filtering.
- Frontend test coverage for masked-transformer motif panel query scoping if an existing suitable test harness exists.

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/workers/queue.py src/humpback/workers/masked_transformer_worker.py src/humpback/api/routers/sequence_models.py src/humpback/services/motif_extraction_service.py tests/workers/test_masked_transformer_worker.py tests/services/test_motif_extraction_service.py tests/integration/test_masked_transformer_api.py`
2. `uv run ruff check src/humpback/workers/queue.py src/humpback/workers/masked_transformer_worker.py src/humpback/api/routers/sequence_models.py src/humpback/services/motif_extraction_service.py tests/workers/test_masked_transformer_worker.py tests/services/test_motif_extraction_service.py tests/integration/test_masked_transformer_api.py`
3. `uv run pyright src/humpback/workers/queue.py src/humpback/workers/masked_transformer_worker.py src/humpback/api/routers/sequence_models.py src/humpback/services/motif_extraction_service.py`
4. `uv run pytest tests/workers/test_masked_transformer_worker.py tests/services/test_motif_extraction_service.py tests/integration/test_masked_transformer_api.py`
5. `cd frontend && npx tsc --noEmit`
