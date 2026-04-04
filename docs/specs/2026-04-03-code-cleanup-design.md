# Code Cleanup: Dead Code, Deduplication, and File Splitting

**Date:** 2026-04-03
**Status:** Approved

## Goal

Reduce codebase complexity through three phased efforts: removing dead code, extracting duplicated patterns into shared utilities, and splitting oversized files into focused modules. All changes are mechanical refactors with no behavior changes.

## Approach

Approach C: one PR per phase for phases 1-2, one PR per split target for phase 3. Each PR is independently reviewable and revertible. Frontend splitting is deferred to the backlog.

## Phase 1: Dead Code Removal (1 PR)

### Targets

| File | Lines | Reason |
|------|-------|--------|
| `frontend/src/components/vocalization/VocalizationResultsBrowser.tsx` | 295 | Not imported anywhere |
| `frontend/src/components/vocalization/VocalizationInferenceForm.tsx` | 284 | Not imported anywhere |
| `tests/test_search_service.py` | 278 | Orphan at root; consolidate into `tests/unit/test_search_service.py` |
| `tests/test_clustering_service.py` | 72 | Orphan at root; consolidate into `tests/unit/` |
| `scripts/noaa_gcs_poc.py` | 260 | POC superseded by production `providers/noaa_gcs.py` |

### Process

- For orphan test files: diff test cases against the existing `tests/unit/` counterparts. Move any non-duplicate test cases into the unit test file before deleting the root-level file.
- For frontend components: verify no dynamic imports or lazy loading references before deletion.
- For the POC script: verify no other scripts or docs reference it.

### Verification

- `uv run pytest tests/`
- `cd frontend && npx tsc --noEmit`

## Phase 2: Utility Extraction / Deduplication (1 PR)

### 2a. Parquet Utilities → `src/humpback/processing/parquet_utils.py`

Extract repeated parquet read/write patterns (~21 call sites) into shared functions:

- `read_embeddings(path: Path) -> np.ndarray` — the common `pq.read_table` + `np.array([row.as_py() for row in table["embedding"]])` pattern
- `read_table_with_embeddings(path: Path) -> tuple[pa.Table, np.ndarray]` — returns both the table and extracted embedding array
- `write_embeddings_table(path: Path, table: pa.Table) -> None` — atomic write with temp-file promotion

Placed in `processing/` alongside existing `parquet_writer.py`. Not every parquet call will consolidate — domain-specific schema handling stays in place. The target is the mechanical read pattern that repeats identically.

### 2b. Job Status Helpers → `src/humpback/workers/job_helpers.py`

Extract the try/except/mark-complete/mark-failed pattern from 8+ worker locations:

- `mark_complete(session, model_class, job_id, summary=None)` — sets status to "complete", updated_at to now(UTC), optional summary JSON
- `mark_failed(session, model_class, job_id, error: str)` — sets status to "failed", error_message, updated_at

Workers call these instead of duplicating the inline `update().where().values()` + commit pattern. Preserves existing status transition semantics exactly.

### 2c. Schema Converters → `src/humpback/schemas/converters.py`

Move the ~14 model-to-Pydantic conversion functions out of router files into a shared module:

- `_model_to_out()`, `_job_to_out()`, `_detection_job_to_out()`, etc. from `api/routers/classifier.py`
- Similar converters from `api/routers/vocalization.py`, `api/routers/clustering.py`, etc.

Routers import from `schemas.converters`. This also simplifies Phase 3 splitting since routers won't each need copies.

### Verification

- `uv run pytest tests/`
- `uv run pyright src/humpback/processing/parquet_utils.py src/humpback/workers/job_helpers.py src/humpback/schemas/converters.py`
- Verify no behavior changes via existing test coverage

## Phase 3: Backend File Splitting (3-5 PRs)

### 3a. Split `api/routers/classifier.py` (1,848 lines → package)

```
api/routers/classifier/
├── __init__.py          # re-exports combined router via include_router()
├── models.py            # model CRUD endpoints
├── training.py          # training job endpoints
├── detection.py         # detection job endpoints, labels, row edits
├── autoresearch.py      # autoresearch candidate endpoints
├── hydrophone.py        # hydrophone detection endpoints
└── embeddings.py        # detection embedding job endpoints
```

`__init__.py` composes sub-routers so `api/app.py` import path does not change.

### 3b. Split `workers/classifier_worker.py` (1,483 lines → package)

```
workers/classifier/
├── __init__.py          # re-exports job runner functions
├── training.py          # run_training_job
├── detection.py         # run_detection_job, run_extraction_job
└── hydrophone.py        # run_hydrophone_detection_job + helpers
```

`queue.py` dispatches to the same function names via the package `__init__`.

### 3c. Split `services/classifier_service.py` (1,483 lines → package)

```
services/classifier/
├── __init__.py          # re-exports public API
├── models.py            # model CRUD
├── training.py          # training job management
├── detection.py         # detection job management
├── autoresearch.py      # candidate import/promotion
└── hydrophone.py        # hydrophone job management
```

### 3d. Split `classifier/detector.py` (1,073 lines) — Optional

Only if 3a-3c complete cleanly. Extract window processing and diagnostic collection into `classifier/detector_utils.py`. The main `detector.py` retains the public API; internal helpers move to the utils module.

### Splitting Rules (all sub-phases)

- `__init__.py` re-exports preserve the public API — most callers don't change
- Existing tests must pass without test rewrites (only import path changes if tests import internal functions directly)
- Each split is one PR: split target + all import updates + test adjustments
- No behavior changes, no new functionality

### Verification (per PR)

- `uv run pytest tests/`
- `uv run pyright` on modified files
- `cd frontend && npx tsc --noEmit` (if any frontend imports changed)

## Backlog (Out of Scope)

- Frontend file splitting (`HydrophoneTab.tsx`, `TrainingTab.tsx`, `types.ts`, `client.ts`)
- Further classifier module splits (`extractor.py`, `s3_stream.py`, `detection_rows.py`, `label_processor.py`)
- Test file splitting (large integration/unit test files)
- Script cleanup beyond POC deletion

## Phase Dependencies

```
Phase 1 (dead code) → Phase 2 (utilities) → Phase 3a (router) → 3b (worker) → 3c (service) → 3d (detector, optional)
```

Each phase depends on the previous completing. Phase 2 converters extraction simplifies Phase 3a router splitting. Phase 3 sub-phases are ordered by decreasing external surface area (router → worker → service → internal module).

## Risk

All phases are mechanical refactoring. Primary risk is missed import updates causing runtime errors. Mitigated by:
- Full test suite runs after each change
- Pyright type checking catches broken imports statically
- `__init__.py` re-exports mean most callers never see the split
