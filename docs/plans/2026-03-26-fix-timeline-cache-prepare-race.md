# Timeline Cache Prepare Race Fix Implementation Plan

**Goal:** Eliminate `FileNotFoundError` during timeline tile preparation by preventing overlapping same-job prepare runs and hardening cache writes against concurrent writers.
**Root cause:** The worker completion hook and the timeline `POST /prepare` endpoint can both call `_prepare_tiles_sync()` for the same job. The current `_preparing` guard is process-local, and `TimelineTileCache.put()` stages writes through a shared `tile_NNNN.tmp` path, so concurrent writers can remove each other's temp file before `os.replace()`.
**Spec:** Bug-fix workflow; no standalone design spec per `AGENTS.md`.

---

### Task 1: Harden timeline cache writes against concurrent writers

**Files:**
- Modify: `src/humpback/processing/timeline_cache.py`
- Modify: `tests/unit/test_timeline_cache.py`

**Acceptance criteria:**
- [x] `TimelineTileCache.put()` uses a unique temp file per write attempt instead of a shared `tile_NNNN.tmp` path
- [x] Concurrent writes to the same tile do not raise `FileNotFoundError`
- [x] Successful writes still leave the final tile in the expected cache path and do not leave stray temp files behind
- [x] Job access sentinels continue to update for successful writes

**Tests needed:**
- Add a regression test that two cache instances can race on the same tile path without shared-temp collisions
- Keep the existing atomic-write and no-stray-temp coverage passing after the write-path change

---

### Task 2: Prevent overlapping prepare runs for the same job across processes

**Files:**
- Modify: `src/humpback/api/routers/timeline.py`
- Modify: `src/humpback/workers/classifier_worker.py`
- Modify: `src/humpback/processing/timeline_cache.py`
- Modify: `tests/integration/test_timeline_api.py`

**Acceptance criteria:**
- [x] The worker pre-render path and the API `POST /timeline/prepare` path coordinate so the same job is not prepared twice at the same time across processes
- [x] The coordination mechanism is external to the module-level `_preparing` set so it works between `api.1` and `worker.1`
- [x] Repeated prepare requests while a job is already being prepared remain idempotent and do not fail the request
- [x] Cache progress and tile availability still behave correctly when a prepare is already in flight

**Tests needed:**
- Add regression coverage for repeated prepare triggers on the same job while tiles are uncached
- Verify the prepare endpoint still returns a stable success response when preparation is already underway

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/processing/timeline_cache.py src/humpback/api/routers/timeline.py src/humpback/workers/classifier_worker.py tests/unit/test_timeline_cache.py tests/integration/test_timeline_api.py`
2. `uv run ruff check src/humpback/processing/timeline_cache.py src/humpback/api/routers/timeline.py src/humpback/workers/classifier_worker.py tests/unit/test_timeline_cache.py tests/integration/test_timeline_api.py`
3. `uv run pyright src/humpback/processing/timeline_cache.py src/humpback/api/routers/timeline.py src/humpback/workers/classifier_worker.py tests/unit/test_timeline_cache.py tests/integration/test_timeline_api.py`
4. `uv run pytest tests/`
