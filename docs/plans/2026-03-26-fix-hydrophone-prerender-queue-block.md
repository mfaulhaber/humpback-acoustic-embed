# Hydrophone Timeline Pre-render Queue Unblock Implementation Plan

**Goal:** Keep queued detection work flowing by preventing best-effort hydrophone timeline pre-render from blocking the single worker after a detection job completes.
**Root cause:** `run_hydrophone_detection_job()` marks a job `complete` and then awaits coarse timeline tile pre-render inline. The UI shows no active detection job, but the worker remains occupied and cannot claim the next queued job until pre-render finishes.
**Spec:** Bug-fix workflow; no standalone design spec per `AGENTS.md`.

---

### Task 1: Move worker timeline pre-render off the detection critical path

**Files:**
- Modify: `src/humpback/workers/classifier_worker.py`

**Acceptance criteria:**
- [ ] Hydrophone detection jobs still persist `result_summary`, alerts, and terminal status before returning success
- [ ] Timeline pre-render remains best-effort and still uses the existing prepare-lock coordination
- [ ] The worker no longer waits for timeline pre-render before it can claim the next queued job
- [ ] `timeline_tiles_ready` is still set when background pre-render finishes successfully

**Tests needed:**
- Add regression coverage that the hydrophone job runner returns without awaiting slow timeline pre-render work
- Verify failures in background pre-render are logged without failing the detection job

---

### Task 2: Add regression coverage for queue visibility and readiness updates

**Files:**
- Modify: `tests/unit/test_classifier_worker.py`

**Acceptance criteria:**
- [ ] Test coverage proves hydrophone job completion does not block on a slow pre-render helper
- [ ] Test coverage proves successful background pre-render marks `timeline_tiles_ready=True`
- [ ] Test coverage proves pre-render errors do not revert the detection job from `complete`

**Tests needed:**
- Add focused async unit tests around the hydrophone completion path with mocked pre-render behavior

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/workers/classifier_worker.py tests/unit/test_classifier_worker.py`
2. `uv run ruff check src/humpback/workers/classifier_worker.py tests/unit/test_classifier_worker.py`
3. `uv run pyright src/humpback/workers/classifier_worker.py tests/unit/test_classifier_worker.py`
4. `uv run pytest tests/unit/test_classifier_worker.py tests/unit/test_queue.py`
