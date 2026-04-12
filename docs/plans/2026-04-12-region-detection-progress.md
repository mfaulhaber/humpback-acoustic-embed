# Region Detection Progress Tracking Implementation Plan

**Goal:** Add chunk artifacts, manifest, resume logic, progress columns, structured logging, and a benchmark script to the hydrophone region detection worker.
**Spec:** [docs/specs/2026-04-12-region-detection-progress-design.md](../specs/2026-04-12-region-detection-progress-design.md)

---

### Task 1: Database Schema — Migration, Model, and API Schema

**Files:**
- Create: `alembic/versions/045_region_detection_progress_columns.py`
- Modify: `src/humpback/models/call_parsing.py`
- Modify: `src/humpback/schemas/call_parsing.py`

**Acceptance criteria:**
- [ ] Alembic migration adds `chunks_total` (nullable Integer) and `chunks_completed` (nullable Integer) to `region_detection_jobs` using `batch_alter_table`
- [ ] `RegionDetectionJob` model has both new `Mapped[Optional[int]]` columns
- [ ] `RegionDetectionJobSummary` schema exposes both fields as `Optional[int]`
- [ ] Migration applies cleanly: `uv run alembic upgrade head`

**Tests needed:**
- Migration round-trip (upgrade/downgrade) against a scratch DB

---

### Task 2: Manifest and Chunk I/O Helpers

**Files:**
- Modify: `src/humpback/call_parsing/storage.py`

**Acceptance criteria:**
- [ ] `write_manifest(job_dir, manifest_dict)` writes `manifest.json` atomically (write tmp, rename)
- [ ] `read_manifest(job_dir)` reads and returns the manifest dict, or `None` if missing
- [ ] `update_manifest_chunk(job_dir, chunk_index, update_dict)` reads manifest, updates the specified chunk entry, writes atomically
- [ ] `write_chunk_trace(job_dir, chunk_index, window_scores)` writes `chunks/{index:04d}.parquet` atomically using the existing `TRACE_SCHEMA`
- [ ] `read_all_chunk_traces(job_dir, total_chunks)` reads all chunk parquets in order, returns concatenated list of `WindowScore`
- [ ] `chunk_parquet_path(job_dir, chunk_index)` returns the path for a given chunk index
- [ ] All writes use the existing `_atomic_write_parquet` pattern (tmp + rename)

**Tests needed:**
- Unit tests for manifest write/read/update round-trip
- Unit tests for chunk parquet write/read/concatenate
- Test that missing chunk parquet is detected (for resume verification)

---

### Task 3: Worker Refactor — Chunk Artifacts, Resume, Progress, and Logging

**Files:**
- Modify: `src/humpback/workers/region_detection_worker.py`

**Acceptance criteria:**
- [ ] Hydrophone path computes chunk edges, then writes initial manifest with all chunks `pending` and sets `chunks_total` in DB
- [ ] Each chunk iteration: fetch audio, score windows, write chunk parquet atomically, update manifest entry (status, completed_at, trace_rows, elapsed_sec), increment `chunks_completed` in DB
- [ ] Resume logic: if manifest exists on job start, verify each "complete" chunk has its parquet on disk; reset missing ones to `pending`; skip verified-complete chunks; update `chunks_completed` in DB to match
- [ ] Final merge: read all chunk parquets in order, concatenate, run hysteresis merge, write `trace.parquet` and `regions.parquet`
- [ ] Structured INFO-level logging: job start (chunks, range, hydrophone), per-chunk fetch/score with timing and rate, resume indicator, merge summary, completion summary
- [ ] `_cleanup_partial_artifacts` updated to also clean up `chunks/` subdirectory and `manifest.json` on failure (only final artifacts and temps — completed chunk parquets are preserved for resume)
- [ ] File-based path (`audio_file_id`) unchanged — no manifest, no chunks, no progress columns

**Tests needed:**
- Unit test: cold-start hydrophone path writes manifest and chunk parquets in correct layout
- Unit test: resume skips completed chunks and picks up from first pending
- Unit test: resume resets chunk to pending when parquet is missing
- Unit test: final merge concatenates chunks and produces correct trace/regions
- Integration test: progress columns are updated during processing

---

### Task 4: Benchmark Script

**Files:**
- Create: `scripts/benchmark_region_detection.py`

**Acceptance criteria:**
- [ ] CLI accepts: hydrophone ID, start/end timestamps (UTC), model config ID, classifier model ID, optional duration override (default 10 minutes)
- [ ] Fetches audio via `iter_audio_chunks` (same path as the worker)
- [ ] Runs `score_audio_windows` and times audio fetch vs scoring separately
- [ ] Reports: total time, per-minute-of-audio rate, window count, extrapolated 24h estimate
- [ ] Does not write any DB rows or job artifacts
- [ ] Passes pyright type checking

**Tests needed:**
- No automated tests (manual benchmark tool), but must pass pyright

---

### Task 5: ADR-052 and Documentation Updates

**Files:**
- Modify: `DECISIONS.md`
- Modify: `CLAUDE.md`

**Acceptance criteria:**
- [ ] ADR-052 appended to DECISIONS.md documenting the decision to restrict chunk artifacts to the hydrophone path only
- [ ] CLAUDE.md §8.7 behavioral constraints updated with chunk artifact crash/resume semantics
- [ ] CLAUDE.md §9.2 latest migration updated to `045`
- [ ] CLAUDE.md §9.1 updated to mention progress tracking and resume capability for region detection

**Tests needed:**
- None (documentation only)

---

### Task 6: Tests

**Files:**
- Modify: `tests/unit/test_call_parsing_workers.py`
- Modify: `tests/integration/test_region_detection_worker.py`

**Acceptance criteria:**
- [ ] Unit tests for all manifest and chunk I/O helpers from Task 2
- [ ] Unit tests for worker resume logic from Task 3 (cold start, resume with all chunks present, resume with missing chunk parquet)
- [ ] Unit tests for final merge producing correct concatenated trace and regions
- [ ] Integration test verifying `chunks_total`/`chunks_completed` columns are populated on a hydrophone job
- [ ] All existing region detection tests still pass

**Tests needed:**
- This task IS the tests

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/models/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/call_parsing/storage.py src/humpback/workers/region_detection_worker.py scripts/benchmark_region_detection.py`
2. `uv run ruff check src/humpback/models/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/call_parsing/storage.py src/humpback/workers/region_detection_worker.py scripts/benchmark_region_detection.py`
3. `uv run pyright src/humpback/models/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/call_parsing/storage.py src/humpback/workers/region_detection_worker.py scripts/benchmark_region_detection.py`
4. `uv run pytest tests/`
