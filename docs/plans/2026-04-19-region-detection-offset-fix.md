# Region Detection Offset Fix — Implementation Plan

**Goal:** Fix the `time_offset_sec` bug in the region detection worker that causes incorrect trace timestamps when `iter_audio_chunks` yields multiple buffers per chunk due to HLS discontinuities.
**Severity:** Data integrity — all hydrophone-sourced region detection jobs produce incorrect trace data and therefore inaccurate region boundaries.

---

### Bug Summary

In `src/humpback/workers/region_detection_worker.py` line 350, the detector calls `score_audio_windows` with `time_offset_sec=chunk_start - start_ts` for every audio buffer yielded by `iter_audio_chunks`. When a chunk has HLS discontinuities, `iter_audio_chunks` yields multiple buffers per chunk (flushing its accumulator at each gap). Each buffer is scored starting from the chunk's start offset, not the buffer's actual start time.

**Result:** Duplicate/overlapping trace timestamps within each chunk. For example, in job `96dff08a`:
- Chunk 4 (7200-9000s): first buffer covers 7200-7849, second buffer covers 7200-8340
- The first buffer's scores for 7200-7849 are overwritten by incorrect scores from different audio
- 23k duplicate rows across the full job trace
- Regions are merged across gaps that shouldn't be bridged; region boundaries are wrong

### Root Cause

`iter_audio_chunks` yields `(audio_buf, seg_start_utc, segs_done, segs_total)`. The `seg_start_utc` is the actual UTC datetime of the buffer's start, but the worker ignores it (stores as `_seg_start_utc`) and hardcodes `time_offset_sec=chunk_start - start_ts`.

---

### Task 1: Fix time_offset_sec to use buffer start epoch

**Files:**
- Modify: `src/humpback/workers/region_detection_worker.py`

**Acceptance criteria:**
- [ ] Rename `_seg_start_utc` to `seg_start_utc` in the `iter_audio_chunks` unpacking
- [ ] Compute `time_offset_sec` from `seg_start_utc` instead of `chunk_start`: `time_offset_sec = seg_start_utc.timestamp() - start_ts`
- [ ] No change to the file-based path (`_load_file_trace`) — it processes the whole file in one call with `time_offset_sec=0.0`, which is correct

**Tests needed:**
- Unit test that simulates two yielded buffers within one chunk (discontinuity scenario) and verifies trace timestamps are non-overlapping and correctly offset
- Unit test that a single continuous buffer still produces correct timestamps

---

### Task 2: Deduplicate existing trace merge logic

**Files:**
- Modify: `src/humpback/workers/region_detection_worker.py`

**Acceptance criteria:**
- [ ] After merging all chunk traces (line ~401: `all_scores = read_all_chunk_traces(...)`), deduplicate by `time_sec` — if two entries share the same timestamp, keep only one (latest chunk wins, or highest score)
- [ ] Log a warning when duplicates are found, including the count, so operators can identify affected jobs

**Tests needed:**
- Unit test: merge with overlapping chunk boundaries produces deduplicated output
- Unit test: merge with no overlaps passes through unchanged

---

### Task 3: Add integration test for discontinuity handling

**Files:**
- Modify: `tests/integration/test_region_detection_worker.py`

**Acceptance criteria:**
- [ ] Test that exercises the hydrophone chunk processing path with a mock provider that has a timeline gap (two non-contiguous segments within a single chunk)
- [ ] Verify the resulting trace has no duplicate timestamps
- [ ] Verify trace timestamps correctly reflect the actual audio positions
- [ ] Verify regions produced from the trace have accurate boundaries

**Tests needed:**
- This task IS the test

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/workers/region_detection_worker.py`
2. `uv run ruff check src/humpback/workers/region_detection_worker.py`
3. `uv run pyright src/humpback/workers/region_detection_worker.py`
4. `uv run pytest tests/unit/test_region_detection_chunks.py tests/integration/test_region_detection_worker.py -v`
5. `uv run pytest tests/`
