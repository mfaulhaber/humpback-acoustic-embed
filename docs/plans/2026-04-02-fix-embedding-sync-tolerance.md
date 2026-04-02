# Fix Embedding Sync Tolerance Bug

**Goal:** Fix the detection embedding sync infinite loop caused by fractional-second timestamps failing the strict tolerance check, and fix the sync worker to preserve sub-second precision in synthetic embedding records.

**Bug report:** Without edits in timeline view, "Sync Embeddings" shows "11 added, 11 removed" and resyncing never resolves it. Detection job `2a5f51f3-b91d-470e-a92e-4900ebedb97d`.

**Root cause analysis:**
- 11 detection rows have `.5` fractional-second `start_utc` values (e.g., `1635756141.5`)
- These had no embeddings from original detection (likely chunk-boundary edge case)
- Sync worker creates synthetic filenames by truncating `start_utc` to integer seconds (`20211101T084221Z.wav`) and sets `record_start_sec = 0.0`
- Reconstructed UTC = `1635756141.0` vs row-store `1635756141.5` → delta of exactly `0.5`
- Tolerance check `abs(delta) < 0.5` is strict less-than, so `0.5 < 0.5` → false → never matches
- Each resync removes the old bad embeddings and creates new identically-bad ones → infinite loop

---

### Task 1: Fix tolerance comparison to use `<=`

**Files:**
- Modify: `src/humpback/classifier/detector.py`

**Acceptance criteria:**
- [ ] `diff_row_store_vs_embeddings` uses `<=` instead of `<` for the tolerance check on line 890
- [ ] Deltas of exactly `_SYNC_TOLERANCE_SEC` are treated as matches

**Tests needed:**
- Unit test for `diff_row_store_vs_embeddings` with row-store and embedding pairs that differ by exactly the tolerance value

---

### Task 2: Fix sync worker to preserve fractional seconds in synthetic embedding records

**Files:**
- Modify: `src/humpback/workers/detection_embedding_worker.py`

**Acceptance criteria:**
- [ ] Hydrophone path (lines ~334-340): compute `record_start_sec = start_utc - base_epoch` instead of `0.0`
- [ ] Audio folder fallback path (lines ~356-362): same fix
- [ ] Generic fallback path (lines ~364-369): same fix
- [ ] The `base_epoch` is derived from the synthetic filename via `_file_base_epoch` to ensure round-trip consistency

**Tests needed:**
- Unit test verifying sync-created embedding records round-trip through `_embedding_utc_pairs` to match the original row-store UTC values, including fractional-second timestamps

---

### Task 3: Add regression test

**Files:**
- Modify: `tests/integration/test_embedding_sync_worker.py` (or appropriate test file)

**Acceptance criteria:**
- [ ] Test that a row store with `.5`-second timestamps produces embeddings that pass the diff after sync
- [ ] Test that running sync twice on such data produces "already in sync" on the second run

**Tests needed:**
- Integration test exercising the full sync → re-diff cycle with fractional-second edge case data

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/classifier/detector.py src/humpback/workers/detection_embedding_worker.py`
2. `uv run ruff check src/humpback/classifier/detector.py src/humpback/workers/detection_embedding_worker.py`
3. `uv run pyright src/humpback/classifier/detector.py src/humpback/workers/detection_embedding_worker.py`
4. `uv run pytest tests/`
