# Code Cleanup Phase 4: Split `classifier/detector.py` — Implementation Plan

**Goal:** Extract utility functions from the 1,073-line `classifier/detector.py` into `classifier/detector_utils.py`, keeping the main detection API in `detector.py` with backward-compatible re-exports.
**Spec:** [docs/specs/2026-04-03-code-cleanup-design.md](../specs/2026-04-03-code-cleanup-design.md) (Phase 3d)

---

### Task 1: Create `detector_utils.py` with extracted functions

**Files:**
- Create: `src/humpback/classifier/detector_utils.py`

**What to extract (in source order):**

Utility:
- `_TS_PATTERN` (module-level compiled regex)
- `_file_base_epoch`

Window processing:
- `merge_detection_spans`
- `merge_detection_events`
- `snap_event_bounds`
- `snap_and_merge_detection_events`
- `_smooth_scores`
- `select_peak_windows_from_events`

TSV I/O:
- `TSV_FIELDNAMES`
- `read_detections_tsv`
- `write_detections_tsv`
- `append_detections_tsv`

Diagnostics I/O:
- `WINDOW_DIAGNOSTICS_SCHEMA`
- `_window_diagnostics_table`
- `write_window_diagnostics`
- `write_window_diagnostics_shard`
- `read_window_diagnostics_table`

Embedding I/O:
- `match_embedding_records_to_row_store`
- `write_detection_embeddings`
- `read_detection_embedding`
- `EmbeddingDiffResult`
- `diff_row_store_vs_embeddings`

Audio resolution:
- `_build_file_timeline`
- `resolve_audio_for_window`
- `resolve_audio_for_window_hydrophone`

Move imports needed by these functions (`csv`, `math`, `os`, `re`, `datetime`, `typing.Any`, `numpy`, `pyarrow`, `pyarrow.parquet`) into `detector_utils.py`. The `logging` import stays in both files.

**Acceptance criteria:**
- [ ] `detector_utils.py` exists with all listed functions, constants, and classes
- [ ] Each function body is identical to the original (no behavior changes)
- [ ] `detector_utils.py` passes `uv run pyright`
- [ ] `detector_utils.py` passes `uv run ruff check`

**Tests needed:**
- No new tests — existing tests cover all functions via re-exports

---

### Task 2: Slim down `detector.py` and add re-exports

**Files:**
- Modify: `src/humpback/classifier/detector.py`

**Changes:**
- Remove all function/class/constant bodies that moved to `detector_utils.py`
- Add `from humpback.classifier.detector_utils import (...)` block importing every moved name
- Keep `AUDIO_EXTENSIONS` constant in `detector.py` (not moved)
- Keep `run_detection()` function in `detector.py` (not moved)
- `run_detection()` already calls `_file_base_epoch`, `merge_detection_events`, `snap_and_merge_detection_events`, `select_peak_windows_from_events` — these now come via the import from `detector_utils`
- Remove imports that are no longer needed directly (`csv`, `math`, `os`, `re`, `datetime`), keeping only what `run_detection()` and module-level code need (`logging`, `numpy`, `sklearn.pipeline.Pipeline`, `pathlib.Path`, processing imports)
- Add `__all__` list to make re-exports explicit

**Acceptance criteria:**
- [ ] `detector.py` is under 300 lines (down from 1,073)
- [ ] `detector.py` re-exports every name that was previously importable from it
- [ ] All 40+ existing import sites across the codebase continue working with zero changes
- [ ] `run_detection()` behavior is identical
- [ ] `uv run pyright` passes on `detector.py`

**Tests needed:**
- No new tests — all existing tests validate the re-exports implicitly

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/classifier/detector.py src/humpback/classifier/detector_utils.py`
2. `uv run ruff check src/humpback/classifier/detector.py src/humpback/classifier/detector_utils.py`
3. `uv run pyright src/humpback/classifier/detector.py src/humpback/classifier/detector_utils.py`
4. `uv run pytest tests/`
