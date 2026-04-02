# Autoresearch Detection Job Integration — Implementation Plan

**Goal:** Extend autoresearch manifest generator and train/eval pipeline to incorporate labeled and unlabeled detection job windows as training data and hard negatives.
**Spec:** [docs/specs/2026-04-01-autoresearch-detection-jobs-design.md](../specs/2026-04-01-autoresearch-detection-jobs-design.md)

---

### Task 1: Extend manifest generator for detection jobs

**Files:**
- Modify: `scripts/autoresearch/generate_manifest.py`

**Acceptance criteria:**
- [ ] New `--detection-job-ids` CLI flag (comma-separated UUIDs)
- [ ] New `--score-range` CLI flag (default: `0.5,0.995`)
- [ ] At least one of `--job-ids` or `--detection-job-ids` must be provided
- [ ] Queries database to verify each detection job has `has_positive_labels=True`; raises error if not
- [ ] Reads `detection_embeddings.parquet` and `detection_rows.parquet` for each detection job
- [ ] Matches rows between the two files by positional index
- [ ] Classifies windows: humpback=1 → label 1; ship/background=1 → label 0 with semantic negative_group; unlabeled within score range → label 0 with score-band negative_group; unlabeled outside score range → excluded
- [ ] Score band boundaries: [0.50, 0.90), [0.90, 0.95), [0.95, 0.99), [0.99, score_max)
- [ ] ID scheme: `det{job_id[:8]}_row{index}`
- [ ] `source_type: "detection_job"` on detection examples
- [ ] `detection_confidence` field stores original classifier score
- [ ] `audio_file_id` set from detection embeddings `filename` column
- [ ] Split by filename using same seeded shuffle as embedding set splits
- [ ] Metadata includes `detection_job_ids` and `score_range`
- [ ] Existing `--job-ids` behavior unchanged; embedding set examples gain `source_type: "embedding_set"`

**Tests needed:**
- Mock DB with detection job (`has_positive_labels=True`), synthetic detection Parquets, verify label classification and score-band grouping
- Verify score-range filtering excludes out-of-range windows
- Verify rejection when `has_positive_labels` is not true
- Verify mixed manifest with both embedding set and detection sources

---

### Task 2: Schema-aware Parquet loading in train_eval

**Files:**
- Modify: `scripts/autoresearch/train_eval.py`

**Acceptance criteria:**
- [ ] `_load_parquet_cache` auto-detects Parquet format: checks for `row_index` column (embedding set) vs `filename` column (detection)
- [ ] Detection format: reads `embedding` column, generates positional indices `[0, 1, 2, ...]`, also stores `filename` per row
- [ ] Embedding set format: unchanged behavior
- [ ] `_build_embedding_lookup` works with both formats via the unified cache
- [ ] Context pooling for detection windows: skips neighbor if neighbor's filename differs from current row's filename (falls back to center)
- [ ] Backward compatible: manifests without `source_type` field work as before

**Tests needed:**
- Write detection-format Parquet, verify auto-detection and positional indexing
- Context pooling with detection Parquet: verify cross-file neighbors are skipped
- Mixed manifest with both source types: verify train_eval runs end-to-end

---

### Task 3: Tests for detection job integration

**Files:**
- Modify: `tests/unit/test_autoresearch.py`
- Modify: `tests/integration/test_autoresearch.py`

**Acceptance criteria:**
- [ ] Unit test: manifest generation with detection job — mock DB, synthetic detection_embeddings.parquet and detection_rows.parquet, verify label classification, score-band groups, score-range filtering
- [ ] Unit test: detection job without `has_positive_labels` is rejected
- [ ] Unit test: detection Parquet auto-detection in `_load_parquet_cache`
- [ ] Unit test: context pooling with cross-file detection neighbors falls back to center
- [ ] Integration test: mixed-source manifest (embedding set + detection), 3 search trials, verify grouped metrics include detection score-band groups
- [ ] All existing tests continue to pass

**Tests needed:**
- This task IS the tests

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check scripts/autoresearch/`
2. `uv run ruff check scripts/autoresearch/`
3. `uv run pyright scripts/autoresearch/`
4. `uv run pytest tests/unit/test_autoresearch.py tests/integration/test_autoresearch.py`
5. `uv run pytest tests/` (full suite, ensure no regressions)
