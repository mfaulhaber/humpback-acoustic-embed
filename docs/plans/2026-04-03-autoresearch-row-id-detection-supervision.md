# Autoresearch Row-ID Detection Supervision — Implementation Plan

**Goal:** Align autoresearch detection-job ingestion with live production artifacts and manual vocalization-label supervision so production hard-negative experiments are trustworthy.
**Spec:** [docs/specs/2026-04-03-autoresearch-row-id-detection-supervision-design.md](../specs/2026-04-03-autoresearch-row-id-detection-supervision-design.md)

---

### Task 1: Rebuild detection-job manifest generation around live row-id supervision

**Files:**
- Modify: `scripts/autoresearch/generate_manifest.py`

**Acceptance criteria:**
- [ ] Detection-job ingestion supports canonical `detection_embeddings.parquet` files keyed by `row_id`
- [ ] Legacy filename-based detection embeddings remain supported for backward compatibility
- [ ] The generator queries `vocalization_labels` for every requested detection job and groups labels by `row_id`
- [ ] Detection rows are classified with explicit precedence: vocalization-positive, vocalization-negative, binary positive fallback, binary negative fallback, unlabeled score-band fallback
- [ ] Rows with contradictory positive and negative supervision across systems are skipped instead of silently coerced
- [ ] Row-id detection examples include `row_id`, `label_source`, and stable IDs derived from detection job ID plus row ID
- [ ] Row-id detection split groups are derived from UTC-hour buckets built from row-store `start_utc`
- [ ] Unlabeled hard-negative mining only uses rows with non-null confidence inside `--score-range`
- [ ] Manifest metadata includes per-job summary counts for included rows and skipped-row reasons
- [ ] Existing embedding-set-only manifest generation remains unchanged

**Tests needed:**
- Unit tests for row-id detection manifests with vocalization-positive and vocalization-negative labels
- Unit tests for conflict skipping between vocalization labels and row-store labels
- Unit tests for null-confidence unlabeled rows being excluded from score-band negatives
- Unit tests for hourly split-group derivation from `start_utc`
- Unit tests proving legacy filename-based detection manifests still work

---

### Task 2: Teach train/eval to read row-id detection examples safely

**Files:**
- Modify: `scripts/autoresearch/train_eval.py`

**Acceptance criteria:**
- [ ] `_load_parquet_cache` recognizes embedding-set, canonical row-id detection, and legacy filename-based detection Parquet schemas
- [ ] `_build_embedding_lookup` resolves examples by `row_id` when present and by `row_index` otherwise
- [ ] Canonical row-id detection examples load successfully without requiring `filename`
- [ ] Context pooling falls back to center-only for row-id detection examples under `mean3` and `max3`
- [ ] Existing context pooling behavior remains unchanged for embedding sets and legacy filename-based detection files
- [ ] Mixed-source manifests continue to train and evaluate without API changes to the search loop

**Tests needed:**
- Unit tests for row-id cache loading and example lookup
- Unit tests for center-only pooling fallback on row-id detection examples
- Regression tests for unchanged embedding-set behavior
- Integration coverage that mixes embedding sets with row-id detection examples

---

### Task 3: Update autoresearch docs and regression coverage for production behavior

**Files:**
- Modify: `scripts/autoresearch/README.md`
- Modify: `tests/unit/test_autoresearch.py`
- Modify: `tests/integration/test_autoresearch.py`

**Acceptance criteria:**
- [ ] README describes the live row-id detection embedding schema and the role of `vocalization_labels`
- [ ] README documents `"(Negative)"` as explicit hard-negative supervision for autoresearch manifests
- [ ] README documents the non-null-confidence requirement for unlabeled score-band negatives
- [ ] README documents the hourly split-group strategy for row-id detection jobs
- [ ] Unit tests cover production-like supervision mixes drawn from row store plus vocalization labels
- [ ] Integration tests verify output artifacts still serialize correctly for row-id detection manifests
- [ ] Existing autoresearch tests continue to pass after the new coverage is added

**Tests needed:**
- This task is the documentation and regression test work

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check scripts/autoresearch/ tests/unit/test_autoresearch.py tests/integration/test_autoresearch.py`
2. `uv run ruff check scripts/autoresearch/ tests/unit/test_autoresearch.py tests/integration/test_autoresearch.py`
3. `uv run pyright scripts/autoresearch/`
4. `uv run pytest tests/unit/test_autoresearch.py tests/integration/test_autoresearch.py`
5. `uv run pytest tests/`
