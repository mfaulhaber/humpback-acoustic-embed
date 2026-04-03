# Autoresearch Replay Parity Implementation Plan

**Goal:** Enable exact replay of AR-v1 autoresearch candidates through production promotion by extracting shared training logic, adding a candidate-replay training path, and verifying produced models against imported candidate metrics.

**Spec:** [docs/specs/2026-04-03-autoresearch-replay-parity-design.md](../specs/2026-04-03-autoresearch-replay-parity-design.md)

---

### Task 1: Extract shared replay module from train_eval.py

**Files:**
- Create: `src/humpback/classifier/replay.py`
- Modify: `scripts/autoresearch/train_eval.py`

**Acceptance criteria:**
- [ ] `replay.py` contains extracted functions: `apply_context_pooling`, `build_feature_pipeline`, `apply_transforms`, `collect_split_arrays`, `build_classifier`, `apply_calibration`, `score_classifier`, `compute_metrics`, `evaluate_on_split` (wrapping score + metrics), `build_replay_pipeline` (top-level: normalization -> PCA -> classifier -> calibration, returns fitted Pipeline + EffectiveConfig)
- [ ] `replay.py` contains supporting types: `ParquetCacheEntry`, `EffectiveConfig` (dataclass), `PoolingReport` (dataclass with `applied_count`, `fallback_count`)
- [ ] `replay.py` contains data-loading helpers: `load_manifest`, `load_parquet_cache`, `build_embedding_lookup`
- [ ] `build_replay_pipeline` returns a single sklearn `Pipeline` with calibration baked in as the final step when `prob_calibration != "none"`
- [ ] `apply_context_pooling` returns `(pooled_lookup, PoolingReport)` — the report tracks how many examples used neighbor pooling vs center-only fallback
- [ ] `train_eval.py` imports from `replay.py` instead of defining these functions locally; its public API (`train_eval`, `fit_autoresearch_classifier`, `evaluate_classifier_on_split`, `prepare_embeddings`, `find_top_false_positives`) remains unchanged
- [ ] `_unpack_cache_entry` helper stays in `train_eval.py` (it's a compatibility shim for older test fixtures)

**Tests needed:**
- Existing `tests/unit/test_autoresearch.py` tests continue to pass unchanged (they exercise `train_eval.py`'s public API which now delegates to `replay.py`)
- New `tests/unit/test_replay.py` with direct tests for:
  - `apply_context_pooling` for center/mean3/max3 across embedding-set, row-id, and filename formats; cross-file neighbor skipping; row-id fallback to center-only; PoolingReport counts
  - `build_replay_pipeline` for config combinations: norm-only, norm+PCA, norm+PCA+calibration(platt), norm+calibration(isotonic) without PCA; verify pipeline step names and predict_proba output shape
  - PCA component clamping when n_samples < pca_dim
  - `evaluate_on_split` produces correct metrics for a small synthetic dataset at a known threshold

---

### Task 2: Add replay_metric_tolerance setting

**Files:**
- Modify: `src/humpback/config.py`

**Acceptance criteria:**
- [ ] `Settings.replay_metric_tolerance` field added, type `float`, default `0.01`
- [ ] Configurable via `HUMPBACK_REPLAY_METRIC_TOLERANCE` env var

**Tests needed:**
- Verify default value is 0.01
- Verify env var override works (can add to existing config tests if they exist, or a small dedicated test)

---

### Task 3: Extend manifest loading for context pooling support

**Files:**
- Modify: `src/humpback/classifier/trainer.py`

**Acceptance criteria:**
- [ ] New function `load_manifest_split_data` that returns a `ManifestSplitData` dataclass containing: `X` (embeddings array), `y` (labels array), `examples` (list of manifest example dicts for the split), `parquet_cache` (loaded parquet dict), `manifest` (full manifest dict), and `source_summary` (existing summary dict)
- [ ] Uses `replay.py`'s `load_parquet_cache` and `build_embedding_lookup` for loading
- [ ] Existing `load_manifest_split_embeddings` preserved for backward compatibility (embedding-set training path still calls it via the worker)

**Tests needed:**
- `load_manifest_split_data` returns correct shapes and example metadata for a synthetic manifest
- Parquet cache and manifest dict are properly populated for downstream context pooling

---

### Task 4: Add candidate-replay training path in worker

**Files:**
- Modify: `src/humpback/workers/classifier_worker.py`

**Acceptance criteria:**
- [ ] When `source_mode == "autoresearch_candidate"`, the worker loads data via `load_manifest_split_data`, applies `apply_context_pooling` from `replay.py`, then calls `build_replay_pipeline` with `promoted_config`
- [ ] The resulting Pipeline is serialized via joblib (same atomic write pattern as today)
- [ ] `promoted_config` is parsed from `job.promoted_config` JSON field and used directly — no translation through `map_autoresearch_config_to_training_parameters`
- [ ] `EffectiveConfig` and `PoolingReport` are included in the training summary under `replay_effective_config` and `replay_pooling_report`
- [ ] Embedding-set training path (`source_mode != "autoresearch_candidate"`) is completely unchanged

**Tests needed:**
- Unit test in `tests/unit/test_classifier_worker.py`: mock manifest loading, verify the candidate branch calls replay module functions in correct order
- Integration test: end-to-end candidate-backed training with a synthetic manifest produces a valid joblib model that can `predict_proba`

---

### Task 5: Add replay verification to candidate training path

**Files:**
- Modify: `src/humpback/workers/classifier_worker.py`
- Modify: `src/humpback/classifier/replay.py` (add `verify_replay` function)

**Acceptance criteria:**
- [ ] `verify_replay(pipeline, manifest, parquet_cache, promoted_config, candidate_split_metrics, threshold, tolerance)` in `replay.py` evaluates val and test splits, compares against imported metrics, returns `ReplayVerification` dict
- [ ] Rate metrics (precision, recall, fp_rate, high_conf_fp_rate) compared with absolute tolerance; counts (tp, fp, fn, tn, example counts) compared exactly
- [ ] `ReplayVerification` includes: status ("verified"/"mismatch"), tolerance used, threshold, per-split expected/actual/deltas/pass, effective_config
- [ ] Worker calls `verify_replay` after `build_replay_pipeline`, stores result in `source_comparison_context` under `replay_verification` key and in `training_summary`
- [ ] Mismatch does not fail the job — model is still saved, job status is still "complete"

**Tests needed:**
- `verify_replay` passes when metrics match within tolerance
- `verify_replay` reports mismatch when a rate metric exceeds tolerance
- `verify_replay` reports mismatch when counts differ
- Tolerance value is recorded in the result
- Effective config (pooling report, PCA components, calibration) is correctly captured

---

### Task 6: Update promotability check

**Files:**
- Modify: `src/humpback/services/classifier_service.py`

**Acceptance criteria:**
- [ ] `_assess_reproducibility` no longer blocks: `pca_dim != None`, `prob_calibration != "none"`, `context_pooling != "center"`
- [ ] Still blocks: `linear_svm`, `hard_negative_fraction > 0`, `mlp` with non-default class weights, unsupported feature_norm values
- [ ] Candidates that were previously "blocked" solely due to PCA/calibration/pooling now import as "promotable"

**Tests needed:**
- AR-v1-style config (logreg + pca_dim=128 + platt + mean3) returns `is_reproducible_exact=True`
- linear_svm config still returns blocked
- hard_negative_fraction > 0 still returns blocked
- mlp with class weights still returns blocked

---

### Task 7: Frontend replay verification display

**Files:**
- Modify: `frontend/src/components/classifier/AutoresearchCandidatesSection.tsx`

**Acceptance criteria:**
- [ ] When a completed training job's `source_comparison_context` contains `replay_verification`, show a badge: green "Exact Replay Verified" for status "verified", amber "Replay Mismatch" for status "mismatch"
- [ ] Below the badge, show effective config summary: context pooling mode with applied/fallback counts, PCA dim (actual components), calibration method, normalization
- [ ] Show a compact split metric comparison: two-row table (val/test) with columns for key metrics, expected vs actual, delta; highlight cells where delta exceeded tolerance
- [ ] Candidates with PCA/calibration/non-center pooling now show "Promotable" badge instead of "Blocked"

**Tests needed:**
- TypeScript compiles (`npx tsc --noEmit`)
- Playwright test: candidate detail view shows replay verification badge when present in response data

---

### Task 8: Update CLAUDE.md and documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] CLAUDE.md §8.7 Behavioral Constraints updated: note that candidate-backed training uses exact replay via `promoted_config`, not `map_autoresearch_config_to_training_parameters`
- [ ] CLAUDE.md §8.8 Classifier API Surface updated: note replay verification in training job/model responses
- [ ] CLAUDE.md §9.1 Implemented Capabilities updated: mention AR-v1 exact replay promotion with verification
- [ ] CLAUDE.md §8.6 Runtime Configuration updated: document `HUMPBACK_REPLAY_METRIC_TOLERANCE`
- [ ] DECISIONS.md: new ADR for exact replay promotion decision

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/classifier/replay.py src/humpback/classifier/trainer.py src/humpback/workers/classifier_worker.py src/humpback/services/classifier_service.py src/humpback/config.py scripts/autoresearch/train_eval.py`
2. `uv run ruff check src/humpback/classifier/replay.py src/humpback/classifier/trainer.py src/humpback/workers/classifier_worker.py src/humpback/services/classifier_service.py src/humpback/config.py scripts/autoresearch/train_eval.py`
3. `uv run pyright src/humpback/classifier/replay.py src/humpback/classifier/trainer.py src/humpback/workers/classifier_worker.py src/humpback/services/classifier_service.py src/humpback/config.py scripts/autoresearch/train_eval.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
