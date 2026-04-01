# Autoresearch Implementation Plan

**Goal:** Build a bounded hyperparameter search loop for binary classifier heads on frozen Perch embeddings, optimizing for low high-confidence false positive rate.
**Spec:** [docs/specs/2026-04-01-autoresearch-design.md](../specs/2026-04-01-autoresearch-design.md)

---

### Task 1: Search space and objectives modules

**Files:**
- Create: `scripts/autoresearch/__init__.py`
- Create: `scripts/autoresearch/search_space.py`
- Create: `scripts/autoresearch/objectives.py`

**Acceptance criteria:**
- [ ] `SEARCH_SPACE` dict contains all 9 dimensions from spec (feature_norm, pca_dim, classifier, class_weight_pos, class_weight_neg, hard_negative_fraction, prob_calibration, threshold, context_pooling)
- [ ] `default_objective(metrics)` returns `recall - 15.0 * high_conf_fp_rate - 3.0 * fp_rate`
- [ ] Objectives are selectable by name via a registry dict
- [ ] A helper function samples a random config from the search space given a seed/RNG

**Tests needed:**
- Verify default objective computation with known inputs
- Verify config sampling produces valid combinations
- Verify all search space values are the expected types

---

### Task 2: Train/eval pipeline

**Files:**
- Create: `scripts/autoresearch/train_eval.py`

**Acceptance criteria:**
- [ ] Loads embeddings from Parquet files using `humpback.processing.embeddings.read_embeddings`
- [ ] Implements context pooling: `center` (passthrough), `mean3` (average of row ± 1), `max3` (element-wise max of row ± 1) with boundary fallback to center
- [ ] Implements feature normalization: `none`, `l2` (sklearn Normalizer), `standard` (StandardScaler fit on train)
- [ ] Implements optional PCA (fit on train, transform both splits)
- [ ] Supports three classifiers: `logreg` (LogisticRegression), `linear_svm` (LinearSVC + CalibratedClassifierCV), `mlp` (MLPClassifier)
- [ ] Class weights applied as `{0: class_weight_neg, 1: class_weight_pos}`
- [ ] Probability calibration (`none`, `platt`, `isotonic`) via CalibratedClassifierCV; skipped for linear_svm
- [ ] Evaluates on validation split: computes threshold, precision, recall, fp_rate, high_conf_fp_rate, tp, fp, fn, tn
- [ ] `high_conf_fp_rate` counts validation negatives with score >= 0.90 regardless of configured threshold
- [ ] Reports `high_conf_fp_rate_by_group` when negative_group metadata is present
- [ ] Saves top N highest-scoring validation negatives as false positive candidates
- [ ] Accepts config as JSON CLI arg, writes metrics JSON to stdout
- [ ] Deterministic with fixed seed; seed recorded in output

**Tests needed:**
- Context pooling with synthetic 5-row Parquet: verify mean3 math, boundary fallback
- Feature transforms: L2 produces unit vectors, PCA reduces dimensionality, StandardScaler fits train only
- Metrics computation: synthetic predictions with known scores, verify high_conf_fp_rate at 0.90 boundary
- End-to-end single run with synthetic linearly-separable embeddings: verify output JSON schema

---

### Task 3: Manifest generator

**Files:**
- Create: `scripts/autoresearch/generate_manifest.py`

**Acceptance criteria:**
- [ ] Accepts `--job-ids`, `--split-ratio`, `--seed`, `--output` CLI args
- [ ] Queries humpback database for ClassifierTrainingJob → positive/negative EmbeddingSet IDs → Parquet paths
- [ ] Enumerates rows per Parquet file without loading full vectors into memory
- [ ] Splits by audio_file_id: all windows from same file in same split
- [ ] Deterministic: same inputs + seed → same manifest
- [ ] ID scheme: `es{embedding_set_id}_row{row_index}`
- [ ] Labels: rows from positive sets → 1, negative sets → 0
- [ ] Output JSON has `metadata` (created_at, source_job_ids, embedding set IDs, split_strategy) and `examples` array
- [ ] negative_group defaults to null

**Tests needed:**
- Mock database session with synthetic training job and embedding sets, verify split grouping by audio file
- Verify deterministic output with same seed
- Verify label assignment from positive/negative sets
- Verify metadata fields in output

---

### Task 4: Search loop

**Files:**
- Create: `scripts/autoresearch/run_autoresearch.py`

**Acceptance criteria:**
- [ ] Accepts `--manifest`, `--trials`, `--objective`, `--seed` CLI args
- [ ] Optional `--hard-negative-from` flag for second-phase runs
- [ ] Loads manifest and caches embeddings in memory, keyed by context_pooling mode
- [ ] Random-samples configs from search space, deduplicates by config hash
- [ ] Calls train_eval in-process (function call, not subprocess)
- [ ] Computes objective from returned metrics using selected objective function
- [ ] Writes `results/search_history.json` incrementally after each trial (JSON array, each entry has config, metrics, objective, trial number, timestamp)
- [ ] Updates `results/best_run.json` when new best found
- [ ] Updates `results/top_false_positives.json` from the best run
- [ ] Prints summary at end: best objective, best config, total trials
- [ ] Creates `results/` directory if it doesn't exist

**Tests needed:**
- Mock train_eval to return canned metrics, verify dedup skips duplicate configs
- Verify best_run tracks actual best objective
- Verify incremental history writes (each entry present after mock run)
- Verify embedding caching: train_eval receives pre-loaded data, Parquet read called once per pooling mode

---

### Task 5: Integration test

**Files:**
- Create: `tests/unit/test_autoresearch.py`
- Create: `tests/integration/test_autoresearch.py`

**Acceptance criteria:**
- [ ] Unit tests cover: objectives, config sampling, context pooling, feature transforms, metrics computation, search loop dedup/tracking
- [ ] Integration test: writes synthetic Parquet files to tmp dir, generates manifest from them (bypassing DB), runs 5 search trials, verifies all output artifacts are valid JSON with expected schema
- [ ] All tests use synthetic numpy arrays (small dim ~16, ~50 examples), no real audio or models
- [ ] Tests pass with `uv run pytest tests/unit/test_autoresearch.py tests/integration/test_autoresearch.py`

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
