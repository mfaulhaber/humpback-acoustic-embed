# MLP Class Weight Promotion Implementation Plan

**Goal:** Enable promotion of MLP hyperparameter search candidates with explicit class weights, and persist comparison artifacts to disk for candidate import.
**Spec:** [docs/specs/2026-04-09-mlp-class-weight-promotion-design.md](../specs/2026-04-09-mlp-class-weight-promotion-design.md)

---

### Task 1: Add sample weight helper and apply in replay pipeline

**Files:**
- Modify: `src/humpback/classifier/replay.py`

**Acceptance criteria:**
- [ ] New function `compute_sample_weight(class_weight: dict, y: ndarray) -> ndarray` that maps per-class weights to per-sample weights
- [ ] Returns `None` when all class weights are 1.0 (uniform)
- [ ] `build_replay_pipeline()` computes sample weight from config's `class_weight_pos`/`class_weight_neg`
- [ ] `pipeline.fit()` call at line ~613 passes `classifier__sample_weight` when MLP and weights are non-uniform
- [ ] `CalibratedClassifierCV.fit()` call at line ~621 also passes `sample_weight` when applicable
- [ ] LogisticRegression and LinearSVC paths unaffected (they use native `class_weight`)

**Tests needed:**
- `compute_sample_weight` returns correct array for non-uniform weights
- `compute_sample_weight` returns None for uniform weights
- `build_replay_pipeline` with MLP + non-uniform weights trains without error
- `build_replay_pipeline` with logreg + non-uniform weights still uses native `class_weight` (no sample_weight)

---

### Task 2: Apply sample weight in trainer pipeline

**Files:**
- Modify: `src/humpback/classifier/trainer.py`

**Acceptance criteria:**
- [ ] `map_autoresearch_config_to_training_parameters()` no longer raises `ValueError` for MLP with explicit class weights; instead includes class_weight dict in returned parameters for MLP
- [ ] `train_binary_classifier()` computes sample weight when classifier_type is "mlp" and class_weight is non-uniform
- [ ] `cross_validate()` call passes `fit_params={"classifier__sample_weight": sw}` for MLP with non-uniform weights
- [ ] `pipeline.fit()` final call passes `classifier__sample_weight` for MLP with non-uniform weights
- [ ] LogisticRegression path unaffected

**Tests needed:**
- `map_autoresearch_config_to_training_parameters` with MLP + non-default weights returns class_weight in parameters without raising
- `train_binary_classifier` with MLP + non-uniform weights produces valid model

---

### Task 3: Remove MLP class weight blocker from reproducibility check

**Files:**
- Modify: `src/humpback/services/classifier_service/autoresearch.py`

**Acceptance criteria:**
- [ ] `_assess_reproducibility()` no longer adds "MLP promotion cannot yet reproduce explicit class weights" blocker
- [ ] Other existing blockers (e.g., hard_negative_fraction) unchanged

**Tests needed:**
- `_assess_reproducibility` with MLP + explicit weights returns reproducible=True (assuming no other blockers)
- `_assess_reproducibility` with non-MLP configs unchanged

---

### Task 4: Write comparison.json to disk in hyperparameter worker

**Files:**
- Modify: `src/humpback/workers/hyperparameter_worker.py`

**Acceptance criteria:**
- [ ] After `compare_classifiers()` returns, write result to `results_dir / "comparison.json"` before DB update
- [ ] Only write when `comparison_result` is not None
- [ ] Existing DB storage of `comparison_result` unchanged

**Tests needed:**
- Worker writes `comparison.json` to results_dir when comparison_result exists
- Worker skips file write when no comparison model configured

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/classifier/replay.py src/humpback/classifier/trainer.py src/humpback/services/classifier_service/autoresearch.py src/humpback/workers/hyperparameter_worker.py`
2. `uv run ruff check src/humpback/classifier/replay.py src/humpback/classifier/trainer.py src/humpback/services/classifier_service/autoresearch.py src/humpback/workers/hyperparameter_worker.py`
3. `uv run pyright src/humpback/classifier/replay.py src/humpback/classifier/trainer.py src/humpback/services/classifier_service/autoresearch.py src/humpback/workers/hyperparameter_worker.py`
4. `uv run pytest tests/`
