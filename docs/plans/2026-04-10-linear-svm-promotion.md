# Linear SVM Candidate Promotion Implementation Plan

**Goal:** Unblock `linear_svm` candidates from the hyperparameter tuning page so they can be imported and promoted via the existing candidate-backed replay path.
**Spec:** [docs/specs/2026-04-10-linear-svm-promotion-design.md](../specs/2026-04-10-linear-svm-promotion-design.md)

---

### Task 1: Unblock linear_svm in the reproducibility assessor

**Files:**
- Modify: `src/humpback/services/classifier_service/autoresearch.py`

**Acceptance criteria:**
- [ ] `_assess_reproducibility` no longer adds a blocker when `config["classifier"] == "linear_svm"`.
- [ ] `_assess_reproducibility` still blocks any unknown classifier string (e.g., `"xgboost"`).
- [ ] `_assess_reproducibility` still blocks `hard_negative_fraction > 0`.
- [ ] `_assess_reproducibility` still blocks unknown `feature_norm` values.

**Tests needed:**
- Rename `tests/unit/test_autoresearch.py::test_linear_svm_still_blocked` to `test_linear_svm_is_reproducible`, flipping assertions to expect `ok is True` and `blockers == []`.
- Add a new assertion in the same test class that an unknown classifier (e.g., `"xgboost"`) still produces a blocker, to guard against the allowlist being accidentally removed entirely.
- Leave `test_hard_negative_fraction_still_blocked` and any `feature_norm` blocker tests untouched.

---

### Task 2: Map linear_svm in the trainer parameter mapper

**Files:**
- Modify: `src/humpback/classifier/trainer.py`

**Acceptance criteria:**
- [ ] `map_autoresearch_config_to_training_parameters` accepts `{"classifier": "linear_svm"}` without raising.
- [ ] The returned dict has `classifier_type == "linear_svm"`.
- [ ] The returned dict still carries `class_weight`, `feature_norm`, and `random_state` fields populated from config.
- [ ] Existing logreg and mlp behavior is unchanged.

**Tests needed:**
- Unit test (locate existing `map_autoresearch_config_to_training_parameters` tests — add to `tests/unit/test_trainer.py` if no dedicated module exists) asserting linear_svm maps correctly and does not raise.
- Keep or add assertions that unknown classifiers still raise `ValueError`.

---

### Task 3: Add replay pipeline determinism coverage for linear_svm

**Files:**
- Modify: `tests/unit/test_replay.py`

**Acceptance criteria:**
- [ ] A new unit test builds a small synthetic training set (e.g., 40 samples, 16 features, two well-separated classes with a fixed numpy seed).
- [ ] The test calls `build_replay_pipeline` with `{"classifier": "linear_svm", "feature_norm": "standard", "prob_calibration": "none", "seed": 42}`.
- [ ] The returned pipeline exposes `predict_proba`.
- [ ] Running `predict_proba` twice on the same input returns identical arrays (determinism sanity check for the cv=3 `CalibratedClassifierCV` wrapping).
- [ ] The existing `test_build_linear_svm` test remains and continues to pass.

**Tests needed:**
- Covered by the task itself.

---

### Task 4: Integration test for end-to-end linear_svm candidate promotion

**Files:**
- Modify: `tests/integration/test_classifier_api.py`
- Potentially modify/create: an autoresearch fixture with a linear_svm best-run config (reuse `tests/fixtures/autoresearch/explicit-negatives/phase1/search_history.json` which already contains linear_svm entries, or add a minimal new fixture if the existing one is not used by the promotion integration test).

**Acceptance criteria:**
- [ ] A new integration test imports a candidate whose `promoted_config.classifier == "linear_svm"`.
- [ ] The test asserts `is_reproducible_exact is True` on the returned candidate detail.
- [ ] The test creates a candidate-backed training job via the service (or API endpoint, matching the pattern of the existing promotion integration test).
- [ ] The test runs the classifier training worker for that job inline to completion.
- [ ] The persisted `ClassifierModel.training_summary` contains `replay_effective_config.classifier_type == "linear_svm"`.
- [ ] The persisted `training_summary.replay_verification.status == "verified"`.

**Tests needed:**
- Covered by the task itself.

---

### Task 5: Frontend display labels for linear_svm models

**Files:**
- Modify: `frontend/src/components/classifier/TrainingTab.tsx`

**Acceptance criteria:**
- [ ] `classifierTag` returns `"SVM"` when `classifier_type === "linear_svm"`.
- [ ] `classifierLabel` returns `"Linear SVM"` when `classifier_type === "linear_svm"`.
- [ ] Existing tags and labels for `"mlp"` and `"logistic_regression"` are unchanged.
- [ ] The fallback (`classifier_type ?? "—"`) still applies for any other unexpected value.

**Tests needed:**
- No new Playwright tests. The existing tuning-page / model-detail flows exercise these code paths; the additions are pure label branches.

---

### Task 6: Documentation updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] CLAUDE.md §9.1 "Autoresearch candidate promotion" bullet lists `linear_svm` alongside PCA / calibration / context pooling as promotable via the shared replay module.
- [ ] DECISIONS.md has a new dated entry (either an addendum to ADR-043 or a small new ADR referencing it) recording:
  - Linear SVM is now promotable via the shared replay path.
  - Trigger: a reviewed research candidate justified the complexity.
  - `hard_negative_fraction > 0` remains deferred.
- [ ] README.md is **not** modified (linear_svm is not a direct-training option).

**Tests needed:**
- None. Documentation-only.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/services/classifier_service/autoresearch.py src/humpback/classifier/trainer.py tests/unit/test_autoresearch.py tests/unit/test_replay.py tests/unit/test_trainer.py tests/integration/test_classifier_api.py`
2. `uv run ruff check src/humpback/services/classifier_service/autoresearch.py src/humpback/classifier/trainer.py tests/unit/test_autoresearch.py tests/unit/test_replay.py tests/unit/test_trainer.py tests/integration/test_classifier_api.py`
3. `uv run pyright src/humpback/services/classifier_service/autoresearch.py src/humpback/classifier/trainer.py tests/unit/test_autoresearch.py tests/unit/test_replay.py tests/unit/test_trainer.py tests/integration/test_classifier_api.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`

### Manual acceptance (after merge)

1. Delete the existing `s-v2 (imported)` candidate via the UI.
2. Re-import the same search as a candidate; verify `is_reproducible_exact=True` and zero blockers.
3. Create a candidate-backed training job from the new candidate; wait for completion.
4. On the Classifier → Training tab, verify the resulting model shows an "SVM" tag and "Linear SVM" label.
5. Verify `replay_verification.status == "verified"` in the model's training summary.
6. Run a small detection job against the new model to confirm end-to-end `predict_proba` works on hydrophone audio.
