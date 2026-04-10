# Linear SVM Candidate Promotion — Design

**Status:** Approved
**Date:** 2026-04-10
**Supersedes aspect of:** ADR-043 (autoresearch replay parity — deferred linear_svm)

## Goal

Allow `linear_svm` candidates from the hyperparameter tuning page to be promoted
to production classifier models via the existing candidate-backed replay path.

The immediate trigger is a blocked candidate in production (`s-v2 (imported)`)
with error `"classifier='linear_svm' is not supported by the production trainer"`.
ADR-043 explicitly noted: *"Revisit linear_svm and hard-negative support if
research results justify complexity."* The research has now produced a candidate
worth promoting.

## Scope

**In scope:**

- Remove the two hard gates that reject `linear_svm` during candidate import and
  promotion.
- Ensure promoted linear_svm models display a friendly label ("Linear SVM" /
  "SVM") in the Classifier → Training tab model detail.
- Backfill tests that currently assert linear_svm is blocked, and add coverage
  proving end-to-end replay works.

**Explicitly out of scope:**

- Adding linear_svm as a first-class option in the Classifier → Training tab
  dropdown for direct (non-tuning) training. Direct training still offers only
  Logistic Regression and MLP. The tuning → candidate → promotion path is the
  only entry point.
- Lifting the `hard_negative_fraction > 0` blocker (remains deferred).
- Refactoring the classifier allowlists into a shared registry (YAGNI — two
  small lists have coexisted stably across multiple ADRs).

## Background: why the blocker exists today

`src/humpback/classifier/replay.py` already has full linear_svm support:

- `build_classifier()` wraps `LinearSVC` in `CalibratedClassifierCV(cv=3,
  method="sigmoid")` so `predict_proba()` works.
- `apply_calibration()` short-circuits for `linear_svm` (already calibrated).
- `build_replay_pipeline()` constructs and fits the complete feature-norm → PCA
  → SVM+calibration pipeline from config.
- `test_build_linear_svm` in `tests/unit/test_replay.py` confirms the pipeline
  builds correctly.

The hyperparameter search itself (`services/hyperparameter_service/train_eval.py`)
already calls `replay.build_classifier`, so linear_svm trials run end-to-end
during tuning and produce comparison results. The candidate-backed training
worker (`workers/classifier_worker/training.py`) uses `build_replay_pipeline`
directly via the `source_mode == "autoresearch_candidate"` branch, bypassing
`train_binary_classifier` entirely.

Both the local detector (`classifier/detector.py:212`) and the hydrophone
detector (`classifier/hydrophone_detector.py:194`) call
`pipeline.predict_proba(all_emb)[:, 1]`, which the wrapped
`CalibratedClassifierCV(LinearSVC)` provides natively.

The only things preventing promotion are two hard-coded allowlists added in
ADR-043 that were deliberately conservative.

## Backend changes

### Gate 1: `_assess_reproducibility`

**File:** `src/humpback/services/classifier_service/autoresearch.py:254-258`

The function rejects any classifier not in `{"logreg", "mlp"}`. Expand the set
to include `"linear_svm"`. The `hard_negative_fraction > 0` and
`feature_norm` validation remain unchanged.

This function is called inline during `import_autoresearch_candidate`; the
boolean result is persisted on the `AutoresearchCandidate` row as
`is_reproducible_exact`. After this change, **newly imported** candidates with
`classifier=linear_svm` will be marked reproducible; existing blocked
candidates remain blocked until re-imported (see Migration below).

### Gate 2: `map_autoresearch_config_to_training_parameters`

**File:** `src/humpback/classifier/trainer.py:131-159`

The function raises `ValueError` for any classifier other than `"logreg"` and
`"mlp"`. Add a third branch that maps `"linear_svm"` to
`classifier_type="linear_svm"`.

This function is called from
`create_candidate_backed_training_job` only to populate the
`ClassifierTrainingJob.parameters` JSON field for display and provenance. The
actual training path in the worker reads `promoted_config` and calls
`build_replay_pipeline` directly; it does not consume the mapped parameters.
No changes to `train_binary_classifier` are needed because candidate-backed
jobs never reach it.

### Not changing

- `train_binary_classifier` in `trainer.py` — never reached for candidate-backed
  jobs.
- The candidate-backed worker path.
- Detector code (uses `predict_proba` generically).
- Database schema (no new columns, no migration).
- Replay pipeline construction or verification.

## Frontend changes

**File:** `frontend/src/components/classifier/TrainingTab.tsx:801-802`

Extend the existing `classifierTag` and `classifierLabel` derivations with a
third branch:

- `classifier_type === "linear_svm"` → tag `"SVM"`, label `"Linear SVM"`

This is the only place a promoted model surfaces a friendly classifier label.
The tuning page's trial history and candidate detail already display raw config
strings, so they light up automatically.

No changes to the classifier training form (linear_svm remains absent from the
dropdown there, by design).

## Test plan

### Unit tests

- **`tests/unit/test_autoresearch.py`** — rename `test_linear_svm_still_blocked`
  → `test_linear_svm_is_reproducible` and flip the assertions to `ok is True`
  and `blockers == []`. Leave `test_hard_negative_fraction_still_blocked`
  unchanged (still blocked).
- **`tests/unit/test_replay.py`** — `test_build_linear_svm` already asserts the
  classifier builds. Add a new test that runs `build_replay_pipeline` on a
  small synthetic fixture with
  `{"classifier": "linear_svm", "feature_norm": "standard", "prob_calibration": "none"}`,
  asserts the returned object has `predict_proba`, and verifies repeated scoring
  with the same seed is deterministic (sanity check for cv=3 calibration).
- **Trainer parameter mapping test** — add a case proving linear_svm config
  maps to `classifier_type == "linear_svm"` without raising. Location: the
  existing `map_autoresearch_config_to_training_parameters` test module (if
  absent, add to `tests/unit/test_trainer.py`).

### Integration tests

- **`tests/integration/test_classifier_api.py`** — extend the existing
  autoresearch candidate promotion test with a linear_svm variant:
  1. Import a candidate from a fixture whose best-run config has
     `classifier: "linear_svm"`.
  2. Assert `is_reproducible_exact == True` on the imported record.
  3. Create a candidate-backed training job from the candidate.
  4. Run the classifier worker inline.
  5. Assert the resulting `ClassifierModel` row persists with
     `training_summary.replay_effective_config.classifier_type == "linear_svm"`
     and `training_summary.replay_verification.status == "verified"`.

### Frontend tests

- No new Playwright tests. The promotion flow and model detail rendering are
  already covered; the change is additive label branches.

## Migration: existing blocked candidate

The existing `s-v2 (imported)` candidate has `is_reproducible_exact=False` and
an error message persisted at import time. After this change ships, the user
will:

1. Delete the candidate via the existing `DELETE
   /classifier/hyperparameter/candidates/{id}` endpoint (UI already exposes it).
2. Re-import from the underlying completed search via `POST
   /classifier/hyperparameter/searches/{id}/import-candidate`.
3. The new record will be marked reproducible and become promotable.

No one-time backfill script, no read-time re-assessment hook. Only one
candidate is affected.

## Documentation updates

Per CLAUDE.md §10.4 doc matrix:

- **CLAUDE.md §9.1** — update the "Autoresearch candidate promotion" bullet to
  list `linear_svm` alongside the existing PCA / calibration / context-pooling
  replay coverage.
- **DECISIONS.md** — append an addendum to ADR-043 (or a short new ADR
  referencing it) recording that linear_svm is now promotable, the trigger (a
  reviewed research candidate), and that `hard_negative_fraction > 0` remains
  deferred.
- **README.md** — no change. linear_svm is not a direct-training option.

## Verification gates

After implementation, run in order:

1. `uv run ruff format --check` on modified Python files
2. `uv run ruff check` on modified Python files
3. `uv run pyright` on modified Python files
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`

## Manual acceptance

1. Delete the existing `s-v2 (imported)` candidate.
2. Re-import the same search; verify `is_reproducible_exact=True`.
3. Create a candidate-backed training job; wait for completion.
4. Verify the resulting model on the Training tab shows "SVM" tag and
   "Linear SVM" label.
5. Verify `replay_verification.status == "verified"` in the training summary.
6. Run a small detection job against the new model to confirm end-to-end
   `predict_proba` works on hydrophone audio.

## Risk

Minimal. Every runtime code path for linear_svm has existed and been
unit-tested since ADR-043. This change opens a gate that was deliberately
closed; everything downstream already works. The primary failure mode is
non-determinism in `CalibratedClassifierCV(cv=3)` causing replay verification
to exceed `replay_metric_tolerance` — the new unit test explicitly guards
against this by asserting deterministic scoring with a fixed seed.
