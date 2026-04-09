# MLP Class Weight Promotion & Comparison Artifact Persistence

## Problem

Hyperparameter search candidates using MLP with non-default class weights
(e.g., `class_weight_neg: 1.5`) cannot be promoted to autoresearch candidates
because `sklearn.MLPClassifier` has no `class_weight` parameter. The
`_assess_reproducibility()` check in `autoresearch.py` blocks these candidates.

Separately, the hyperparameter worker stores comparison results only in the
database (`comparison_result` column on `HyperparameterSearchJob`) but never
writes `comparison.json` to the search results directory. The candidate import
endpoint looks for this file on disk and, finding nothing, imports the candidate
without production comparison data.

## Solution

### Fix 1: MLP sample weights

Use `sample_weight` in `MLPClassifier.fit()` to reproduce class weighting
behavior. Given `class_weight = {0: w_neg, 1: w_pos}`, compute a per-sample
weight array: `sample_weight[i] = class_weight[y[i]]`.

This is the same mechanism sklearn uses internally for classifiers that support
`class_weight` natively. When weights are uniform (`{0: 1.0, 1: 1.0}`), no
`sample_weight` is passed (no behavioral change).

Apply in two pipelines:
- `replay.py:build_replay_pipeline()` — the candidate-backed replay path
- `trainer.py:train_binary_classifier()` — the standard training path

Both `pipeline.fit()` and `cross_validate()` accept `fit_params` for passing
`sample_weight` through to the classifier step via
`classifier__sample_weight`.

Remove the MLP class weight blocker in `_assess_reproducibility()` and the
`ValueError` in `map_autoresearch_config_to_training_parameters()`.

### Fix 2: Comparison artifact persistence

In `hyperparameter_worker.py`, write the comparison result dict to
`results_dir / "comparison.json"` immediately after `compare_classifiers()`
returns, before storing it in the database. The import endpoint already looks
for this file — no changes needed on the import side.

## Scope

- No UI changes
- No database migrations
- No new API endpoints
- Existing LogisticRegression and LinearSVC paths unchanged
