# Autoresearch Replay Parity for AR-v1 — Design Spec

**Date:** 2026-04-03
**Status:** Approved

## Goal

Extend candidate-backed trainer/promotion so an imported autoresearch winner like
`AR-v1` can be promoted into a production classifier without silently changing its
learned behavior. A promoted model must preserve the candidate's feature transforms
and score calibration, then verify on the imported manifest's `val` and `test` splits
that it matches the reviewed candidate within a tight tolerance.

## Problem Context

The current promotion path blocks `AR-v1` because the production trainer cannot replay:

- `pca_dim=128`
- `prob_calibration="platt"`
- `context_pooling="mean3"`

Current promotion reduces candidate configs into a smaller `parameters` shape intended
for manual embedding-set training. That design was acceptable for the first promotable
subset, but it is the wrong abstraction for exact replay of autoresearch winners.

## Non-Goals

- Running autoresearch from the web UI
- Making the manual embedding-set training form feature-parity with autoresearch
- Supporting every search-space feature immediately
- Recomputing a new source-model comparison if replay verification passes
- Changing detection-time thresholds or rollout policy

## Scope

This feature targets the current AR-v1 family of candidates:

- classifier: `logreg`
- feature normalization: `none`, `l2`, `standard`
- optional `pca_dim`
- probability calibration: `none`, `platt`, `isotonic`
- context pooling: `center`, `mean3`, `max3`
- explicit logistic class weights

This feature does not make these promotable:

- `linear_svm`
- `hard_negative_fraction > 0`
- any replay mode that requires data resampling not encoded in the manifest
- unsupported future transforms outside the AR-v1 family

## Decision

Option C: Extend trainer/promotion to exact replay of AR-v1-style configs.

Candidate-backed promotion will use an exact-replay path whose source of truth is the
imported autoresearch config, not the legacy manual-training parameter shape. The system
will only mark promotion as exact when the produced model replays the candidate and
verifies against the imported manifest metrics.

## Key Design Decisions

1. **Shared replay module**: Extract training logic from `scripts/autoresearch/train_eval.py`
   into `src/humpback/classifier/replay.py`. Both production trainer and `train_eval.py`
   import from it. `train_eval.py` keeps its existing public API (minimal refactor, swaps
   internals to shared imports).

2. **Context pooling on row-id manifests**: No upfront promotability block. Allow it through,
   let replay verification catch discrepancies. Effective-config reports fallback counts.

3. **Pipeline serialization**: Calibration baked into the sklearn Pipeline as the final step.
   `predict_proba()` returns calibrated probabilities directly. Detection code unchanged.

4. **Replay verification tolerances**: Configurable threshold via
   `HUMPBACK_REPLAY_METRIC_TOLERANCE` (default 0.01). Exact match for counts, tolerance
   for rates. Threshold used is recorded in verification result for auditability.

5. **Autoresearch script update**: `train_eval.py` imports shared functions from
   `replay.py`, keeping its own public API and orchestration logic intact.

## Exact Replay Contract

For `source_mode="autoresearch_candidate"` jobs:

- `promoted_config` is the authoritative training config
- The trainer must use the same transform order as autoresearch:
  1. Context pooling (data-level, before pipeline)
  2. Feature normalization
  3. PCA
  4. Classifier fit
  5. Optional probability calibration
- Detection-format manifests must keep the same neighbor rules as autoresearch
- Row-id examples must keep the same center-only fallback used by autoresearch
- The promoted model must be evaluated on the imported `val` and `test` splits at the
  candidate threshold
- Replay verification must compare produced metrics against imported candidate metrics
  and persist the result

If replay verification fails, the model artifact still exists, but is not presented as
an exact replay.

## Backend Design

### 1. Shared Replay Module (`src/humpback/classifier/replay.py`)

Extracted from `scripts/autoresearch/train_eval.py`:

| Function | Purpose |
|---|---|
| `apply_context_pooling()` | Pool neighbor embeddings (center/mean3/max3), track fallback counts |
| `build_feature_pipeline()` | Build ordered sklearn transform steps: normalization -> PCA |
| `build_classifier()` | Construct LogisticRegression or MLP with class weights |
| `apply_calibration()` | Wrap fitted classifier in CalibratedClassifierCV (baked into pipeline) |
| `evaluate_on_split()` | Score a split at a given threshold, return metrics dict |
| `collect_split_arrays()` | Load manifest examples for a split, map to embeddings |

Top-level functions:

- `build_replay_pipeline(config, X_train, y_train) -> (Pipeline, EffectiveConfig)` —
  runs normalization -> PCA -> classifier -> calibration, returns fitted pipeline and
  effective-config record.
- `apply_context_pooling(embeddings, manifest_examples, config) -> (np.ndarray, PoolingReport)` —
  returns pooled embeddings and a report of neighbor-vs-center-only counts.

### 2. Candidate-Replay Training Path

In `classifier_worker.py`'s `run_training_job()`, the `source_mode="autoresearch_candidate"`
branch:

1. Load manifest via extended `load_manifest_split_embeddings()` returning
   `ManifestSplitData(X, y, examples, parquet_cache)`
2. `apply_context_pooling()` with candidate's config -> pooled train embeddings + PoolingReport
3. `build_replay_pipeline()` with `promoted_config` -> fitted Pipeline + EffectiveConfig
4. Serialize pipeline via joblib
5. Run replay verification
6. Persist model with provenance

Bypasses `train_binary_classifier()` entirely for candidate-backed jobs.
Embedding-set training path unchanged.

### 3. Promotability Check Updates

Remove these as blockers in `_assess_reproducibility()`:

- `pca_dim != None`
- `prob_calibration != "none"`
- `context_pooling != "center"`

Still blocked: `linear_svm`, `hard_negative_fraction > 0`, `mlp` with class weights.

### 4. Replay Verification

After training, the worker:

1. Loads `val` and `test` splits from manifest
2. Applies same context pooling
3. Transforms through fitted pipeline
4. Scores at candidate threshold via `evaluate_on_split()`
5. Compares against imported `split_metrics`
6. Produces `ReplayVerification` result

ReplayVerification structure:
```json
{
    "status": "verified" | "mismatch",
    "tolerance": 0.01,
    "threshold": "<candidate threshold>",
    "splits": {
        "val": {
            "expected": { "precision": "...", "recall": "..." },
            "actual": { "precision": "...", "recall": "..." },
            "deltas": { "precision": 0.0, "recall": 0.0 },
            "pass": true
        },
        "test": { "..." }
    },
    "effective_config": {
        "context_pooling": "mean3",
        "context_pooling_fallback_count": 42,
        "context_pooling_applied_count": 158,
        "feature_norm": "l2",
        "pca_dim": 128,
        "pca_components_actual": 128,
        "prob_calibration": "platt",
        "classifier_type": "logistic_regression",
        "class_weight": { "0": 1.0, "1": 2.5 }
    }
}
```

Persisted in existing JSON fields:
- `ClassifierTrainingJob.source_comparison_context` — updated with `replay_verification` key
- `ClassifierModel.training_summary` — includes verification result

### 5. Effective-Config Reporting

Training summary and promotion provenance report both requested and effective values for:
context_pooling, feature_norm, pca_dim, prob_calibration, classifier_type, class_weight.

## API and UI Behavior

No new endpoints. Existing response payloads enriched:

- Replay verification summary in `source_comparison_context` and `training_summary`
- Effective context-pooling summary

Frontend Training page updates:

- **Badge**: "Exact Replay Verified" (green) / "Replay Mismatch" (amber) / absent
- **Effective config summary**: Context pooling applied/fallback counts, PCA, calibration
- **Split metric comparison table**: Expected vs actual with deltas, tolerance exceedances highlighted

Manual embedding-set training form unchanged. Candidate list shows updated promotability.

## Data Model Strategy

No new columns. Replay information persisted in:
- `ClassifierTrainingJob.source_comparison_context`
- `ClassifierModel.promotion_provenance`
- `ClassifierModel.training_summary`

New setting: `HUMPBACK_REPLAY_METRIC_TOLERANCE` -> `Settings.replay_metric_tolerance` (default 0.01).

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Promotion and autoresearch drift again | Shared replay module; replay verification on every candidate-backed job |
| Context pooling partially applicable for some manifests | Preserve autoresearch fallback behavior; surface effective counts in provenance/UI |
| Calibration adds nested CV complexity | Scope to AR-v1 family; focused unit tests; unsupported families stay blocked |
| Users confuse exact replay with deployment approval | Replay proves parity only; does not auto-activate or replace standard model |

## Testing Strategy

**Unit tests for replay.py:**
- Context pooling parity for center/mean3/max3 across format types
- Row-id fallback to center-only with correct counts
- Pipeline construction for all config combinations (norm, PCA, calibration)
- PCA component clamping
- Calibration wrapping (platt, isotonic, none)
- Metric computation against known values

**Unit tests for replay verification:**
- Pass when metrics match within tolerance
- Mismatch when rate exceeds tolerance
- Mismatch when counts differ
- Tolerance recorded in result

**Integration tests:**
- Import fixture candidate with PCA+calibration+mean3 -> promote -> train -> verify
- Resulting model loads and predict_proba() works
- Detection code scores with resulting pipeline

**Regression tests:**
- Embedding-set training unchanged
- linear_svm and hard_negative_fraction still blocked
- train_eval.py produces identical results after shared import swap

## Follow-Up Work

1. Add a `promotable-only` autoresearch profile searching only the exact-replay subset.
2. Add optional manual-training UI parity for PCA, calibration, context-pooling controls.
3. Revisit `linear_svm` and hard-negative support if research results justify complexity.
