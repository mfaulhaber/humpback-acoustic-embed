# Classifier Training Provenance — Design Spec

## Problem

The binary classifier training UI has two data gaps:

1. **Detection-manifest models** — the expanded row's "Training Data" section is empty because `get_training_data_summary()` has no `detection_manifest` branch. It falls through to the embedding-set path which returns empty results (detection-manifest jobs have `positive_embedding_set_ids = []`).

2. **Autoresearch-candidate (promoted) models** — the table columns (Samples, Accuracy, AUC, Precision, F1) show dashes because the training worker builds a replay-only summary without standard metric fields. The metrics exist in `best_run_metrics` on the candidate record and `promotion_provenance` on the model, but the frontend reads from `training_summary` which lacks them.

## Approach

Backend normalization: ensure all source modes produce a consistent `training_summary` format so the frontend has one code path.

## Design

### 1. Worker Normalization — Promoted Model Metrics

In the `autoresearch_candidate` branch of the classifier training worker, after building the replay summary, merge standard metric fields computed from available data:

| Standard field | Source |
|---|---|
| `n_positive` | `training_data_source.positive_count` |
| `n_negative` | `training_data_source.negative_count` |
| `balance_ratio` | `positive_count / negative_count` |
| `cv_accuracy` | `(tp+tn)/(tp+fp+fn+tn)` from `best_run_metrics` |
| `cv_precision` | `best_run_metrics.precision` |
| `cv_recall` | `best_run_metrics.recall` |
| `cv_f1` | `2 * (precision * recall) / (precision + recall)` |
| `cv_roc_auc` | `None` (not available in autoresearch format) |
| `classifier_type` | `trainer_parameters.classifier_type` |
| `class_weight_strategy` | derived from `trainer_parameters.class_weight` |
| `effective_class_weights` | `trainer_parameters.class_weight` |
| `train_confusion` | `{tp, fp, fn, tn}` from `best_run_metrics` |
| `score_separation` | `None` (not computed) |

No `_std` fields since these are test-split metrics, not cross-validated. `n_cv_folds` stays absent so the UI can distinguish "5-fold cross-validation" from single-split evaluation.

### 2. Worker Normalization — Detection-Manifest Per-Job Breakdown

In the `detection_manifest` branch of the classifier training worker, after loading manifest split embeddings, compute per-job label breakdowns and include them in `training_data_source`:

```
training_data_source.per_job_counts = [
    { "detection_job_id": "...", "positive_count": N, "negative_count": N },
    ...
]
```

This avoids re-reading the manifest file at query time.

### 3. Training Data Summary Service — Detection-Manifest Branch

Add a `detection_manifest` branch to `get_training_data_summary()` that:

1. Reads `detection_job_ids` from `training_summary`
2. Loads `DetectionJob` records from DB for hydrophone_name, start/end timestamps
3. Reads per-job counts from `training_data_source.per_job_counts` (or falls back to totals if not available)
4. Returns them as `detection_sources` in the response

### 4. Response Schema

Add to `schemas/classifier.py`:

- `DetectionSourceInfo`: detection_job_id, hydrophone_name, start_timestamp (epoch UTC), end_timestamp (epoch UTC), positive_count, negative_count
- Add optional `detection_sources: list[DetectionSourceInfo] | None` to `TrainingDataSummaryResponse`

### 5. Frontend — Detection-Manifest Training Data

In the expanded model row's Training Data section, add a `detection_manifest` rendering path:

- Two-column layout matching the existing embedding-set style
- Left column: Positive total, Negative total, Balance ratio
- Right column: "Detection Jobs" header, then a list of jobs formatted as "Hydrophone — Start-End Datetime UTC"

### 6. Frontend — Training Source Label

Show "Detection Jobs" instead of "Embedding Sets" when `training_source_mode === "detection_manifest"` in the Training Parameters section.

### 7. Backfill Script

One-time Python script (`scripts/backfill_training_summary.py`) that:

1. Finds all `autoresearch_candidate` models, loads their `promotion_provenance`, computes standard metric fields, merges into `training_summary`
2. Finds all `detection_manifest` models, reads their manifest files, computes per-job label breakdowns, patches `training_data_source`
3. Updates DB rows
4. Prints a summary of what changed

## No Schema Changes

All fixes are JSON content normalization within existing columns plus a new response field. No Alembic migration needed.
