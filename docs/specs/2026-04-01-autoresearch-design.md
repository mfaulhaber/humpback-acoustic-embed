# Perch Embedding Autoresearch — Design Spec

**Date:** 2026-04-01
**Status:** Approved

## Goal

Build a bounded autoresearch-style experiment loop for the binary classifier trained on frozen Perch-style audio embeddings. The primary optimization target is reducing high-confidence false positives (score >= 0.90 on negatives). Secondary goals: preserve useful recall, keep the system simple, reproducible, and cheap to run.

## Problem Context

Current pipeline: 5-second audio window → frozen Perch embedding → binary classifier head → probability score → thresholded decision.

The classifier performs well overall, but some false positives occur with very high confidence. These high-confidence false positives are more damaging than borderline errors. The system must optimize for this tail-risk failure mode rather than generic accuracy or F1.

## Architecture Decisions

### Integration: Hybrid scripts importing from humpback

Scripts live in `scripts/autoresearch/` and import low-level humpback utilities (`read_embeddings`, database models for manifest generation) but maintain an independent training pipeline. This avoids duplicating Parquet I/O while keeping the research code free to experiment without risking the production classifier.

### Data: Manifest generator + manual override

A generator script queries the humpback database (classifier training job IDs → embedding sets → Parquet paths) and produces a standalone `data_manifest.json`. After generation, the manifest is a plain JSON file that can be edited — adding negative group labels, adjusting splits, removing examples. The autoresearch system only reads this JSON; it never queries the database at runtime.

### Context neighbors: Inferred from row index

Adjacent-window context for pooling (mean3, max3) is inferred at train time from sequential row indices within each Parquet file. Row N-1 is left, N+1 is right. Boundary windows (first/last in a file) fall back to center-only. No extra artifact needed.

### Results: Advisory only (manual transfer)

The autoresearch system tells you what works. Winning configurations are transferred manually to the platform's classifier training parameters. No automatic model export or registration in v1.

## File Layout

```
scripts/autoresearch/
├── generate_manifest.py   # Queries humpback DB → writes data_manifest.json
├── search_space.py        # Declarative dict of allowed hyperparameter values
├── train_eval.py          # Train one config, emit JSON metrics to stdout
├── run_autoresearch.py    # Sample configs, run train_eval, track best
├── objectives.py          # Objective functions (pluggable, default FP-penalizing)
└── results/               # Created at runtime
    ├── search_history.json
    ├── best_run.json
    └── top_false_positives.json
```

## Data Manifest

### Structure

```json
{
  "metadata": {
    "created_at": "2026-04-01T...",
    "source_job_ids": [12, 15],
    "positive_embedding_set_ids": [3, 7],
    "negative_embedding_set_ids": [4, 8],
    "split_strategy": "by_audio_file"
  },
  "examples": [
    {
      "id": "es3_row42",
      "split": "train",
      "label": 1,
      "parquet_path": "/abs/path/to/embeddings.parquet",
      "row_index": 42,
      "audio_file_id": 17,
      "negative_group": null
    }
  ]
}
```

### ID scheme

`es{embedding_set_id}_row{row_index}` — unique, stable, traceable back to platform data.

### Split strategy

Group by `audio_file_id` so all windows from one recording land in the same split. Default ratio 70/15/15 train/val/test. Deterministic assignment: sorted file IDs, seeded shuffle. Regenerating from the same inputs produces the same manifest.

### Negative groups

Null by default. Populated manually or with a helper if metadata about negative subtypes exists (vessel, rain, other whale, etc.).

## Search Space

```python
SEARCH_SPACE = {
    "feature_norm": ["none", "l2", "standard"],
    "pca_dim": [None, 32, 64, 128, 256],
    "classifier": ["logreg", "linear_svm", "mlp"],
    "class_weight_pos": [1.0, 1.5, 2.0, 3.0],
    "class_weight_neg": [1.0, 1.5, 2.0, 3.0],
    "hard_negative_fraction": [0.0, 0.1, 0.2, 0.4],
    "prob_calibration": ["none", "platt", "isotonic"],
    "threshold": [0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97],
    "context_pooling": ["center", "mean3", "max3"],
}
```

Kept in `search_space.py` as a plain dict. Easy to add or remove dimensions.

## Training Pipeline (`train_eval.py`)

Pipeline stages in order:

1. **Load embeddings** — read Parquet files, filter to requested split, build `{id: vector}` lookup
2. **Context pooling** — for `mean3`/`max3`, retrieve row_index ± 1 from same Parquet file. Missing neighbors → center fallback. One pooled vector per example.
3. **Feature normalization** — `none`, `l2` (sklearn `Normalizer`), or `standard` (sklearn `StandardScaler`, fit on train only)
4. **PCA** — optional, fit on train split only. `None` or integer dim (32, 64, 128, 256)
5. **Classifier training**:
   - `logreg`: `LogisticRegression(C=1.0, class_weight={0: neg_w, 1: pos_w}, max_iter=1000, solver='lbfgs')`
   - `linear_svm`: `LinearSVC` wrapped in `CalibratedClassifierCV` for probability output
   - `mlp`: `MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True, max_iter=500)`
6. **Probability calibration** — `none`, `platt` (sigmoid), or `isotonic` via `CalibratedClassifierCV` on held-out fold. Skipped for SVM (already calibrated in step 5).
7. **Evaluation on validation split** — compute all metrics at configured threshold

### Class weights

Specified as `class_weight_pos` and `class_weight_neg` floats, assembled into sklearn's `{0: neg, 1: pos}` dict.

### Hard negative fraction

When > 0, top N% highest-scoring validation negatives from a previous run are added to training set. V1 supports this structurally; the search loop runs it as a second phase.

### Determinism

Random seed passed through config, default 42. All sklearn estimators receive the seed. Seed recorded in output metrics.

## Metrics & Objective

### Base metrics (validation split)

```json
{
  "threshold": 0.93,
  "precision": 0.96,
  "recall": 0.91,
  "fp_rate": 0.004,
  "high_conf_fp_rate": 0.0007,
  "tp": 1820, "fp": 7, "fn": 46, "tn": 2751,
  "objective": 0.8875
}
```

### High-confidence false positive rate

```
high_conf_fp_rate = count(negatives with score >= 0.90) / total_negatives
```

Always computed regardless of configured threshold.

### Default objective

```python
objective = recall - 15.0 * high_conf_fp_rate - 3.0 * fp_rate
```

15x penalty on high-confidence FPs reflects deployment priority. Objective is pluggable via named functions in `objectives.py`.

### Grouped metrics

When `negative_group` is populated, additionally report `high_conf_fp_rate_by_group`. Not used in v1 objective but reported when available. Code structure supports a grouped objective later.

### Top false positives

After each run, save the N highest-scoring validation negatives with IDs, scores, and negative_group. Feeds future hard-negative mining.

## Search Loop (`run_autoresearch.py`)

### Execution flow

1. Load manifest once; cache embeddings in memory per context-pooling mode
2. For each trial (default budget: 200):
   - Sample config from search space (random search)
   - Skip if exact config already run (dedup by config hash)
   - Call `train_eval()` in-process (no subprocess)
   - Compute objective, append to `results/search_history.json`
   - Update `results/best_run.json` if new best
   - Update `results/top_false_positives.json` from best
3. Print summary

### Embedding caching

Embeddings loaded and context-pooled once per pooling mode. Dict keyed by `context_pooling` value holds precomputed matrices.

### Incremental persistence

`search_history.json` appended after every trial. Crash loses at most one run.

### CLI

```bash
uv run scripts/autoresearch/run_autoresearch.py \
  --manifest data_manifest.json \
  --trials 200 \
  --objective default \
  --seed 42
```

Single-run mode:

```bash
uv run scripts/autoresearch/train_eval.py \
  --manifest data_manifest.json \
  --config '{"classifier":"logreg","feature_norm":"l2","pca_dim":128,...}'
```

### Hard-negative phase

After initial search, run a second pass with `--hard-negative-from results/top_false_positives.json` to add those examples to training. Manual and explicit.

## Manifest Generator (`generate_manifest.py`)

### CLI

```bash
uv run scripts/autoresearch/generate_manifest.py \
  --job-ids 12,15 \
  --split-ratio 70,15,15 \
  --seed 42 \
  --output data_manifest.json
```

### Process

1. Query classifier training jobs by ID → get positive/negative embedding set IDs
2. Resolve Parquet paths from embedding sets
3. Enumerate all rows per Parquet file (row count, not loading vectors)
4. Collect unique `audio_file_id` values, sort, seeded shuffle
5. Assign files to train/val/test by ratio
6. Every row inherits its file's split
7. Rows from positive sets get `label: 1`, negative sets get `label: 0`
8. Write JSON manifest

### Imports from humpback

`database.get_session`, `ClassifierTrainingJob` and `EmbeddingSet` ORM models, `read_embeddings` (row count only).

## Testing Strategy

### Unit tests (`tests/unit/test_autoresearch.py`)

- Manifest generation: mock DB, verify split grouping by audio file, deterministic output, label assignment
- Context pooling: synthetic Parquet with 5 rows, verify mean3 averages neighbors, boundary fallback
- Feature transforms: L2 produces unit vectors, PCA reduces dim, StandardScaler fits train only
- Metrics: synthetic predictions, verify high_conf_fp_rate at 0.90 boundary, confusion matrix
- Objective: verify default penalizes high-conf FP, verify pluggable selection
- Search loop: mock train_eval, verify dedup, best tracking, incremental writes

### Integration test (`tests/integration/test_autoresearch.py`)

- End-to-end: generate manifest from test fixtures, run 5 trials with small synthetic embeddings, verify all output artifacts are valid JSON with expected schema

Tests use synthetic numpy arrays (small dim, ~50 examples). No real audio or model files needed.

## Non-Goals for v1

- LLM-assisted mutation of search space
- End-to-end encoder finetuning
- Distributed search
- Model export to platform
- UI integration
- Complex orchestration frameworks

## Future Extensions

- Grouped objective using `max_group_high_conf_fp_rate`
- Hard-negative mining chained across search rounds
- LLM agent with bounded edit permissions over `search_space.py` and objective weights
