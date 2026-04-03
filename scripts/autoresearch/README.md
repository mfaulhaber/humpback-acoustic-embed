# Autoresearch: Bounded Classifier Head Search

Constrained hyperparameter search over binary classifier heads trained on frozen Perch embeddings. The primary optimization target is reducing **high-confidence false positives** (negatives scored >= 0.90) while preserving acceptable recall.

## Process Overview

The autoresearch workflow has three stages:

**1. Build a data manifest** — Pull embeddings from the platform into a stable, inspectable JSON file with fixed train/val/test splits. Two data sources are available:
- **Embedding sets** from classifier training jobs (curated positives and negatives)
- **Detection jobs** with human labels (deployment-realistic positives, labeled negatives like ship/background, and unlabeled high-confidence windows as hard negatives)

**2. Run the search** — The search loop randomly samples classifier configurations from a bounded hyperparameter space (normalization, PCA, classifier type, class weights, calibration, threshold, context pooling). Each trial trains a classifier, evaluates on the validation split, and scores the result using a custom objective that heavily penalizes high-confidence false positives. Results are persisted incrementally.

**3. Analyze and iterate** — Review `best_run.json` for the winning config and `top_false_positives.json` for the most confusing negatives. Optionally run a second search phase that feeds those false positives back as hard negatives in training. Compare the winning config against a live production model such as `LR-v12`, then either promote the reviewed candidate through the platform UI/API when it is exactly reproducible or fall back to manual transfer for unsupported configs.

```
generate_manifest.py        run_autoresearch.py          Analyze results
        |                          |                          |
  DB + Parquet files    -->   data_manifest.json   -->   results/best_run.json
  (embedding sets,            (fixed splits,              results/search_history.json
   detection jobs)             all sources)               results/top_false_positives.json
                                    |
                                    v
                          Optional: phase 2 search
                          with --hard-negative-from
                                    |
                                    v
                        Compare vs production (e.g. LR-v12)
                                    |
                                    v
                    Import artifacts into Training tab candidate review
                    and promote when the config is exactly reproducible
```

## Quick Start

```bash
# 1. Generate a manifest from existing classifier training jobs
uv run scripts/autoresearch/generate_manifest.py \
  --job-ids <job-id-1>,<job-id-2> \
  --output data_manifest.json

# 2. Run the search loop
uv run scripts/autoresearch/run_autoresearch.py \
  --manifest data_manifest.json \
  --trials 200
```

## Scripts

### `generate_manifest.py` — Build a Data Manifest

Queries the humpback database for classifier training jobs (embedding sets) and/or detection jobs (labeled + unlabeled windows), then produces a stable `data_manifest.json` with train/val/test splits grouped by source-specific split buckets.

```bash
# From training jobs only (embedding sets)
uv run scripts/autoresearch/generate_manifest.py \
  --job-ids <comma-separated training job IDs> \
  --output data_manifest.json

# From detection jobs (explicitly labeled positives/negatives only)
uv run scripts/autoresearch/generate_manifest.py \
  --detection-job-ids <comma-separated detection job IDs> \
  --score-range 0.5,0.995 \
  --output data_manifest.json

# Opt in to unlabeled score-band hard negatives
uv run scripts/autoresearch/generate_manifest.py \
  --detection-job-ids <comma-separated detection job IDs> \
  --score-range 0.9,0.995 \
  --include-unlabeled-hard-negatives \
  --output data_manifest.json

# Both sources combined
uv run scripts/autoresearch/generate_manifest.py \
  --job-ids <training job IDs> \
  --detection-job-ids <detection job IDs> \
  --output data_manifest.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--job-ids` | none | Classifier training job IDs (embedding set sources) |
| `--detection-job-ids` | none | Detection job IDs (must have positive labels) |
| `--score-range` | `0.5,0.995` | Min,max confidence for unlabeled hard negatives when that opt-in path is enabled |
| `--split-ratio` | `70,15,15` | Train/val/test ratio |
| `--seed` | `42` | Random seed for split assignment |
| `--include-unlabeled-hard-negatives` | off | Opt in to using unlabeled detections inside `--score-range` as hard negatives |
| `--output` | required | Output path for the manifest JSON |

At least one of `--job-ids` or `--detection-job-ids` is required.

**Detection job data:** Live production detection embeddings are the canonical row-id schema `(row_id, embedding, confidence)`. Legacy filename-based detection embeddings remain supported for older artifacts.

Detection jobs with human labels contribute examples using this precedence:
- **Manual vocalization positives** — any non-`"(Negative)"` label in `vocalization_labels` becomes a positive
- **Manual vocalization negatives** — `"(Negative)"` becomes a negative with `negative_group = "vocalization_negative"`
- **Row-store fallback positives** — `humpback=1` or `orca=1` when no contradictory vocalization label exists
- **Row-store fallback negatives** — `ship=1` or `background=1` when no contradictory vocalization label exists

By default, unlabeled detections are excluded from the manifest. If you explicitly pass `--include-unlabeled-hard-negatives`, rows unlabeled by both systems and inside `--score-range` are admitted as score-band negatives (`det_0.50_0.90`, `det_0.90_0.95`, `det_0.95_0.99`, `det_0.99_1.00`).

Rows with contradictory positive and negative supervision are skipped and counted in `metadata.detection_job_summaries` rather than silently coerced.

For canonical row-id detection jobs, split grouping uses a synthetic hourly UTC bucket derived from row-store `start_utc`:

`det{job_id[:8]}:{YYYY-MM-DDTHH}`

This value is stored in `audio_file_id` for backward compatibility with the rest of the pipeline.

The output manifest is a plain JSON file. After generation you can edit it directly to:
- Add `negative_group` labels (e.g. `"vessel"`, `"rain"`, `"other_whale"`)
- Adjust split assignments
- Remove specific examples

### `train_eval.py` — Run a Single Experiment

Trains one classifier configuration and prints metrics to stdout.

```bash
uv run scripts/autoresearch/train_eval.py \
  --manifest data_manifest.json \
  --config '{"classifier":"logreg","feature_norm":"l2","pca_dim":128,"threshold":0.93,"context_pooling":"mean3","class_weight_pos":2.0,"class_weight_neg":1.0,"prob_calibration":"none","hard_negative_fraction":0.0}'
```

For canonical row-id detection examples, `context_pooling=mean3` and `context_pooling=max3` fall back to center-only because Parquet row order is not treated as a stable temporal-neighbor contract.

### `run_autoresearch.py` — Search Loop

Samples random configs from the search space, runs `train_eval` for each, and tracks the best result.

```bash
uv run scripts/autoresearch/run_autoresearch.py \
  --manifest data_manifest.json \
  --trials 200 \
  --objective default \
  --seed 42 \
  --results-dir results
```

| Flag | Default | Description |
|------|---------|-------------|
| `--manifest` | required | Path to `data_manifest.json` |
| `--trials` | `200` | Number of search trials |
| `--objective` | `default` | Objective function name |
| `--seed` | `42` | Random seed |
| `--results-dir` | `results` | Output directory |
| `--hard-negative-from` | none | Path to `top_false_positives.json` for hard-negative mining |

### `compare_classifiers.py` — Compare Against Production

Retrains the winning autoresearch config on the manifest train split, loads a production classifier from `classifier_models`, and evaluates both on the same requested split(s).

When you want to compare against live production models such as `LR-v12`, source the repo `.env` first so `HUMPBACK_DATABASE_URL` points at the production database and classifier artifact paths:

```bash
set -a && source .env
uv run scripts/autoresearch/compare_classifiers.py \
  --manifest data_manifest.json \
  --best-run results/phase1/best_run.json \
  --production-classifier-name LR-v12 \
  --splits val,test \
  --output results/phase1/lr_v12_comparison.json
```

For a phase-2 winner, also pass the false-positive source file that defined the replay pool so the script can rebuild the replay-adjusted manifest correctly:

```bash
set -a && source .env
uv run scripts/autoresearch/compare_classifiers.py \
  --manifest data_manifest.json \
  --best-run results/phase2/best_run.json \
  --hard-negative-from results/phase1/top_false_positives.json \
  --production-classifier-name LR-v12 \
  --splits val,test \
  --output results/phase2/lr_v12_comparison.json
```

The output JSON includes split-level metrics for both models, metric deltas (`autoresearch - production`), top false positives for each model, and the largest prediction disagreements between them. That comparison artifact is what the platform import flow uses to populate candidate review details and promotion evidence.

## Import and Promote a Reviewed Candidate

After you have a manifest, `best_run.json`, and optionally a production comparison plus top-false-positive preview, import the bundle into the Classifier Training page:

1. Open the `Training` tab and find `Autoresearch Candidates`.
2. Import the server-visible paths for:
   - `manifest.json`
   - `best_run.json`
   - optional comparison JSON such as `phase1/lr-v12-comparison.json`
   - optional `top_false_positives.json`
3. Review the candidate summary and expanded detail:
   - source model and phase
   - split-level comparison metrics and deltas
   - disagreement preview and top false positives
   - reproducibility warnings
4. If the candidate is marked `promotable`, enter a new model name and start candidate-backed training.

Candidate-backed promotion trains directly from the manifest `train` split and preserves the imported comparison provenance on the resulting training job and model.

### Candidate Promotion vs Legacy Retrain

These are two different workflows:

- **Candidate-backed promotion** starts from imported autoresearch artifacts and manifest examples. It does not reconstruct folder roots and is intended for reviewed search winners.
- **Legacy retrain-from-folders** starts from an existing trained model that was originally built from embedding-set imports tied back to source folders. It reimports folders, reprocesses audio, and then retrains.

Today, candidate-backed models do not support the folder-root retrain flow because the promotion path is manifest-backed rather than folder-backed.

## Current Promotion Limits

The platform only marks a candidate as `promotable` when the current production trainer can replay the autoresearch config faithfully. The following settings are promotable today:

- `classifier`: `logreg`, `mlp`
- `feature_norm`: `none`, `l2`, `standard`
- `context_pooling`: `center`
- `prob_calibration`: `none`
- `pca_dim`: `null`
- `hard_negative_fraction`: `0.0`
- MLP class weights: `class_weight_pos=1.0` and `class_weight_neg=1.0`

The following features currently block promotion and leave the candidate in `blocked` status:

- unsupported classifier families
- any non-`center` context pooling such as `mean3` or `max3`
- any probability calibration (`platt`, `isotonic`)
- any PCA projection (`pca_dim != null`)
- replay-adjusted hard-negative sampling (`hard_negative_fraction > 0`)
- explicit MLP class weights other than `1.0`

Blocked candidates are still useful review artifacts, but their winning configs must still be transferred manually if you want to reproduce them outside the current promotion contract.

## Vendored Fixture Bundle

For UI and API development, the repo vendors a stable explicit-negative artifact bundle under [`scripts/autoresearch/output/`](./output/). In particular:

- [`scripts/autoresearch/output/explicit-negatives/manifest.json`](./output/explicit-negatives/manifest.json)
- [`scripts/autoresearch/output/explicit-negatives/phase1/best_run.json`](./output/explicit-negatives/phase1/best_run.json)
- [`scripts/autoresearch/output/explicit-negatives/phase1/lr-v12-comparison.json`](./output/explicit-negatives/phase1/lr-v12-comparison.json)

Those fixtures are the reference bundle used by the candidate import API tests and the Training-tab promotion UI coverage.

## Results

After a search run, the results directory contains:

| File | Description |
|------|-------------|
| `search_history.json` | Every trial: config, metrics, objective, timestamp |
| `best_run.json` | The single best trial by objective score |
| `top_false_positives.json` | Highest-scoring validation negatives from the best run |

## Objective Function

The default objective penalizes high-confidence false positives heavily:

```
objective = recall - 15.0 * high_conf_fp_rate - 3.0 * fp_rate
```

Where `high_conf_fp_rate` is the fraction of validation negatives scored >= 0.90. This is computed regardless of the configured decision threshold.

Custom objectives can be added to `objectives.py`.

## Search Space

Defined in `search_space.py`:

| Dimension | Values |
|-----------|--------|
| `feature_norm` | `none`, `l2`, `standard` |
| `pca_dim` | `None`, `32`, `64`, `128`, `256` |
| `classifier` | `logreg`, `linear_svm`, `mlp` |
| `class_weight_pos` | `1.0`, `1.5`, `2.0`, `3.0` |
| `class_weight_neg` | `1.0`, `1.5`, `2.0`, `3.0` |
| `hard_negative_fraction` | `0.0`, `0.1`, `0.2`, `0.4` |
| `prob_calibration` | `none`, `platt`, `isotonic` |
| `threshold` | `0.50` ... `0.97` (8 values) |
| `context_pooling` | `center`, `mean3`, `max3` |

## Hard-Negative Mining

After an initial search, use the top false positives to enrich training data for a second pass:

```bash
# Phase 1: baseline search
uv run scripts/autoresearch/run_autoresearch.py \
  --manifest data_manifest.json \
  --trials 200 \
  --results-dir results/phase1

# Phase 2: re-search with hard negatives added to training
uv run scripts/autoresearch/run_autoresearch.py \
  --manifest data_manifest.json \
  --trials 200 \
  --hard-negative-from results/phase1/top_false_positives.json \
  --results-dir results/phase2
```

When `--hard-negative-from` is present, each trial deterministically replays only the configured `hard_negative_fraction` of that false-positive pool into `train`. Unsampled replay candidates are moved to `unused`, not left in `val`/`test`, so all phase-2 trials are scored on the same evaluation set.

When `--hard-negative-from` is absent, `hard_negative_fraction` is normalized to `0.0` and does not create distinct trials.

## Limitations

- Embeddings are frozen (no encoder finetuning)
- Search is random sampling, not Bayesian optimization
- Only exactly reproducible candidates can be promoted automatically; unsupported configs still require manual transfer
