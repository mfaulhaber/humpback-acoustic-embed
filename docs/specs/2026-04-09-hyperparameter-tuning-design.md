# Classifier Hyperparameter Tuning Page — Design Spec

**Date:** 2026-04-09
**Status:** Approved

## Overview

Promote the CLI-based autoresearch scripts (`scripts/autoresearch/`) into a new
Classifier/Hyperparameter Tuning page in the web UI. The page replaces the manual
script workflow with a three-stage UI: manifest generation → hyperparameter search →
production comparison, plus the relocated autoresearch candidate section.

## Motivation

The current workflow requires running four separate CLI scripts with careful argument
passing and file path management. Moving this to the UI makes the workflow accessible,
observable, and persistent — matching how every other platform workflow operates.

## Data Model

### `hyperparameter_manifests` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK (UUID) | |
| `name` | TEXT | User-provided name |
| `status` | TEXT | `queued` / `running` / `complete` / `failed` |
| `training_job_ids` | TEXT (JSON array) | Source classifier training job IDs (embedding set sources) |
| `detection_job_ids` | TEXT (JSON array) | Source detection job IDs (labeled windows) |
| `split_ratio` | TEXT (JSON array) | e.g., `[70, 15, 15]` |
| `seed` | INTEGER | Random seed for split assignment |
| `manifest_path` | TEXT nullable | Path to the generated `manifest.json` artifact on disk |
| `example_count` | INTEGER nullable | Total examples in the manifest |
| `split_summary` | TEXT nullable (JSON) | Per-split label counts |
| `detection_job_summaries` | TEXT nullable (JSON) | Per-detection-job inclusion/skip counts |
| `error_message` | TEXT nullable | Failure reason |
| `created_at` | DATETIME | |
| `completed_at` | DATETIME nullable | |

### `hyperparameter_search_jobs` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK (UUID) | |
| `name` | TEXT | User-provided name |
| `status` | TEXT | `queued` / `running` / `complete` / `failed` |
| `manifest_id` | TEXT FK → `hyperparameter_manifests.id` | |
| `search_space` | TEXT (JSON) | Customized search space grid |
| `n_trials` | INTEGER | Number of trials to run |
| `seed` | INTEGER | Random seed |
| `objective_name` | TEXT | Fixed to `"default"` for now |
| `results_dir` | TEXT nullable | Path to results directory on disk |
| `trials_completed` | INTEGER default 0 | Progress counter updated periodically |
| `best_objective` | REAL nullable | Best objective score so far |
| `best_config` | TEXT nullable (JSON) | Config of the best trial |
| `best_metrics` | TEXT nullable (JSON) | Metrics of the best trial |
| `comparison_model_id` | TEXT nullable FK → `classifier_models.id` | Production model to compare against |
| `comparison_threshold` | REAL nullable | Decision threshold for the production model |
| `comparison_result` | TEXT nullable (JSON) | Full comparison output |
| `error_message` | TEXT nullable | |
| `created_at` | DATETIME | |
| `completed_at` | DATETIME nullable | |

## Worker Jobs

### Priority Order

Manifest generation and hyperparameter search are the lowest-priority worker jobs:

```
search -> processing -> clustering -> classifier training -> detection -> extraction ->
detection embedding generation -> label processing -> retrain -> vocalization training ->
vocalization inference -> manifest generation -> hyperparameter search
```

### Manifest Generation Worker

1. Claims queued `hyperparameter_manifests` row
2. Calls the manifest generation service function (refactored from `generate_manifest.py`)
3. Writes manifest JSON to `{storage_root}/hyperparameter/manifests/{manifest_id}/manifest.json`
4. Updates row with `manifest_path`, `example_count`, `split_summary`, `detection_job_summaries`
5. Sets status to `complete` or `failed`

Only human-annotated labels are included: vocalization positives/negatives and binary
row-store labels from detection jobs, plus embedding set examples from training jobs.
No `include_unlabeled_hard_negatives`, no score-band negatives.

### Hyperparameter Search Worker

1. Claims queued `hyperparameter_search_jobs` row
2. Loads manifest from disk
3. Pre-caches embeddings for all pooling modes in the search space
4. Runs the search loop with the job's custom `search_space`
5. Updates `trials_completed`, `best_objective`, `best_config`, `best_metrics` in DB every ~10 trials
6. Writes `search_history.json`, `best_run.json`, `top_false_positives.json` to results dir
7. If `comparison_model_id` is set, runs comparison and stores in `comparison_result`
8. Sets status to `complete` or `failed`

No `hard_negative_fraction` dimension. No phase 2 workflow.

## API Endpoints

All under `/classifier/hyperparameter`.

### Manifests

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/classifier/hyperparameter/manifests` | Create and queue manifest generation |
| `GET` | `/classifier/hyperparameter/manifests` | List all manifests |
| `GET` | `/classifier/hyperparameter/manifests/{id}` | Full manifest detail |
| `DELETE` | `/classifier/hyperparameter/manifests/{id}` | Delete (blocked if referenced by search) |

### Searches

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/classifier/hyperparameter/searches` | Create and queue search job |
| `GET` | `/classifier/hyperparameter/searches` | List all searches |
| `GET` | `/classifier/hyperparameter/searches/{id}` | Full detail |
| `GET` | `/classifier/hyperparameter/searches/{id}/history` | Full trial history |
| `DELETE` | `/classifier/hyperparameter/searches/{id}` | Delete search + artifacts |

### Search Space Defaults

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/classifier/hyperparameter/search-space-defaults` | Return default search space for UI pre-population |

### Candidates (relocated)

Existing autoresearch candidate endpoints move to `/classifier/hyperparameter/candidates/*`.
Old paths aliased for backward compatibility. New endpoint:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/classifier/hyperparameter/searches/{id}/import-candidate` | Create candidate from completed search artifacts |

## Service Layer

### New package: `src/humpback/services/hyperparameter_service/`

```
hyperparameter_service/
├── __init__.py          (public API)
├── manifest.py          (from generate_manifest.py)
├── search.py            (from run_autoresearch.py)
├── comparison.py        (from compare_classifiers.py)
└── search_space.py      (SEARCH_SPACE defaults, sample_config, config_hash)
```

The `objectives.py` content (single default objective) folds into `search.py`.
The `train_eval.py` logic already delegates to `humpback.classifier.replay`.

Scripts in `scripts/autoresearch/` become thin CLI wrappers importing from the service.

## Storage Paths

```
{storage_root}/hyperparameter/
├── manifests/{manifest_id}/
│   └── manifest.json
└── searches/{search_id}/
    ├── search_history.json
    ├── best_run.json
    └── top_false_positives.json
```

## Frontend

### Route

`/app/classifier/tuning` — new "Tuning" tab in the Classifier section nav, after "Embeddings".

### Page Layout

Three collapsible sections:

**1. Manifests** — table with name, status, source summary, example count, split ratio,
created date. "New Manifest" button opens a dialog with name, multi-select for training
jobs and detection jobs, split ratio, seed. Completed rows expand to show split summary
and detection job breakdown. Delete blocked if referenced by a search.

**2. Searches** — table with name, status + progress (45/200), manifest name, best
objective, comparison model, created date. "New Search" button opens a dialog with name,
manifest dropdown, search space configurator (each dimension as checkboxes, all checked
by default), trial count, seed, optional comparison model + threshold. Completed rows
expand to show best config/metrics, comparison deltas, and "Import as Candidate" button.

**3. Candidates** — the existing `AutoresearchCandidatesSection` component relocated
from the Training tab. Still supports file-path-based import for external artifacts.

### Polling

TanStack Query refetch for job lists. Search detail polls `trials_completed` while running.

## Scope Boundaries

**Not included:**
- Promotion workflow (future design session)
- Hard-negative mining / phase 2 / unlabeled score-band negatives
- Custom objective functions
- Bayesian optimization
- Search space presets/templates
- Manifest editing after generation
- Trial-level detail table in UI (available via API)

## Objective Function

Fixed to the default: `recall - 15 * high_conf_fp_rate - 3 * fp_rate`. Displayed on
the page for transparency.
