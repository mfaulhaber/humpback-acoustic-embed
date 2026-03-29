# Multi-Label Vocalization Type Classifier

**Date**: 2026-03-29
**Status**: Approved

## 1. Problem

The labeling UI allows users to create sub-5-second annotations within 5-second
detection windows, labeling individual vocalization types (whup, moan, shriek,
etc.). However, the current vocalization training pipeline ignores sub-window
offsets and treats each window as single-label — whichever label it encounters
first becomes the window's only label. This loses information when multiple call
types co-occur in one window, and dilutes training signal when short calls occupy
a small fraction of the 5-second embedding.

Two problems need solving:

1. **Multi-label windows** — a single 5s window containing multiple vocalization
   types collapses to one label, losing information.
2. **Label precision** — the 5s embedding captures audio outside the annotated
   region, diluting the training signal for short vocalizations.

## 2. Approach: Binary Relevance with Independent Logistic Regression

Based on research into multi-label bioacoustic classification (BirdNET, Perch/Chirp,
DCASE challenges), the standard and most practical approach is **binary relevance**:
each vocalization type gets an independent binary classifier (present/absent) with
sigmoid output, rather than a single multi-class softmax.

This means:
- A 5s window can output `whup=0.85, moan=0.62, click=0.15` simultaneously
- Each type's classifier trains independently, so rare types (30 examples) don't
  drag down common types (500 examples)
- `class_weight='balanced'` handles per-type class imbalance naturally
- Works with both TFLite (Perch) and TF2 SavedModel embeddings — always operates
  on the full 5s embedding vector regardless of model type

**Why not isolate sub-window audio?** Perch/TFLite has a fixed 5s input contract
(128x128 mel spectrogram). Shorter clips would need zero-padding, which creates
out-of-distribution inputs. Instead, we accept window-level multi-hot labels and
let logistic regression learn "windows containing this call type have this
embedding signature." With enough examples the vocalization signal dominates the
surrounding audio context.

## 3. UI Separation Constraint

The binary whale/no-whale classifier and vocalization type classifier must be
logically and UI-separate:

- **Classifier** section (unchanged): Training binary detection models, running
  detection jobs, labeling humpback/orca/ship/background
- **Vocalization** section (new): Managed vocabulary, training per-type classifiers,
  running vocalization inference, browsing multi-label results

The existing vocalization label UI on the binary classifier page moves to the
Vocalization section.

## 4. Data Model

### 4.1 Managed Vocabulary

**`vocalization_types`** table:
- `id` (UUID PK)
- `name` (str, unique, normalized lowercase) — e.g., "whup", "moan", "shriek"
- `description` (optional str) — user-provided notes about the type
- `created_at`, `updated_at`

Populated via auto-import from embedding set folder names. User selects which
embedding sets to scan — this naturally skips binary positive/negative sets that
lack call-type folder structure. Users can also add/edit/delete types manually.

### 4.2 Model & Training

**`vocalization_models`** table:
- `id` (UUID PK)
- `name` (str) — user-provided or auto-generated
- `model_dir_path` (str) — directory containing N `.joblib` files (one per type)
- `vocabulary_snapshot` (JSON) — the type names this model was trained on
- `per_class_thresholds` (JSON) — `{"whup": 0.42, "moan": 0.61, ...}` auto-optimized
- `per_class_metrics` (JSON) — per-type AP, F1, precision, recall from CV
- `training_summary` (JSON) — overall stats, sample counts, filtered types
- `is_active` (bool, default false) — at most one active model
- `created_at`

**`vocalization_training_jobs`** table:
- `id` (UUID PK)
- `status` (str) — queued/running/complete/failed
- `source_config` (JSON) — which embedding sets and/or detection jobs to pull from
- `parameters` (JSON) — classifier_type, l2_normalize, class_weight,
  min_examples_per_type (default 4)
- `vocalization_model_id` (nullable FK) — set on completion
- `result_summary` (JSON) — types trained, types filtered, per-class counts
- `created_at`, `updated_at`

### 4.3 Inference

**`vocalization_inference_jobs`** table:
- `id` (UUID PK)
- `status` (str) — queued/running/complete/failed
- `vocalization_model_id` (FK)
- `source_type` (str) — `"detection_job"`, `"embedding_set"`, or `"rescore"`
- `source_id` (str) — detection_job_id or embedding_set_id
- `output_path` (str) — path to results parquet
- `result_summary` (JSON) — counts per type, total windows scored
- `created_at`, `updated_at`

## 5. Training Pipeline

### 5.1 Data Collection

Training collects embeddings + multi-hot labels from two sources:

1. **Curated embedding sets** — user selects embedding sets with call-type folder
   structure. Each subfolder name maps to a vocabulary type. Every embedding in the
   `whup/` subfolder becomes a positive example for the whup classifier.

2. **Detection job labels** — vocalization labels and labeling annotations from
   detection results. Sub-window annotation offsets are ignored for training — only
   the type label matters at the window level.

When both sources are combined, embeddings are deduplicated by
`(filename, start_sec, end_sec)` to avoid double-counting.

### 5.2 Multi-Label-Aware Negative Construction

**This is critical to the design.** For each per-type binary classifier:

- **Positives**: all windows labeled with this type
- **Negatives**: all windows NOT labeled with this type

**Multi-label rule**: a window labeled with both "whup" and "moan" is a positive
for the whup classifier AND a positive for the moan classifier. It is a negative
for NEITHER. Only windows that are explicitly not labeled with a given type serve
as negatives for that type's classifier.

This prevents contradictory training signals when types co-occur. A window
containing both a whup and a moan must not be used as a negative example for
either type.

### 5.3 Per-Type Classifier Training

For each vocalization type in the vocabulary that meets `min_examples_per_type`
(default 4):

1. Collect positives and negatives per the multi-label-aware rule above
2. Build sklearn Pipeline: optional L2 Normalizer -> StandardScaler ->
   LogisticRegression (class_weight='balanced')
3. Run 5-fold stratified CV, compute per-class AP, F1, precision, recall
4. Auto-optimize threshold: find threshold maximizing F1 on held-out folds
5. Save `.joblib` to `{model_dir}/{type_name}.joblib`

Types below `min_examples_per_type` are skipped and reported in `result_summary`
as filtered, with their example counts.

### 5.4 Model Artifact

```
/vocalization_models/{model_id}/
  whup.joblib
  moan.joblib
  shriek.joblib
  ...
  metadata.json    (vocabulary snapshot, thresholds, training params)
```

Each `.joblib` is an independent sklearn Pipeline.

## 6. Inference Pipeline

### 6.1 Input Sources

1. **Detection job** — score all positive-labeled windows from a binary detection
   job. Loads embeddings from the detection job's embedding source.
2. **Embedding set** — score all windows in an arbitrary embedding set.
3. **Rescore** — re-run inference on windows that already have predictions from a
   previous inference job. Used after retraining to compare old vs new model.

### 6.2 Execution

For each window:
1. Load its embedding vector
2. Run through each per-type `.joblib` pipeline independently
3. Collect raw sigmoid scores into a score vector
4. Apply per-class thresholds (from model, with user overrides if set)

### 6.3 Output

Results persisted as Parquet:
```
/vocalization_inference/{job_id}/predictions.parquet
```

Columns: `filename`, `start_sec`, `end_sec`, `start_utc`, `end_utc` (when
available from detection source), plus one float column per vocabulary type
containing the raw sigmoid score.

### 6.4 Threshold Application

Per-class thresholds are stored on the model but applied at query time, not baked
into the parquet. This means:
- The API re-applies thresholds without re-running inference
- UI threshold sliders work instantly
- Exports can use either auto-optimized or user-tuned thresholds

## 7. API Surface

New router at `/vocalization/`.

### 7.1 Vocabulary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/vocalization/types` | List all vocalization types |
| POST | `/vocalization/types` | Add a new type |
| PUT | `/vocalization/types/{id}` | Rename/edit a type |
| DELETE | `/vocalization/types/{id}` | Remove a type (fails if used by active model) |
| POST | `/vocalization/types/import` | Auto-import from selected embedding set folder names |

### 7.2 Training

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/vocalization/models` | List all models (with active flag) |
| GET | `/vocalization/models/{id}` | Model detail (per-class metrics, thresholds, vocabulary) |
| PUT | `/vocalization/models/{id}/activate` | Set as active model |
| POST | `/vocalization/training-jobs` | Queue training job |
| GET | `/vocalization/training-jobs` | List training jobs |
| GET | `/vocalization/training-jobs/{id}` | Training job status/detail |

### 7.3 Inference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/vocalization/inference-jobs` | Queue inference job |
| GET | `/vocalization/inference-jobs` | List inference jobs |
| GET | `/vocalization/inference-jobs/{id}` | Job status/detail |
| GET | `/vocalization/inference-jobs/{id}/results` | Paginated results with threshold params |
| GET | `/vocalization/inference-jobs/{id}/export` | TSV/CSV download with applied thresholds |

The `/results` endpoint accepts optional `thresholds` query params to override
per-class thresholds at query time.

## 8. Frontend

### 8.1 Navigation

Add "Vocalization" as a top-level nav section:

```
/app/vocalization/training    — Vocalization/Training
/app/vocalization/labeling    — Vocalization/Labeling
```

Existing Classifier section unchanged.

### 8.2 Training Page (`/app/vocalization/training`)

Three panels:

1. **Vocabulary Manager** — table of defined types with add/edit/delete, plus
   "Import from Embedding Set" button that opens a selector dialog. Shows example
   count per type when training data is available.

2. **Train Model** — source selector (pick embedding sets and/or detection jobs),
   parameter controls (classifier_type, l2_normalize, class_weight,
   min_examples_per_type), "Train" button. Active training jobs show progress.

3. **Models** — table of trained models with columns: name, created date, types
   count, mean F1, active badge. Expandable rows showing per-class metrics (AP,
   F1, precision, recall, sample count), per-class threshold values, and
   "Activate" / "Delete" actions.

### 8.3 Labeling Page (`/app/vocalization/labeling`)

Two panels:

1. **Inference Jobs** — queue new inference (select active model + source), list
   of completed jobs with result summaries. Expandable rows show per-type tag
   counts at current thresholds.

2. **Results Browser** — when a job is selected, paginated table of windows with:
   - Spectrogram thumbnail
   - Audio playback
   - Type tags (above-threshold types as colored badges)
   - Per-type threshold sliders (adjustable, applied client-side to stored scores)
   - Export button (TSV with applied thresholds)

## 9. Worker Integration

### 9.1 Priority Order

Vocalization jobs slot into the existing worker loop after the current pipeline:

```
search -> processing -> clustering -> classifier training -> detection ->
extraction -> label processing -> retrain -> vocalization training ->
vocalization inference
```

### 9.2 Job Lifecycle

Both training and inference jobs follow the standard lifecycle:

```
queued -> running -> complete
queued -> running -> failed
queued -> canceled
```

No pause/resume needed — vocalization training and inference operate on
pre-computed embeddings and are fast (seconds to low minutes).

### 9.3 Claim Semantics

Same atomic compare-and-set pattern as other job types. Stale job recovery
extended to cover both vocalization job types.

## 10. Storage Layout

```
/vocalization_models/
  {model_id}/
    whup.joblib
    moan.joblib
    ...
    metadata.json

/vocalization_inference/
  {job_id}/
    predictions.parquet
```

## 11. Migration

Single Alembic migration adding four tables: `vocalization_types`,
`vocalization_models`, `vocalization_training_jobs`,
`vocalization_inference_jobs`. Uses `op.batch_alter_table()` for SQLite
compatibility. No changes to existing tables.

## 12. Cleanup: Remove Sub-Window Annotation System

The sub-window annotation system is replaced by window-level multi-hot labels.

**Remove:**
- `LabelingAnnotation` model from `src/humpback/models/labeling.py`
- `labeling_annotations` table (Alembic migration to drop)
- Annotation CRUD API endpoints from labeling router
- `AnnotationOverlay` component and annotation mode in labeling UI
- Annotation schemas from `src/humpback/schemas/labeling.py`
- Annotation fallback in `_run_vocalization_training_job` worker code

**Keep:**
- `VocalizationLabel` model and `vocalization_labels` table
- Vocalization label CRUD endpoints (move to new `/vocalization/` router)

## 13. Testing

- **Unit tests**: per-type classifier training with synthetic embeddings,
  multi-label-aware negative construction, threshold optimization, vocabulary
  import from folder names, min_examples filtering
- **Integration tests**: API endpoints for vocabulary CRUD, training job
  lifecycle, inference job lifecycle, results with threshold overrides
- **Key assertion**: a window labeled with types A and B is positive for both
  A and B classifiers, and negative for neither — verify multi-label-aware
  negative logic explicitly

## 14. Out of Scope

- Sub-window embedding isolation (padding short annotations to 5s) — future work
- Attention-based MIL over sub-window embeddings — future Phase 2
- Sound Event Detection (SED) frame-level classification — future Phase 3
- Vocalization type detection on raw audio (always requires binary detection first
  or pre-computed embeddings)
- Modifying the existing binary classifier pipeline
