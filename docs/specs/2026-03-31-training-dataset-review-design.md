# Training Dataset Review & Editing View

**Date:** 2026-03-31
**Status:** Approved

## Problem

Vocalization model training data comes from two source types (detection jobs and embedding sets) with different label storage mechanisms. Users cannot inspect what data went into a trained model, identify mislabeled windows, or edit training labels without re-running the full labeling workflow. Embedding set labels are inferred from folder structure and not editable at all.

## Decision

Introduce a **training dataset** as a first-class concept — a durable, editable snapshot of all embeddings and labels used to train a vocalization model. All sources (detection jobs and embedding sets) are unified into a single parquet file with a consistent schema. Labels are stored in SQL and fully editable. A new "Training Data" view on the Vocalization page lets users filter by type, compare positives and negatives with large inline spectrograms, and edit labels with the same save/cancel batch workflow as the labeling page.

Backward compatibility with existing vocalization models is not required. Existing models can be discarded.

## Data Model

### `training_datasets` table

| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | Auto-generated |
| name | VARCHAR | User-facing name |
| source_config | TEXT (JSON) | `{embedding_set_ids: [...], detection_job_ids: [...]}` for provenance |
| parquet_path | VARCHAR | Path to unified embeddings parquet |
| total_rows | INTEGER | Count of windows in the dataset |
| vocabulary | TEXT (JSON) | Array of type names present at snapshot time |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |

### `training_dataset_labels` table

| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | Auto-generated |
| training_dataset_id | VARCHAR FK | References training_datasets.id |
| row_index | INTEGER | Row in the dataset parquet |
| label | VARCHAR | Type name or "(Negative)" |
| source | VARCHAR | "snapshot" (from original) or "manual" (user edit) |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |
| **Index** | | (training_dataset_id, row_index) |

### Training dataset parquet schema

| Column | Type | Notes |
|--------|------|-------|
| row_index | int32 | Sequential 0-based |
| embedding | list\<float32\> | The embedding vector |
| source_type | string | "detection_job" or "embedding_set" |
| source_id | string | Original detection_job_id or embedding_set_id |
| filename | string | Original audio filename (for audio/spectrogram resolution) |
| start_sec | float32 | File-relative start offset |
| end_sec | float32 | File-relative end offset |
| confidence | float32 | From detection (null for embedding set rows) |

### Schema changes to existing tables

- `vocalization_models`: add `training_dataset_id` (nullable FK to training_datasets)
- `vocalization_training_jobs`: add `training_dataset_id` (nullable FK to training_datasets)

## Snapshot Process

When a training job is created:

1. Create a `training_dataset` record.
2. For each **detection job** source:
   - Read `detection_embeddings.parquet` (filename, start_sec, end_sec, embedding, confidence).
   - Query `vocalization_labels` for that detection job.
   - Match labels to embedding rows by UTC key (parse_recording_timestamp + start_sec).
   - Write matched rows into the unified parquet with `source_type="detection_job"`.
   - Copy labels into `training_dataset_labels` with `source="snapshot"`.
3. For each **embedding set** source:
   - Read embedding parquet (row_index, embedding).
   - Look up EmbeddingSet and AudioFile to get filename and processing params.
   - Compute window positions from row_index, window_size, and hop size.
   - Write rows with `source_type="embedding_set"` and computed start_sec/end_sec.
   - Create `training_dataset_labels` with folder-inferred type name, `source="snapshot"`.
4. Write unified parquet to `{storage_root}/training_datasets/{dataset_id}/embeddings.parquet`.
5. Train model from the dataset.
6. Link: `training_job.training_dataset_id = dataset_id`, `model.training_dataset_id = dataset_id`.

**Unlabeled detection job rows** (no vocalization_labels) are excluded from the snapshot — only explicitly labeled windows are included.

## Extend Dataset

An existing training dataset can be extended with new sources via `POST /vocalization/training-datasets/{id}/extend`. This:

1. Reads new source embeddings and labels using the same snapshot logic.
2. Appends rows to the existing parquet (new row_index values continue from total_rows).
3. Inserts new `training_dataset_labels` rows with `source="snapshot"`.
4. Updates `total_rows` and `vocabulary` on the dataset record.

This supports the Labeling page retrain workflow: label a new detection job, extend the model's training dataset with it, then retrain.

## Training & Retrain

**First train:** `POST /vocalization/training-jobs` with `source_config`. Creates a new training dataset, snapshots, then trains. Response includes the training_dataset_id.

**Retrain (same data, edited labels):** `POST /vocalization/training-jobs` with `training_dataset_id` (no source_config). Reads embeddings from existing parquet + current `training_dataset_labels`. Produces a new model linked to the same dataset.

**Retrain with new data (from Labeling page):** Extend the dataset first, then create a training job with `training_dataset_id`.

The training worker reads from:
- `training_datasets/{id}/embeddings.parquet` for embedding vectors
- `training_dataset_labels` for multi-hot label sets

This replaces the current worker logic that walks embedding sets and detection jobs directly.

## Spectrogram & Audio Serving

New endpoints on the training dataset router delegate to original source audio:

- `GET /vocalization/training-datasets/{id}/spectrogram?row_index=N` — reads source_type and source_id from the parquet row, then resolves audio using the existing detection job audio resolver (handles both local files and hydrophone archives) or embedding set AudioFile path.
- `GET /vocalization/training-datasets/{id}/audio-slice?row_index=N` — same delegation for audio playback.

No audio is duplicated. The original source audio must remain accessible.

## API Endpoints

### Training Datasets

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/vocalization/training-datasets` | List all datasets |
| GET | `/vocalization/training-datasets/{id}` | Dataset details |
| GET | `/vocalization/training-datasets/{id}/rows` | Paginated rows with labels |
| GET | `/vocalization/training-datasets/{id}/spectrogram` | Spectrogram PNG for a row |
| GET | `/vocalization/training-datasets/{id}/audio-slice` | Audio for a row |
| POST | `/vocalization/training-datasets/{id}/extend` | Append rows from new sources |

**`/rows` query params:**
- `type` — filter to a specific vocalization type
- `group` — `"positive"` or `"negative"` (relative to selected type)
- `offset`, `limit` — pagination (default limit 50)

Returns: row_index, filename, start_sec, end_sec, source_type, source_id, confidence, labels (array of label strings).

### Training Dataset Labels

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/vocalization/training-datasets/{id}/labels` | Create label (row_index, label) |
| DELETE | `/vocalization/training-datasets/{id}/labels/{label_id}` | Remove label |

### Changes to Existing Endpoints

- `POST /vocalization/training-jobs` accepts either `source_config` (new dataset) or `training_dataset_id` (retrain).
- `GET /vocalization/models/{id}` response includes `training_dataset_id`.

## Frontend View

### Location

New "Training Data" tab section on the Vocalization page, accessible when a model with a training dataset is selected.

### Layout

1. **Model selector** — dropdown of trained models (reuse existing). Selecting a model loads its training dataset.

2. **Type filter bar** — horizontal pills showing vocabulary types plus "(All)". Selecting a type enables positive/negative grouping.

3. **Positive/Negative toggle** — visible when a type is selected. Defaults to "Positive". Shows rows labeled with the type (positive) or rows labeled "(Negative)" / not labeled with the type (negative).

4. **Row list** — paginated (50 per page). Each row:
   - **Large inline spectrogram** (~400px wide) for direct visual comparison.
   - **Label tags** — colored pills with three visual states (saved, pending-add, pending-remove). Same interaction as labeling page.
   - **Source badge** — small indicator showing detection job name or embedding set name.
   - **Confidence** — if available.
   - Click spectrogram to open full-width dialog with audio playback.

5. **Label editing** — same batch interaction pattern as labeling page:
   - Click label to mark for removal (strikethrough).
   - "+" button to add type labels from vocabulary.
   - "(Negative)" mutual exclusivity enforced.
   - Edits accumulate locally as pending changes.

6. **Sticky Save/Cancel bar** — visible when dirty. Shows pending change count. Save batches API calls. Cancel discards.

7. **Retrain button** — in footer, enabled only when no pending edits.

### Not included

- Inference scores (this view shows raw training labels, not model predictions).
- Sort by uncertainty/score (no inference context; rows in row-index order).

## Migration

**Alembic migration 032:**
- Create `training_datasets` table.
- Create `training_dataset_labels` table.
- Add `training_dataset_id` (nullable FK) to `vocalization_models`.
- Add `training_dataset_id` (nullable FK) to `vocalization_training_jobs`.

**Cleanup:**
- Existing vocalization models and training jobs remain in DB but have null `training_dataset_id` (orphaned, non-functional for retrain).
- Worker training data assembly replaced: reads from training dataset parquet + training_dataset_labels.
- Current source-walking logic in vocalization_worker.py (embedding set folder inference, detection job label matching) moves into the snapshot step.

**Storage layout:**
```
data/training_datasets/{dataset_id}/
    embeddings.parquet
```

## What Stays the Same

- `vocalization_labels` table — still used for labeling detection job windows on the Labeling page.
- `vocalization_types` table — managed vocabulary.
- Inference jobs — model + source produces predictions.
- Labeling page — same workflow; retrain action now extends dataset + retrains.
