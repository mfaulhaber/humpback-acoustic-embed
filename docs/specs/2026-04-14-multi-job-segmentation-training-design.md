# Multi-Job Segmentation Training Datasets

**Date:** 2026-04-14
**Status:** Approved

## Problem

Segmentation model training currently supports only a single segmentation job as
the source of boundary corrections. To build robust models, users need to combine
corrections from multiple 24-hour detection jobs — first across days on the same
hydrophone, then eventually across hydrophone locations.

Additionally, two parallel training paths exist (standard `SegmentationTrainingJob`
and feedback `EventSegmentationTrainingJob`) that do essentially the same thing
with different persistence models. This creates maintenance overhead.

## Solution

1. Extend the dataset-from-corrections endpoint to accept multiple segmentation
   job IDs.
2. Consolidate on the standard `SegmentationTrainingDataset` →
   `SegmentationTrainingJob` path as the single canonical training pipeline.
3. Remove the feedback training path (`EventSegmentationTrainingJob`, its worker,
   endpoints, and frontend components).
4. Rework the Segment Training page with a multi-select job picker, dataset
   management, and paginated job listing.

## API & Schema Changes

### Modified endpoint

`POST /call-parsing/segmentation-training-datasets/from-corrections`

Request body changes from:

```json
{ "segmentation_job_id": "...", "name": "...", "description": "..." }
```

to:

```json
{ "segmentation_job_ids": ["...", "..."], "name": "...", "description": "..." }
```

- `segmentation_job_ids`: list of completed segmentation job IDs (min_length=1).
- Each job is validated as complete. Jobs with zero corrections are silently
  skipped (valid but nothing to extract).
- Raises 400 if no samples are collected across all provided jobs.
- Each `SegmentationTrainingSample.source_ref` records which job the sample came
  from, preserving per-job lineage.
- Default dataset name: `corrections-{N}jobs-{first_id[:8]}` for multi-job, or
  `corrections-{id[:8]}` for single-job.

### New endpoint

`GET /call-parsing/segmentation-jobs/with-correction-counts`

Returns all completed segmentation jobs annotated with their boundary correction
count. Powers the frontend job picker without N+1 queries.

Response: list of objects with the existing `EventSegmentationJobSummary` fields
plus `correction_count: int`.

### Removed endpoints

- `POST /call-parsing/segmentation-feedback-training-jobs`
- `GET /call-parsing/segmentation-feedback-training-jobs`
- `GET /call-parsing/segmentation-feedback-training-jobs/{job_id}`
- `DELETE /call-parsing/segmentation-feedback-training-jobs/{job_id}`

## Service Layer Changes

### `create_dataset_from_corrections()`

Refactored to accept `segmentation_job_ids: list[str]`. Loops over each job ID,
validates each is complete, calls `collect_corrected_samples()` per job, and
accumulates all samples into one `SegmentationTrainingDataset`. Raises if no
samples are collected across all jobs combined.

### New: `create_dataset_and_train()`

Convenience function for the SegmentReviewWorkspace single-job retrain flow.
Creates a dataset from one job's corrections and immediately queues a
`SegmentationTrainingJob` against it. Atomic — both created in the same
transaction. Exposed as:

`POST /call-parsing/segmentation-training/quick-retrain`

Request body: `{ "segmentation_job_id": "..." }`

Returns the created dataset ID, training job ID, and sample count.

### Removed service functions

- `create_segmentation_feedback_training_job()`
- `list_segmentation_feedback_training_jobs()`
- `get_segmentation_feedback_training_job()`
- `delete_segmentation_feedback_training_job()`

## Database Changes

### No new tables or columns

The existing schema supports multi-job datasets as-is:
- `SegmentationTrainingDataset` is a named container — works unchanged.
- `SegmentationTrainingSample.source_ref` records the originating job ID per
  sample — multi-job datasets simply have samples with different `source_ref`
  values.
- `SegmentationTrainingJob` trains from a dataset — unchanged.

### Table removal

Alembic migration to drop the `event_segmentation_training_jobs` table. Existing
`SegmentationModel` rows whose `training_job_id` pointed to feedback training
jobs retain that value as a historical breadcrumb — the model files and config
are self-contained.

## Worker Changes

### Unchanged

`segmentation_training_worker.py` — already reads samples from a dataset and
trains. No modifications needed.

### Removed

- `event_segmentation_feedback_worker.py` (entire file)
- Worker dispatch branch that claims `EventSegmentationTrainingJob` rows

## Frontend Changes

### SegmentTrainingPage (reworked)

Three sections replacing the current two-table layout:

**1. Job picker + dataset creation**

Paginated table of completed segmentation jobs (fixed page size, prev/next
navigation). Columns:

| Column | Description |
|--------|-------------|
| Checkbox | Multi-select |
| Hydrophone | Short-id moniker (same as review page) |
| Date range | Job time span |
| Correction count | Number of boundary corrections |

Below the table: name/description inputs and a "Create Training Dataset" button
that calls the updated endpoint with checked job IDs.

**2. Training datasets table (new)**

Lists existing datasets: name, sample count, source job count, creation date.
"Train" button per dataset queues a `SegmentationTrainingJob`.

**3. Models table**

Existing `SegmentModelTable`, unchanged.

### SegmentReviewWorkspace retrain button

Changes from creating a feedback training job to calling `create_dataset_and_train()`
(create single-job dataset + queue training job). Same UX — one click, toast
confirmation, poll for completion.

### Removed frontend components

- `FeedbackTrainingJobTable` component
- `useSegmentationFeedbackTrainingJobs` hook
- `useCreateSegmentationFeedbackTrainingJob` hook
- `useDeleteSegmentationFeedbackTrainingJob` hook
- Related API types (`SegmentationFeedbackTrainingJob`)

### New frontend hooks

- `useSegmentationJobsWithCorrectionCounts()` — fetches job picker data
- `useCreateSegmentationTrainingDataset()` — creates multi-job dataset
- `useSegmentationTrainingDatasets()` — lists datasets
- `useCreateSegmentationTrainingJob()` — queues training from a dataset

## Removals Summary

| Layer | What | File/Location |
|-------|------|---------------|
| Model | `EventSegmentationTrainingJob` class | `models/feedback_training.py` |
| Worker | Feedback training worker | `workers/event_segmentation_feedback_worker.py` |
| Worker | Dispatch branch for feedback jobs | Worker loop |
| API | 4 feedback training endpoints | `api/routers/call_parsing.py` |
| Service | 4 feedback training functions | `services/call_parsing.py` |
| Schemas | Request + response models | `schemas/call_parsing.py` |
| Frontend | `FeedbackTrainingJobTable` | `components/call-parsing/` |
| Frontend | Feedback training hooks + types | `hooks/queries/`, `api/types` |
| DB | `event_segmentation_training_jobs` table | Alembic migration |

## What Stays

- `EventBoundaryCorrection` model and all correction endpoints — source data
- `SegmentationModel` and model management endpoints — unchanged
- `SegmentationTrainingDataset`, `SegmentationTrainingSample`,
  `SegmentationTrainingJob` models — unchanged
- `segmentation_training_worker.py` — unchanged
- `collect_corrected_samples()` extraction logic — unchanged (called per job)
- `train_model()` trainer — unchanged (receives samples, produces checkpoint)
