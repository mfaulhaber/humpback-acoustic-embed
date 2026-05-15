# Segment Training Page Rework Design

**Date:** 2026-05-15
**Status:** Draft

## Problem

The Call Parsing Segment Training page currently exposes the implementation
detail behind Pass 2 retraining: users first create a persisted training dataset
from selected segmentation jobs, then train a model from that dataset. This
creates a layout and mental model that diverges from the Classifier Training
page, where trained models, a training form, and previous jobs are presented as
one cohesive workflow.

The page also cannot show queued or running segmentation training jobs because
there is no frontend read path for `segmentation_training_jobs`. A user who
starts training sees no queued/training item in the page until the worker
finishes and creates a segmentation model row.

## Goals

- Rework the Segment Training page to match the Classifier Training page
  hierarchy:
  - top panel: existing segmentation models
  - second panel: train a new segmentation model
  - following panel: previous segmentation training jobs with status
- Let users train directly from selected completed segmentation jobs with
  corrections.
- Remove the visible training-dataset creation and management UI from this
  page.
- Keep the existing dataset tables as the backend contract used by the worker.
- Add an advanced section to the training form for overriding
  `SegmentationFeatureConfig` and `SegmentationTrainingConfig` defaults from
  `src/humpback/schemas/call_parsing.py`.
- Ensure queued and running training jobs appear immediately in the previous
  jobs panel with their status.

## Non-Goals

- No database migration.
- No deletion UI for segmentation training jobs.
- No removal of the existing training dataset API endpoints, tables, or service
  helpers; they remain useful as the persistence layer and for compatibility.
- No change to correction semantics: dataset extraction continues to use
  segmentation-scoped corrections first with the existing safe legacy fallback.
- No change to Pass 2 inference job creation or the Segment Review quick-retrain
  button beyond shared invalidation behavior.

## Current State

`SegmentTrainingPage` renders three independent sections:

- `SegmentationJobPicker`: selects completed segmentation jobs and creates a
  saved training dataset.
- `TrainingDatasetTable`: lists saved datasets and queues training jobs.
- `SegmentModelTable`: lists trained segmentation models.

The backend route `POST /call-parsing/segmentation-training-jobs` accepts only
`training_dataset_id`, so the frontend must expose dataset creation before it
can queue a job. There is no `GET /call-parsing/segmentation-training-jobs`
route for the previous-jobs panel.

`SegmentationTrainingConfig` stores optimizer, split, and model-architecture
knobs, while the worker constructs `SegmentationFeatureConfig` with only
`n_mels` overridden from the training config. The UI has no way to override the
feature-extractor defaults such as `sample_rate`, `n_fft`, `hop_length`,
`fmin`, `fmax`, or `normalize`.

## Recommended Design

### Page Layout

Rework `SegmentTrainingPage` into this order:

1. `SegmentModelTable`
2. New direct training panel
3. New previous training jobs panel

The model table remains visually equivalent to today, just moved to the top.
The training panel uses the same compact form-and-advanced-section pattern as
`classifier/TrainingTab.tsx`.

### Direct Training Flow

The training panel lists completed segmentation jobs from
`GET /call-parsing/segmentation-jobs/with-correction-counts`. Users can select
one or more rows, then click a single train button. The selected jobs are sent
to `POST /call-parsing/segmentation-training-jobs`.

The backend keeps persisted training datasets as an implementation detail:

1. Validate selected jobs using the existing dataset-from-corrections path.
2. Create the saved `SegmentationTrainingDataset` and samples internally.
3. Queue a `SegmentationTrainingJob` against the created dataset.
4. Return the queued job response.

The existing dataset endpoints stay in place for compatibility, but the Segment
Training page no longer calls them directly.

### Request Schema

Extend `CreateSegmentationTrainingJobRequest` so it supports either:

- `training_dataset_id`: existing compatibility path
- `segmentation_job_ids`: new direct page path

Exactly one source mode must be provided. `segmentation_job_ids` requires
settings access in the router so the service can extract correction-backed
samples. The service should surface the same 400/404 behavior as the existing
dataset creation endpoint for invalid or empty selections.

### Advanced Configuration

Add a nested `feature_config` field to `SegmentationTrainingConfig` with
`SegmentationFeatureConfig` defaults. Preserve compatibility with existing job
JSON that only contains the old flat training fields by materializing
`feature_config.n_mels` from the flat `n_mels` field when the nested config is
absent.

The training worker should use `training_config.feature_config` instead of
constructing `SegmentationFeatureConfig(n_mels=training_config.n_mels)`.
Validation should keep the model architecture `n_mels` aligned with the feature
extractor `n_mels`.

The advanced UI should expose the conservative set of defaults users are likely
to tune:

- optimization: epochs, batch size, learning rate, weight decay, early stopping
  patience, grad clip, validation fraction, seed
- architecture: n mels, convolution channels, GRU hidden size, GRU layers
- feature extraction: sample rate, FFT size, hop length, min frequency, max
  frequency, normalization

The form should submit config overrides only when advanced values differ from
defaults, but the backend can also accept complete configs safely.

### Previous Jobs Panel

Add `GET /call-parsing/segmentation-training-jobs` returning
`SegmentationTrainingJobResponse[]` ordered by newest first. The frontend hook
polls this endpoint while any job is queued or running.

The previous jobs panel shows:

- status badge
- source dataset ID or short ID
- produced model ID when available
- created timestamp
- completion timestamp or error message
- compact config summary for advanced overrides

When the train mutation succeeds, invalidate both segmentation training jobs
and segmentation models. The queued job will appear immediately in Previous
Jobs, then the completed model appears in the top Models panel when the worker
finishes.

## Alternatives Considered

### Keep Dataset Management UI and Reorder Sections

This would be the smallest UI change, but users would still need to understand
dataset creation as a separate product step. It does not satisfy the request to
remove dataset-building UI.

### Add a New `/from-corrections` Training Endpoint

A dedicated endpoint would avoid widening the existing training job request,
but it would split "create a segmentation training job" across two public
routes. Extending the existing endpoint keeps the frontend hook and route name
aligned with the product action.

### Remove Training Dataset Tables Entirely

This would simplify the conceptual model but requires a migration and worker
contract rewrite. ADR-050 deliberately made persisted training datasets the
Pass 2 training contract, and keeping that layer avoids churn while still
improving the page.

## Tests

- Backend router tests for:
  - listing segmentation training jobs
  - creating a training job directly from segmentation job IDs
  - preserving creation from `training_dataset_id`
  - rejecting requests with both or neither source modes
  - accepting advanced feature/training config overrides
- Worker or schema tests for:
  - old config JSON without nested `feature_config`
  - aligned nested feature config being used by the worker
  - mismatched `n_mels` validation
- Frontend tests for:
  - models panel appears above the training panel
  - training form posts selected segmentation job IDs directly
  - advanced section exposes defaults and submits overrides
  - previous jobs panel shows queued/running status
- Verification:
  - targeted call parsing backend tests
  - frontend TypeScript check
  - call parsing Playwright smoke if practical
  - full backend pytest suite before session end
