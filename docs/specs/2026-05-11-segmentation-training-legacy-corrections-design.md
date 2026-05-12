# Segmentation Training Dataset Legacy Correction Inclusion

**Date:** 2026-05-11
**Status:** Draft

## Problem

The Segment Training page is intended to create datasets from all available
human boundary corrections on selected completed segmentation jobs. The current
UI/API split makes that promise unreliable for historical data:

- `GET /call-parsing/segmentation-jobs/with-correction-counts` counts boundary
  corrections by `region_detection_job_id`, so it includes older
  region-scoped rows whose `event_segmentation_job_id` is null.
- `POST /call-parsing/segmentation-training-datasets/from-corrections` calls
  `collect_corrected_samples()`, which only reads corrections where
  `event_boundary_corrections.event_segmentation_job_id` equals the selected
  segmentation job id.
- A selected job with only legacy region-scoped corrections is therefore shown
  as available in the picker but silently contributes zero samples to the
  dataset.

Observed production example:

- Dataset `test-3-datasets` selected three Orcasound segmentation jobs.
- The saved dataset contains 137 samples from two source jobs:
  `49e94a4a-61bc-4a20-98e8-b1675286c985` contributed 111 samples, and
  `c8db1123-00a0-4f4a-9b81-0ab8594dc276` contributed 26 samples.
- The missing selected job, `431749b6-afd4-415d-9167-b886f7e6e7b8`, has 291
  legacy region-scoped corrections and no segmentation-scoped corrections.
  Its region detection job has only this one segmentation job, so the legacy
  corrections are unambiguous for dataset extraction.

The result is confusing: the UI reports that three jobs were selected, the
dataset summary correctly reports two contributing source jobs, and the sample
count is lower than expected.

## Goals

- Include all useful human boundary corrections in new Segment Training
  datasets, including legacy region-scoped corrections when they can be safely
  interpreted for the selected segmentation job.
- Preserve ADR-062's current model: modern boundary corrections are owned by
  `event_segmentation_job_id`, and immutable parquet artifacts are not mutated.
- Avoid silently skipping selected jobs that have correction rows.
- Make the creation response and UI copy distinguish selected jobs from
  contributing jobs.
- Add regression coverage for mixed modern and legacy correction datasets.

## Non-Goals

- No automatic database-wide migration or backfill of historical correction
  rows.
- No attempt to infer ambiguous legacy correction intent when multiple
  segmentation jobs share the same region detection job.
- No recreation or mutation of existing datasets. A user may delete and
  recreate an affected dataset after the fix.
- No change to training sample semantics: samples remain corrected region crops,
  not one row per correction.

## Context

ADR-062 established segmentation-scoped effective event identity. New boundary
correction writes populate `event_segmentation_job_id` and `source_event_id`,
and downstream event-aware consumers must explicitly choose raw or effective
event semantics.

The codebase already keeps a compatibility path in
`load_corrected_events()`: if scoped correction rows exist for a segmentation
job, it uses those; otherwise it can overlay legacy rows where
`event_segmentation_job_id IS NULL` for the region detection job. Segment
Training dataset extraction should use equivalent compatibility semantics.

## Approaches Considered

### Approach A: Keep Extraction Strictly Segmentation-Scoped

Change only the picker/query to exclude legacy region-scoped corrections, so
users can no longer select jobs that extraction will skip.

**Pros**

- Smallest backend behavior change.
- Aligns strictly with ADR-062's preferred modern ownership model.
- Avoids any risk of applying old corrections to the wrong segmentation output.

**Cons**

- Throws away useful historical human corrections from early review sessions.
- Does not satisfy the product goal of building a dataset from all available
  human corrections.
- Would make the observed `431749b6...` corrections unavailable despite being
  unambiguous.

**Verdict:** rejected. This fixes the display mismatch by hiding data that is
still useful.

### Approach B: Compatibility Fallback During Dataset Extraction

Teach `collect_corrected_samples()` to resolve correction rows in this order:

1. Use segmentation-scoped corrections for the selected
   `event_segmentation_job_id` when any exist.
2. If no scoped rows exist, look for legacy region-scoped corrections for the
   job's `region_detection_job_id`.
3. Use legacy rows only when they are unambiguous for the selected segmentation
   job. The initial guard is: the region detection job has exactly one
   segmentation job. If there are multiple segmentation jobs, skip legacy rows
   for that source and report that they were skipped as ambiguous.

Dataset rows still use `source="boundary_correction"` and
`source_ref=<selected segmentation_job_id>`, so the resulting dataset summary
counts contributing source jobs the same way it does today.

**Pros**

- Recovers useful historical corrections without mutating the correction table.
- Matches the existing `load_corrected_events()` compatibility intent.
- Keeps modern scoped rows authoritative when both modern and legacy rows exist.
- Provides a clear ambiguity rule for older data.
- Small, targeted change to the extraction path.

**Cons**

- Requires care in response/UI messaging so users know when selected jobs were
  skipped because they had no usable corrections.
- Legacy rows remain legacy; future consumers still need compatibility handling
  or explicit migration if they want to use them outside dataset extraction.

**Verdict:** recommended.

### Approach C: Backfill Legacy Corrections In Place

Add a migration or one-off command that sets `event_segmentation_job_id` on
legacy correction rows whenever the owning region detection job has exactly one
segmentation job.

**Pros**

- Normalizes the data so all future consumers can use one scoped query.
- Makes correction counts and extraction naturally agree.

**Cons**

- Mutates historical review data and requires the production database backup
  workflow.
- Still needs ambiguity handling for region jobs with multiple segmentation
  jobs.
- ADR-062 explicitly treated historical cleanup/backfill as manual.
- Larger operational surface than needed to fix dataset creation.

**Verdict:** defer. A later cleanup command can be useful, but dataset creation
does not need it.

## Recommended Design

Implement Approach B.

### Correction Resolution

Add a small helper in `humpback.call_parsing.segmentation.extraction` that
returns the correction rows used for a selected segmentation job and a short
resolution mode:

- `scoped`: one or more rows matched `event_segmentation_job_id`.
- `legacy_unambiguous`: no scoped rows matched, legacy rows matched the region
  detection job, and the region detection job has exactly one segmentation job.
- `legacy_ambiguous`: no scoped rows matched, legacy rows exist, but the region
  detection job has multiple segmentation jobs.
- `none`: no scoped or legacy rows exist.

`collect_corrected_samples()` should use rows from `scoped` or
`legacy_unambiguous`. It should return no samples for `legacy_ambiguous` or
`none`.

Modern scoped rows remain authoritative. If a job has both scoped and legacy
rows, extraction uses only scoped rows to avoid double-applying stale legacy
intent.

### Creation Accounting

Extend `create_dataset_from_corrections()` internally to track per-selected-job
outcomes:

- selected job id
- number of samples contributed
- correction resolution mode
- optional skipped reason

The database row model does not need to change. The public create response
should be expanded to include:

- `sample_count`
- `selected_job_count`
- `source_job_count`
- `skipped_job_count`

Optionally include a compact `skipped_jobs` list with job ids and reasons. This
is useful for the toast and for debugging, but the first implementation can keep
it simple if the team prefers not to expose per-job details yet.

`list_segmentation_training_datasets()` can continue deriving
`source_job_count` from distinct `SegmentationTrainingSample.source_ref`, since
that reflects actual dataset provenance.

### Frontend Copy

Update `SegmentationJobPicker` success copy from:

`N samples from M jobs`

to wording based on backend accounting, for example:

`N samples from K contributing jobs`

If `skipped_job_count > 0`, add a second sentence:

`S selected jobs had no usable corrections.`

This prevents the toast from claiming selected jobs contributed when extraction
skipped them.

### Existing Dataset Handling

Existing datasets are immutable snapshots for practical purposes. The fix
applies to new dataset creation. For `test-3-datasets`, deleting and recreating
the dataset after the fix is the clean path. Based on current production data,
the recreated dataset should include the two modern-scoped jobs plus the
unambiguous legacy `431749b6...` corrections, likely increasing from 137 samples
to roughly 167 samples.

## Tests

Backend targeted tests:

- Add an integration test where one selected job has modern
  segmentation-scoped corrections and one selected job has only legacy
  region-scoped corrections with exactly one segmentation job for the region.
  Assert the created dataset includes samples from both source job ids.
- Add an integration test where legacy region-scoped corrections exist but the
  region detection job has multiple segmentation jobs. Assert the ambiguous job
  contributes no samples and the response reports a skipped job.
- Add a regression assertion that modern scoped rows win when both scoped and
  legacy rows exist for the same region detection job.
- Update create-response schema tests for `selected_job_count`,
  `source_job_count`, and `skipped_job_count`.

Frontend targeted tests:

- Update `SegmentationJobPicker` or relevant E2E expectations so the success
  toast reports contributing jobs, not selected jobs.

Suggested verification:

- `uv run pytest tests/integration/test_dataset_from_corrections.py -q`
- `uv run pytest tests/integration/test_call_parsing_router.py -q`
- `cd frontend && npx tsc --noEmit`

Final backend gate remains `uv run pytest tests/`.

## Acceptance Criteria

- Creating a dataset from all three available Orcasound jobs includes samples
  from the unambiguous legacy-corrected job `431749b6...`.
- Dataset summaries continue to report actual contributing source jobs.
- Dataset creation no longer silently hides selected jobs with correction rows;
  skipped jobs are reflected in the response and UI messaging.
- No correction rows or parquet artifacts are mutated during dataset creation.
