# Segmentation Dataset Extraction from Corrections — Design Spec

**Date:** 2026-04-13
**Status:** Approved

## Problem

The Pass 2 segmentation feedback training loop does not produce usable models.
Fine-tuning the bootstrap CRNN on <50 correction samples destroys its learned
discrimination — output range compresses from 0.0001–0.9999 to a narrow
~0.24–0.84 band regardless of approach (frozen conv, unfrozen, pretrained init).

The bootstrap model itself has good spectral discrimination but was trained on
synthetic 5-second vocalization label blobs, not real call boundaries. Training
from scratch on human-annotated call boundaries is the path forward.

## Solution

Build an API-driven extraction pipeline that reads human boundary corrections
from a completed segmentation job, converts them into a
`SegmentationTrainingDataset` with `SegmentationTrainingSample` rows, and feeds
that dataset into the existing `SegmentationTrainingJob` pipeline for
from-scratch training.

## Design Decisions

1. **Only corrected regions are included.** Uncorrected regions are excluded
   entirely — no ambiguous labels. 18 of 38 regions in the initial job have
   corrections (608 total: 586 adds, 22 adjustments).

2. **Negative samples come from implicit gaps.** Gaps between annotated events
   within corrected regions provide natural ocean-background negatives. No
   external negative sources needed — corrected regions contain real ambient
   noise, avoiding z-score normalization artifacts from pure silence.

3. **Extraction writes to `segmentation_training_samples`.** The existing table
   has the right shape (audio source, crop bounds, `events_json`). The existing
   `train_model()` pipeline is reused unchanged.

4. **API endpoint, not CLI script.** Fits the existing pattern where dataset
   creation and training jobs are API-driven. Enables future UI integration.

5. **Re-extraction creates a new dataset.** Running extraction again after
   correcting more regions produces a new `SegmentationTrainingDataset` (not
   update-in-place). Old datasets remain for reproducibility.

6. **Feedback worker left in place.** The existing `EventSegmentationTrainingJob`
   path is not removed — it may be useful for future fine-tuning experiments.

7. **No decoder threshold changes.** The new model's output distribution is
   unknown. Evaluate after training, then tune thresholds.

8. **No conv-freezing flag needed.** The trainer only freezes conv layers when a
   pretrained checkpoint is provided. From-scratch training has no checkpoint,
   so freezing doesn't trigger.

## Architecture

### Shared Extraction Module

**New file:** `src/humpback/call_parsing/segmentation/extraction.py`

Three functions extracted from the feedback worker into a shared module:

- `apply_corrections(original_events, corrections)` — applies delete/adjust/add
  corrections to a region's original events, returns corrected event dicts.

- `subdivide_region(crop_start, crop_end, corrected_events, hydrophone_id,
  start_timestamp, end_timestamp)` — splits long regions into 30-second crops
  with 15-second hops. Returns `CorrectedSample` dataclass instances.

- `collect_corrected_samples(session, segmentation_job_id, storage_root)` —
  reads regions and corrections for a single segmentation job, applies
  corrections, subdivides into crops, returns samples for corrected regions
  only.

`CorrectedSample` is a plain dataclass replacing the worker's private
`_FeedbackSample`: `hydrophone_id`, `start_timestamp`, `end_timestamp`,
`crop_start_sec`, `crop_end_sec`, `events_json`.

### API Endpoint

`POST /call-parsing/segmentation-training-datasets/from-corrections`

Request body:
```json
{
  "segmentation_job_id": "01f3d822-...",
  "name": "corrections-01f3d822",
  "description": "Human-annotated call boundaries from segmentation review"
}
```

`name` is optional (auto-generated as `"corrections-{job_id[:8]}"` if omitted).
`description` is optional.

Behavior:
1. Validate the segmentation job exists and is complete
2. Call `collect_corrected_samples()` from the extraction module
3. Return 400 if no corrected regions found
4. Create `SegmentationTrainingDataset` row
5. Create `SegmentationTrainingSample` rows (one per crop) with
   `source="boundary_correction"` and `source_ref=segmentation_job_id`
6. Return dataset ID and sample count

### Service Layer

`create_dataset_from_corrections(session, segmentation_job_id, name, description)`
in `src/humpback/services/call_parsing.py`:

1. Fetch `EventSegmentationJob`, verify exists and complete
2. Resolve upstream `RegionDetectionJob` for hydrophone context
3. Call `collect_corrected_samples()`
4. Raise if empty
5. Create `SegmentationTrainingDataset` row
6. Bulk-insert `SegmentationTrainingSample` rows
7. Return the dataset

### Feedback Worker Refactor

Minimal change to `event_segmentation_feedback_worker.py`:
- Remove `_apply_corrections`, `_subdivide_region`, `_FeedbackSample`
- Import `apply_corrections`, `subdivide_region`, `CorrectedSample` from
  `extraction.py`
- Update `_collect_samples` to delegate per-region logic to shared functions

Training execution, model saving, and error handling untouched.

## Iterative Workflow

The intended usage cycle:

1. Run segmentation on a detection job with the current best model
2. Review events in Segment Review UI, correct boundaries
3. `POST .../from-corrections` to extract a dataset
4. `POST .../segmentation-training-jobs` to train from scratch
5. Use the new model for step 1 on different detection jobs
6. Repeat — corrections accumulate, datasets grow, models improve

## Key Findings to Preserve

- Decoder thresholds matter more than model changes when the model already has
  discrimination
- Fine-tuning a segmentation CRNN on <50 samples destroys learned features
  regardless of approach
- The bootstrap model's 0.0001–0.9999 range means it learned useful spectral
  features from the 158-sample bootstrap — but with wrong temporal targets
- Z-score normalization on pure silence produces artifacts — implicit negatives
  from real ocean background avoid this
