# Bootstrap Hydrophone Support — Design Spec

**Date:** 2026-04-12
**Status:** Approved

## Problem

The call parsing pipeline's bootstrap scripts (`bootstrap_segmentation_dataset.py`, `bootstrap_event_classifier_dataset.py`) and the segmentation training worker only support file-based detection jobs. All vocalization-labeled detection data in this project is hydrophone-sourced (Orcasound Lab HLS streams). The call parsing workflow will only support hydrophone audio — no file-based path is needed.

## Scope

Three components need hydrophone support:

1. **`scripts/bootstrap_segmentation_dataset.py`** — creates `SegmentationTrainingSample` rows from vocalization-labeled detection rows
2. **Segmentation training worker** (`src/humpback/workers/segmentation_training_worker.py`) — loads audio for training samples
3. **`scripts/bootstrap_event_classifier_dataset.py`** — runs Pass 2 segmentation on detection windows to discover events for Pass 3 training

## Design

### Simplification: hydrophone-only

Since the call parsing workflow is hydrophone-only, all file-based audio resolution code is removed from both bootstrap scripts:
- `_build_audio_index`, `_resolve_file_for_row`, `_file_base_epoch` — deleted
- `AudioFile` lookups — deleted
- `audio_folder` checks — deleted

Detection jobs must have `hydrophone_id` set; jobs without it are skipped with a clear warning.

### Component 1: `bootstrap_segmentation_dataset.py`

This script never loads audio — it writes crop metadata to `SegmentationTrainingSample` rows.

**Hydrophone flow:**
- Detection job provides `hydrophone_id`, `start_timestamp`, `end_timestamp`
- Each detection row provides absolute `start_utc`/`end_utc` from the parquet row store
- Crop window: center on the detection event, extend to `--crop-seconds`, clamp to the detection job's `[start_timestamp, end_timestamp]` range
- Sample row fields:
  - `hydrophone_id` = detection job's hydrophone_id
  - `start_timestamp` = crop start (absolute UTC epoch)
  - `end_timestamp` = crop end (absolute UTC epoch)
  - `crop_start_sec` = 0.0
  - `crop_end_sec` = crop duration
  - `events_json` = event bounds relative to crop start (i.e., `[{"start_sec": event_start - crop_start, "end_sec": event_end - crop_start}]`)
  - `audio_file_id` = None

### Component 2: Segmentation training worker

Add a hydrophone branch to `_build_audio_loader()`.

When a sample has `hydrophone_id` (instead of `audio_file_id`):
- Call `resolve_timeline_audio()` with:
  - `hydrophone_id` = sample's hydrophone_id
  - `job_start_timestamp` = sample's `start_timestamp`
  - `job_end_timestamp` = sample's `end_timestamp`
  - `start_sec` = `crop_start_sec` (will be 0.0 for bootstrap samples)
  - `duration_sec` = `crop_end_sec - crop_start_sec`
  - `target_sr` from feature config
- Return the buffer directly (the fetch IS the crop)

The `resolve_timeline_audio()` function handles provider dispatch (Orcasound HLS / NOAA GCS), segment caching, and gap-filling with silence.

### Component 3: `bootstrap_event_classifier_dataset.py`

This script loads audio to run segmentation inference on each detection window.

**Hydrophone flow:**
- Detection job provides `hydrophone_id`, `start_timestamp`, `end_timestamp`
- For each qualifying window, fetch audio via `resolve_timeline_audio()` using the window's `start_utc`/`end_utc`
- Run segmentation inference on the fetched buffer (same as file-based)
- Output samples with `hydrophone_id` + absolute UTC start/end instead of `audio_file_id`

### Testing

- Unit tests for bootstrap logic with mocked DB sessions and synthetic detection rows/labels
- Integration validation: dry-run against real DB to verify row discovery and crop computation
