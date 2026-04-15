# Bootstrap Event Classifier with Synthetic Labels

**Date:** 2026-04-14
**Status:** Draft

## Problem

The event classifier feedback loop has a circular dependency: training requires labeled data from human corrections on Pass 3 output, but Pass 3 requires a trained model. There is no way to produce a first model without breaking this cycle.

## Solution

A one-off CLI script that creates a deliberately bad model by assigning random vocalization type labels to events from an existing segmentation training dataset (which has good human-corrected boundaries but no type labels). This model is bad but functional — it produces predictions that humans can correct in the Classify Review workspace, bootstrapping the feedback training loop.

## Workflow Context

```
Segmentation training dataset (human-corrected boundaries)
    ↓
Bootstrap script (random type labels → train → register model)
    ↓
Pass 3 inference with bootstrap model (mostly wrong predictions)
    ↓
Human corrections in Classify Review workspace
    ↓
Feedback training → real model
```

## Design

### Script Interface

`scripts/bootstrap_classifier.py`

**Arguments:**
- `dataset_id` (required) — UUID of a `SegmentationTrainingDataset` to source events from

**Output:**
- Trained model registered in `vocalization_models` table
- Prints model ID on completion

### Data Flow

1. **Read DB** — query `SegmentationTrainingSample` rows for the given dataset ID, plus all `VocalizationType` names from the `vocalization_types` table.

2. **Flatten events** — parse each sample's `events_json` to extract individual events with `(start_sec, end_sec)`. Each event inherits its parent sample's `hydrophone_id`, `start_timestamp`, and `end_timestamp`.

3. **Coordinate offset** — event times in `events_json` are relative to the crop. Convert to region-relative times by adding the sample's `crop_start_sec`: `event.start_sec + sample.crop_start_sec`.

4. **Assign random types** — for each event, assign a random vocalization type. Use round-robin assignment first to guarantee every type has at least `min_examples_per_type` (default 10) events, then assign remaining events uniformly at random. This prevents types from being filtered out during training.

5. **Build `_ClassifierSample` objects** — each sample exposes `.start_sec`, `.end_sec`, `.type_index`, `.hydrophone_id`, `.start_timestamp`, `.end_timestamp`, matching the interface expected by `train_event_classifier()`.

6. **Train** — call `train_event_classifier()` from `humpback.call_parsing.event_classifier.trainer` with default `EventClassifierTrainingConfig` and the full vocabulary.

7. **Register model** — insert a `VocalizationClassifierModel` row with:
   - `model_family='pytorch_event_cnn'`
   - `input_mode='segmented_event'`
   - `vocabulary_snapshot` — JSON array of type names
   - `checkpoint_path` — model directory under `storage_root/vocalization_models/<model_id>/`

### Audio Loading

Reuses the same `resolve_timeline_audio()` pattern from the feedback worker. Each sample's hydrophone metadata provides the audio source; the loader fetches audio with context padding around each event for z-score normalization.

### Existing Infrastructure Reused

| Component | Source |
|-----------|--------|
| Trainer | `humpback.call_parsing.event_classifier.trainer.train_event_classifier()` |
| Feature extraction | `humpback.call_parsing.segmentation.features` (log-mel + z-score) |
| Audio loading | `humpback.processing.timeline_audio.resolve_timeline_audio()` |
| Model architecture | `humpback.call_parsing.event_classifier.model.EventClassifierCNN` |
| Train/val split | `humpback.call_parsing.segmentation.trainer.split_by_audio_source()` |

### What the Script Does NOT Do

- Run Pass 3 inference (done manually via UI afterward)
- Create job records in the database (no `EventClassifierTrainingJob` row)
- Persist synthetic labels (they exist only in memory during training)
- Produce a good model (the model is intentionally bad)
