# Audio Loader Consolidation — Follow-up Report

**Date:** 2026-04-15
**Context:** Discovered during event classifier training collapse fix (feature/training-collapse-fix-v2)

## Problem

Five independent `_build_audio_loader` implementations exist across four workers. Each re-derives the same pattern (resolve timeline audio for a time range, return an array) with subtle differences in coordinate handling, return types, and caching strategy. The training collapse bug was caused by one of these copies getting the relative-vs-absolute coordinate conversion wrong.

## Current Implementations

| Worker | Function | Bounds convention | Return type | Caching |
|--------|----------|-------------------|-------------|---------|
| `event_classification_worker.py` | `_build_hydrophone_audio_loader` | Relative, adds `job_start_ts` | `(audio, load_start)` | Pre-loads full event span once |
| `event_classification_worker.py` | `_build_audio_loader` | File-based, offset=0 | `(audio, 0.0)` | Pre-loads full file once |
| `segmentation_training_worker.py` | `_build_audio_loader` | Relative, adds `start_ts` | `np.ndarray` (no offset) | Per-sample, no caching |
| `event_classifier_feedback_worker.py` | `_build_audio_loader` | Relative, adds `start_ts` (fixed) | `(audio, rel_offset)` | Per-sample, no caching |
| `event_segmentation_worker.py` | `_build_file_audio_loader` | File-based | closure over pre-loaded audio | Pre-loads full file once |
| `event_segmentation_worker.py` | `_build_hydrophone_audio_loader` | Absolute epoch | closure | Pre-loads region span |

## Risks of Current State

1. **Coordinate bugs** — each copy independently converts between relative/absolute timestamps. The training collapse was caused by one copy getting this wrong. The same bug existed in `scripts/bootstrap_classifier.py`.
2. **Protocol drift** — `AudioLoader` in `dataset.py` was updated to `tuple[np.ndarray, float]` but `segmentation_training_worker.py` still returns bare `np.ndarray`. The next consumer to share an `AudioLoader` type will hit this mismatch.
3. **No caching in training loaders** — the feedback and segmentation training workers call `resolve_timeline_audio` per-sample. The classification worker pre-loads the full span once. Training workers should do the same for performance.

## Suggested Consolidation

A single factory function in a shared module (e.g., `src/humpback/processing/audio_loader.py`) that:

1. Accepts `hydrophone_id`, `job_start_ts`, `job_end_ts`, `settings`, and an optional list of events/samples for pre-loading
2. Returns a `Callable[[sample], tuple[np.ndarray, float]]` matching the `AudioLoader` protocol
3. Handles the `start_ts + relative_offset` conversion in one place
4. Supports both per-sample and pre-loaded-span strategies via a parameter
5. File-based sources return `(audio, 0.0)` for protocol compatibility

This would eliminate all five private implementations and centralize the coordinate conversion that caused the training collapse.

## Files to Change

- `src/humpback/workers/event_classification_worker.py` (lines 66-114)
- `src/humpback/workers/event_classifier_feedback_worker.py` (lines 207-244)
- `src/humpback/workers/segmentation_training_worker.py` (lines 52-81)
- `src/humpback/workers/event_segmentation_worker.py` (lines 93-168)
- `scripts/bootstrap_classifier.py` (lines 137-166)
- `src/humpback/call_parsing/event_classifier/dataset.py` (AudioLoader type)
