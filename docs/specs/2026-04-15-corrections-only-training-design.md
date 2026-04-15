# Corrections-Only Training Mode for Event Classifier Feedback

**Date:** 2026-04-15
**Status:** Approved

## Problem

When the event classifier is bootstrapped (cold-start), it assigns the same
type to all or most events with high confidence. If the user corrects a subset
of events and retrains, the current `_resolve_event_labels` logic falls back to
machine inference for uncorrected events. This means ~18 correct labels are
overwhelmed by ~1250 identical bootstrap pseudo-labels, producing a model that
reproduces the bootstrap's poor behavior.

## Design

Add a `corrections_only` boolean (default `True`) to the training config. When
enabled, `_resolve_event_labels` only includes events with explicit human type
corrections — uncorrected events are labeled `None` and excluded from training.

### Backend changes

1. Add `corrections_only: bool = True` to:
   - `EventClassifierTrainingConfig` Pydantic model in `schemas/call_parsing.py`
   - `EventClassifierTrainingConfig` dataclass in `event_classifier/trainer.py`

2. In `event_classifier_feedback_worker.py`:
   - Pass `corrections_only` from the deserialized config through to
     `_collect_samples` and then to `_resolve_event_labels`
   - When `corrections_only=True`, the `else` branch in `_resolve_event_labels`
     sets `labels[event_id] = None` instead of using inference predictions

No migration needed — the field is stored inside the existing `config_json`
TEXT column, and `True` default preserves backward compatibility (old jobs
without the field in their JSON get the safe default when deserialized).

### Frontend changes

1. **ClassifyReviewWorkspace** retrain button: add a checkbox "Corrected events
   only" (default checked) near the Retrain button. Pass the value in the
   `config` field of `CreateClassifierTrainingJobRequest`.

2. **ClassificationJobPicker** training page: add the same checkbox (default
   checked) near the Train Model button.

### What does NOT change

- The `event_boundary_corrections` or `event_type_corrections` tables
- The `EventClassifierTrainingJob` model or schema
- The audio loading or feature extraction pipeline
- Segmentation feedback training (separate worker, unaffected)
