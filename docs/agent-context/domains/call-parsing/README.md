# Call Parsing Domain

Load this domain for region detection, event segmentation, event
classification, corrections, feedback training, window classification, or the
Call Parsing UI.

## Primary Paths

- `src/humpback/call_parsing/`
- `src/humpback/api/routers/call_parsing.py`
- `src/humpback/services/call_parsing.py`
- `src/humpback/workers/region_detection_worker.py`
- `src/humpback/workers/event_segmentation_worker.py`
- `src/humpback/workers/event_classification_worker.py`
- `src/humpback/workers/event_classifier_feedback_worker.py`
- `src/humpback/workers/segmentation_training_worker.py`
- `src/humpback/workers/window_classification_worker.py`
- `frontend/src/components/call-parsing/`
- `frontend/e2e/call-parsing-*.spec.ts`

## Artifact Roots

- `call_parsing/regions/{job_id}/`
- `call_parsing/segmentation/{job_id}/`
- `call_parsing/classification/{job_id}/`
- `call_parsing/window_classification/{job_id}/`
- Segmentation and classifier model artifact roots are referenced by database
  rows and model registry paths.

## Current UI Notes

- Segment Training queues Pass 2 model training directly from selected
  completed segmentation jobs. Persisted segmentation training datasets remain
  the internal worker contract, but dataset creation and management are not a
  visible page step.

## Likely Neighbors

- `signal-timeline` for audio resolution, playback, and timeline overlays.
- `vocalization-clustering` for type vocabularies and classifier feedback.
- `sequence-models` for Continuous Embedding upstream event/region sources.
- `core-platform` for schema, queue, and storage changes.

## Before Editing

1. Identify the pass: Pass 1 regions, Pass 2 segmentation, Pass 3
   classification, sidecar window classification, or feedback training.
2. Decide whether downstream consumers should read raw artifacts or effective
   corrected overlays.
