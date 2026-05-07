# Sequence Models Domain

Load this domain for Continuous Embedding jobs, Event Encoder tokenization jobs,
Sequence Models API, Sequence Models worker/service behavior, CRNN region chunk
helpers, or the retained Sequence Models UI.

## Primary Paths

- `src/humpback/sequence_models/`
- `src/humpback/api/routers/sequence_models.py`
- `src/humpback/services/continuous_embedding_service.py`
- `src/humpback/services/event_encoder_service.py`
- `src/humpback/workers/continuous_embedding_worker.py`
- `src/humpback/workers/event_encoder_worker.py`
- `src/humpback/models/sequence_models.py`
- `src/humpback/schemas/sequence_models.py`
- `frontend/src/components/sequence-models/`
- `frontend/src/api/sequenceModels.ts`
- `frontend/e2e/sequence-models/continuous-embedding.spec.ts`
- `frontend/e2e/sequence-models/event-encoder.spec.ts`

## Frontend Scope

- Active Sequence Models UI means Continuous Embedding and Event Encoder jobs,
  create forms, job tables, and detail pages.
- `DiscreteSequenceBar`, `RegionNavBar`, `SpanNavBar`,
  `MotifTimelineLegend`, `MotifHighlightOverlay`, and `CollapsiblePanelCard`
  are retained generic visualization primitives for future analysis/review
  surfaces. Do not treat them as active downstream Sequence Models workflows.
- Focused component tests document the retained generic primitive contracts.

## Artifact Roots

- `continuous_embeddings/{job_id}/`
- `continuous_embeddings/{job_id}/embeddings.parquet`
- `continuous_embeddings/{job_id}/manifest.json`
- `event_encoders/{job_id}/`
- `event_encoders/{job_id}/event_vectors.parquet`
- `event_encoders/{job_id}/event_tokens.parquet`
- `event_encoders/{job_id}/token_sequences.parquet`
- `event_encoders/{job_id}/manifest.json`
- `event_encoders/{job_id}/report.json`

## Likely Neighbors

- `call-parsing` for upstream Pass 1/Pass 2 source validation.
- `signal-timeline` for audio resolution and chunk/window timing semantics.
- `core-platform` for idempotent job rows, queue claims, and storage helpers.
- `frontend-shell` for navigation and query hooks.

## Before Editing

1. Identify whether the change affects SurfPerch event-padded windows, CRNN
   region chunks, Event Encoder event tokenization, or shared job lifecycle.
2. Load `call-parsing` for upstream region/segmentation validation changes.
