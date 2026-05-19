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
- `frontend/src/components/sequence-models/EventEncoderTimelinePanel.tsx`
- `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`
- `frontend/src/components/sequence-models/EventEncoderClusterProjectionPanel.tsx`
- `frontend/src/components/sequence-models/EventEncoderTokenOverlay.tsx`
- `frontend/e2e/sequence-models/continuous-embedding.spec.ts`
- `frontend/e2e/sequence-models/event-encoder.spec.ts`
- `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts`

## Frontend Scope

- Active Sequence Models UI means Continuous Embedding and Event Encoder jobs,
  create forms, job tables, detail pages, the Event Encoder detail timeline
  viewer, and the dedicated Event Encoder piano roll route.
- The Event Encoder timeline viewer is read-only. It renders completed
  `event_tokens.parquet` assignments through a dedicated timeline endpoint and
  uses Call Parsing region tiles/audio for context. Its selected-event feature
  table is also read-only and uses timeline response descriptor metadata plus
  `event_vectors.parquet` descriptor-vector values. Do not treat token labels
  as globally stable across Event Encoder jobs.
- The Event Encoder piano roll is also read-only and uses the same timeline
  endpoint. It renders tokenized events on a time-by-frequency canvas using
  job-local, k-local token ids, descriptor values, and Call Parsing region
  audio slices for playback. It includes a collapsible bottom spectrogram strip
  backed by Call Parsing region timeline tiles; the piano roll's smooth
  viewport state remains the source of truth for that strip.
- Event Encoder timeline previous/next navigation can be token-scoped by
  toggling the selected event's token badge. This is a frontend-only affordance
  derived from the currently loaded selected-k timeline rows; it does not hide
  other events and must not imply token ids are stable across jobs or k values.
- The Event Encoder cluster projection panel is read-only and artifact-backed.
  Its projection endpoint joins completed `event_tokens.parquet` assignments to
  persisted `event_vectors.parquet` event vectors for the selected job-local
  `k`, then returns UMAP or PCA plot points colored with the same token palette
  as the timeline overlay.
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

Event Encoder manifests record ordered `descriptor_feature_names`. The active
14-entry non-CRNN descriptor block includes duration, energy, spectral shape,
`ridge_log_frequency_slope`, `gap_to_previous`, F0 descriptors
(`median_f0`, `f0_range`, `voicing_fraction`), contour complexity
(`inflection_count`), and pulse descriptors (`pulse_rate`,
`pulse_rate_slope`). Full STFT matrices, ridge traces, and F0 contours are not
stored in Continuous Embedding artifacts.
Descriptor vectors are robust-z normalized and clipped by the Event Encoder
preprocessing config (`descriptor_clip_value`, default 3.0) before weighting and
concatenation.

## Likely Neighbors

- `call-parsing` for upstream Pass 1/Pass 2 source validation.
- `signal-timeline` for audio resolution and chunk/window timing semantics.
- `core-platform` for idempotent job rows, queue claims, and storage helpers.
- `frontend-shell` for navigation and query hooks.

## Before Editing

1. Identify whether the change affects SurfPerch event-padded windows, CRNN
   region chunks, Event Encoder event tokenization, or shared job lifecycle.
2. Load `call-parsing` for upstream region/segmentation validation changes.
