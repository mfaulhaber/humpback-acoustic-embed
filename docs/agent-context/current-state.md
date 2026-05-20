# Current State By Domain

This file carries active project state that should not be loaded globally in
full. Read the relevant domain section when planning or implementing.

## Core Platform

- SQLite is the primary database through SQLAlchemy and Alembic.
- Latest migration in the tree is `075_remove_hmm_mt_sequence_models.py`.
- Worker processing is SQL-backed and polling-based.
- Stale job recovery runs on worker startup and periodically while running.
- Global retained-idempotency guarantees include detection embeddings per
  `(detection_job_id, model_version)` and Continuous Embedding per
  `encoding_signature`.

## Signal Timeline

- Timeline spectrogram rendering uses PCEN normalization for visualization.
- Playback audio is RMS-normalized and soft-clipped for listening; this is
  intentionally separate from spectrogram normalization.
- Timeline provider components own playback, zoom, pan, keyboard handling,
  clipped overlay layout, and tile redraw notifications.
- Timeline APIs and UI are used by detection, labeling, call parsing, and
  sequence-model review surfaces.

## Ingest Detection

- Detection-job-backed classifier training is retained.
- Perch v2 is a first-class embedding model family with model-versioned
  re-embedding.
- Hydrophone streaming supports Orcasound HLS and NOAA sources.
- Detection window selection supports NMS, prominence, and tiling modes.
- Detection jobs write a Parquet row store; TSV downloads are generated from
  retained rows.

## Vocalization Clustering

- Vocalization labeling supports managed vocalization type vocabulary.
- `"(Negative)"` is a reserved mutually-exclusive label used as negative for
  all vocalization types.
- Multi-label vocalization training consumes retained detection-job sources.
- Vocalization / Clustering jobs operate on detection-job embeddings.

## Call Parsing

- Four-pass call parsing has Passes 1-3 active and Pass 4 deferred.
- Pass 1 region detection can run from local audio or hydrophone ranges.
- Pass 2 event segmentation uses CRNN framewise segmentation and writes
  immutable raw `events.parquet`.
- Pass 3 event classification writes immutable `typed_events.parquet`.
- Human boundary and type corrections are stored in SQL overlays rather than
  mutating parquet artifacts.
- Window classification is a sidecar over cached Pass 1 Perch embeddings.

## Sequence Models

- The active Sequence Models runtime surface is Continuous Embedding.
- HMM, Masked Transformer, and motif-extraction runtime surfaces have been
  retired from active code.
- Continuous Embedding supports SurfPerch event-padded windows and CRNN
  region-based chunks.
- Continuous Embedding jobs are idempotent on `encoding_signature`.
- Sequence Models depends on Call Parsing for upstream segmentation/region
  source identity and on Signal Timeline for audio resolution semantics.
- Piano Roll Notes is a sidecar worker that decorates completed Event Encoder
  jobs with per-event MIDI notes. The worker is idempotent on
  `(event_encoder_job_id, extractor_version)`, auto-enqueues on encoder
  completion, and persists `event_notes_{version}.parquet` next to the encoder
  artifacts. The UI surfaces a `Notes` pill on the Event Encoder job table and
  in the piano roll toolbar, plus a `Generate notes` / `Re-run` action.

## Frontend Shell

- Frontend is a React 18 + Vite + TypeScript + Tailwind + shadcn/ui SPA.
- Frontend package operations use `npm` from `frontend/`.
- App shell navigation covers Classifier, Vocalization, Call Parsing, Sequence
  Models, and Admin.
- Shared query hooks and UI components are consumed across feature domains.
