# Sequence Models Domain

Load this domain for Continuous Embedding jobs, Event Encoder tokenization jobs,
Sequence Models API, Sequence Models worker/service behavior, CRNN region chunk
helpers, or the retained Sequence Models UI.

## Primary Paths

- `src/humpback/sequence_models/`
- `src/humpback/api/routers/sequence_models.py`
- `src/humpback/services/continuous_embedding_service.py`
- `src/humpback/services/event_encoder_service.py`
- `src/humpback/services/piano_roll_notes_service.py`
- `src/humpback/processing/piano_roll_cqt.py`
- `src/humpback/processing/piano_roll_tracker.py`
- `src/humpback/workers/continuous_embedding_worker.py`
- `src/humpback/workers/event_encoder_worker.py`
- `src/humpback/workers/piano_roll_notes_worker.py`
- `src/humpback/workers/piano_roll_midi_export_worker.py`
- `src/humpback/services/piano_roll_midi_export_service.py`
- `src/humpback/processing/midi_synthesis.py`
- `src/humpback/models/sequence_models.py`
- `src/humpback/models/piano_roll_notes.py`
- `src/humpback/models/piano_roll_midi_export.py`
- `src/humpback/schemas/sequence_models.py`
- `src/humpback/schemas/piano_roll_notes.py`
- `src/humpback/schemas/piano_roll_midi_export.py`
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
  viewport state remains the source of truth for that strip. For v3 Event
  Encoder artifacts it defaults to Ridge mode, using trimmed ridge low/high
  frequency descriptors to set one token rectangle's vertical band, with
  conservative spectral-envelope top expansion for broad harmonic events.
- When a Piano Roll Notes sidecar exists for the Event Encoder job, the page
  defaults to a new Notes view mode that draws one bar per MIDI note from the
  sidecar on an 88-key log-frequency Y axis (MIDI 21–108) with semitone
  gridlines, octave labels (C0…C8), and black-key shading. If the
  `.../notes` fetch fails, the page reverts to the previous rectangle mode
  with a non-blocking toast and leaves the Notes selector greyed out.
- The Piano Roll page exposes an "Export MIDI" button to the left of the
  Notes status badge. Clicking it enqueues an asynchronous MIDI export job
  (`piano_roll_midi_exports` table) whose worker reads the notes parquet,
  synthesizes a Standard MIDI File via `humpback.processing.midi_synthesis`,
  and persists it under
  `<storage_root>/exports/event_encoders/{job_id}/notes_{version}.mid`.
  The button is disabled until notes status is `complete`. Status pill
  states (`absent → queued → running → complete`) drive the button label;
  on `complete` the button becomes "Download MIDI" and an overflow menu
  exposes a "Re-export" affordance that POSTs with `force=true`. MIDI
  conventions: SMF Type 1, 480 PPQ, constant 120 BPM. The file uses a
  slim seven-channel layout — F0, 2nd, 3rd, 4th, 5th harmonics each on
  their own channel, a combined "higher harmonics" channel for
  `partial_index ≥ 5`, and an "unmatched" channel for tracks the
  harmonic prior could not label. The GM drum channel (channel 10,
  1-indexed) is intentionally empty. Each channel is rendered as its own
  SMF track with a `track_name` meta-event and a distinct GM
  `program_change` so DAWs present each partial as a named, distinctly
  voiced lane. Time origin is shifted to the earliest note's
  `start_utc`. Pitch-bend is deferred; the underlying `mido` library
  already supports the MPE primitives needed for the future extension.
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
- `event_encoders/{job_id}/event_notes_{extractor_version}.parquet` (Piano Roll Notes sidecar; current default is `v2` — the per-frame harmonic labeler from ADR-067; legacy `v1` artifacts may coexist until manually deleted)
- `exports/event_encoders/{job_id}/notes_{extractor_version}.mid` (Piano Roll Notes MIDI export artifact produced on demand by the export worker)

Event Encoder manifests record ordered `descriptor_feature_names`. The active
v3 22-entry non-CRNN descriptor block includes duration, energy, spectral
shape, `ridge_log_frequency_slope`, `gap_to_previous`, F0 descriptors
(`median_f0`, `f0_range`, `voicing_fraction`), contour complexity
(`inflection_count`), pulse descriptors (`pulse_rate`, `pulse_rate_slope`),
and ridge display descriptors (`ridge_median_frequency`,
`ridge_low_frequency`, `ridge_high_frequency`, `ridge_frequency_span`,
`ridge_coverage`, `ridge_energy_ratio`, `band_limited_peak_frequency`,
`high_band_energy_ratio`). Older v2 artifacts with the 14-entry descriptor
block remain readable through their manifests. Full STFT matrices, ridge
traces, and F0 contours are not stored in Continuous Embedding artifacts.
Descriptor vectors are robust-z normalized and clipped by the Event Encoder
preprocessing config (`descriptor_clip_value`, default 3.0) before weighting and
concatenation.

## Relevant ADRs

- ADR-056: Sequence Models track parallel to Call Parsing pipeline.
- ADR-057: CRNN region-based chunk embeddings as second Sequence Models source.
- ADR-063: Event Encoder v3 ridge frequency descriptors for piano-roll display.
- ADR-064: Piano Roll Notes sidecar worker.
- ADR-065: Extended Piano Roll Notes pitch range (deferred placeholder).
- ADR-066: User-initiated async MIDI export for Piano Roll Notes.
- ADR-067: Per-frame harmonic labeling and channelized MIDI export.

## Likely Neighbors

- `call-parsing` for upstream Pass 1/Pass 2 source validation.
- `signal-timeline` for audio resolution and chunk/window timing semantics.
- `core-platform` for idempotent job rows, queue claims, and storage helpers.
- `frontend-shell` for navigation and query hooks.

## Before Editing

1. Identify whether the change affects SurfPerch event-padded windows, CRNN
   region chunks, Event Encoder event tokenization, or shared job lifecycle.
2. Load `call-parsing` for upstream region/segmentation validation changes.
