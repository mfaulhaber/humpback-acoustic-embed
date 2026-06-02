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
- `src/humpback/processing/piano_roll_tracker.py` (legacy v1/v2 only; retired by ADR-069)
- `src/humpback/processing/ridge_path.py` (shared STFT ridge tracker used by the encoder and the v3 notes extractor)
- `src/humpback/processing/note_extractor_v3.py` (legacy v3 ridge-aware extractor; frozen after ADR-070)
- `src/humpback/processing/note_extractor_v4.py` (HPS-based F0 + harmonics with 30 Hz STFT ridge floor; ADR-070; legacy after ADR-071)
- `src/humpback/processing/note_extractor_v5.py` (harmonic-Viterbi F0 over CQT with pad-based background subtraction; ADR-071; current default)
- `tools/piano_roll_notes_debug.py` (permanent debug test-bed: renders spectrogram + per-variant piano-roll PNG for one encoder-job event; first-stop tool for any Piano Roll Notes investigation)
- `tools/piano_roll_notes_registry.py` (algorithm registry consumed by the debug CLI; not imported by the worker)
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
  defaults to the Notes view mode. Under ADR-069 (extractor `v3`) it draws
  curved ribbons that trace each note's per-frame `cents_from_pitch` contour
  on a MIDI 12–120 log-frequency Y axis with semitone gridlines, octave
  labels C0…G9, black-key shading, and a desaturated tint on the
  extended bands outside the 88-key range (MIDI 12–20 and 109–120). Contours
  are fetched in batches via `/notes/contours` and cached per `note_uid` so
  panning does not refetch already-loaded notes. Notes without a fetched
  contour render as the existing flat bar at `midi_pitch` and hydrate into
  ribbons when the contour resolves; a 500 from `/notes/contours` is a
  terminal flat-bar state for the session with a single non-blocking toast.
  Hover tooltip includes a `Δpitch: ±N¢` summary; hit-testing uses ≤ 6 px
  polyline distance. Legacy v1/v2 sidecars on disk still render as bars
  because their rows have no `note_uid` and the contour endpoint returns
  422. If the `.../notes` fetch fails, the page reverts to the previous
  rectangle mode with a non-blocking toast and leaves the Notes selector
  greyed out.
- `PianoRollNotesStatusPill` surfaces a "v3 available" badge when the
  encoder has a v2 (or older) notes sidecar but no v3 yet; clicking it
  POSTs a v3 notes job. The "Download MIDI" tooltip names MPE / DAW
  compatibility; the export status panel displays `Format: MPE v3` below
  the file size when the latest export is v3.
- The Piano Roll page exposes a windowed bundled "Export view" action
  to the left of the Notes status badge. Clicking it enqueues an
  asynchronous Piano Roll export job (`piano_roll_midi_exports` table)
  whose worker (a) filters the notes parquet to the viewer's current
  `timeRange` (`window_start_utc`, `window_end_utc`), synthesizes a
  Standard MIDI File whose tick-0 origin is the window start via
  `humpback.processing.midi_synthesis.notes_table_to_midi_bytes(
  notes, time_origin_utc=window_start_utc)`, and (b) resolves the source
  audio for the same window through `resolve_timeline_audio()` and
  writes a 16-bit PCM FLAC clip (no loudness normalization) alongside
  the MIDI. Artifacts live at
  `<storage_root>/exports/event_encoders/{job_id}/notes_{version}.mid`
  and `audio_{version}.flac` and are written atomically as a pair.
  The export button is disabled until notes status is `complete` and
  when the requested window duration exceeds 30 minutes (1800 s; see
  ADR-068 for the cap and rationale). On `complete` the UI renders a
  compact panel with the exported window text, a "Download MIDI" link,
  a "Download audio (FLAC)" link, and a "Re-export view" affordance
  whose emphasis tracks whether the current viewport matches the
  persisted window within ~50 ms. Re-export sends `force=true`.
  MIDI conventions follow ADR-069 when the resolved notes-job version is
  `v3`: SMF Type 1, 480 PPQ, constant 120 BPM, **MPE Lower Zone** with 15
  member channels and per-member ±24-semitone pitch-bend range. The first
  non-tempo track is the MPE Master (RPN 6 = 15, per-member RPN 0/0 + Data
  Entry MSB 24, plus per-note `MetaMessage("text", "pN")` events tagging
  partial identity). Voices rotate across channels 1–15 via a deterministic
  longest-idle allocator with FIFO voice steal; the steal count is recorded
  in `params_json`. Per-note `program_change` (F0→0, H2→11, H3→12, H4→10,
  H5→8, H6..H16→88) and CC 74 (= `partial_index * 16` clamped) preserve
  partial identity through the voice rotation. Pitch-bend events are emitted
  per contour frame whose `cents_from_pitch` differs from the last-emitted
  bend by ≥ 4¢ (≈14 bend units at ±24 semitones). Harmonic notes inherit
  their parent F0's bend stream in cents (cents conservation); the measured
  CQT peak validates harmonic presence only and does not drive the bend.
  17 tracks total (tempo + master + 15 voice tracks); empty voice tracks
  still emit `track_name` and `end_of_track`. Legacy v1/v2 exports stay on
  the slim seven-channel layout from ADR-067 — F0, 2nd, 3rd, 4th, 5th
  harmonics each on their own channel, a combined "higher harmonics"
  channel for `partial_index ≥ 5`, and an "unmatched" channel for tracks
  the harmonic prior could not label. The GM drum channel (channel 10,
  1-indexed) is intentionally empty in both layouts.
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
- `event_encoders/{job_id}/event_notes_{extractor_version}.parquet` (Piano Roll Notes sidecar; current default is `v7` — v6's de-spiked harmonic-Viterbi extractor plus residual discontinuity splitting and ridge-guided flat-segment rescue from ADR-074. Legacy `v1` through `v6` artifacts remain readable on disk until manually deleted)
- `event_encoders/{job_id}/event_ridges_{tokenizer_version}.parquet` (per-event STFT ridge contours produced by the encoder worker. One row per frame per event with `event_id`, `frame_index`, `frame_time_offset_s`, `log_frequency`, `strength`, `energy_ratio`. Consumed by Piano Roll Notes v3/v4 and by v7 ridge rescue; v3 falls back to in-process ridge recompute when this sidecar is absent — ADR-069 / ADR-074)
- `event_encoders/{job_id}/event_note_contours_v3.parquet` (per-frame note contour sidecar for v3 notes. One row per frame per note keyed on `note_uid` with `time_offset_s`, `cents_from_pitch`, `harmonic_strength`, `subharmonic_octave`. Consumed by the MPE MIDI synthesizer and the frontend ribbon renderer — ADR-069)
- `event_encoders/{job_id}/event_note_contours_v4.parquet` (per-frame note contour sidecar for v4 notes. Schema identical to the v3 sidecar; the `subharmonic_octave` column stores `chosen_divisor − 1` (0..5) in v4 rather than the v3 octave-halving count (0..3) — ADR-070)
- `event_encoders/{job_id}/event_notes_v5.parquet` and `event_note_contours_v5.parquet` (per-frame note + contour sidecars for v5 notes. Schemas identical to v3/v4; the `subharmonic_octave` column is reserved / unused in v5 (always 0) — ADR-071)
- `event_encoders/{job_id}/event_notes_v6.parquet` and `event_note_contours_v6.parquet` (per-frame note + contour sidecars for v6 notes. Schemas identical to v3/v4/v5; the F0 contour is v5's de-spiked contour and `subharmonic_octave` is reserved / unused (always 0) — ADR-072)
- `event_encoders/{job_id}/event_notes_v7.parquet` and `event_note_contours_v7.parquet` (per-frame note + contour sidecars for v7 notes. Schemas identical to v3-v6; v7 splits residual high-slope branch jumps and can rescue flat decoded F0 from the persisted ridge sidecar — ADR-074)
- `exports/event_encoders/{job_id}/notes_{extractor_version}.mid` (Piano Roll Notes MIDI export artifact for the last-exported window. `v3` is MPE Lower Zone with per-voice pitch bend; legacy `v1`/`v2` remain on the slim seven-channel layout from ADR-067)
- `exports/event_encoders/{job_id}/audio_{extractor_version}.flac` (co-exported 32 kHz mono FLAC clip for the same exported window)

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
- ADR-068: Piano Roll windowed bundled export (MIDI + FLAC).
- ADR-069: Ridge-aligned F0 + harmonics extractor and MPE Piano Roll MIDI export.
- ADR-070: Piano Roll Notes v4 — HPS F0 selection with extended low band.
- ADR-071: Piano Roll Notes v5 — harmonic-Viterbi F0 with pad-based background subtraction.

## Likely Neighbors

- `call-parsing` for upstream Pass 1/Pass 2 source validation.
- `signal-timeline` for audio resolution and chunk/window timing semantics.
- `core-platform` for idempotent job rows, queue claims, and storage helpers.
- `frontend-shell` for navigation and query hooks.

## Before Editing

1. Identify whether the change affects SurfPerch event-padded windows, CRNN
   region chunks, Event Encoder event tokenization, or shared job lifecycle.
2. Load `call-parsing` for upstream region/segmentation validation changes.
