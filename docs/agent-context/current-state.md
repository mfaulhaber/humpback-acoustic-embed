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
- Piano Roll exports are windowed and bundled (ADR-068). The same
  user-initiated async export worker (`piano_roll_midi_exports` table)
  now writes a pair of artifacts under
  `<storage_root>/exports/event_encoders/{job_id}/`:
  `notes_{version}.mid` (MIDI whose tick-0 origin equals the requested
  `window_start_utc`) and `audio_{version}.flac` (32 kHz mono 16-bit
  PCM clip of the same `[window_start_utc, window_end_utc)`, NOT
  loudness-normalized so the clip matches what the player rendered).
  The piano roll toolbar exposes an "Export view" button that captures
  the viewer's current `timeRange`. On `complete`, the UI surfaces
  exported-window text plus "Download MIDI" and "Download audio (FLAC)"
  download links and a "Re-export view" affordance that is emphasized
  when the current viewport differs from the persisted window by more
  than ~50 ms. Window duration is capped at 1800 s (30 min) at the
  schema, service, API, and button layers. The MIDI's SMF Type 1 /
  480 PPQ / 120 BPM framing is unchanged.
- Piano Roll Notes now produces MPE-ready ridge-aligned contours; export
  uses MPE Lower Zone (ADR-069). The encoder worker persists per-event
  ridge contours, the notes worker writes a per-frame contour sidecar,
  and the Notes view defaults to curved-ribbon rendering on a MIDI
  12–120 Y axis. Legacy v1/v2/v3 sidecars remain readable; the export
  resolver picks the highest complete version per string comparison.
- v5 (ADR-071), superseded as default by v6 (ADR-072): the v5 extractor
  replaces v4's ridge-locked HPS divisor selection with direct
  harmonic-sum F0 estimation over the CQT plus log-frequency Viterbi
  smoothing — temporal smoothness is part of the cost function rather
  than a post-filter. A pad-only background-subtraction stage cleans
  chronic low-frequency noise (ship hum, hydrophone self-noise) from
  the voicing oracle by sampling per-bin CQT magnitudes from the pad
  zones outside the segmented event. v5 emits `event_notes_v5.parquet`
  and `event_note_contours_v5.parquet` (schemas identical to v3/v4;
  the `subharmonic_octave` column is reserved / unused in v5 and
  always written as 0). Worker `pad_seconds` default for v5 is 0.25 s
  (was 0.05 s in v3/v4) so background subtraction has noise frames
  to sample. MIDI export auto-resolves to v5 when a complete v5 row
  exists; no auto-backfill of v5 for completed v4 jobs.
- `tools/piano_roll_notes_debug.py` is the permanent debug surface
  for Piano Roll Notes investigations established during ADR-071's
  v5 iteration: it loads any event of any encoder job and renders a
  CQT spectrogram + ridge overlay alongside a stacked piano-roll
  panel per registered algorithm variant. Use it as the first stop
  for any rendering / pitch / F0 issue.
- v6 (ADR-072), superseded as default by v7 (ADR-074): v6 is v5's decode plus a
  slope-based F0 contour de-spike pass that runs before note building.
  v5 occasionally leaves a surviving spike (a short out-and-back F0
  excursion such as the ~15-semitone plunge at t≈1.2 s on event
  `669849340bff411390e5eaaf1ec9b9e9`); v6 excises the spike frames and
  linearly bridges log-frequency across the gap so the note stays one
  continuous contour, and harmonic ribbons are corrected automatically
  via cents conservation. Only out-and-back excursions that return to the
  anchor's slope envelope are bridged; an excursion that never returns
  within `max_spike_frames` is a genuine level change (register jump /
  signal drop) and is left untouched, not joined (return-to-baseline
  guard added after the `cb23dfcd…` over-bridging finding — ADR-072
  amendment). A short non-returning excursion at the very *end* of a
  segment is the exception: it is trimmed (≤ `max_trailing_trim_frames`)
  so an energy-fade tail that drops to a sub-fundamental does not show as
  an end slope-drop (ADR-072 Amendment 2, events `2054e6de…`/`c82fa1fc…`).
  `DespikeParams`: `max_slope_oct_per_s = 6.0`,
  `max_spike_frames = 12`, `max_trailing_trim_frames = 4`,
  `enabled = True`; `enabled = False` is
  byte-identical to v5. v6 emits
  `event_notes_v6.parquet` / `event_note_contours_v6.parquet` (schemas
  identical to v3–v5; `subharmonic_octave` reserved / written as 0) and
  inherits v5's worker defaults (30 Hz STFT floor,
  `min_break_frames = 6`, `pad_seconds = 0.25`). MIDI export auto-resolves
  to v6 by lexicographic version ordering; no auto-backfill of v6 for
  completed v5 jobs. `tools/piano_roll_notes_debug.py` gains a `"v6"`
  variant for v5-vs-v6 before/after rendering.
- `DEFAULT_EXTRACTOR_VERSION = "v7"` (ADR-074): v7 keeps the v6
  harmonic-Viterbi decode and de-spike, then adds two post-decode passes
  before note building. Residual discontinuity splitting cuts adjacent
  retained F0 frames whose actual slope exceeds
  `max_continuous_slope_oct_per_s = 6.0`, so non-returning branch jumps
  become note boundaries rather than continuous MPE bends. Ridge-guided
  rescue rewrites flat decoded segments when the persisted Event Encoder
  STFT ridge overlaps sufficiently, moves at least 5 semitones, and is
  smooth enough after a linear trend fit; the rescued F0 is the ridge
  divided by a median carrier harmonic. v7 emits `event_notes_v7.parquet`
  / `event_note_contours_v7.parquet` with the v3-v6 schemas, inherits
  v5/v6 worker defaults (30 Hz STFT floor, `min_break_frames = 6`,
  `pad_seconds = 0.25`), consumes the ridge sidecar when present, and
  remains explicitly comparable through the debug tool's `"v7"` variant.

## Frontend Shell

- Frontend is a React 18 + Vite + TypeScript + Tailwind + shadcn/ui SPA.
- Frontend package operations use `npm` from `frontend/`.
- App shell navigation covers Classifier, Vocalization, Call Parsing, Sequence
  Models, and Admin.
- Shared query hooks and UI components are consumed across feature domains.
