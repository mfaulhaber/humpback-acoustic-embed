# Sequence Models Invariants

- Continuous Embedding and Event Encoder are the active Sequence Models runtime
  surfaces.
- HMM, Masked Transformer, and motif-extraction runtime surfaces are retired.
- One canonical Continuous Embedding row/output exists per `encoding_signature`.
- Re-submitting an in-flight or complete signature reuses the existing row.
- Re-submitting a failed or canceled signature resets that row to `queued`.
- SurfPerch Continuous Embedding can read raw or effective event source modes.
- Effective event mode must include correction revision identity in the
  `encoding_signature`.
- CRNN region Continuous Embedding requires completed Pass 1 and matching Pass 2
  context plus a segmentation CRNN model.
- CRNN region source rejects effective event mode because the artifact is
  region-scoped rather than event-padded.
- One canonical Event Encoder row/output exists per `tokenization_signature`.
- Event Encoder reuses in-flight or complete signatures and resets failed or
  canceled signatures to `queued`.
- Event Encoder requires a completed Pass 2 segmentation job and a completed
  matching `region_crnn` Continuous Embedding job.
- Event Encoder raw mode reads immutable Pass 2 `events.parquet`; effective
  mode reads `load_effective_events()` and includes correction revision identity
  in the `tokenization_signature`.
- Event Encoder recomputes event/chunk overlap from timestamps and never treats
  CRNN `nearest_event_id` as authoritative.
- Completed Event Encoder timeline views are artifact-authoritative: they read
  the job's token/vector parquet artifacts and do not reload current Pass 2
  raw or effective events.
- Event Encoder token ids and `Txx` labels are job-local and k-local, not a
  stable global vocabulary.
- The active v3 Event Encoder non-CRNN descriptor order is `duration`,
  `log_energy`, `peak_frequency`, `spectral_centroid`, `bandwidth`,
  `spectral_entropy`, `ridge_log_frequency_slope`, `gap_to_previous`,
  `median_f0`, `f0_range`, `voicing_fraction`, `inflection_count`,
  `pulse_rate`, `pulse_rate_slope`, `ridge_median_frequency`,
  `ridge_low_frequency`, `ridge_high_frequency`, `ridge_frequency_span`,
  `ridge_coverage`, `ridge_energy_ratio`, `band_limited_peak_frequency`,
  `high_band_energy_ratio`.
- Existing v2 Event Encoder artifacts with the 14-entry descriptor order remain
  readable because timeline endpoints derive descriptor fields from each
  artifact manifest.
- Event Encoder descriptor vectors are robust-z normalized, clipped to
  `descriptor_clip_value` (default 3.0, null disables clipping), then multiplied
  by `descriptor_weight`.
- Event Encoder ridge descriptors persist scalar summaries only, including
  trimmed low/high ridge frequency bounds for piano roll display. Full STFT
  matrices and frame-level ridge contours are not stored in Continuous
  Embedding artifacts for this feature.
- Event Encoder piano-roll Ridge mode may expand a trusted ridge band's display
  top to the spectral centroid when scalar spectral-envelope descriptors show a
  broad tonal high-band event; this remains display-only and artifact-backed.
- Piano Roll Notes is a sidecar worker keyed on
  `(event_encoder_job_id, extractor_version)`. It reads completed Event Encoder
  outputs and the source audio, writes per-event MIDI notes to
  `event_notes_{extractor_version}.parquet`, and never modifies Event Encoder
  outputs. A complete Event Encoder job auto-enqueues a Piano Roll Notes job at
  the current default `extractor_version`; the auto-enqueue hook swallows
  conflicts so an in-flight or completed sidecar never blocks the encoder
  completion.
- The Piano Roll Notes harmonic prior (`label_harmonics` in
  `src/humpback/processing/piano_roll_tracker.py`) selects the lowest-bin
  track in each cluster as the F0 anchor and labels other tracks by per-frame
  frequency ratios: each shared frame contributes a `round(ratio)` value, and
  the candidate is labeled the Nth harmonic when the median nearest-integer
  lies in `[2, max_harmonic]` and the median absolute cents deviation
  against that integer multiple is ≤ `cents_tolerance`. Tracks that fail
  the harmonic check are left unprocessed so they remain eligible to anchor
  their own clusters on later iterations; tracks that match are consumed.
  `max_harmonic = 16`, `cents_tolerance = 75`, `min_overlap_frames = 3` are
  the v2 defaults.
- The Piano Roll MIDI export uses a fixed slim seven-channel layout
  (F0 → channel 1, 2nd–5th harmonics → channels 2–5, higher harmonics
  collapsed onto channel 6, unmatched onto channel 7, GM drum channel 10
  intentionally empty). Each channel is its own SMF track with a
  `program_change` and `track_name` header.
