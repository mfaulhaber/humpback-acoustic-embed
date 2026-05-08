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
- The active Event Encoder non-CRNN descriptor order is `duration`,
  `log_energy`, `peak_frequency`, `spectral_centroid`, `bandwidth`,
  `spectral_entropy`, `ridge_log_frequency_slope`, `gap_to_previous`,
  `median_f0`, `f0_range`, `voicing_fraction`, `inflection_count`,
  `pulse_rate`, `pulse_rate_slope`.
- Event Encoder descriptor vectors are robust-z normalized, clipped to
  `descriptor_clip_value` (default 3.0, null disables clipping), then multiplied
  by `descriptor_weight`.
- Event Encoder ridge slope persists only a scalar descriptor value; full STFT
  matrices are not stored in Continuous Embedding artifacts for this feature.
