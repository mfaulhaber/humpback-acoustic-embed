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
