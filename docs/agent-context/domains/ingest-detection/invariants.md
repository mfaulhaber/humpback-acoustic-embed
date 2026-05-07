# Ingest Detection Invariants

- Detection jobs write a Parquet row store; TSV is generated for downloads.
- Detection rows are linked to labels by stable UUID `row_id`.
- Detection embedding output is canonical per `(detection_job_id, model_version)`.
- Sync/full detection embedding generation updates the same canonical output.
- Hydrophone batch work should prefer local cached listings where supported.
- Window selection modes are `nms`, `prominence`, and `tiling`.
- Prominence mode works in logit space and includes gap filling.
- Perch v2 embeddings are model-versioned and have vector dimensions distinct
  from older model families.
- Candidate-backed replay training must use exact replay behavior and preserve
  comparison provenance.
