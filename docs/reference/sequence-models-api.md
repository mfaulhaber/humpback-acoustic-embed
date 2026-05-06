# Sequence Models API Surface

The active Sequence Models track now contains only **Continuous Embedding**.
All endpoints are mounted under `/sequence-models/`, and job status
transitions follow the standard `queued -> running -> complete|failed|canceled`
pattern.

## Continuous Embedding

Continuous Embedding jobs produce time-aligned embedding parquet artifacts for
downstream analysis. The producer has two retained source families:

- **`surfperch-tensorflow2`**: pass an `event_segmentation_job_id` from a
  completed Pass 2 job. The producer emits SurfPerch embeddings around each
  event with configurable hop and padding.
- **`crnn-call-parsing-pytorch`**: pass a completed Pass 1
  `region_detection_job_id`, a matching completed Pass 2
  `event_segmentation_job_id` disambiguator, and a segmentation CRNN model.
  The producer slices CRNN BiGRU activations into region-scoped chunks with
  configurable projection.

### Endpoints

- `POST /sequence-models/continuous-embeddings` creates or reuses a job,
  idempotent on `encoding_signature`.

  SurfPerch requests accept `event_segmentation_job_id`,
  `event_source_mode` (`"raw"` or `"effective"`), `model_version`,
  `hop_seconds`, and `pad_seconds`.

  CRNN requests accept `region_detection_job_id`,
  `event_segmentation_job_id`, `crnn_segmentation_model_id`,
  `chunk_size_seconds`, `chunk_hop_seconds`, `projection_kind`,
  `projection_dim`, and `model_version`.

  `event_source_mode="effective"` is supported for the SurfPerch source and
  rejected for the CRNN source. Responses return `ContinuousEmbeddingJob` with
  `201` for a new row, `200` for a reused or reset row, and `422` for invalid
  upstream/source configuration.

- `GET /sequence-models/continuous-embeddings` lists jobs newest-first, with an
  optional `?status=` filter.

- `GET /sequence-models/continuous-embeddings/{id}` returns
  `{ job: ContinuousEmbeddingJob, manifest: ContinuousEmbeddingJobManifest | null }`.

- `POST /sequence-models/continuous-embeddings/{id}/cancel` cancels a queued or
  running job. Terminal jobs return `409`.

- `DELETE /sequence-models/continuous-embeddings/{id}` deletes the job row and
  its `continuous_embeddings/{id}/` disk artifacts.

### Schemas

`ContinuousEmbeddingJob` exposes the retained DB row fields: status,
`event_segmentation_job_id`, `event_source_mode`, `model_version`, source
parameters, `target_sample_rate`, `feature_config_json`,
`encoding_signature`, summary counters, `parquet_path`, `error_message`, and
UTC timestamps.

SurfPerch-specific fields include `window_size_seconds`, `hop_seconds`,
`pad_seconds`, `total_events`, `merged_spans`, and `total_windows`.

CRNN-specific fields include `region_detection_job_id`,
`chunk_size_seconds`, `chunk_hop_seconds`, `crnn_checkpoint_sha256`,
`crnn_segmentation_model_id`, `projection_kind`, `projection_dim`,
`total_regions`, and `total_chunks`.

SurfPerch idempotency includes `event_source_mode`. Effective-mode signatures
also include a correction revision fingerprint computed from correction row IDs
and `updated_at` values for the selected segmentation job, so changing reviewed
event boundaries produces a distinct Continuous Embedding job while raw mode
preserves the existing `events.parquet` signature behavior.

`ContinuousEmbeddingJobManifest` matches the `manifest.json` sidecar written
next to `embeddings.parquet`. SurfPerch manifests include `spans` with
per-merged-span window summaries. CRNN manifests include `regions` with
per-region chunk-count summaries plus CRNN checkpoint, projection config, and
the parent Pass 1 / Pass 2 job ids.

The parquet artifact schema is source-specific:

- SurfPerch rows contain `merged_span_id`, `event_id`,
  `window_index_in_span`, `audio_file_id`, `start_timestamp`,
  `end_timestamp`, `is_in_pad`, and `embedding`.
- CRNN rows contain `region_id`, `audio_file_id`, `hydrophone_id`,
  `chunk_index_in_region`, `start_timestamp`, `end_timestamp`, `is_in_pad`,
  `call_probability`, `event_overlap_fraction`, `nearest_event_id`,
  `distance_to_nearest_event_seconds`, `tier`, and `embedding`.

See `src/humpback/schemas/sequence_models.py` for the full Pydantic
definitions.
