# Sequence Models API Surface

The Sequence Models track operates parallel to the four-pass call parsing
pipeline. PR 1 lands the **Continuous Embedding** producer — region-bounded,
hydrophone-only 1-second-hop SurfPerch embeddings padded around Pass-1
region detections. Per-job status transitions follow the standard
`queued → running → complete|failed|canceled` pattern.

All endpoints are mounted under `/sequence-models/`.

## Continuous Embedding (PR 1)

- `POST /sequence-models/continuous-embeddings` — create or reuse a
  continuous-embedding job, idempotent on `encoding_signature`. Accepts:
  - `region_detection_job_id` (required) — completed hydrophone-backed Pass-1 job FK
  - `model_version` (default `"surfperch-tensorflow2"`)
  - `hop_seconds` (default `1.0`, must be > 0)
  - `pad_seconds` (default `10.0`, must be >= 0)

  Returns `ContinuousEmbeddingJob`. Status `201` when a new row is
  created; `200` when an existing complete or in-flight (`queued` /
  `running`) row with the same signature is returned, or when a failed /
  canceled row with the same signature is reset back to `queued`.
  `400` on validation errors (missing region detection job, incomplete
  source job, non-hydrophone source, unsupported `model_version`).

- `GET /sequence-models/continuous-embeddings` — list jobs newest-first
  with optional `?status=` filter.

- `GET /sequence-models/continuous-embeddings/{id}` — return job detail
  plus the `manifest.json` sidecar (when the parquet artifact exists).
  Response shape: `{ job: ContinuousEmbeddingJob, manifest: ContinuousEmbeddingJobManifest | null }`.
  `404` if the job is missing.

- `POST /sequence-models/continuous-embeddings/{id}/cancel` — flip a
  `queued` or `running` job to `canceled`. `404` if the job is missing.
  `409` if the job is in a terminal state (`complete` / `failed` /
  `canceled`).

### Schemas

`ContinuousEmbeddingJob` exposes the producer DB row: status, all
producer parameters (`hop_seconds`, `pad_seconds`, `window_size_seconds`,
`target_sample_rate`, `model_version`, `feature_config_json`),
`encoding_signature`, post-completion summary stats (`vector_dim`,
`total_regions`, `merged_spans`, `total_windows`, `parquet_path`),
`error_message`, and UTC timestamps.

`ContinuousEmbeddingJobManifest` matches the `manifest.json` sidecar
written next to `embeddings.parquet`, with per-merged-span window-count
summaries (`merged_span_id`, `start_time_sec`, `end_time_sec`,
`window_count`, `source_region_ids`).

The parquet artifact stores one row per embedded window with
`source_region_ids` preserved as UUID strings, allowing downstream
sequence-model consumers to trace each window back to its contributing
Pass-1 region rows.

See `src/humpback/schemas/sequence_models.py` for the full Pydantic
definitions.
