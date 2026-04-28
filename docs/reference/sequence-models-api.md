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

- `DELETE /sequence-models/continuous-embeddings/{id}` — permanently
  delete a continuous-embedding job and its disk artifacts (parquet,
  manifest). `204` on success, `404` if the job is missing.

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

## HMM Sequence Jobs (PR 2)

HMM sequence jobs train a Gaussian HMM on PCA-reduced continuous
embeddings, decode Viterbi state sequences, and produce summary
statistics and visualizations for latent-state discovery.

- `POST /sequence-models/hmm-sequences` — create an HMM sequence job.
  Requires a completed `ContinuousEmbeddingJob` as source. Accepts:
  - `continuous_embedding_job_id` (required)
  - `n_states` (required, >= 2)
  - `pca_dims` (default `50`, >= 1)
  - `pca_whiten` (default `false`)
  - `l2_normalize` (default `true`)
  - `covariance_type` (default `"diag"`, one of `"diag"` / `"full"`)
  - `n_iter` (default `100`, >= 1)
  - `random_seed` (default `42`)
  - `min_sequence_length_frames` (default `10`, >= 1)
  - `tol` (default `1e-4`, > 0)

  Returns `HMMSequenceJob`. Status `201` on creation, `400` on
  validation errors (missing or non-complete source job).

- `GET /sequence-models/hmm-sequences` — list jobs newest-first with
  optional `?status=` and `?continuous_embedding_job_id=` filters.

- `GET /sequence-models/hmm-sequences/{id}` — return job detail plus
  `state_summary.json` sidecar (when artifacts exist). Response shape:
  `{ job: HMMSequenceJob, summary: HMMStateSummary[] | null }`.
  `404` if the job is missing.

- `GET /sequence-models/hmm-sequences/{id}/states` — paginated
  `states.parquet` rows as JSON. Query params: `offset` (default 0),
  `limit` (default 500, max 5000). Returns `{ total, offset, limit,
  items }`. `404` if job or parquet not found.

- `GET /sequence-models/hmm-sequences/{id}/transitions` — transition
  matrix as `{ n_states, matrix: number[][] }`. `404` if not found.

- `GET /sequence-models/hmm-sequences/{id}/dwell` — dwell-time
  histograms as `{ n_states, histograms: { "0": [...], ... } }`.
  `404` if not found.

- `POST /sequence-models/hmm-sequences/{id}/cancel` — flip a `queued`
  or `running` job to `canceled`. `404` if not found, `409` if terminal.

- `DELETE /sequence-models/hmm-sequences/{id}` — permanently delete an
  HMM sequence job and its disk artifacts (states parquet, models,
  transition matrix, interpretation artifacts). `204` on success, `404`
  if the job is missing.

### Schemas

`HMMSequenceJob` exposes the DB row: status, all hyperparameters
(`n_states`, `pca_dims`, `pca_whiten`, `l2_normalize`, `covariance_type`,
`n_iter`, `random_seed`, `min_sequence_length_frames`, `tol`, `library`),
training stats (`train_log_likelihood`, `n_train_sequences`,
`n_train_frames`, `n_decoded_sequences`), `artifact_dir`,
`error_message`, and UTC timestamps.

`HMMStateSummary` provides per-state `occupancy`, `mean_dwell_frames`,
and `dwell_histogram` (list of run-length counts).

The `states.parquet` artifact stores one row per decoded window with
columns from the source `embeddings.parquet` (minus `embedding`) plus
`viterbi_state`, `state_posterior`, `max_state_probability`, and
`was_used_for_training`.

## Interpretation Visualizations (PR 3)

Interpretation endpoints are mounted on completed HMM sequence jobs.
Overlay and exemplar artifacts are auto-generated by the worker on job
completion; label distribution is computed on-demand (since vocalization
labels change over time).

- `GET /sequence-models/hmm-sequences/{id}/overlay` — paginated
  `pca_overlay.parquet` rows as `{ total, items: OverlayPoint[] }`.
  Query params: `offset` (default 0), `limit` (default 5000, max
  50000). Each point includes `pca_x`, `pca_y`, `umap_x`, `umap_y`,
  `viterbi_state`, and `max_state_probability`. `404` if job or
  artifact not found.

- `GET /sequence-models/hmm-sequences/{id}/label-distribution` —
  per-state label distribution from center-time join with
  `vocalization_labels`. Returns `{ n_states, total_windows,
  states: { "0": { "label_a": count, ... }, ... } }`. Computed
  on-demand if no cached `label_distribution.json` exists. `400` if
  job not complete. `404` if job not found.

- `GET /sequence-models/hmm-sequences/{id}/exemplars` — per-state
  exemplar windows (high-confidence, nearest-to-centroid,
  boundary-low-confidence). Returns `{ n_states, states: { "0":
  [ExemplarRecord, ...], ... } }`. Each record includes
  `audio_file_id`, time range, `max_state_probability`, and
  `exemplar_type`. `404` if job or artifact not found.

- `POST /sequence-models/hmm-sequences/{id}/generate-interpretations`
  — regenerate all three interpretation artifacts (overlay, label
  distribution, exemplars) for a completed job. Returns
  `{ status: "ok", job_id }`. `400` if job not complete. `404` if
  not found.

### Interpretation Schemas

`OverlayPoint`: `merged_span_id`, `window_index_in_span`,
`start_time_sec`, `end_time_sec`, `pca_x`, `pca_y`, `umap_x`,
`umap_y`, `viterbi_state`, `max_state_probability`.

`LabelDistributionResponse`: `n_states`, `total_windows`,
`states` (dict of state index → dict of label → count).

`ExemplarRecord`: `merged_span_id`, `window_index_in_span`,
`audio_file_id`, `start_time_sec`, `end_time_sec`,
`max_state_probability`, `exemplar_type`.
