# Sequence Models API Surface

The active Sequence Models track contains **Continuous Embedding** producers and
the **Event Encoder** tokenizer. All endpoints are mounted under
`/sequence-models/`, and job status transitions follow the standard
`queued -> running -> complete|failed|canceled` pattern.

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

## Event Encoder

Event Encoder jobs tokenize Pass 2 events using a completed CRNN
Continuous Embedding artifact. The worker recomputes event/chunk overlap
against the selected raw or effective event set, pools CRNN frame embeddings
inside each event, adds acoustic descriptors, preprocesses the event vectors,
and fits one k-means tokenizer per feasible k.

### Endpoints

- `POST /sequence-models/event-encoders` creates or reuses a job, idempotent on
  `tokenization_signature`.

  Requests require `event_segmentation_job_id` and
  `continuous_embedding_job_id`. The segmentation job must be complete. The
  Continuous Embedding job must be complete, must be the `region_crnn` source
  family, and must reference the same segmentation job as its Pass 2
  disambiguator.

  Optional fields:

  - `event_source_mode`: `"raw"` (default) or `"effective"`.
  - `tokenizer_version`: default `"crnn-event-encoder-v2"`.
  - `pooling.enabled_pools`: defaults to `mean_pool`, `top_k_pool`,
    `start_pool`, `middle_pool`, and `end_pool`.
  - `pooling.top_k_fraction`: default `0.25`.
  - `pooling.min_overlap_fraction`: default `0.25`.
  - `pooling.min_chunks_per_event`: default `1`.
  - `descriptor.target_sample_rate`: default `16000`; `n_fft`: `1024`;
    `hop_length`: `512`; `eps`: `1e-12`;
    `ridge_min_frequency_hz`: `100.0`; `ridge_max_frequency_hz`: `3000.0`;
    `ridge_candidate_count`: `5`; `ridge_smoothness_penalty`: `8.0`;
    `ridge_peak_prominence_ratio`: `0.0`.
  - `preprocessing.l2_normalize_pools`: default `true`.
  - `preprocessing.pca_dim`: `64` or `128`, default `128`.
  - `preprocessing.embedding_weight` and `descriptor_weight`: default `1.0`.
  - `k_values`: default `[50, 100, 200]`.
  - `random_seed`: default `0`.

  Responses return `EventEncoderJob` with `201` for a new row, `200` for a
  reused or reset row, and `422` for invalid upstream/source configuration.

- `GET /sequence-models/event-encoders` lists jobs newest-first, with an
  optional `?status=` filter.

- `GET /sequence-models/event-encoders/{id}` returns
  `{ job: EventEncoderJob, manifest: object | null, report: object | null }`.

- `GET /sequence-models/event-encoders/{id}/timeline` returns tokenized events
  for the Event Encoder detail timeline viewer. The endpoint reads
  `event_tokens.parquet` from the completed job and treats that artifact as
  authoritative; it does not reload current raw or effective Pass 2 events.
  An optional `?k=` filters to one tokenization. If omitted, the lowest valid
  k in the artifact is selected. Responses include source provenance,
  `region_detection_job_id`, region job timestamp bounds for tiles/audio,
  `selected_k`, all available `valid_k_values`, ordered
  `descriptor_feature_names`, descriptor unit labels, and compact event rows
  with absolute timestamps, token id/label, confidence, centroid-distance
  diagnostics, raw `descriptor_values`, and standardized
  `descriptor_vector_values`. The descriptor vector values are joined from
  `event_vectors.parquet` by source sequence, sequence index, and event id. If
  the vector artifact is missing or unreadable, timeline events still render
  with empty descriptor-vector maps so the detail page can show an unavailable
  state for selected features. Missing jobs or token artifacts return `404`,
  incomplete jobs and corrupt non-`region_crnn` provenance return `409`, and
  unavailable k values return `422`.

- `POST /sequence-models/event-encoders/{id}/cancel` cancels a queued or
  running job. Terminal jobs return `409`.

- `DELETE /sequence-models/event-encoders/{id}` deletes the job row and its
  `event_encoders/{id}/` disk artifacts.

### Schemas And Artifacts

`EventEncoderJob` exposes source provenance, tokenizer version, serialized
configs, `k_values_json`, `random_seed`, `tokenization_signature`, counters,
artifact paths, error state, and timestamps. Effective-mode signatures include
the event-boundary correction revision fingerprint, matching ADR-062 raw versus
effective event semantics.

The active Event Encoder descriptor order is:
`duration`, `log_energy`, `peak_frequency`, `spectral_centroid`, `bandwidth`,
`spectral_entropy`, `ridge_log_frequency_slope`, and `gap_to_previous`.
`ridge_log_frequency_slope` is measured in octaves per second. It is computed
inside Event Encoder descriptor extraction from each event crop using a
band-limited ridge tracker over the event STFT; full STFT matrices are not
persisted in Continuous Embedding artifacts for this feature.

`event_vectors.parquet` contains one row per encoded event with source
sequence key, sequence index, timestamps, descriptors, pooled embedding vector,
scaled descriptor vector, and final event vector. Descriptor columns follow the
active descriptor order and include `ridge_log_frequency_slope`.

`event_tokens.parquet` contains one row per valid k/event with `event_id`,
timing, `token_id`, `token_label`, `distance_to_centroid`,
`second_centroid_distance`, `token_confidence`, and descriptors, including
`ridge_log_frequency_slope`.
Token ids and labels are job-local and k-local; colors or labels in one Event
Encoder job should not be interpreted as globally stable vocabulary entries.

`token_sequences.parquet` contains ordered token streams per valid
`k` and `source_sequence_key`.

`manifest.json` records configs, source job ids, continuous embedding
provenance, ordered `descriptor_feature_names`, valid and invalid k values,
vector dimensions, and encode/skip counters.

`report.json` is the UI-friendly summary: encode counts, token distributions,
closest exemplar event ids per token, ordered descriptor feature names,
descriptor summaries, and a compact token sequence preview.
