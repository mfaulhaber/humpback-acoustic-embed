# Sequence Models API Surface

The Sequence Models track operates parallel to the four-pass call parsing
pipeline. PR 1 lands the **Continuous Embedding** producer — region-bounded,
hydrophone-only 1-second-hop SurfPerch embeddings padded around Pass-1
region detections. Per-job status transitions follow the standard
`queued → running → complete|failed|canceled` pattern.

All endpoints are mounted under `/sequence-models/`.

## Continuous Embedding (PR 1, ADR-056 + ADR-057)

The producer ships in two source families discriminated on
`model_version` (ADR-057):

- **`surfperch-tensorflow2`** (event-padded SurfPerch chunks): pass an
  `event_segmentation_job_id` from a completed Pass 2 job; the producer
  emits 1-second-hop SurfPerch embeddings padded around each event.
- **`crnn-call-parsing-pytorch`** (Pass 1 region-scoped chunks): pass
  `region_detection_job_id`, `event_segmentation_job_id` (Pass 2
  disambiguator), `crnn_segmentation_model_id`, plus chunk geometry and
  projection config; the producer slices the segmentation CRNN's BiGRU
  activations into 250 ms chunks per Pass 1 region.

- `POST /sequence-models/continuous-embeddings` — create or reuse a
  continuous-embedding job, idempotent on `encoding_signature`.

  **SurfPerch source** accepts:
  - `event_segmentation_job_id` (required) — completed Pass 2 job FK
  - `model_version` (default `"surfperch-tensorflow2"`)
  - `hop_seconds` (default `1.0`, must be > 0)
  - `pad_seconds` (default `2.0`, must be >= 0)

  **CRNN region-based source** accepts:
  - `region_detection_job_id` (required) — completed Pass 1 job FK
  - `event_segmentation_job_id` (required disambiguator) — completed
    Pass 2 job whose `region_detection_job_id` matches the submitted
    Pass 1 job
  - `crnn_segmentation_model_id` (required) — `segmentation_models` row
  - `chunk_size_seconds`, `chunk_hop_seconds` (required, > 0;
    typical `0.250` / `0.250` or `0.250` / `0.125`)
  - `projection_kind` (required; `"identity"` | `"random"` | `"pca"`;
    default `"identity"`)
  - `projection_dim` (required, > 0; default `1024` for identity)
  - `model_version` (default `"crnn-call-parsing-pytorch"`)

  XOR rule (Pydantic): when `region_detection_job_id` is set the request
  is the CRNN source; the disambiguator and CRNN-only fields are
  required, and conversely the CRNN-only fields must be absent on
  SurfPerch requests.

  Returns `ContinuousEmbeddingJob`. Status `201` when a new row is
  created; `200` when an existing complete or in-flight (`queued` /
  `running`) row with the same signature is returned, or when a failed /
  canceled row with the same signature is reset back to `queued`.
  `422` on validation errors (missing source job, incomplete upstream
  job, parent mismatch, unsupported `model_version`, CRNN-only fields
  on a SurfPerch source).

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

`ContinuousEmbeddingJob` exposes the producer DB row: status,
`model_version`, source-kind-specific parameters (SurfPerch:
`event_segmentation_job_id`, `hop_seconds`, `pad_seconds`,
`window_size_seconds`; CRNN: `region_detection_job_id`,
`event_segmentation_job_id`, `chunk_size_seconds`, `chunk_hop_seconds`,
`crnn_checkpoint_sha256`, `crnn_segmentation_model_id`,
`projection_kind`, `projection_dim`), `target_sample_rate`,
`feature_config_json`, `encoding_signature`, post-completion summary
stats (SurfPerch: `vector_dim`, `total_events`, `merged_spans`,
`total_windows`; CRNN: `total_regions`, `total_chunks`, `vector_dim`),
`parquet_path`, `error_message`, and UTC timestamps. SurfPerch-only
columns are null on CRNN rows and vice versa.

`ContinuousEmbeddingJobManifest` matches the `manifest.json` sidecar
written next to `embeddings.parquet`. SurfPerch-source manifests
include `spans` with per-merged-span window summaries
(`merged_span_id`, `start_timestamp`, `end_timestamp`, `window_count`,
`event_id`, `region_id`); CRNN-source manifests include `regions` with
per-region chunk-count summaries plus `crnn_checkpoint_sha256`,
projection config, and the parent Pass 1 / Pass 2 job ids.

The parquet artifact schema is source-specific:

- **SurfPerch source** stores one row per embedded window with
  `merged_span_id`, `event_id`, `window_index_in_span`,
  `audio_file_id`, `start_timestamp`, `end_timestamp`, `is_in_pad`, and
  `embedding`.
- **CRNN source** stores one row per chunk with `region_id`,
  `audio_file_id`, `hydrophone_id`, `chunk_index_in_region`,
  `start_timestamp`, `end_timestamp`, `is_in_pad`, `call_probability`,
  `event_overlap_fraction`, `nearest_event_id` (nullable),
  `distance_to_nearest_event_seconds` (nullable), `tier`
  (`event_core` | `near_event` | `background`), and `embedding`.

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

  **CRNN-source jobs** additionally accept (rejected with `422` on
  SurfPerch sources):
  - `training_mode` (`"full_region"` | `"event_balanced"` |
    `"event_only"`; default `"event_balanced"` when source is CRNN)
  - `event_core_overlap_threshold` (default `0.5`)
  - `near_event_window_seconds` (default `5.0`)
  - `event_balanced_proportions` (default
    `{"event_core": 0.40, "near_event": 0.35, "background": 0.25}`;
    must sum to 1.0 ± 1e-6)
  - `subsequence_length_chunks` (default `32`)
  - `subsequence_stride_chunks` (default `16`)
  - `target_train_chunks` (default `200_000`)
  - `min_region_length_seconds` (default `2.0`)

  Returns `HMMSequenceJob`. Status `201` on creation, `422` on
  validation errors (missing or non-complete source job, CRNN-only
  fields on a SurfPerch source, `event_balanced_proportions` not
  summing to 1.0).

- `GET /sequence-models/hmm-sequences` — list jobs newest-first with
  optional `?status=` and `?continuous_embedding_job_id=` filters.

- `GET /sequence-models/hmm-sequences/{id}` — return job detail plus
  `state_summary.json` sidecar (when artifacts exist). Response shape:
  `{ job: HMMSequenceJob, region_detection_job_id: string, region_start_timestamp: number | null, region_end_timestamp: number | null, summary: HMMStateSummary[] | null, tier_composition: StateTierComposition[] | null, source_kind: "surfperch" | "region_crnn" }`.
  Source metadata is resolved from the parent continuous embedding job and
  region detection job. `tier_composition` is only populated for
  CRNN-source jobs. `404` if the job is missing.

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
  `vocalization_labels`. Returns the unified nested shape
  `{ n_states, total_windows,
  states: { "0": { "tier": { "label_a": count, ... } }, ... } }`
  (ADR-060). SurfPerch jobs use the synthetic `"all"` tier key; CRNN
  jobs use the per-chunk tier values (`event_core` / `near_event` /
  `background`). Computed on-demand if no cached
  `label_distribution.json` exists. `400` if job not complete. `404` if
  job not found.

- `GET /sequence-models/hmm-sequences/{id}/exemplars` — per-state
  exemplar windows (high-confidence, nearest-to-centroid,
  boundary-low-confidence). Returns `{ n_states, states: { "0":
  [ExemplarRecord, ...], ... } }`. Each record includes
  `audio_file_id`, time range, `max_state_probability`, and
  `exemplar_type`. `404` if job or artifact not found.

- `POST /sequence-models/hmm-sequences/{id}/generate-interpretations`
  — regenerate all three interpretation artifacts (overlay, label
  distribution, exemplars) for a completed job. Runs unconditionally
  for both source kinds (ADR-060). Returns
  `{ status: "ok", job_id, label_distribution_generated: true }`.
  `400` if job not complete. `404` if not found.

### Interpretation Schemas (ADR-059, source-agnostic)

`OverlayPoint`: `sequence_id` (string; SurfPerch stringifies its int span
id, CRNN passes its region UUID), `position_in_sequence` (int; window
index for SurfPerch, chunk index for CRNN), `start_timestamp`,
`end_timestamp`, `pca_x`, `pca_y`, `umap_x`, `umap_y`, `viterbi_state`,
`max_state_probability`.

`LabelDistributionResponse`: `n_states`, `total_windows`, `states` (dict
of state index → dict of tier key → dict of label → count). Unified
across sources under ADR-060: SurfPerch jobs collapse the tier dimension
to a single synthetic `"all"` key on disk; CRNN jobs persist real tier
keys (`event_core` / `near_event` / `background`). The frontend chart
collapses the tier dimension in `useMemo` so the visual is identical for
both sources; the tier-stratified data is preserved on disk for a future
tier-aware UI.

`ExemplarRecord`: `sequence_id`, `position_in_sequence`,
`audio_file_id` (nullable for hydrophone-only jobs), `start_timestamp`,
`end_timestamp`, `max_state_probability`, `exemplar_type`, plus an
`extras: dict[str, str | int | float | None]` channel for
source-specific metadata. CRNN-source records populate `extras["tier"]`
with one of `"event_core"` / `"near_event"` / `"background"`; SurfPerch
records leave `extras` empty.

**Legacy read-time adapter (transitional).** Pre-ADR-059 SurfPerch
overlay parquets and exemplars JSON files remain on disk with the old
`merged_span_id` (int) / `window_index_in_span` (int) field names. The
overlay and exemplars GET endpoints translate those legacy column / key
names in-memory to the unified shape before serializing the response;
disk files are not rewritten by the adapter. The existing Refresh
button (POST `/generate-interpretations/{id}`) rewrites them in unified
form on demand. The adapter is a structural no-op when the on-disk
artifact is already in unified shape.

The same transitional pattern covers pre-ADR-060 flat
`label_distribution.json` files (`states[state] = {label: count}`): the
GET `/label-distribution` endpoint projects them to
`states[state] = {"all": {label: count}}` in-memory; the file on disk is
unchanged until a Refresh rewrites it in unified form.

## Motif Extraction Jobs

Motif extraction jobs consume completed HMM sequence jobs and mine recurring
symbolic n-grams over collapsed Viterbi state sequences.

- `POST /sequence-models/motif-extractions` — create or reuse a motif
  extraction job. Body fields: `hmm_sequence_job_id`, `min_ngram`,
  `max_ngram`, `minimum_occurrences`, `minimum_event_sources`,
  `frequency_weight`, `event_source_weight`, `event_core_weight`,
  `low_background_weight`, and optional `call_probability_weight`.
  Returns `201` for a new queued job, `200` for an existing queued,
  running, or complete job with the same config signature. Returns `422`
  for invalid configs or non-complete source HMM jobs.

- `GET /sequence-models/motif-extractions` — list motif jobs newest-first
  with optional `?status=` and `?hmm_sequence_job_id=` filters.

- `GET /sequence-models/motif-extractions/{id}` — return
  `{ job, manifest }`, where `manifest` is null until artifacts exist.

- `POST /sequence-models/motif-extractions/{id}/cancel` — cancel queued or
  running jobs. `409` for terminal jobs.

- `DELETE /sequence-models/motif-extractions/{id}` — delete the job row and
  `motif_extractions/{id}` artifacts.

- `GET /sequence-models/motif-extractions/{id}/motifs` — paginated motif
  summary rows from `motifs.parquet`. Query params: `offset` (default 0),
  `limit` (default 100, max 5000). Job must be complete.

- `GET /sequence-models/motif-extractions/{id}/motifs/{motif_key}/occurrences`
  — paginated occurrence rows for one motif from `occurrences.parquet`.

`MotifSummary` rows include `motif_key`, `states`, `length`,
`occurrence_count`, `event_source_count`, `audio_source_count`,
`event_core_fraction`, `background_fraction`, nullable
`mean_call_probability`, duration stats, `rank_score`, and
`example_occurrence_ids`.

`MotifOccurrence` rows include source/group keys, token/raw row ranges,
absolute timestamps, event/background fractions, nullable call probability,
and event-midpoint alignment fields (`anchor_event_id`, `anchor_timestamp`,
`relative_start_seconds`, `relative_end_seconds`, `anchor_strategy`).
