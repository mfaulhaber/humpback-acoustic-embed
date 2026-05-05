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
  - `event_source_mode` (default `"raw"`) — `"raw"` reads immutable
    `events.parquet`; `"effective"` reads canonical reviewed events via
    `load_effective_events()`
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

  `event_source_mode="effective"` is rejected for the CRNN region-based
  source because that producer is region-scoped rather than event-padded.

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
`model_version`, `event_source_mode`, source-kind-specific parameters (SurfPerch:
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

SurfPerch idempotency includes `event_source_mode`. Effective-mode
signatures also include a correction revision fingerprint computed from
correction row IDs and `updated_at` values for the selected segmentation
job, so changing reviewed event boundaries produces a distinct continuous
embedding job while raw mode preserves the existing `events.parquet`
signature behavior.

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
  - `event_classification_job_id` (optional; ADR-063) — when omitted the
    server picks the most recent completed `EventClassificationJob`
    whose `event_segmentation_job_id` matches the upstream segmentation;
    rejected with `422` when no completed Classify exists. When provided,
    the value must reference a completed Classify job whose segmentation
    matches the upstream segmentation; otherwise `422`.
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
  per-state label distribution sourced from the bound
  `EventClassificationJob` (Pass 3 Classify) plus `VocalizationCorrection`
  overlay (ADR-063, supersedes ADR-060). Returns the simplified shape
  `{ n_states, total_windows,
  states: { "0": { "label_a": count, ... }, ... } }`. The
  `(background)` bucket holds windows whose center falls outside any
  effective event (or inside an event whose corrections wiped every
  type). Computed on-demand if no cached `label_distribution.json`
  exists. `400` if job not complete. `404` if job not found.

- `GET /sequence-models/hmm-sequences/{id}/exemplars` — per-state
  exemplar windows (high-confidence, nearest-to-centroid,
  boundary-low-confidence). Returns `{ n_states, states: { "0":
  [ExemplarRecord, ...], ... } }`. Each record includes
  `audio_file_id`, time range, `max_state_probability`, and
  `exemplar_type`. `404` if job or artifact not found.

- `POST /sequence-models/hmm-sequences/{id}/generate-interpretations`
  — legacy regenerate; equivalent to
  `regenerate-label-distribution` with no body. Returns
  `{ status: "ok", job_id, label_distribution_generated: true }`.
  `400` if job not complete. `404` if not found.

- `POST /sequence-models/hmm-sequences/{id}/regenerate-label-distribution`
  — rebuild label-distribution + exemplar artifacts (and the
  overlay-parquet recomputation), with an optional Classify re-bind
  (ADR-063). Body:
  `{ "event_classification_job_id"?: string }`. When omitted, the
  current FK is used. When provided, the new Classify job's
  `event_segmentation_job_id` must match the HMM job's upstream
  segmentation; otherwise `400` and nothing changes on disk or in the
  DB. The endpoint validates → writes artifacts via per-file
  temp-then-rename → commits the FK update in a single transaction.
  A failure during the artifact-write step leaves the FK and existing
  files untouched (in-memory FK swap is reverted). Returns
  `{ status: "ok", job_id, event_classification_job_id, label_distribution }`.
  `400` for validation/state errors. `404` if not found.

### Interpretation Schemas (ADR-059, source-agnostic)

`OverlayPoint`: `sequence_id` (string; SurfPerch stringifies its int span
id, CRNN passes its region UUID), `position_in_sequence` (int; window
index for SurfPerch, chunk index for CRNN), `start_timestamp`,
`end_timestamp`, `pca_x`, `pca_y`, `umap_x`, `umap_y`, `viterbi_state`,
`max_state_probability`.

`LabelDistributionResponse`: `n_states`, `total_windows`, `states` (dict
of state index → dict of label → int count). Simplified shape
(ADR-063, supersedes ADR-060) — no tier dimension. Both SurfPerch and
CRNN sources produce the same shape. The reserved `(background)` bucket
holds windows whose center falls outside any effective event or inside
an event whose corrections wiped every type. CRNN's `extras.tier` on
`decoded.parquet` and exemplars is unchanged and continues to drive the
per-state tier-composition strip.

`ExemplarRecord`: `sequence_id`, `position_in_sequence`,
`audio_file_id` (nullable for hydrophone-only jobs), `start_timestamp`,
`end_timestamp`, `max_state_probability`, `exemplar_type`, plus an
`extras` channel for source-specific metadata. CRNN-source records
populate `extras["tier"]` with one of `"event_core"` / `"near_event"` /
`"background"`. ADR-063 also adds `extras["event_id"]` (nullable
string), `extras["event_types"]` (list of strings; empty for
background), and `extras["event_confidence"]` (`{type: float}`) so the
detail-page chips can click through to Classify Review for the
underlying event. SurfPerch records carry the same Classify-binding
fields and only omit `tier`.

**Legacy read-time adapter (transitional).** Pre-ADR-059 SurfPerch
overlay parquets and exemplars JSON files remain on disk with the old
`merged_span_id` (int) / `window_index_in_span` (int) field names. The
overlay and exemplars GET endpoints translate those legacy column / key
names in-memory to the unified shape before serializing the response;
disk files are not rewritten by the adapter. The existing Refresh
button (POST `/regenerate-label-distribution/{id}`) rewrites them in
unified form on demand. The adapter is a structural no-op when the
on-disk artifact is already in unified shape.

ADR-063 supersedes ADR-060: the tier dimension is dropped from
`label_distribution.json`, no read-time tier-shape adapter is shipped.
Existing rows had been deleted via the UI before the cutover and the
artifact directories were wiped, so no migration loader is required. A
fresh regenerate produces the simplified shape directly.

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

### Generalized parent (ADR-061)

Motif extraction now accepts either an HMM job or a masked-transformer
job as the parent. `MotifExtractionJobCreate` carries:

- `parent_kind` (required, `"hmm"` | `"masked_transformer"`)
- `hmm_sequence_job_id` (required when `parent_kind="hmm"`, must be
  null otherwise)
- `masked_transformer_job_id` (required when
  `parent_kind="masked_transformer"`, must be null otherwise)
- `k` (required when `parent_kind="masked_transformer"`, must be in the
  parent job's `k_values`; must be null otherwise)

XOR is enforced at the Pydantic level and again by a SQL CHECK
constraint. `config_signature()` includes `parent_kind`, the parent FK,
and `k` so masked-transformer motif jobs are idempotent per `(parent
job, k, config)` tuple.

The worker constructs the source-agnostic loader (ADR-059) with the
appropriate decoded-artifact path: HMM →
`<hmm_job_dir>/decoded.parquet`; masked-transformer →
`<mt_job_dir>/k<N>/decoded.parquet`. The extraction algorithm,
ranking, and on-disk artifact shape are parent-agnostic — existing
HMM-parent motif jobs render unchanged after the generalization.

`?hmm_sequence_job_id=` and `?masked_transformer_job_id=` filters on
`GET /motif-extractions` are mutually exclusive in practice (each motif
job has exactly one parent FK populated).

## Masked Transformer Jobs (ADR-061)

Masked-transformer jobs train a masked-span transformer encoder on a
completed CRNN region-based continuous-embedding job, extract
contextualized embeddings (Z), and fit one k-means tokenizer per k in
the configured `k_values` sweep. Per-k artifacts (decoded tokens,
overlay, exemplars, label distribution, run-lengths) live under `k<N>/`
subdirs of the job dir; the detail page selects k via the URL `?k=`
parameter.

- `POST /sequence-models/masked-transformers` — create or reuse a
  masked-transformer job. Idempotent on `training_signature` (which
  excludes `k_values` so the k-sweep can be extended without
  retraining). Body fields:

  - `continuous_embedding_job_id` (required) — must reference a
    completed `continuous_embedding_jobs` row with
    `model_version="crnn-call-parsing-pytorch"` (CRNN region-based
    source); SurfPerch sources are rejected with `422` at v1.
  - `event_classification_job_id` (optional; ADR-063) — same semantics
    as on the HMM submit. Defaults to the most recent completed
    Classify for the upstream segmentation; `422` when none exists or
    when the explicit value mismatches.
  - `preset` (default `"default"`, one of `"small"` | `"default"` |
    `"large"`) — selects `(d_model, num_layers, num_heads, ff_dim)`.
  - `k_values` (default `[100]`) — non-empty list of ints ≥ 2;
    deduplicated.
  - `mask_fraction` (default `0.20`)
  - `span_length_min` (default `2`), `span_length_max` (default `6`)
  - `dropout` (default `0.1`)
  - `mask_weight_bias` (default `true`) — when set and the source
    carries tier metadata, scales the per-position loss by tier weight
    (event_core 1.5, near_event 1.2, background 0.5).
  - `cosine_loss_weight` (default `0.0`) — MSE-only when zero.
  - `retrieval_head_enabled` (default `false`) — when true, trains a
    small projection head, writes `retrieval_embeddings.parquet`, and
    fits per-k k-means tokenizers from the retrieval vectors instead of
    contextual hidden states.
  - `retrieval_dim` (default `128` when enabled; null when disabled)
    and `retrieval_hidden_dim` (default `512` when enabled; null when
    disabled) — positive integers controlling the projection head.
  - `retrieval_l2_normalize` (default `true`) — L2-normalizes retrieval
    vectors before persistence and downstream tokenization.
  - `sequence_construction_mode` (default `"region"`, one of
    `"region"` | `"event_centered"` | `"mixed"`) — controls which
    windows feed training. Region mode trains on full CRNN region
    sequences. Event-centered mode trains on one effective-event-centered
    window per boundary-corrected event. Mixed mode combines full-region
    and event-centered candidates deterministically under `seed`.
  - `event_centered_fraction` — normalized to `0.0` in region mode and
    `1.0` in event-centered mode. Mixed mode requires
    `0.0 < event_centered_fraction < 1.0`.
  - `pre_event_context_sec`, `post_event_context_sec` — event-centered
    and mixed modes default omitted values to `2.0`; region mode clears
    both to null. Values must be non-negative.
  - `contrastive_loss_weight` (default `0.0`) — when positive, adds
    human-correction supervised contrastive loss to masked loss. Positive
    values require `retrieval_head_enabled=true` and
    `contrastive_label_source="human_corrections"`. Positive contrastive
    loss also requires `sequence_construction_mode` to be
    `"event_centered"` or `"mixed"` so event-level human labels can align
    to training windows.
  - `contrastive_temperature` (default `0.07`) — positive supervised
    contrastive temperature.
  - `contrastive_label_source` (default `"none"`, one of `"none"` |
    `"human_corrections"`). Model Classify labels are not valid
    contrastive positives.
  - `contrastive_min_events_per_label` (default `4`) and
    `contrastive_min_regions_per_label` (default `2`) — positive support
    thresholds before a human label participates in contrastive masks.
  - `require_cross_region_positive` (default `true`) — when same-label
    different-region positives exist for an anchor, same-region positives
    are excluded for that anchor.
  - `related_label_policy_json` (optional) — JSON policy for related-label
    negative exclusions. Omitted enabled jobs use the Phase 3 defaults.
  - `batch_size` (default `8`) — training mini-batch size. Non-default
    values participate in `training_signature`.
  - `max_epochs`, `early_stop_patience` (default `3`), `val_split`
    (default `0.1`), `seed` (default `42`).

  Returns `MaskedTransformerJob`. Status `201` for new rows; `200` when
  an existing row with the same `training_signature` is returned. `422`
  on validation errors (missing or non-completed upstream, non-CRNN
  upstream, empty `k_values`, k below 2, unknown `preset`).

- `GET /sequence-models/masked-transformers` — list jobs newest-first
  with optional `?status=` and `?continuous_embedding_job_id=` filters.

- `GET /sequence-models/masked-transformers/{id}` — return job detail
  plus tier-composition strip (CRNN sources only) and resolved upstream
  region/embedding metadata. Response shape mirrors the HMM detail
  shape with masked-transformer-specific fields (`k_values`,
  `retrieval_head_enabled`, `retrieval_dim`, `retrieval_hidden_dim`,
  `retrieval_l2_normalize`, `sequence_construction_mode`,
  `event_centered_fraction`, `pre_event_context_sec`,
  `post_event_context_sec`, `batch_size`, contrastive fields
  (`contrastive_loss_weight`, `contrastive_temperature`,
  `contrastive_label_source`, `contrastive_min_events_per_label`,
  `contrastive_min_regions_per_label`, `require_cross_region_positive`,
  `related_label_policy_json`), `chosen_device`, `fallback_reason`,
  `final_train_loss`, `final_val_loss`, `total_epochs`,
  `total_sequences`, `total_chunks`).

- `POST /sequence-models/masked-transformers/{id}/extend-k-sweep` —
  body `{ "additional_k": list[int] }`. Only valid for
  `status="completed"` (`409` otherwise). Dedupes against the existing
  `k_values`; queues a follow-up worker pass that fits + decodes only
  the new k values and triggers `generate_interpretations` per new k.
  `transformer.pt` and `contextual_embeddings.parquet` are not
  retrained.

- `POST /sequence-models/masked-transformers/{id}/cancel` — flip a
  `queued` or `running` job to `cancelled`. `409` for terminal jobs.

- `DELETE /sequence-models/masked-transformers/{id}` — permanently
  delete the job row and `masked_transformer_jobs/{id}/` artifacts
  (transformer, contextual embeddings, all per-k subdirs). `204` on
  success.

- `POST /sequence-models/masked-transformers/{id}/generate-interpretations`
  — body `{ "k_values": list[int] | null }`. Null regenerates per-k
  interpretation artifacts (overlay, exemplars, label distribution) for
  every configured k; a list scopes regeneration to the listed k. Only
  valid for `status="completed"`.

- `POST /sequence-models/masked-transformers/{id}/regenerate-label-distribution?k=N`
  — rebuild **all** `k<N>/label_distribution.json` files in one shot
  with optional Classify re-bind (ADR-063). Body:
  `{ "event_classification_job_id"?: string }`. Effective events load
  exactly once per call regardless of how many k values are configured.
  The `?k=` query parameter selects which payload is returned in the
  response; defaults to the first configured k. Same atomic write
  ordering as the HMM regenerate (validate → temp-then-rename →
  commit FK). Returns `{ status: "ok", job_id, k, event_classification_job_id, label_distribution }`.

- `POST /sequence-models/masked-transformers/{id}/nearest-neighbor-report`
  — run retrieval diagnostics for a completed masked-transformer job.
  The endpoint uses backend source logic in
  `humpback.sequence_models.retrieval_diagnostics`; it does not shell
  out to a standalone script. Body fields:

  - `k` (optional, default first configured k; unknown configured k
    returns `404`)
  - `embedding_space` (default `"contextual"`, one of `"contextual"` |
    `"retrieval"`). Phase 0 supports contextual artifacts for existing
    jobs; requesting `"retrieval"` before Phase 1 writes
    `retrieval_embeddings.parquet` returns `409`.
  - `samples` (default `50`, >= 1), `topn` (default `10`, >= 1),
    `seed` (default `20260504`)
  - `retrieval_modes` (default all): `"unrestricted"`,
    `"exclude_same_event"`, `"exclude_same_event_and_region"`
  - `embedding_variants` (default all): `"raw_l2"`, `"centered_l2"`,
    `"remove_pc1"`, `"remove_pc3"`, `"remove_pc5"`, `"remove_pc10"`,
    `"whiten_pca"`
  - `include_query_rows`, `include_neighbor_rows`, and
    `include_event_level` booleans (all default `false`)

  Response is structured JSON with job metadata, resolved options,
  artifact paths, label coverage, aggregate metrics by retrieval mode
  and embedding variant, representative good/risky query summaries, and
  optional detail rows. Human-label metrics are derived only from
  `VocalizationCorrection` add/remove rows overlapped with
  boundary-corrected effective events; Classify model `TypedEvent`
  labels are not used for retrieval positives or negatives.

  Status codes: `404` when the job or requested k is missing, `409`
  when the job is incomplete or required artifacts are missing, and
  `422` for invalid report options.

- `GET /sequence-models/masked-transformers/{id}/loss-curve` — returns
  `{ "epochs": list[int], "train_loss": list[float], "val_loss":
  list[float | null] }` from `loss_curve.json`. `train_loss` and
  `val_loss` are total-loss compatibility aliases; Phase 3 artifacts
  also include explicit `train_masked_loss`, `train_contrastive_loss`,
  `train_total_loss`, `val_masked_loss`, `val_contrastive_loss`,
  `val_total_loss`, and contrastive skipped-batch series when present.

- `GET /sequence-models/masked-transformers/{id}/reconstruction-error`
  — paginated `(sequence_id, position, score)` rows from
  `reconstruction_error.parquet`; same response shape as the
  `ConfidenceStrip` consumer expects.

Per-k endpoints take a required `?k=` query param (default = first
entry of `k_values`); `404` on unknown k:

- `GET /sequence-models/masked-transformers/{id}/tokens?k=N` —
  paginated `decoded.parquet` rows: `(sequence_id, position, label,
  confidence)`.
- `GET /sequence-models/masked-transformers/{id}/overlay?k=N` —
  paginated 2-D PCA + UMAP projection rows colored by token label.
  Same `OverlayPoint` shape as HMM (`label` reuses the
  `viterbi_state` field for serialization parity).
- `GET /sequence-models/masked-transformers/{id}/exemplars?k=N` —
  per-token exemplar windows. Same `ExemplarRecord` shape as HMM.
- `GET /sequence-models/masked-transformers/{id}/label-distribution?k=N`
  — per-token vocabulary distribution. Simplified `states[token][label] = count`
  shape (ADR-063, supersedes ADR-060), same as HMM.
- `GET /sequence-models/masked-transformers/{id}/run-lengths?k=N` —
  per-token run-length arrays from `run_lengths.json`. Same shape as
  HMM dwell histograms.

### Schemas

`MaskedTransformerJob` exposes the DB row: status, status_reason,
`continuous_embedding_job_id` FK, `training_signature`, all training
hyperparameters (`preset`, `mask_fraction`, `span_length_min`,
`span_length_max`, `dropout`, `mask_weight_bias`, `cosine_loss_weight`,
`batch_size`, `retrieval_head_enabled`, `retrieval_dim`, `retrieval_hidden_dim`,
`retrieval_l2_normalize`, `sequence_construction_mode`,
`event_centered_fraction`, `pre_event_context_sec`,
`post_event_context_sec`, `max_epochs`, `early_stop_patience`,
`val_split`, `seed`), tokenization config (`k_values`), device +
outcomes (`chosen_device`, `fallback_reason`, `final_train_loss`,
`final_val_loss`, `total_epochs`), storage (`job_dir`,
`total_sequences`, `total_chunks`), and UTC timestamps.

`MaskedTransformerJobDetail` adds the tier-composition strip (CRNN
only) and resolved upstream region metadata, mirroring
`HMMSequenceDetail`.

`LossCurveResponse`: `{ epochs: list[int], train_loss: list[float],
val_loss: list[float | null] }`.

`ReconstructionErrorResponse`: paginated `{ total, items:
ReconstructionErrorPoint[] }` where `ReconstructionErrorPoint` is
`(sequence_id, position, score)` for direct consumption by
`ConfidenceStrip`.

`ExtendKSweepRequest`: `{ additional_k: list[int] }`.

`MaskedTransformerNearestNeighborReportRequest`: `{ k?: int,
embedding_space?: "contextual" | "retrieval", samples?: int, topn?: int,
seed?: int, retrieval_modes?: list[str], embedding_variants?: list[str],
include_query_rows?: bool, include_neighbor_rows?: bool,
include_event_level?: bool }`.

`MaskedTransformerNearestNeighborReportResponse`: structured retrieval
diagnostics with job metadata, label coverage, aggregate metrics by
retrieval mode and embedding variant, representative queries, optional
query rows, and optional neighbor rows.

Reused across HMM and masked-transformer detail pages: `OverlayResponse`,
`ExemplarsResponse`, `LabelDistribution`, `StateTierComposition`.

### Behavioral notes

- `training_signature` is computed from
  `(continuous_embedding_job_id, preset, mask_fraction,
  span_length_min, span_length_max, dropout, mask_weight_bias,
  cosine_loss_weight, retrieval-head config when enabled,
  non-region sequence-construction config, max_epochs,
  early_stop_patience, val_split, seed)` — `k_values` is intentionally
  excluded so a completed job can extend its k-sweep. Disabled
  retrieval-head jobs preserve the pre-067 signature shape; default
  region-mode jobs preserve the pre-068 signature shape.
- Event-centered and mixed sequence construction affect training only.
  The worker still extracts contextual embeddings, retrieval embeddings,
  reconstruction error, and per-k decoded tokens over the original
  full-region CRNN sequences, so artifact row counts and downstream
  token consumers remain unchanged.
- Retrieval-head jobs continue writing `contextual_embeddings.parquet`
  for compatibility and diagnostics, and additionally write
  `retrieval_embeddings.parquet`; their per-k `decoded.parquet` bundles
  are tokenized from retrieval embeddings.
- Device validation runs forward+backward on a fixed synthetic batch on
  both CPU and the chosen accelerator before training. On tolerance
  failure the worker records `fallback_reason` and proceeds on CPU.
  `chosen_device` and `fallback_reason` surface as a UI badge.
- K-means token confidence is `max(softmax(−‖z − μ‖² / τ))` with τ
  auto-fit per job from the median pairwise centroid distance. The
  confidence value is persisted in the same `max_state_probability`
  column the HMM uses, so the confidence-strip and overlay-color UI
  components are reused without bespoke code paths.
- Per-k subdirs are written atomically: the worker stages to
  `k<N>.tmp/` and renames to `k<N>/` only after every per-k artifact
  is on disk. A failure mid-write leaves no half-written `k<N>/`.
