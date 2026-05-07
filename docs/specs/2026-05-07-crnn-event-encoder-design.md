# CRNN Event Encoder Tokenization - Design

**Date:** 2026-05-07
**Status:** Draft
**Primary domain:** Sequence Models
**Neighbor domains:** Call Parsing, Core Platform, Frontend Shell

## 1. Goal

Add a Sequence Models workflow that converts Pass 2 segmented events into
discrete acoustic tokens. The first version consumes a completed Call Parsing
`EventSegmentationJob` plus a matching CRNN Continuous Embedding job, pools CRNN
chunk embeddings and acoustic descriptors into one vector per event, runs
k-means tokenization, and writes a reportable token sequence such as:

```text
recording_001:
  T17, T42, T17, T08, T31, T42, T17
```

This is an analysis/report workflow, not a new Call Parsing pass. It belongs
under the Sequence Models navigation alongside Continuous Embedding.

## 2. Scope

### In scope

- New retained `EventEncoderJob` under Sequence Models.
- Input from one completed Pass 2 `EventSegmentationJob`.
- Explicit `event_source_mode`: `raw` or `effective`.
- CRNN embedding source from one completed `ContinuousEmbeddingJob` whose source
  kind is `region_crnn` and whose `event_segmentation_job_id` matches the
  selected segmentation job.
- Event vectors built from:
  - CRNN embedding pools: `mean_pool`, `top_k_pool`, `start_pool`,
    `middle_pool`, `end_pool`.
  - Acoustic descriptors: `duration`, `log_energy`, `peak_frequency`,
    `spectral_centroid`, `bandwidth`, `spectral_entropy`,
    `frequency_slope`, `gap_to_previous`.
- Preprocessing:
  - optional L2 normalization of each embedding pool, default true;
  - PCA over the concatenated embedding-pool block to 64 or 128 dimensions;
  - robust z-score scaling for acoustic descriptors;
  - configurable feature weights before final concatenation.
- Tokenization with k-means for one or more k values, default `[50, 100, 200]`.
- Artifact outputs: event vectors, event token assignments, token sequences,
  tokenizer models, manifest, and report.
- UI page under Sequence Models for create/list/detail report flows.
- Idempotent job creation, cancellation, deletion, worker claim, and retry
  behavior mirroring Continuous Embedding.

### Non-goals

- No new CRNN model architecture or Pass 2 training changes.
- No mutation of `events.parquet` or correction rows.
- No global token vocabulary shared across jobs.
- No timeline overlay in v1; the detail page is report-first.
- No automatic downstream model training from tokens.
- No support for SurfPerch event-padded embeddings in v1.

## 3. Existing Contracts Used

- ADR-056 keeps Sequence Models parallel to Call Parsing. The encoder is a
  downstream analysis job, not Pass 4 of Call Parsing.
- ADR-057 already provides `region_crnn` Continuous Embedding rows with
  `region_id`, `chunk_index_in_region`, timestamps, call probability, event
  overlap metadata, and CRNN chunk embeddings.
- ADR-062 requires event-aware consumers to choose raw or effective event
  semantics explicitly. Effective mode must include correction revision identity
  in idempotency.
- Continuous Embedding idempotency is canonical on `encoding_signature`. Event
  Encoder should use the upstream embedding row as immutable input and create
  its own `tokenization_signature`.

## 4. Approaches Considered

### Approach A: Pool Existing CRNN Continuous Embedding Chunks

Use the existing `region_crnn` `embeddings.parquet` as the CRNN feature source.
For each selected event, join chunks by `(region_id, timestamp overlap)`, pool
the chunk embeddings, compute descriptors from event audio, then tokenize.

Pros:

- Reuses the retained Continuous Embedding producer and idempotency boundary.
- Avoids touching Pass 2 CRNN inference internals.
- Lets raw and effective event modes reuse the same region-scoped embedding
  source because chunks are timestamped over full regions.
- Fastest path to a report UI.

Cons:

- Pools chunk embeddings, not raw frame embeddings. The current CRNN chunks are
  eight-frame BiGRU concat/projection vectors.
- Event boundary changes can alter pooling without changing the upstream
  Continuous Embedding job, so the encoder must carry its own correction
  revision in `tokenization_signature`.

Verdict: recommended for v1.

### Approach B: Extend Continuous Embedding to Emit Event-Scoped Pools

Add an optional event-pooling sidecar to the CRNN Continuous Embedding worker.
The producer would write both chunk embeddings and event-level vectors.

Pros:

- Keeps CRNN inference and event pooling in one worker pass.
- Could access exact frame activations before chunk projection.

Cons:

- Couples tokenization experiments to the embedding producer.
- Requires rerunning expensive CRNN extraction when only k-means or descriptor
  weighting changes.
- Makes `continuous_embedding_jobs` responsible for a modeling/report output,
  weakening the current "producer only" boundary.

Verdict: defer unless chunk pooling proves insufficient.

### Approach C: Re-run CRNN Over Event Crops in a New Worker

The Event Encoder worker would load the segmentation CRNN and run inference over
each event crop directly.

Pros:

- Exact per-event frame embeddings and simpler event joins.
- Independent of Continuous Embedding artifact shape.

Cons:

- Duplicates expensive CRNN feature extraction and source audio resolution.
- Reintroduces Pass 2 internal coupling in a second worker.
- More GPU/device failure modes.

Verdict: rejected for v1.

## 5. Recommended Architecture

Add a new Sequence Models job family:

```text
EventSegmentationJob (Pass 2)
        +
ContinuousEmbeddingJob (region_crnn, complete)
        |
        v
EventEncoderJob
  -> event_vectors.parquet
  -> event_tokens.parquet
  -> token_sequences.parquet
  -> report.json
```

The Event Encoder worker does not call the CRNN. It reads:

- selected events from `events.parquet` or `load_effective_events()`;
- CRNN chunks from the completed Continuous Embedding parquet;
- source audio through the same Call Parsing audio source chain for acoustic
  descriptors.

The job is retained and idempotent. Re-submitting an in-flight or complete
`tokenization_signature` returns the existing row. Re-submitting a failed or
canceled signature resets the row to `queued`.

## 6. Data Model

Add `EventEncoderJob` in `src/humpback/models/sequence_models.py`.

| Column | Type | Notes |
|---|---|---|
| `id` | string UUID | standard job id |
| `status` | text | `queued`, `running`, `complete`, `failed`, `canceled` |
| `event_segmentation_job_id` | string | required Pass 2 source |
| `event_source_mode` | text | `raw` or `effective`, default `raw` |
| `continuous_embedding_job_id` | string | required completed CRNN source |
| `continuous_embedding_signature` | text | copied from upstream row for provenance |
| `tokenizer_version` | text | default `crnn-event-encoder-v1` |
| `pooling_config_json` | text | enabled pools, top-k settings, min coverage |
| `descriptor_config_json` | text | descriptor settings |
| `preprocessing_config_json` | text | L2, PCA dim, feature weights |
| `k_values_json` | text | sorted list such as `[50,100,200]` |
| `random_seed` | integer | deterministic PCA/k-means seed |
| `tokenization_signature` | text unique | idempotency key |
| `event_vector_dim` | integer nullable | final vector dimension |
| `total_events` | integer nullable | selected source events |
| `encoded_events` | integer nullable | events with usable vectors |
| `skipped_events` | integer nullable | events skipped for coverage/duration |
| `event_vectors_path` | text nullable | artifact path |
| `event_tokens_path` | text nullable | artifact path |
| `token_sequences_path` | text nullable | artifact path |
| `manifest_path` | text nullable | artifact path |
| `report_path` | text nullable | artifact path |
| `error_message` | text nullable | terminal failure detail |
| `created_at`, `updated_at` | datetime | standard timestamps |

Migration uses Alembic and `op.batch_alter_table()` if existing tables are
altered. This feature mainly adds a table, so SQLite batch operations should be
limited to indexes/constraints only if required.

## 7. Idempotency Signature

`tokenization_signature = sha256(canonical_json(...))` with:

- `tokenizer_version`;
- `event_segmentation_job_id`;
- `event_source_mode`;
- `correction_revision` when `event_source_mode="effective"`;
- `continuous_embedding_signature`;
- CRNN source provenance copied from the embedding row:
  `model_version`, `crnn_checkpoint_sha256`, `chunk_size_seconds`,
  `chunk_hop_seconds`, `projection_kind`, `projection_dim`;
- `pooling_config`;
- `descriptor_config`;
- `preprocessing_config`;
- sorted `k_values`;
- `random_seed`.

Use the same reset/reuse semantics as Continuous Embedding:

- `complete`, `running`, `queued`: return existing row.
- `failed`, `canceled`: reset counters and artifact paths, set `queued`.

## 8. API Surface

Extend `src/humpback/api/routers/sequence_models.py`.

- `POST /sequence-models/event-encoders`
  - creates or reuses an Event Encoder job;
  - returns 201 for new rows, 200 for reused/reset rows;
  - validates source rows and config, returning 422 for invalid inputs.
- `GET /sequence-models/event-encoders`
  - list newest-first, optional `?status=`.
- `GET /sequence-models/event-encoders/{job_id}`
  - returns `{ job, manifest, report }`, with `manifest/report` nullable while
    queued/running.
- `POST /sequence-models/event-encoders/{job_id}/cancel`
  - mirrors Continuous Embedding cancellation.
- `DELETE /sequence-models/event-encoders/{job_id}`
  - deletes row and `event_encoders/{job_id}/` artifacts.

### Create Request Defaults

| Field | Default |
|---|---|
| `event_source_mode` | `raw` |
| `pooling.enabled` | all five pools |
| `pooling.top_k_fraction` | `0.25` |
| `pooling.min_overlap_fraction` | `0.25` |
| `pooling.min_chunks_per_event` | `1` |
| `preprocessing.l2_normalize_pools` | `true` |
| `preprocessing.pca_dim` | `128` |
| `preprocessing.embedding_weight` | `1.0` |
| `preprocessing.descriptor_weight` | `1.0` |
| `k_values` | `[50, 100, 200]` |
| `random_seed` | `0` |

Validation:

- selected segmentation job must exist and be complete;
- selected Continuous Embedding job must exist, be complete, be `region_crnn`,
  and match `event_segmentation_job_id`;
- `k_values` must be positive, unique, and each `k <= encoded_events` at worker
  time; impossible k values are skipped and reported rather than failing the
  whole job if at least one k is valid;
- `pca_dim` must be 64 or 128 for v1;
- worker computes `effective_pca_dim = min(requested_pca_dim,
  encoded_events - 1, embedding_pool_dim)` and records any clamp in the report;
- effective mode computes a segmentation-scoped correction revision, following
  ADR-062.

## 9. Storage Layout

Add helpers to `src/humpback/storage.py`:

```text
event_encoders/{job_id}/
  manifest.json
  report.json
  event_vectors.parquet
  event_tokens.parquet
  token_sequences.parquet
  preprocess.joblib
  kmeans_k50.joblib
  kmeans_k100.joblib
  kmeans_k200.joblib
```

All writes should be atomic: write same-directory temp files, then `os.replace`.
On cancellation or failure, partial artifacts are removed or left unreachable
from row paths until a successful completion updates the job row.

## 10. Artifact Schemas

### `event_vectors.parquet`

One row per encoded event.

| Column | Type | Notes |
|---|---|---|
| `event_id` | string | raw or effective event id |
| `region_id` | string | source region |
| `source_sequence_key` | string | audio file id or hydrophone range key |
| `sequence_index` | int32 | event order within source |
| `start_timestamp` | float64 | source timeline timestamp |
| `end_timestamp` | float64 | source timeline timestamp |
| `segmentation_confidence` | float64 | from selected event set |
| `duration` | float32 | seconds |
| `log_energy` | float32 | descriptor |
| `peak_frequency` | float32 | Hz |
| `spectral_centroid` | float32 | Hz |
| `bandwidth` | float32 | Hz |
| `spectral_entropy` | float32 | normalized entropy |
| `frequency_slope` | float32 | Hz/sec |
| `gap_to_previous` | float32 | seconds, 0 for first event |
| `chunk_count` | int32 | chunks used for pooling |
| `coverage_fraction` | float32 | event duration covered by selected chunks |
| `embedding_vector` | list<float32> | post-PCA embedding block |
| `descriptor_vector` | list<float32> | robust-scaled descriptor block |
| `event_vector` | list<float32> | final weighted vector |

### `event_tokens.parquet`

One row per `(k, event_id)`.

| Column | Type | Notes |
|---|---|---|
| `k` | int32 | k-means cluster count |
| `event_id` | string | joins to `event_vectors.parquet` |
| `region_id` | string | copied for report convenience |
| `source_sequence_key` | string | copied for report convenience |
| `sequence_index` | int32 | copied for report convenience |
| `start_timestamp` | float64 | copied for report convenience |
| `end_timestamp` | float64 | copied for report convenience |
| `token_id` | int32 | stable job-local token id |
| `token_label` | string | display label, e.g. `T17` |
| `distance_to_centroid` | float32 | nearest-centroid distance |
| `second_centroid_distance` | float32 nullable | second-nearest distance |
| `token_confidence` | float32 | `1 - d1 / (d2 + eps)`, clipped to `[0, 1]` |
| descriptor columns | float32 | same acoustic descriptor values |

Token ids are job-local. They are stable for the same
`tokenization_signature`, but should not be compared across different encoder
jobs as a shared vocabulary.

### `token_sequences.parquet`

One row per token occurrence, sorted by source and event start.

| Column | Type | Notes |
|---|---|---|
| `k` | int32 | k-means cluster count |
| `source_sequence_key` | string | recording/range key |
| `position` | int32 | token position in sequence |
| `event_id` | string | source event |
| `token_id` | int32 | token id |
| `token_label` | string | e.g. `T17` |
| `start_timestamp` | float64 | source timeline timestamp |
| `end_timestamp` | float64 | source timeline timestamp |
| `gap_to_previous` | float32 | seconds |

### `report.json`

The detail API loads this directly. Include:

- source provenance and configs;
- event counts and skip reasons;
- valid/invalid k values;
- token distribution per k;
- k-means inertia and sampled silhouette score when feasible;
- per-token descriptor summaries;
- per-token exemplar event ids nearest to centroid;
- sequence preview per source, truncated for UI display.

## 11. Event Vector Algorithm

### Event selection

- Raw mode reads `call_parsing/segmentation/{job_id}/events.parquet`.
- Effective mode calls `load_effective_events(event_segmentation_job_id)`.
- Events are sorted by `(source_sequence_key, start_timestamp, end_timestamp,
  event_id)`.
- `gap_to_previous` is computed within each `source_sequence_key`; the first
  event uses `0.0`.

### Chunk join

For each event:

1. Select CRNN chunks with the same `region_id`.
2. Compute overlap with `[event.start, event.end)`.
3. Keep chunks with positive overlap.
4. Compute total event coverage from the union of selected chunk overlaps.
5. Skip the event when coverage is below `pooling.min_overlap_fraction` and the
   positive-overlap chunk count is below `pooling.min_chunks_per_event`.
6. If no chunks qualify, skip the event and record the reason.

Do not rely on the upstream `nearest_event_id` column for pooling, because that
column was computed against the raw Pass 2 events used by the Continuous
Embedding job. The encoder recomputes overlap against the selected raw or
effective event set.

### Embedding pools

Let `E_i` be the selected chunk embeddings and `w_i` be overlap-duration weights.

- `mean_pool`: weighted mean over all selected chunks.
- `top_k_pool`: weighted mean over the top fraction of chunks by
  `call_probability`, default top 25 percent with at least one chunk.
- `start_pool`: weighted mean over chunks overlapping the first third of the
  event.
- `middle_pool`: weighted mean over chunks overlapping the middle third.
- `end_pool`: weighted mean over chunks overlapping the final third.

If a segment pool has no chunks, fall back to the nearest selected chunk by
chunk center. This keeps vector dimensions stable for short events.

### Acoustic descriptors

Descriptors are computed from event audio, using the source resolved through
the selected `EventSegmentationJob -> RegionDetectionJob` chain.

- `duration`: `end - start`.
- `log_energy`: log mean square energy with epsilon.
- `peak_frequency`: frequency bin with maximum mean magnitude.
- `spectral_centroid`: magnitude-weighted mean frequency.
- `bandwidth`: magnitude-weighted frequency standard deviation.
- `spectral_entropy`: entropy of normalized mean spectrum.
- `frequency_slope`: least-squares slope of per-frame peak frequency over time.
- `gap_to_previous`: seconds since previous event in the same source sequence.

Use the segmentation model feature sample rate when available; otherwise use
the same target sample-rate conventions as Pass 2 audio loading.

### Preprocessing and tokenization

1. Optionally L2-normalize each pool vector independently.
2. Concatenate enabled pools in fixed order:
   `mean`, `top_k`, `start`, `middle`, `end`.
3. Fit PCA on the embedding-pool block, with `random_seed`; clamp the effective
   dimension when the event count is smaller than the requested PCA dimension.
4. Robust-scale descriptor columns using median and MAD:
   `z = (x - median) / max(1.4826 * MAD, eps)`.
5. Concatenate:
   `[embedding_weight * pca_embedding, descriptor_weight * descriptor_z]`.
6. Fit k-means for each valid k.
7. Remap cluster labels deterministically by sorting centroids along the first
   PCA dimension and then centroid norm; emit `token_id` and `token_label`.
8. Compute `distance_to_centroid`, `second_centroid_distance`, and
   `token_confidence`.

Use `sklearn` components already available in the project. For v1, use
deterministic `KMeans` for moderate event counts and allow a later switch to
`MiniBatchKMeans` if large hydrophone ranges make runtime unacceptable.

## 12. Worker Flow

1. Claim queued `EventEncoderJob` with the standard atomic status update.
2. Merge the row into the worker session and validate that upstream rows still
   exist and are complete.
3. Load selected events and CRNN chunks.
4. Resolve the parent audio source and build an audio loader.
5. Build event vectors, recording skips with explicit reason counts.
6. If no events can be encoded, fail the job with a clear `error_message`.
7. Fit preprocessing and k-means models.
8. Write artifacts atomically.
9. Refresh the job row; if canceled, cleanup partial artifacts and return.
10. Update counters, artifact paths, and `status="complete"`.
11. On exception, cleanup partial artifacts and mark `failed`.

Cancellation checks should happen after loading inputs, during vector building
between regions, and between k-means fits.

## 13. Frontend Design

Add routes:

- `/app/sequence-models/event-encoder`
- `/app/sequence-models/event-encoder/:jobId`

Update SideNav and Breadcrumbs so Sequence Models contains:

- Continuous Embedding
- Event Encoder

### Jobs Page

Use the same layout pattern as Continuous Embedding:

- create panel at the top;
- active jobs table;
- previous jobs table.

Create form controls:

- Pass 2 segmentation job selector, complete jobs only;
- Event source segmented control: Raw / Effective;
- CRNN Continuous Embedding selector filtered to completed `region_crnn` jobs
  matching the selected segmentation job;
- PCA dimension segmented control: 64 / 128;
- k values as checkboxes for 50, 100, 200;
- advanced collapsible panel for pool selection, top-k fraction, min coverage,
  weights, random seed.

If no compatible CRNN Continuous Embedding exists, show an inline callout with a
link to Continuous Embedding and prefilled guidance. Do not auto-create the
embedding job in v1.

### Detail Page

Report-first view:

- source provenance and config summary;
- status/error panel;
- event count summary: total, encoded, skipped;
- token distribution chart per selected k;
- token sequence preview per source;
- token descriptor summary table;
- exemplar event ids per token;
- artifact path table for parquet/report outputs.

Polling behavior matches Continuous Embedding detail: active jobs refetch every
3 seconds, terminal jobs stop polling.

## 14. Tests

Backend targeted tests:

- Pydantic schema validation for defaults, invalid k values, PCA dimension, and
  invalid source combinations.
- Service idempotency:
  - new row on first submit;
  - reuse for queued/running/complete;
  - reset failed/canceled;
  - effective-mode correction revision changes signature.
- Service validation:
  - segmentation job missing/not complete;
  - Continuous Embedding missing/not complete;
  - Continuous Embedding is not `region_crnn`;
  - embedding source segmentation id mismatch.
- Worker pure-function tests:
  - chunk/event overlap join uses selected effective events, not upstream
    `nearest_event_id`;
  - short events fall back for empty start/middle/end pools;
  - first event gap is zero and later gaps are correct;
  - token confidence handles close and distant second centroids.
- Worker artifact tests:
  - writes all expected files atomically;
  - invalid k values are reported and skipped;
  - no encodable events fails clearly;
  - cancellation cleans partial artifacts.
- Migration test for the new `event_encoder_jobs` table and unique signature.
- API integration tests for create/list/detail/cancel/delete.

Frontend tests:

- TypeScript compile.
- Playwright route/nav test for Event Encoder.
- Create form filters compatible CRNN Continuous Embedding jobs after a
  segmentation job selection.
- Detail page renders report summaries and terminal errors.

Targeted verification before full gate:

```text
uv run pytest tests/sequence_models tests/services/test_event_encoder_service.py tests/workers/test_event_encoder_worker.py tests/integration/test_sequence_models_api.py -q
cd frontend && npx tsc --noEmit
cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts
```

Final gate remains `uv run pytest tests/`.

## 15. Risks and Mitigations

- **Chunk pooling may be too coarse for very short events.** Mitigate with
  fallback segment pools, skip reporting, and later Approach B if frame-level
  fidelity is required.
- **Effective event boundaries may not align with raw CRNN event metadata.**
  Mitigate by recomputing event/chunk overlap against the selected event set.
- **Token IDs can look more stable than they are.** Mitigate with job-local
  token labeling and report text that treats token IDs as local to one
  signature.
- **Large hydrophone ranges may make k-means expensive.** Mitigate first with
  PCA and deterministic sampling for silhouette/report metrics; switch fitting
  to `MiniBatchKMeans` only when measured.
- **Descriptor scaling can over-weight sparse descriptors.** Mitigate with
  explicit `descriptor_weight`, persisted robust scaler stats, and report
  descriptor summaries by token.

## 16. Open Questions

- Should v1 reject non-identity CRNN Continuous Embedding projection sources, or
  allow them and record that the tokenization used projected chunk embeddings?
  Recommendation: allow them, but default the UI toward identity/1024-d sources.
- Should the detail page include audio playback links for exemplar events in
  v1? Recommendation: defer; report event ids are enough for the first slice.
- Should k-means run all default k values in one job or exactly one selected k?
  Recommendation: allow multiple k values in one job because preprocessing is
  shared and the report benefits from side-by-side comparison.

## 17. Documentation Updates

Implementation should update:

- `docs/reference/data-model.md` with `EventEncoderJob`.
- `docs/reference/storage-layout.md` with `event_encoders/{job_id}/`.
- `docs/reference/sequence-models-api.md` with Event Encoder endpoints and
  artifact schemas.
- `docs/reference/behavioral-constraints.md` with Event Encoder idempotency and
  raw/effective source-mode semantics.
- `docs/agent-context/domains/sequence-models/README.md`,
  `invariants.md`, `references.md`, and `tests.md`.
