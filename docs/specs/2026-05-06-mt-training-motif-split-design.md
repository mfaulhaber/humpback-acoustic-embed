# MT Training And MT Motif Split - Design

**Date:** 2026-05-06
**Status:** Implemented

## 1. Purpose

Split the current Masked Transformer workflow into two user-facing surfaces:

1. **MT Training** - train a masked-transformer model from multiple
   continuous-embedding plus Classify job pairs, without exposing vocalization
   contrastive learning.
2. **MT Motif** - inference and motif creation. For the first implementation,
   this is only a rename of the current Masked Transformer page; the deeper
   inference/motif refactor is deferred.

The immediate goal is to make model training a first-class multi-source
workflow and to move training analysis into a detail page that is not dominated
by timeline and motif UI.

The user has deleted existing continuous-embedding and masked-transformer jobs,
so historical job-row/data backfill is out of scope. Schema changes can assume
there are no production MT job rows to migrate, though Alembic migrations are
still required for the local database schema.

## 2. Current State

Current anchors:

- `frontend/src/components/sequence-models/MaskedTransformerJobsPage.tsx`
  combines job creation, job listing, and links to the current detail page.
- `frontend/src/components/sequence-models/MaskedTransformerCreateForm.tsx`
  accepts exactly one CRNN continuous-embedding job plus one matching Classify
  job. It exposes retrieval-head, sequence-construction, contrastive, sampler,
  and ablation controls.
- `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`
  combines training analysis, token timeline, motif extraction, exemplar
  badges, and label distribution.
- `src/humpback/models/sequence_models.py::MaskedTransformerJob` stores one
  `continuous_embedding_job_id` and one `event_classification_job_id`.
- `src/humpback/services/masked_transformer_service.py` owns job identity,
  Classify binding, k-sweep extension, and per-k interpretation artifacts.
- `src/humpback/workers/masked_transformer_worker.py` trains the transformer,
  writes model and embedding artifacts, fits per-k tokenizers, and generates
  interpretation artifacts.
- `POST /sequence-models/masked-transformers/{job_id}/nearest-neighbor-report`
  already exposes Phase 0 retrieval diagnostics, including optional geometry
  diagnostics via `include_geometry_report`.

Current detail panels:

- Token timeline
- Motifs
- Loss curve
- Run-length histograms
- UMAP overlay
- Exemplars, including vocalization label badges
- Label distribution

The MT Training detail should keep the analysis-oriented pieces but remove the
timeline, motif creation, label distribution, and exemplar vocalization label
badges.

## 3. Goals

1. Add an MT Training page that creates jobs from multiple source pairs:
   `(continuous_embedding_job_id, event_classification_job_id)`.
2. Train one model over all selected sources.
3. Persist model, tokenizer, metadata, and diagnostic artifacts needed for a
   later MT Motif inference page.
4. Keep the new MT Training flow free of vocalization contrastive-learning
   controls and behavior.
5. Preserve existing backend contrastive functionality for research or legacy
   callers; do not delete it as part of this split.
6. Rename the current Masked Transformer page/navigation to MT Motif without
   refactoring its internals yet.
7. Add an Analysis button to MT Training detail that runs the full Phase 0
   nearest-neighbor diagnostics endpoint with geometry diagnostics included.
8. Display the analysis report on a child page of the MT Training detail in
   table form, with green/yellow/red indicators wherever the measurement's
   direction is clear.

## 4. Non-Goals

- Do not build the full MT Motif inference workflow in this implementation.
- Do not remove contrastive training code, schemas, worker support, or tests.
- Do not migrate historical MT jobs or embedding jobs.
- Do not rewrite the shared timeline system.
- Do not change motif extraction public schemas unless the later MT Motif
  refactor requires it.
- Do not redesign Phase 0 diagnostic math in this split.

## 5. User Experience

### 5.1 Navigation

Sequence Models navigation becomes:

- Continuous Embedding
- HMM Sequence
- MT Training
- MT Motif

`MT Motif` is a rename of the current Masked Transformer page. Its route may be
renamed to `/app/sequence-models/mt-motif`, with redirects from the current
`/app/sequence-models/masked-transformer` route to avoid broken links. The
content can remain the current combined page until the later refactor.

`MT Training` gets new routes:

- `/app/sequence-models/mt-training`
- `/app/sequence-models/mt-training/:jobId`
- `/app/sequence-models/mt-training/:jobId/analysis`

### 5.2 MT Training Create Page

The create form focuses on model training:

- Source pair selector, allowing 1 or more rows.
- Each row selects a completed CRNN region-based continuous-embedding job.
- Each row selects a completed Classify job for the same segmentation as that
  embedding job.
- The form prevents duplicate source pairs.
- The form validates compatibility across selected embedding jobs.
- Training controls remain: preset, k values, max epochs, mask fraction, span
  lengths, dropout, cosine loss weight, batch size, retrieval head enablement,
  retrieval head architecture, retrieval dimensions, L2 normalization,
  sequence-construction mode, event-centered context, early stopping, validation
  split, and seed.
- Contrastive controls are not rendered.
- Projection-head-only ablation controls are not rendered.

The submitted payload always normalizes contrastive configuration to:

- `contrastive_loss_weight = 0.0`
- `contrastive_label_source = "none"`
- `training_freeze_mode = "none"`
- `source_masked_transformer_job_id = null`
- `negative_label_family_policy_json = null`

### 5.3 Source Compatibility Rules

All selected source pairs must satisfy:

- Continuous-embedding job status is `complete`.
- Continuous-embedding source kind is `region_crnn`.
- Every embedding job has a non-null `event_segmentation_job_id`.
- The selected Classify job is `complete`.
- The selected Classify job belongs to the selected embedding job's
  `event_segmentation_job_id`.
- All selected embedding jobs share the same vector dimension.
- All selected embedding jobs share compatible CRNN projection/input metadata:
  `model_version`, `chunk_size_seconds`, `chunk_hop_seconds`,
  `projection_kind`, and `projection_dim`.

Implementation resolution: `crnn_checkpoint_sha256` must match when every
selected source has a non-null value. Null checkpoint values from older rows do
not block otherwise compatible source sets.

### 5.4 MT Training List Page

The training list shows active and previous jobs, similar to the current MT job
table, but source display changes from one source ID to a compact summary:

- source count
- total chunks
- preset
- k values
- retrieval head mode
- status/device
- actions

Rows open the MT Training detail page.

### 5.5 MT Training Detail Page

The detail page shows:

- job metadata and status
- source-pair table
- model/artifact summary
- loss curve
- run-length histograms by selected k
- UMAP/token overlay by selected k
- exemplar gallery by selected k, without vocalization label badges
- Analysis button

The detail page does not show:

- token timeline
- spectrogram playback
- motif extraction panel
- motif occurrence navigation
- label distribution
- exemplar vocalization label badges

The exemplar gallery may still show token, time, probability, tier, and
exemplar type. It must not show `extras.event_types` chips on MT Training.

### 5.6 Analysis Child Page

The Analysis button runs the Phase 0 endpoint:

`POST /sequence-models/masked-transformers/{job_id}/nearest-neighbor-report`

with "full report" options:

- all default retrieval modes
- all default embedding variants
- `include_event_level = true`
- `include_geometry_report = true`
- `include_query_rows = true`
- `include_neighbor_rows = false` by default, unless the UI adds an advanced
  option for detailed neighbor rows
- default geometry spaces, which include contextual and retrieval variants
- selected k, defaulting to the first configured k

On success, navigate to:

`/app/sequence-models/mt-training/:jobId/analysis`

The child page displays:

- report metadata table
- label coverage table
- aggregate retrieval metrics table, grouped by retrieval mode and embedding
  variant
- event-level metrics table when present
- geometry diagnostics table
- representative good queries table
- representative risky queries table

The implementation persists the latest report artifact so the analysis child
page can reload independently of React Query cache state:

- `masked_transformer_jobs/{job_id}/analysis/latest_report.json`
- `GET /sequence-models/masked-transformers/{job_id}/nearest-neighbor-report/latest`

If persistence is added, the POST still calls the same Phase 0 report builder
and writes its returned payload; no second diagnostic implementation should be
introduced.

## 6. Data Model

### 6.1 Recommended Shape

Keep `masked_transformer_jobs` as the training job table, and add a child table:

`masked_transformer_job_sources`

Fields:

- `id`
- `masked_transformer_job_id`
- `source_order`
- `continuous_embedding_job_id`
- `event_classification_job_id`
- `source_alias`, nullable display name
- `created_at`
- `updated_at`

Constraints:

- unique `(masked_transformer_job_id, source_order)`
- unique `(masked_transformer_job_id, continuous_embedding_job_id,
  event_classification_job_id)`

For implementation simplicity, existing single-source columns on
`masked_transformer_jobs` can be retained and populated with the first source
pair as the primary pair. The child table is authoritative for MT Training.

Because existing MT jobs have been deleted, no historical backfill is required.

### 6.2 Create Schema

Add a multi-source create schema for the MT Training page:

- `sources: list[MaskedTransformerJobSourceCreate]`
- each source contains `continuous_embedding_job_id` and
  `event_classification_job_id`

The existing single-source fields may remain accepted by the legacy endpoint
for compatibility, but the MT Training page should use `sources`.

Validation:

- `sources` must be non-empty.
- duplicate source pairs are rejected.
- contrastive and ablation fields are rejected or normalized off for
  multi-source MT Training submissions.
- `k_values` remains non-empty and deduplicated.

### 6.3 Job Identity

The training signature should include:

- normalized source-pair list
- training hyperparameters
- retrieval-head config
- sequence-construction config
- seed

The training signature should continue to exclude `k_values` so a completed
training job can extend tokenization sweeps without retraining.

Since Classify jobs are explicit source-pair inputs for MT Training and feed
label-aware diagnostics/artifacts, include `event_classification_job_id` in the
source-pair signature even though contrastive loss is disabled.

## 7. Worker And Artifacts

### 7.1 Multi-Source Loading

The worker loads every source pair, reads each continuous-embedding parquet,
and groups regions as it does today. To prevent region ID collisions across
sources, every sequence ID is namespaced:

`<source_index>:<region_id>`

Persist source identity on all output rows where practical:

- `source_index`
- `continuous_embedding_job_id`
- `event_classification_job_id`
- original `region_id`

The model trains over the concatenated training sequences.

### 7.2 Training Outputs

The MT Training job writes:

- `transformer.pt`
- `loss_curve.json`
- `reconstruction_error.parquet`
- `contextual_embeddings.parquet`
- `retrieval_embeddings.parquet`, when retrieval head is enabled
- `retrieval_head_outputs.parquet`, when retrieval head is enabled
- per-k `kmeans.joblib`
- per-k `decoded.parquet`
- per-k `run_lengths.json`
- per-k `overlay.parquet`
- per-k `exemplars.json`
- optional per-k `label_distribution.json`, if generation remains bundled for
  compatibility, but the MT Training UI does not display it
- `inference_manifest.json`

### 7.3 Inference Manifest

`inference_manifest.json` is the contract consumed by the future MT Motif
inference page. It should include:

- schema version
- training job ID
- model artifact path
- model config from `transformer.pt`
- input vector dimension
- required source kind (`region_crnn`)
- compatible embedding metadata
- retrieval-head output space used for tokenization
- k values and per-k tokenizer paths
- sequence-construction config
- artifact creation timestamp

This lets the later MT Motif page run:

1. choose a completed MT Training job
2. choose target embedding/Classify source(s)
3. extract transformer embeddings with the trained model
4. decode tokens using the selected per-k tokenizer
5. create motifs from inference tokens

The actual inference job/resource is deferred.

## 8. Analysis Report Tables

### 8.1 Metric Tables

Aggregate retrieval tables are keyed by:

- retrieval mode
- embedding variant

Recommended columns:

- same human label
- exact human label set
- same event
- same region
- adjacent 1s
- nearby 5s
- same token
- similar duration
- without human label
- low event overlap
- median cosine
- verdict counts

Color rules where direction is clear:

- Higher is good: `same_human_label`, `exact_human_label_set`,
  `similar_duration`
  - green: `>= 0.50`
  - yellow: `>= 0.25` and `< 0.50`
  - red: `< 0.25`
- Higher is bad for cross-region retrieval: `same_event`, `same_region`,
  `adjacent_1s`, `nearby_5s`, `without_human_label`, `low_event_overlap`
  - green: `< 0.25`
  - yellow: `>= 0.25` and `< 0.50`
  - red: `>= 0.50`
- Verdict counts:
  - green when `good` is the largest bucket
  - red when `bad_time_adjacent` or `bad_unlabeled_or_background` is the
    largest bucket
  - yellow otherwise

Deferred open question: thresholds for `same_token`, `avg_cosine`, and
`median_cosine` are not clearly good or bad without a baseline for the selected
embedding space and variant. Display them without green/yellow/red indicators
for now, or color them only when they appear in existing geometry warnings.

### 8.2 Label Coverage Table

Recommended columns:

- embedding rows
- sampled queries
- human-labeled query pool rows
- human-labeled effective events
- unlabeled effective events
- single-label effective events
- multi-label effective events
- vocalization correction rows

Color rules:

- human-labeled query pool rows:
  - green when `>= samples`
  - yellow when `> 0` but `< samples`
  - red when `0`
- multi-label effective events:
  - no color for now. Multi-label events may be valid, and this split does not
    define whether they imply better or worse training data.

### 8.3 Geometry Diagnostics Table

Geometry rows are keyed by requested geometry space, for example:

- `contextual.raw_l2`
- `contextual.remove_pc10`
- `contextual.whiten_pca`
- `retrieval.raw_l2`
- `retrieval.remove_pc10`
- `retrieval.whiten_pca`

Recommended columns:

- availability
- row count
- vector dimension
- random-pair cosine p50/p75/p95
- mean vector norm and band
- effective rank and band
- effective rank fraction
- PCA pc1, pc1-5, pc1-10 explained variance
- dimension std near-zero fraction
- dimension std dominance ratio
- pre-L2 norm p50/p95 when available
- warnings

Color rules using existing backend bands/warnings:

- `mean_vector_band`
  - `good`: green
  - `okay`: yellow
  - `suspicious`, `collapse_risk`: red
- `effective_rank_band`
  - `broad`, `plausible`: green
  - `weak`: yellow
  - `severe_collapse`: red
- geometry summary:
  - `retrieval_raw_saturated = true`: red
  - `lambda_sweeps_blocked = true`: red
  - no warnings: green
  - warnings without saturation: yellow
- unavailable artifact rows:
  - yellow when the missing space is expected to be absent, such as retrieval
    geometry for a job without retrieval head
  - red when the missing artifact is required by the selected job config

Deferred open question: precise green/yellow/red thresholds for PCA dominance,
dimension std dominance ratio, and pre-L2 norm distribution should follow more
observed runs. For now, show existing warning strings and avoid inventing
additional color thresholds beyond the backend warning semantics.

## 9. Approaches Considered

### Option A - Evolve Existing MT Job Into MT Training Job

Add source-pair support to the existing `masked_transformer_jobs` resource and
worker. Create the new MT Training pages on top of that resource. Rename the
current page to MT Motif as a temporary legacy surface.

Pros:

- Reuses the current worker, artifact layout, API endpoints, and tests.
- Lowest implementation cost.
- Keeps existing contrastive code available without surfacing it in MT
  Training.
- Aligns with the fact that existing MT job data has been deleted.

Cons:

- During the transition, MT Motif is still backed by a training-style job
  resource.
- The existing route name `masked-transformers` remains slightly broader than
  the new UI language.

### Option B - Add A New MT Training Resource

Create separate `mt_training_jobs` tables, services, endpoints, and frontend
types while leaving `masked_transformer_jobs` untouched for the renamed MT
Motif page.

Pros:

- Very clean conceptual split.
- MT Motif legacy behavior is isolated.

Cons:

- Duplicates large amounts of worker and artifact logic.
- More tests and higher regression risk.
- The old resource would still need to be retired later.

### Option C - Store Multi-Source Inputs As JSON On The Job Row

Add a `sources_json` column to `masked_transformer_jobs` and avoid a source
child table.

Pros:

- Fast schema change.
- Simple to serialize in the create API.

Cons:

- Harder to validate and query.
- Harder to show source-level status, counts, or future source weighting.
- Weaker database-level idempotency guarantees.

## 10. Decision

Use Option A with a source child table.

This gives MT Training a real multi-source data model while preserving the
current backend investment. The MT Motif page can be renamed now and refactored
later into a proper inference/motif workflow that consumes
`inference_manifest.json`.

## 11. Testing Strategy

Backend tests:

- schema validation for multi-source create payloads
- duplicate source-pair rejection
- source compatibility validation
- Classify job must match the embedding job segmentation
- training signature includes source pairs and excludes k values
- contrastive fields are forced off or rejected for MT Training submissions
- worker namespaces source region IDs and preserves source metadata in output
  artifacts
- Phase 0 analysis endpoint still returns geometry diagnostics

Frontend tests:

- MT Training route and nav render
- create form adds/removes source rows
- classify dropdown is scoped to each selected embedding job
- contrastive controls do not appear on MT Training
- submit payload contains `sources` and contrastive-off values
- detail page omits timeline, motifs, label distribution, and exemplar label
  badges
- Analysis button posts full Phase 0 options and navigates to the child page
- analysis child page renders metric, coverage, and geometry tables with
  expected color classes for clear cases

Verification gates remain the project gates from `CLAUDE.md`:

- `uv run ruff format --check ...`
- `uv run ruff check ...`
- `uv run pyright ...`
- `uv run pytest tests/`
- frontend `npm` checks from `frontend/`
