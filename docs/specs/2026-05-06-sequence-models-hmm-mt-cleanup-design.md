# Sequence Models HMM/MT Cleanup Design

**Status:** Approved for planning
**Date:** 2026-05-06
**Track:** Sequence Models cleanup

---

## 1. Goal

Remove the active HMM Sequence, Masked Transformer, and motif-extraction
features from the product surface and runtime stack while preserving Sequence
Models / Continuous Embedding.

The cleanup should remove HMM/MT dependencies across:

- FastAPI endpoints, schemas, services, workers, queue claiming, and storage helpers.
- SQLAlchemy models and Alembic-managed database schema.
- Frontend routes, navigation entries, query hooks, pages, forms, and MT/HMM-only charts.
- Test coverage that exercises removed behavior.
- Generated artifact roots on disk.
- Package dependencies that exist only for the retired HMM/MT functionality.

The end state is a smaller Sequence Models track with one active job family:
`ContinuousEmbeddingJob`.

---

## 2. Non-Goals

- Do not remove `continuous_embedding_jobs`, its idempotency guarantees, its API,
  its worker, or its artifact root at `continuous_embeddings/{job_id}/`.
- Do not remove CRNN-region Continuous Embedding support by default. It was added
  during the HMM/MT era, but it is a continuous-embedding producer and remains
  useful without HMM or MT consumers.
- Do not rewrite historical design documents or plans. They should remain as
  project history. Active reference docs and the active ADR index should be
  updated.
- Do not redesign the Timeline Viewer core, playback, spectrogram rendering,
  tile loading, zoom/pan behavior, or call-parsing overlays.
- Do not remove shared visualization primitives that can support future motif or
  state timelines.
- Do not make Alembic responsible for deleting filesystem artifacts. Database
  migration and artifact cleanup should be separate, explicit operations.

---

## 3. Current State Inventory

### 3.1 Backend Runtime Surfaces

The active Sequence Models router currently combines four distinct concerns in
`src/humpback/api/routers/sequence_models.py`:

- Continuous Embedding jobs.
- HMM Sequence jobs and HMM interpretation endpoints.
- Motif Extraction jobs for HMM and MT parents.
- Masked Transformer training, inference, k-sweep, diagnostics, and analysis
  endpoints.

The cleanup should slim the router to Continuous Embedding only.

Backend services to remove:

- `src/humpback/services/hmm_sequence_service.py`
- `src/humpback/services/masked_transformer_service.py`
- `src/humpback/services/motif_extraction_service.py`

Backend workers to remove:

- `src/humpback/workers/hmm_sequence_worker.py`
- `src/humpback/workers/masked_transformer_worker.py`
- `src/humpback/workers/motif_extraction_worker.py`

Queue and runner cleanup:

- Remove stale recovery for `HMMSequenceJob`, `MaskedTransformerJob`, and
  `MotifExtractionJob` from `src/humpback/workers/queue.py`.
- Remove `claim_hmm_sequence_job`, `claim_masked_transformer_job`, and
  `claim_motif_extraction_job`.
- Remove the HMM, motif, and MT polling blocks from
  `src/humpback/workers/runner.py`.

Storage helpers to remove from `src/humpback/storage.py`:

- `hmm_sequence_*`
- `masked_transformer_*`
- `motif_extraction_*`

Keep the Continuous Embedding helpers:

- `continuous_embedding_dir`
- `continuous_embedding_parquet_path`
- `continuous_embedding_manifest_path`

### 3.2 Domain Modules

Retain modules used by Continuous Embedding, including CRNN-region embedding:

- `src/humpback/sequence_models/chunk_projection.py`
- `src/humpback/sequence_models/crnn_features.py`
- `src/humpback/sequence_models/event_overlap_join.py`

Remove HMM/MT/motif-only modules:

- `src/humpback/sequence_models/hmm_decoder.py`
- `src/humpback/sequence_models/hmm_trainer.py`
- `src/humpback/sequence_models/pca_pipeline.py`
- `src/humpback/sequence_models/region_sampling.py`
- `src/humpback/sequence_models/summary.py`
- `src/humpback/sequence_models/overlay.py`
- `src/humpback/sequence_models/exemplars.py`
- `src/humpback/sequence_models/label_distribution.py`
- `src/humpback/sequence_models/motifs.py`
- `src/humpback/sequence_models/tokenization.py`
- `src/humpback/sequence_models/masked_transformer.py`
- `src/humpback/sequence_models/masked_transformer_sequences.py`
- `src/humpback/sequence_models/contrastive_labels.py`
- `src/humpback/sequence_models/contrastive_loss.py`
- `src/humpback/sequence_models/retrieval_diagnostics.py`
- `src/humpback/sequence_models/retrieval_sweeps.py`
- `src/humpback/sequence_models/loaders/`

After removal, `src/humpback/sequence_models/__init__.py` should describe the
remaining package as Continuous Embedding support utilities rather than HMM/MT
modeling.

### 3.3 Database Schema

Keep:

- `continuous_embedding_jobs`

Drop:

- `motif_extraction_jobs`
- `masked_transformer_job_sources`
- `masked_transformer_jobs`
- `hmm_sequence_jobs`

The drop order matters because motif and MT source tables reference the parent
job tables.

`continuous_embedding_jobs` should retain the CRNN source columns introduced by
the CRNN continuous-embedding path, including `region_detection_job_id`,
`chunk_size_seconds`, `chunk_hop_seconds`, `crnn_checkpoint_sha256`,
`crnn_segmentation_model_id`, `projection_kind`, `projection_dim`,
`total_regions`, and `total_chunks`.

### 3.4 Disk Artifacts

Keep:

- `continuous_embeddings/{job_id}/`

Remove retired artifact roots:

- `hmm_sequences/{job_id}/`
- `masked_transformer_jobs/{job_id}/`
- `motif_extractions/{job_id}/`

Artifact deletion must be dry-run-first and path-safe. It should not happen as a
side effect of importing models, starting the API, or applying the Alembic
migration.

### 3.5 Frontend Surfaces

Keep active pages:

- `ContinuousEmbeddingJobsPage.tsx`
- `ContinuousEmbeddingCreateForm.tsx`
- `ContinuousEmbeddingJobTable.tsx`
- `ContinuousEmbeddingDetailPage.tsx`

Remove active routes and pages for:

- HMM Sequence
- MT Training
- MT Motif / Masked Transformer
- Motif Extraction panels tied to HMM/MT endpoints

Modify:

- `frontend/src/App.tsx`
- `frontend/src/components/layout/SideNav.tsx`
- `frontend/src/components/layout/Breadcrumbs.tsx`
- `frontend/src/api/sequenceModels.ts`

`/app/sequence-models` should continue to route to Continuous Embedding.

### 3.6 Timeline Viewer Controls

The Timeline Viewer should be treated carefully. The cleanup should avoid broad
changes to timeline provider state, spectrogram rendering, tile loading, and
playback behavior.

Keep these shared/future-ready visualization primitives:

- `DiscreteSequenceBar.tsx`
- `RegionNavBar.tsx`
- `SpanNavBar.tsx`
- `MotifTimelineLegend.tsx`
- `MotifHighlightOverlay.tsx`
- `frontend/src/components/sequence-models/constants.ts`
- `frontend/src/lib/motifColor.ts`

The retained primitives should be decoupled from HMM/MT API types. For example,
`MotifHighlightOverlay.tsx` currently imports `MotifOccurrence` from
`@/api/sequenceModels`. That type should move to a neutral timeline or
visualization module so the overlay can remain without keeping motif API hooks.

Remove HMM/MT-specific wrappers and panels:

- `HMMStateBar.tsx`
- `MotifExtractionPanel.tsx`
- `MotifExampleAlignment.tsx`
- `MotifTokenCountSelector.tsx`
- `KPicker.tsx`
- MT/HMM-only Plotly chart components under `components/sequence-models/`

---

## 4. Cleanup Approaches

### Approach A: Hard Removal, Preserve Continuous Embedding

Remove HMM/MT/motif code, routes, workers, DB tables, and artifact roots. Keep
Continuous Embedding, including SurfPerch and CRNN continuous-embedding source
modes.

Pros:

- Best matches the requested cleanup.
- Removes the worker/API dependency chain cleanly.
- Drops stale database and artifact surfaces.
- Reduces test and maintenance burden immediately.

Cons:

- HMM/MT historical jobs become inaccessible after migration.
- Downgrade can recreate empty tables but cannot restore dropped data or deleted
  artifacts.
- Requires careful route/test cleanup to avoid dangling imports.

### Approach B: Soft Hide / Read-Only Legacy

Remove frontend navigation and stop workers, but keep DB tables and read-only API
endpoints.

Pros:

- Preserves historical records.
- Lower data-loss risk.

Cons:

- Does not fully remove HMM/MT API dependencies.
- Keeps schemas, storage helpers, and tests alive.
- Leaves a confusing unsupported surface.

### Approach C: Archive Legacy Feature Set

Move HMM/MT/motif code into an archive namespace or branch and remove it from
the main runtime.

Pros:

- Makes old implementation easier to retrieve.
- Can preserve reference code without runtime imports.

Cons:

- Adds archive maintenance and discovery rules.
- Still requires nearly all of Approach A.
- Historical git history already provides recovery.

### Recommendation

Use Approach A.

Keep the implementation direct: hard-remove runtime support from `main`, leave
historical specs/plans in place, remove retired HMM/MT/motif ADR entries from
the active decision index, and provide an explicit artifact cleanup script with
a manifest.

---

## 5. Chosen Design

### 5.1 Product Behavior

The Sequence Models UI exposes only Continuous Embedding. Existing HMM, MT, and
motif URLs should no longer be registered frontend routes. Existing API endpoints
under `/sequence-models/hmm-sequences`, `/sequence-models/masked-transformers`,
and `/sequence-models/motif-extractions` should return FastAPI's normal 404
because no route exists.

Continuous Embedding behavior remains unchanged:

- Creating jobs remains idempotent on `encoding_signature`.
- Listing, detail, cancel, and delete continue to work.
- Worker output remains under `continuous_embeddings/{job_id}/`.
- Delete continues to remove only the selected Continuous Embedding job artifact
  directory.

### 5.2 Backend API and Schemas

`src/humpback/api/routers/sequence_models.py` should be reduced to:

- Continuous Embedding create.
- Continuous Embedding list.
- Continuous Embedding detail.
- Continuous Embedding cancel.
- Continuous Embedding delete.

`src/humpback/schemas/sequence_models.py` should keep only:

- `ContinuousEmbeddingJobCreate`
- `ContinuousEmbeddingJobOut`
- `ContinuousEmbeddingJobDetail`
- `ContinuousEmbeddingManifest`
- Small nested summaries needed by Continuous Embedding detail responses.

Remove schemas for:

- HMM job creation, detail, transition matrix, dwell histograms, overlays,
  exemplars, and label distribution.
- Motif extraction jobs, motif summaries, and occurrence pages.
- Masked Transformer jobs, source rows, k-sweeps, diagnostics, loss curves,
  decoded tokens, overlays, exemplars, run lengths, and retrieval reports.

### 5.3 ORM Models

`src/humpback/models/sequence_models.py` should export only:

- `ContinuousEmbeddingJob`
- `JobStatus` if still useful locally

Remove ORM classes:

- `HMMSequenceJob`
- `MaskedTransformerJob`
- `MaskedTransformerJobSource`
- `MotifExtractionJob`

`src/humpback/models/__init__.py` already imports only `ContinuousEmbeddingJob`
from this module, so it should require little or no change.

### 5.4 Worker Runtime

The worker runner should continue polling Continuous Embedding jobs after the
existing call-parsing, training, clustering, and hyperparameter jobs. It should
then proceed to idle wait; no HMM/MT/motif polling should remain.

`recover_stale_jobs()` should no longer touch dropped tables. This is important
because otherwise API or worker startup will fail immediately after the migration
when `hmm_sequence_jobs`, `masked_transformer_jobs`, or `motif_extraction_jobs`
are absent.

### 5.5 Database Migration

Create a new Alembic revision:

- `alembic/versions/075_remove_hmm_mt_sequence_models.py`

Upgrade should:

1. Drop `motif_extraction_jobs`.
2. Drop `masked_transformer_job_sources`.
3. Drop `masked_transformer_jobs`.
4. Drop `hmm_sequence_jobs`.

Use project migration conventions for SQLite. If an index or constraint has to
be removed before dropping a table, use `op.batch_alter_table()`. The migration
must not touch `continuous_embedding_jobs`.

This migration is a destructive retirement of the HMM/MT/motif schema. Downgrade
should not recreate the retired tables as active schema. If the project requires
a downgrade function for Alembic hygiene, it should raise a clear unsupported
downgrade error that directs operators to restore from the required database
backup. The revision docstring should say that dropped rows and filesystem
artifacts are not restorable by Alembic.

### 5.6 Disk Artifact Cleanup

Add a standalone script:

- `scripts/cleanup_sequence_model_artifacts.py`

Default behavior:

- Dry-run only.
- Read `settings.storage_root` by default.
- Accept an explicit `--storage-root` for tests or emergency use.
- Target only these root directories under storage root:
  - `hmm_sequences`
  - `masked_transformer_jobs`
  - `motif_extractions`
- Count candidate directories, files, and total bytes.
- Write a JSON manifest under
  `cleanup-manifests/{timestamp}-sequence-models-hmm-mt.json`.
- Refuse to follow symlinks or delete paths outside `storage_root`.
- Require `--apply` for deletion.

Recommended operation order:

1. Stop API and workers.
2. Back up the database.
3. Run the artifact cleanup script in dry-run mode and inspect the manifest.
4. Apply the Alembic migration.
5. Run the artifact cleanup script with `--apply`.
6. Start API and workers.

This order keeps artifact discovery independent of the database schema while
still making the destructive deletion explicit and reviewable.

### 5.7 Dependency Cleanup

Remove from `pyproject.toml` and regenerate `uv.lock`:

- `hmmlearn`

Keep:

- `umap-learn`, because clustering still imports UMAP in
  `src/humpback/clustering/reducer.py`.
- `hdbscan`, because clustering still imports it.
- `torch`, because call parsing and CRNN Continuous Embedding still use it.
- `scikit-learn`, `joblib`, `numpy`, `pyarrow`, and related scientific packages
  used outside the retired HMM/MT path.

Keep frontend Plotly dependencies for now:

- `react-plotly.js`
- `plotly.js-basic-dist-min`
- `@types/react-plotly.js`

Although several Sequence Models charts will be removed, the vocalization UMAP
view still imports `react-plotly.js`.

### 5.8 Documentation

Update active docs:

- `docs/reference/sequence-models-api.md`: document Continuous Embedding only;
  remove active HMM/MT/motif endpoint references.
- `docs/reference/frontend.md`: Sequence Models nav should list only Continuous
  Embedding and retained shared visualization primitives.
- `DECISIONS.md`: remove ADR entries that describe the retired HMM Sequence,
  Masked Transformer, and motif-extraction product surfaces. Preserve or rewrite
  any remaining Continuous Embedding decision text that is still active.

Leave prior specs, plans, and other historical docs intact as historical
records.

---

## 6. Frontend Cleanup Details

### 6.1 Routes and Navigation

Modify `frontend/src/App.tsx`:

- Keep `/app/sequence-models` redirecting to
  `/app/sequence-models/continuous-embedding`.
- Keep `/app/sequence-models/continuous-embedding`.
- Keep `/app/sequence-models/continuous-embedding/:jobId`.
- Remove HMM Sequence, MT Training, MT Motif, and Masked Transformer routes.

Modify `frontend/src/components/layout/SideNav.tsx`:

- Keep a Sequence Models section with Continuous Embedding as the only child.
- Alternatively, make Sequence Models a direct link to Continuous Embedding. The
  section-with-one-child option is preferred because it preserves room for
  future sequence visualization routes without a nav rewrite.

Modify `frontend/src/components/layout/Breadcrumbs.tsx`:

- Keep Continuous Embedding crumbs.
- Remove HMM/MT/motif crumbs.

### 6.2 API Client

Slim `frontend/src/api/sequenceModels.ts` to Continuous Embedding types and
hooks only.

Remove hooks for:

- HMM Sequence jobs.
- Motif Extraction jobs.
- Masked Transformer jobs.
- MT analysis reports and diagnostics.

Move any retained visualization-only types out of this API module. The API
client should not export motif types for endpoints that no longer exist.

### 6.3 Component Retention

Retain and adjust tests for:

- Continuous Embedding components.
- `DiscreteSequenceBar`.
- `RegionNavBar`.
- `SpanNavBar`, if still imported or intentionally kept.
- `MotifTimelineLegend`.
- `MotifHighlightOverlay`.
- `motifColor` utilities.

Delete HMM/MT-only components and tests. Representative files include:

- `HMMSequenceCreateForm.tsx`
- `HMMSequenceDetailPage.tsx`
- `HMMSequenceJobTable.tsx`
- `HMMSequenceJobsPage.tsx`
- `HMMStateBar.tsx`
- `MaskedTransformerCreateForm.tsx`
- `MaskedTransformerDetailPage.tsx`
- `MaskedTransformerJobsPage.tsx`
- `MTTrainingCreateForm.tsx`
- `MTTrainingDetailPage.tsx`
- `MTTrainingJobsPage.tsx`
- `MTTrainingAnalysisPage.tsx`
- `MTAnalysisReportTables.tsx`
- `MTAnalysisSummaryPanel.tsx`
- `LossCurveChart.tsx`
- `TokenRunLengthHistograms.tsx`
- `KPicker.tsx`
- `MotifExtractionPanel.tsx`
- `MotifExampleAlignment.tsx`
- `MotifTokenCountSelector.tsx`

### 6.4 Timeline Viewer Safety Rules

Implementation should avoid broad CSS or state changes in:

- `frontend/src/components/timeline/provider/`
- `frontend/src/components/timeline/Spectrogram*`
- `frontend/src/components/timeline/TileCanvas*`
- Core overlay positioning and playback code.

Only targeted changes should be made to decouple retained motif/state
visualization primitives from removed Sequence Models API types.

---

## 7. Test Plan for Implementation

### 7.1 Backend Tests to Keep or Update

Keep and update Continuous Embedding tests:

- Continuous Embedding service tests.
- Continuous Embedding worker tests.
- Sequence Models API tests that exercise only Continuous Embedding.
- CRNN Continuous Embedding tests for chunk projection, CRNN features, and event
  overlap joins.

Add or update migration tests:

- Verify migration `075` drops HMM/MT/motif tables.
- Verify `continuous_embedding_jobs` remains present and preserves rows.
- Verify the migration's downgrade behavior is intentionally unsupported for the
  destructive table retirement, if a downgrade function is present.

Add artifact cleanup script tests:

- Dry-run records only the three retired roots.
- `--apply` deletes retired roots.
- `continuous_embeddings/` is not deleted.
- Symlink/out-of-root paths are refused.

### 7.2 Backend Tests to Remove

Remove or rewrite tests that import removed modules or endpoints:

- HMM service, worker, decoder, trainer, PCA, summary, overlay, exemplar, label
  distribution, and region-sampling tests.
- Motif extraction service, worker, API, and motif-mining tests.
- Masked Transformer service, worker, API, diagnostics, retrieval sweep,
  tokenization, contrastive loss, and training tests.
- Classify-binding tests that exist only to validate HMM/MT label-source
  requirements.

### 7.3 Frontend Tests to Keep or Update

Keep:

- Continuous Embedding component tests.
- `frontend/e2e/sequence-models/continuous-embedding.spec.ts`.
- Shared visualization primitive tests for retained timeline/motif/state
  controls.

Remove:

- `frontend/e2e/sequence-models/hmm-sequence.spec.ts`
- `frontend/e2e/sequence-models/masked-transformer.spec.ts`
- `frontend/e2e/sequence-models/masked-transformer-motif-ux.spec.ts`
- `frontend/e2e/sequence-models/mt-training.spec.ts`
- `frontend/e2e/sequence-models/classify-binding.spec.ts` if it only covers
  retired HMM/MT behavior.
- Vitest files for deleted HMM/MT/motif components.

---

## 8. Verification Gates

Run the normal project gates after implementation.

Backend:

1. `uv run ruff format --check src/humpback tests scripts alembic`
2. `uv run ruff check src/humpback tests scripts alembic`
3. `uv run pyright src/humpback scripts tests`
4. `uv run pytest tests/`
5. `uv run alembic upgrade head` against a disposable SQLite database.

Frontend:

1. `cd frontend && npm run build`
2. `cd frontend && npx vitest run`
3. `cd frontend && npx playwright test frontend/e2e/sequence-models/continuous-embedding.spec.ts`

Artifact cleanup:

1. `uv run python scripts/cleanup_sequence_model_artifacts.py --dry-run`
2. Script unit tests covering `--apply` should run against a temporary storage
   root, not the developer's real storage root.

Search-based sanity checks:

- No non-doc import of `HMMSequenceJob`, `MaskedTransformerJob`,
  `MaskedTransformerJobSource`, or `MotifExtractionJob`.
- No non-doc import of deleted HMM/MT/motif services or workers.
- No registered frontend route for HMM, MT, or motif extraction.
- No `hmmlearn` entry in `pyproject.toml` or `uv.lock`.
- `continuous_embedding_jobs` and `continuous_embeddings/` references remain.

---

## 9. Acceptance Criteria

- Sequence Models UI exposes Continuous Embedding only.
- Continuous Embedding create/list/detail/cancel/delete still work.
- Continuous Embedding worker still runs SurfPerch and CRNN source modes.
- API and worker startup do not touch dropped HMM/MT/motif tables.
- Alembic head removes `hmm_sequence_jobs`, `masked_transformer_jobs`,
  `masked_transformer_job_sources`, and `motif_extraction_jobs`.
- Alembic head preserves `continuous_embedding_jobs` and existing rows.
- HMM/MT/motif artifact cleanup script dry-runs by default and never targets
  `continuous_embeddings/`.
- HMM/MT/motif artifact cleanup `--apply` deletes only the retired roots under
  the configured storage root.
- `hmmlearn` is removed from project dependencies.
- UMAP/HDBSCAN/Torch/Plotly remain where still used by clustering,
  call-parsing/CRNN, or vocalization views.
- Retained Timeline Viewer primitives compile and pass tests without depending
  on retired API types.
- Full backend and frontend verification gates pass.

---

## 10. Risks and Mitigations

### Risk: Accidental Continuous Embedding Regression

Mitigation:

- Keep `ContinuousEmbeddingJob` schema and ORM changes tightly scoped.
- Keep both SurfPerch and CRNN Continuous Embedding tests.
- Add migration coverage that proves `continuous_embedding_jobs` survives.

### Risk: Startup Failure After Dropping Tables

Mitigation:

- Remove stale recovery and worker polling for retired tables in the same change
  as the migration.
- Run API and worker startup tests against a migrated database.

### Risk: Over-Deleting Artifacts

Mitigation:

- Artifact cleanup is a separate dry-run-first script.
- Delete only allowlisted root names under `settings.storage_root`.
- Refuse symlinks and out-of-root resolved paths.
- Write a manifest before deletion.

### Risk: Timeline Viewer Churn

Mitigation:

- Avoid changes to timeline provider/playback/spectrogram/tile internals.
- Keep motif/state primitives but move their types out of the removed API module.
- Run retained component tests and at least the Continuous Embedding e2e test.

### Risk: Dependency Removal Breaks Other Tracks

Mitigation:

- Remove only `hmmlearn` initially.
- Keep UMAP, HDBSCAN, Torch, and Plotly because current non-retired code still
  imports them.

---

## 11. Resolved Questions

1. Should CRNN-region Continuous Embedding remain enabled in the UI now that HMM
   and MT consumers are removed? Decision: keep Continuous Embedding enabled,
   including the CRNN-region source mode, because it is part of the retained
   Continuous Embedding track.
2. Should downgrade recreate empty retired tables, or should the migration mark
   downgrade unsupported? Decision: remove the retired tables. Do not recreate
   them as active schema on downgrade; require backup restore for rollback.
3. Should old HMM/MT reference docs be deleted or marked historical? Decision:
   update active reference docs, remove related ADRs from `DECISIONS.md`, and
   leave historical docs/specs in place.
