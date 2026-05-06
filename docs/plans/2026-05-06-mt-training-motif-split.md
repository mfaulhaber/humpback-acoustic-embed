# MT Training And MT Motif Split Implementation Plan

**Goal:** Split Masked Transformer training into a multi-source MT Training workflow, rename the existing page to MT Motif, and add a Phase 0 analysis child page for MT Training jobs.
**Spec:** `docs/specs/2026-05-06-mt-training-motif-split-design.md`

---

### Task 1: Add MT Training Source Data Model

**Files:**
- Create: `alembic/versions/074_masked_transformer_job_sources.py`
- Create: `tests/unit/test_migration_074_masked_transformer_job_sources.py`
- Modify: `src/humpback/models/sequence_models.py`
- Modify: `src/humpback/schemas/sequence_models.py`

**Acceptance criteria:**
- [x] Before running migration commands against a local database, back up the configured SQLite DB from `.env`: run `DB_PATH=$(awk -F= '/^HUMPBACK_DATABASE_URL=/{print $2}' .env | sed 's#^sqlite+aiosqlite:///##; s#^sqlite:///##')`, run `BACKUP="${DB_PATH}.$(date -u +%Y%m%dT%H%M%SZ).bak"`, run `cp "$DB_PATH" "$BACKUP"`, and run `test -s "$BACKUP"`.
- [x] Migration `074` creates `masked_transformer_job_sources` with source order, parent job ID, continuous-embedding job ID, event-classification job ID, optional source alias, and timestamps.
- [x] Migration uses `op.batch_alter_table()` for any changes to existing SQLite tables.
- [x] Migration adds uniqueness guarantees for `(masked_transformer_job_id, source_order)` and `(masked_transformer_job_id, continuous_embedding_job_id, event_classification_job_id)`.
- [x] SQLAlchemy model exposes `MaskedTransformerJobSource` and a relationship from `MaskedTransformerJob`.
- [x] Pydantic schemas include source create/output models and allow MT create payloads to carry `sources`.
- [x] Existing single-source fields on `masked_transformer_jobs` remain available and continue to represent the first source pair.
- [x] No historical row backfill is required because existing MT and embedding jobs have been deleted.

**Tests needed:**
- Migration upgrade/downgrade tests assert the new table, constraints, nullable fields, and downgrade removal.
- Schema tests validate non-empty sources and duplicate source-pair rejection.

---

### Task 2: Implement Multi-Source Service And API Validation

**Files:**
- Modify: `src/humpback/services/masked_transformer_service.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `tests/services/test_masked_transformer_service.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`
- Modify: `tests/integration/test_masked_transformer_api.py`
- Modify: `tests/integration/test_sequence_models_submit.py`

**Acceptance criteria:**
- [x] `create_masked_transformer_job` accepts either the legacy single-source fields or the new `sources` list.
- [x] Multi-source create requires every continuous-embedding job to be complete, `region_crnn`, and backed by an event-segmentation job.
- [x] Every selected Classify job must be complete and belong to the same segmentation as its paired embedding job.
- [x] Duplicate source pairs are rejected before the DB insert.
- [x] Source compatibility validation requires matching vector dimension, `model_version`, `chunk_size_seconds`, `chunk_hop_seconds`, `projection_kind`, and `projection_dim`.
- [x] Open spec question is resolved conservatively: `crnn_checkpoint_sha256` must match when every selected source has a non-null value; null values do not block mixed older rows.
- [x] Source rows are persisted in the selected order, and the first pair is also mirrored into the existing `continuous_embedding_job_id` and `event_classification_job_id` columns.
- [x] Training signature includes the normalized source-pair list and excludes `k_values`.
- [x] When `sources` is present, contrastive and projection-head-only ablation settings are forced to the non-contrastive defaults or rejected with a clear validation error.
- [x] Legacy single-source creation keeps existing contrastive functionality and existing idempotency behavior.

**Tests needed:**
- Service tests for multi-source success, source ordering, duplicate rejection, compatibility failures, segmentation mismatch, signature identity, and k-sweep exclusion.
- API tests for `POST /sequence-models/masked-transformers` with a `sources` payload and legacy single-source payload.

---

### Task 3: Train And Tokenize Across Multiple Sources

**Files:**
- Modify: `src/humpback/workers/masked_transformer_worker.py`
- Modify: `src/humpback/services/masked_transformer_service.py`
- Modify: `src/humpback/storage.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`
- Modify: `tests/services/test_masked_transformer_service.py`

**Acceptance criteria:**
- [x] Worker loads all persisted source rows for a job and falls back to the mirrored single-source columns only for legacy single-source jobs.
- [x] Region sequence IDs are namespaced as `<source_index>:<region_id>` to avoid collisions.
- [x] Output rows in contextual embeddings, retrieval embeddings, retrieval head outputs, reconstruction error, and decoded token parquets preserve source metadata where practical: source index, continuous-embedding job ID, event-classification job ID, and original region ID.
- [x] The transformer trains once over the concatenated training sequences from all selected sources.
- [x] Per-k k-means tokenizers are fitted over the combined token embedding space.
- [x] Extend-k-sweep continues to reuse the trained model and embedding artifacts without retraining.
- [x] Per-k overlay, exemplar, and run-length artifacts remain generated for MT Training detail analysis.
- [x] Label distribution generation remains available for legacy/single-source jobs, but MT Training UI does not depend on it.
- [x] Worker writes `inference_manifest.json` with the model, tokenizer, source compatibility, and sequence-construction metadata required by the future MT Motif inference page.

**Tests needed:**
- Worker tests with two synthetic source parquets assert one combined training run, namespaced sequence IDs, source metadata columns, per-k bundles, and inference manifest contents.
- Extend-k-sweep tests assert multi-source jobs only add missing k bundles.

---

### Task 4: Make Phase 0 Diagnostics Multi-Source Aware

**Files:**
- Modify: `src/humpback/sequence_models/retrieval_diagnostics.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/storage.py`
- Modify: `tests/sequence_models/test_retrieval_diagnostics.py`
- Modify: `tests/integration/test_masked_transformer_retrieval_diagnostics_api.py`

**Acceptance criteria:**
- [x] Nearest-neighbor diagnostics read source metadata from MT artifacts and source rows when present.
- [x] Label coverage and neighbor metrics resolve human labels using each row's own `event_classification_job_id`.
- [x] Existing single-source diagnostics continue to work unchanged.
- [x] `include_geometry_report=true` still returns contextual and retrieval geometry diagnostics, including unavailable retrieval rows when a job has no retrieval head.
- [x] Analysis POST can persist the latest report to `masked_transformer_jobs/{job_id}/analysis/latest_report.json`.
- [x] A GET endpoint returns the latest persisted report for page reloads, returning 404 when no report has been generated.
- [x] Persisted reports are produced by the existing Phase 0 report builder; no second diagnostics implementation is introduced.

**Tests needed:**
- Unit tests for multi-source label joins and mixed source metadata.
- Integration tests for POST full analysis options, geometry diagnostics, persisted latest report, and latest-report 404.

---

### Task 5: Add Frontend API Types, Routes, Navigation, And MT Motif Rename

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/layout/SideNav.tsx`
- Modify: `frontend/src/components/layout/Breadcrumbs.tsx`
- Modify: `frontend/e2e/sequence-models/masked-transformer.spec.ts`
- Modify: `frontend/e2e/sequence-models/masked-transformer-motif-ux.spec.ts`
- Create: `frontend/e2e/sequence-models/mt-training.spec.ts`

**Acceptance criteria:**
- [x] Sequence Models nav shows `MT Training` and `MT Motif`.
- [x] Current Masked Transformer routes redirect to or remain compatible with the renamed MT Motif route.
- [x] MT Training routes exist for list, detail, and analysis child pages.
- [x] Breadcrumbs distinguish MT Training job detail, MT Training analysis, and MT Motif.
- [x] Frontend API types include source create/output payloads and latest analysis report fetch/post hooks.
- [x] Existing MT Motif behavior remains available under the renamed nav entry.

**Tests needed:**
- E2E tests for nav visibility, MT Motif route compatibility, MT Training route rendering, and breadcrumb labels.
- API hook type coverage where existing frontend tests exercise MT create payloads.

---

### Task 6: Build MT Training Create, List, And Detail Pages

**Files:**
- Create: `frontend/src/components/sequence-models/MTTrainingJobsPage.tsx`
- Create: `frontend/src/components/sequence-models/MTTrainingCreateForm.tsx`
- Create: `frontend/src/components/sequence-models/MTTrainingDetailPage.tsx`
- Create: `frontend/src/components/sequence-models/MTAnalysisReportTables.tsx`
- Modify: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/MaskedTransformerCreateForm.test.tsx`
- Create: `frontend/src/components/sequence-models/MTTrainingCreateForm.test.tsx`
- Create: `frontend/src/components/sequence-models/MTTrainingDetailPage.test.tsx`
- Modify: `frontend/e2e/sequence-models/mt-training.spec.ts`

**Acceptance criteria:**
- [x] MT Training create form supports adding and removing one or more source-pair rows.
- [x] Each source row filters Classify choices to the selected embedding job's segmentation.
- [x] Create form blocks duplicate pairs and incompatible selected sources before submit.
- [x] Create form does not render contrastive or projection-head-only ablation controls.
- [x] Submit payload sends `sources` and contrastive-off defaults.
- [x] MT Training list displays active and previous jobs with source count, total chunks, preset, k values, retrieval head mode, status/device, and actions.
- [x] MT Training detail shows metadata, source-pair table, model/artifact summary, loss curve, run-length histograms, UMAP/token overlay, exemplar gallery, and Analysis button.
- [x] MT Training detail omits token timeline, spectrogram playback, motif extraction, motif occurrence navigation, label distribution, and exemplar vocalization label badges.
- [x] Existing Masked Transformer detail code is reused or extracted where practical without changing MT Motif behavior.

**Tests needed:**
- Component tests for source row interactions, filtered Classify dropdowns, submit payload, missing contrastive controls, and detail-page omission of disallowed sections.
- E2E tests for creating a mocked MT Training job and opening its detail page.

---

### Task 7: Build MT Training Analysis Child Page

**Files:**
- Create: `frontend/src/components/sequence-models/MTTrainingAnalysisPage.tsx`
- Create: `frontend/src/components/sequence-models/MTAnalysisReportTables.tsx`
- Create: `frontend/src/components/sequence-models/MTAnalysisReportTables.test.tsx`
- Modify: `frontend/src/components/sequence-models/MTTrainingDetailPage.tsx`
- Modify: `frontend/e2e/sequence-models/mt-training.spec.ts`

**Acceptance criteria:**
- [x] Analysis button posts to the Phase 0 nearest-neighbor report endpoint with all default retrieval modes, all default embedding variants, `include_event_level=true`, `include_geometry_report=true`, `include_query_rows=true`, and selected k.
- [x] Successful analysis navigates to `/app/sequence-models/mt-training/:jobId/analysis`.
- [x] Analysis child page loads the latest persisted report when the React Query cache is empty.
- [x] Report metadata, label coverage, aggregate retrieval metrics, event-level metrics, geometry diagnostics, representative good queries, and representative risky queries render as tables.
- [x] Green/yellow/red indicators are applied only to metrics with clear directionality from the spec.
- [x] Deferred metrics without clear directionality render without invented color thresholds.
- [x] Geometry diagnostic rows include backend bands, warnings, and unavailable-artifact status.

**Tests needed:**
- Component tests for color classification rules and deferred uncolored metrics.
- E2E test that the Analysis button sends the expected POST body and the child page renders metric, coverage, and geometry tables.

---

### Task 8: Final Integration And Documentation Touches

**Files:**
- Modify: `docs/specs/2026-05-06-mt-training-motif-split-design.md`
- Modify: `docs/plans/2026-05-06-mt-training-motif-split.md`
- Modify: `DECISIONS.md`, if implementation resolves a durable architecture decision that should be recorded as an ADR

**Acceptance criteria:**
- [x] Spec open questions resolved during implementation are updated or explicitly left deferred.
- [x] Plan checkboxes are updated to reflect completed work.
- [x] No new durable architecture decision required a `DECISIONS.md` entry.
- [x] No unrelated documentation churn is introduced.

**Tests needed:**
- Documentation-only changes do not need dedicated tests beyond the full verification gates.

---

### Verification

Run in order after all tasks:

1. `uv run alembic upgrade head`
2. `uv run ruff format --check alembic/versions/074_masked_transformer_job_sources.py src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/workers/masked_transformer_worker.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/storage.py tests/unit/test_migration_074_masked_transformer_job_sources.py tests/unit/test_sequence_models_schemas.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/sequence_models/test_retrieval_diagnostics.py tests/integration/test_masked_transformer_api.py tests/integration/test_sequence_models_submit.py tests/integration/test_masked_transformer_retrieval_diagnostics_api.py`
3. `uv run ruff check alembic/versions/074_masked_transformer_job_sources.py src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/workers/masked_transformer_worker.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/storage.py tests/unit/test_migration_074_masked_transformer_job_sources.py tests/unit/test_sequence_models_schemas.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/sequence_models/test_retrieval_diagnostics.py tests/integration/test_masked_transformer_api.py tests/integration/test_sequence_models_submit.py tests/integration/test_masked_transformer_retrieval_diagnostics_api.py`
4. `uv run pyright src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/workers/masked_transformer_worker.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/storage.py`
5. `uv run pytest tests/unit/test_migration_074_masked_transformer_job_sources.py tests/unit/test_sequence_models_schemas.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/sequence_models/test_retrieval_diagnostics.py tests/integration/test_masked_transformer_api.py tests/integration/test_sequence_models_submit.py tests/integration/test_masked_transformer_retrieval_diagnostics_api.py`
6. `uv run pytest tests/`
7. `cd frontend && npx vitest run src/components/sequence-models/MTTrainingCreateForm.test.tsx src/components/sequence-models/MTTrainingDetailPage.test.tsx src/components/sequence-models/MTAnalysisReportTables.test.tsx`
8. `cd frontend && npx tsc --noEmit`
9. `cd frontend && npx playwright test e2e/sequence-models/mt-training.spec.ts e2e/sequence-models/masked-transformer.spec.ts e2e/sequence-models/masked-transformer-motif-ux.spec.ts`
