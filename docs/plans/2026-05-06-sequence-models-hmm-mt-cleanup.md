# Sequence Models HMM/MT Cleanup Implementation Plan

**Goal:** Remove active HMM Sequence, Masked Transformer, and motif-extraction functionality while preserving Continuous Embedding.
**Spec:** `docs/specs/2026-05-06-sequence-models-hmm-mt-cleanup-design.md`

---

### Task 1: Retire Database Schema, ORM Models, and Storage Helpers

**Files:**
- Create: `alembic/versions/075_remove_hmm_mt_sequence_models.py`
- Create: `tests/unit/test_migration_075_remove_hmm_mt_sequence_models.py`
- Modify: `src/humpback/models/sequence_models.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/storage.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`
- Delete: `tests/unit/test_migration_063_masked_transformer_jobs.py`
- Delete: `tests/unit/test_migration_064_motif_extraction_parent.py`
- Delete: `tests/unit/test_migration_067_masked_transformer_retrieval_head.py`
- Delete: `tests/unit/test_migration_068_masked_transformer_event_centered_sequences.py`
- Delete: `tests/unit/test_migration_069_masked_transformer_contrastive_training.py`
- Delete: `tests/unit/test_migration_070_masked_transformer_batch_size.py`
- Delete: `tests/unit/test_migration_072_masked_transformer_projection_head_ablation.py`
- Delete: `tests/unit/test_migration_073_masked_transformer_retrieval_head_arch.py`
- Delete: `tests/unit/test_migration_074_masked_transformer_job_sources.py`

**Acceptance criteria:**
- [ ] Before creating or running the Alembic migration, complete the CLAUDE.md §3.5 database backup step from `.env`: run `DATABASE_URL=$(grep -E '^DATABASE_URL=' .env | cut -d= -f2-)`, run `DB_PATH=${DATABASE_URL#sqlite+aiosqlite:///}` and if unchanged run `DB_PATH=${DATABASE_URL#sqlite:///}`, run `BACKUP_PATH="$DB_PATH.$(date -u +%Y%m%dT%H%M%SZ).bak"`, run `cp "$DB_PATH" "$BACKUP_PATH"`, and run `test -s "$BACKUP_PATH"`.
- [ ] Alembic upgrade drops `motif_extraction_jobs`, `masked_transformer_job_sources`, `masked_transformer_jobs`, and `hmm_sequence_jobs`.
- [ ] Alembic upgrade preserves `continuous_embedding_jobs`, its existing rows, and its uniqueness/idempotency constraints.
- [ ] Alembic downgrade for this destructive retirement is intentionally unsupported or raises a clear backup-restore instruction.
- [ ] `src/humpback/models/sequence_models.py` exports only retained Continuous Embedding ORM state.
- [ ] `src/humpback/schemas/sequence_models.py` exports only Continuous Embedding schemas and detail helpers.
- [ ] `src/humpback/storage.py` retains only Continuous Embedding storage helpers from the retired Sequence Models family.

**Tests needed:**
- Migration test proving retired tables are removed and `continuous_embedding_jobs` survives with rows intact.
- Schema tests updated to cover Continuous Embedding only.
- Existing Continuous Embedding service/worker tests should continue to pass without retired models registered.

---

### Task 2: Remove HMM/MT/Motif Backend Runtime Surfaces

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `src/humpback/workers/queue.py`
- Modify: `src/humpback/workers/runner.py`
- Delete: `src/humpback/services/hmm_sequence_service.py`
- Delete: `src/humpback/services/masked_transformer_service.py`
- Delete: `src/humpback/services/motif_extraction_service.py`
- Delete: `src/humpback/workers/hmm_sequence_worker.py`
- Delete: `src/humpback/workers/masked_transformer_worker.py`
- Delete: `src/humpback/workers/motif_extraction_worker.py`
- Modify: `tests/integration/test_sequence_models_api.py`
- Modify: `tests/integration/test_sequence_models_submit.py`
- Delete: `tests/integration/test_masked_transformer_api.py`
- Delete: `tests/integration/test_masked_transformer_retrieval_diagnostics_api.py`
- Delete: `tests/integration/test_motif_extraction_api.py`
- Delete: `tests/integration/test_sequence_models_regenerate.py`
- Delete: `tests/services/test_hmm_sequence_service.py`
- Delete: `tests/services/test_masked_transformer_service.py`
- Delete: `tests/services/test_motif_extraction_service.py`
- Delete: `tests/workers/test_hmm_sequence_worker.py`
- Delete: `tests/workers/test_masked_transformer_worker.py`
- Delete: `tests/workers/test_motif_extraction_worker.py`

**Acceptance criteria:**
- [ ] The Sequence Models router registers only Continuous Embedding endpoints.
- [ ] Requests to retired HMM, MT, and motif endpoint paths receive the normal unregistered-route 404.
- [ ] `recover_stale_jobs()` does not query dropped HMM/MT/motif tables.
- [ ] The worker runner polls Continuous Embedding and then idles without importing retired workers.
- [ ] API and worker startup succeed against a database migrated through revision `075`.

**Tests needed:**
- Update Sequence Models integration tests to exercise Continuous Embedding create/list/detail/cancel/delete only.
- Remove retired service and worker tests.
- Add or update a startup-oriented test if existing coverage does not catch stale recovery against dropped tables.

---

### Task 3: Remove Retired Sequence Modeling Modules and Dependencies

**Files:**
- Modify: `src/humpback/sequence_models/__init__.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Delete: `src/humpback/sequence_models/hmm_decoder.py`
- Delete: `src/humpback/sequence_models/hmm_trainer.py`
- Delete: `src/humpback/sequence_models/pca_pipeline.py`
- Delete: `src/humpback/sequence_models/region_sampling.py`
- Delete: `src/humpback/sequence_models/summary.py`
- Delete: `src/humpback/sequence_models/overlay.py`
- Delete: `src/humpback/sequence_models/exemplars.py`
- Delete: `src/humpback/sequence_models/label_distribution.py`
- Delete: `src/humpback/sequence_models/motifs.py`
- Delete: `src/humpback/sequence_models/tokenization.py`
- Delete: `src/humpback/sequence_models/masked_transformer.py`
- Delete: `src/humpback/sequence_models/masked_transformer_sequences.py`
- Delete: `src/humpback/sequence_models/contrastive_labels.py`
- Delete: `src/humpback/sequence_models/contrastive_loss.py`
- Delete: `src/humpback/sequence_models/retrieval_diagnostics.py`
- Delete: `src/humpback/sequence_models/retrieval_sweeps.py`
- Delete: `src/humpback/sequence_models/loaders/__init__.py`
- Delete: `src/humpback/sequence_models/loaders/crnn_region.py`
- Delete: `src/humpback/sequence_models/loaders/surfperch.py`
- Delete: `tests/sequence_models/test_contrastive_labels.py`
- Delete: `tests/sequence_models/test_contrastive_loss.py`
- Delete: `tests/sequence_models/test_exemplars.py`
- Delete: `tests/sequence_models/test_hmm_decoder.py`
- Delete: `tests/sequence_models/test_hmm_trainer.py`
- Delete: `tests/sequence_models/test_label_distribution.py`
- Delete: `tests/sequence_models/test_load_effective_event_labels.py`
- Delete: `tests/sequence_models/test_loaders.py`
- Delete: `tests/sequence_models/test_masked_transformer.py`
- Delete: `tests/sequence_models/test_masked_transformer_sequences.py`
- Delete: `tests/sequence_models/test_motifs.py`
- Delete: `tests/sequence_models/test_overlay.py`
- Delete: `tests/sequence_models/test_pca_pipeline.py`
- Delete: `tests/sequence_models/test_region_sampling.py`
- Delete: `tests/sequence_models/test_retrieval_diagnostics.py`
- Delete: `tests/sequence_models/test_retrieval_sweep_outputs.py`
- Delete: `tests/sequence_models/test_retrieval_sweeps.py`
- Delete: `tests/sequence_models/test_summary.py`
- Delete: `tests/sequence_models/test_tokenization.py`
- Delete: `tests/scripts/test_masked_transformer_retrieval_sweep.py`
- Delete: `tests/unit/test_masked_transformer_contrastive_sampler_migration.py`

**Acceptance criteria:**
- [ ] Retained `src/humpback/sequence_models/` modules are limited to Continuous Embedding support utilities.
- [ ] `chunk_projection.py`, `crnn_features.py`, and `event_overlap_join.py` remain and keep their tests.
- [ ] `hmmlearn` is removed from `pyproject.toml` and `uv.lock`.
- [ ] UMAP, HDBSCAN, Torch, Plotly, scikit-learn, joblib, numpy, and pyarrow remain where active non-retired features still need them.
- [ ] No non-doc import references deleted HMM/MT/motif domain modules.

**Tests needed:**
- Keep and run `tests/sequence_models/test_chunk_projection.py`, `tests/sequence_models/test_crnn_features.py`, and `tests/sequence_models/test_event_overlap_join.py`.
- Run dependency lock verification through the normal `uv` workflow.
- Run search-based import hygiene checks for deleted module names.

---

### Task 4: Clean Up Frontend Routes, Navigation, and API Hooks

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/layout/SideNav.tsx`
- Modify: `frontend/src/components/layout/Breadcrumbs.tsx`
- Delete: `frontend/src/api/sequenceModels.useMotifsByLength.test.ts`
- Delete: `frontend/e2e/sequence-models/classify-binding.spec.ts`
- Delete: `frontend/e2e/sequence-models/hmm-sequence.spec.ts`
- Delete: `frontend/e2e/sequence-models/masked-transformer-motif-ux.spec.ts`
- Delete: `frontend/e2e/sequence-models/masked-transformer.spec.ts`
- Delete: `frontend/e2e/sequence-models/mt-training.spec.ts`

**Acceptance criteria:**
- [ ] `/app/sequence-models` redirects to `/app/sequence-models/continuous-embedding`.
- [ ] Continuous Embedding list and detail routes remain registered.
- [ ] HMM Sequence, MT Training, MT Motif, and Masked Transformer routes are removed.
- [ ] Side navigation exposes Sequence Models with Continuous Embedding only.
- [ ] Breadcrumbs handle Continuous Embedding only for the Sequence Models track.
- [ ] `frontend/src/api/sequenceModels.ts` exports only Continuous Embedding types and hooks.

**Tests needed:**
- Keep and run `frontend/e2e/sequence-models/continuous-embedding.spec.ts`.
- Update frontend unit tests that import `sequenceModels.ts` so no retired API hooks are referenced.
- Build the frontend to catch removed route imports.

---

### Task 5: Delete Retired Frontend Components While Preserving Timeline Primitives

**Files:**
- Create: `frontend/src/components/timeline/overlays/motifTypes.ts`
- Modify: `frontend/src/components/timeline/overlays/MotifHighlightOverlay.tsx`
- Modify: `frontend/src/components/timeline/index.ts`
- Modify: `frontend/src/components/sequence-models/DiscreteSequenceBar.tsx`
- Modify: `frontend/src/components/sequence-models/DiscreteSequenceBar.test.ts`
- Modify: `frontend/src/components/sequence-models/MotifTimelineLegend.tsx`
- Modify: `frontend/src/components/sequence-models/MotifTimelineLegend.test.tsx`
- Modify: `frontend/src/components/sequence-models/RegionNavBar.tsx`
- Modify: `frontend/src/components/sequence-models/RegionNavBar.test.tsx`
- Modify: `frontend/src/components/sequence-models/constants.ts`
- Modify: `frontend/src/lib/motifColor.ts`
- Delete: `frontend/src/components/sequence-models/HMMSequenceCreateForm.tsx`
- Delete: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`
- Delete: `frontend/src/components/sequence-models/HMMSequenceJobTable.tsx`
- Delete: `frontend/src/components/sequence-models/HMMSequenceJobsPage.tsx`
- Delete: `frontend/src/components/sequence-models/HMMStateBar.tsx`
- Delete: `frontend/src/components/sequence-models/HMMStateBar.test.ts`
- Delete: `frontend/src/components/sequence-models/KPicker.tsx`
- Delete: `frontend/src/components/sequence-models/KPicker.test.tsx`
- Delete: `frontend/src/components/sequence-models/LossCurveChart.tsx`
- Delete: `frontend/src/components/sequence-models/LossCurveChart.test.tsx`
- Delete: `frontend/src/components/sequence-models/MTAnalysisReportTables.tsx`
- Delete: `frontend/src/components/sequence-models/MTAnalysisReportTables.test.tsx`
- Delete: `frontend/src/components/sequence-models/MTAnalysisSummaryPanel.tsx`
- Delete: `frontend/src/components/sequence-models/MTAnalysisSummaryPanel.test.tsx`
- Delete: `frontend/src/components/sequence-models/MTTrainingAnalysisPage.tsx`
- Delete: `frontend/src/components/sequence-models/MTTrainingCreateForm.tsx`
- Delete: `frontend/src/components/sequence-models/MTTrainingCreateForm.test.tsx`
- Delete: `frontend/src/components/sequence-models/MTTrainingDetailPage.tsx`
- Delete: `frontend/src/components/sequence-models/MTTrainingDetailPage.test.tsx`
- Delete: `frontend/src/components/sequence-models/MTTrainingJobsPage.tsx`
- Delete: `frontend/src/components/sequence-models/MaskedTransformerCreateForm.tsx`
- Delete: `frontend/src/components/sequence-models/MaskedTransformerCreateForm.test.tsx`
- Delete: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`
- Delete: `frontend/src/components/sequence-models/MaskedTransformerJobsPage.tsx`
- Delete: `frontend/src/components/sequence-models/MotifExampleAlignment.tsx`
- Delete: `frontend/src/components/sequence-models/MotifExampleAlignment.test.tsx`
- Delete: `frontend/src/components/sequence-models/MotifExtractionPanel.tsx`
- Delete: `frontend/src/components/sequence-models/MotifTokenCountSelector.tsx`
- Delete: `frontend/src/components/sequence-models/MotifTokenCountSelector.test.tsx`
- Delete: `frontend/src/components/sequence-models/TokenRunLengthHistograms.tsx`
- Delete: `frontend/src/components/sequence-models/TokenRunLengthHistograms.test.tsx`

**Acceptance criteria:**
- [ ] Retained motif/state visualization primitives no longer import retired API types.
- [ ] `MotifHighlightOverlay` uses a neutral visualization-only motif occurrence type.
- [ ] Timeline provider, spectrogram, tile canvas, playback, and core overlay positioning code are not broadly changed.
- [ ] Deleted components are not imported by any active route, test, or barrel file.
- [ ] Plotly dependencies remain because vocalization UMAP still uses them.

**Tests needed:**
- Run retained tests for `DiscreteSequenceBar`, `RegionNavBar`, `MotifTimelineLegend`, and motif color utilities.
- Run frontend typecheck/build to catch deleted imports.
- Run Continuous Embedding e2e as the Sequence Models route smoke test.

---

### Task 6: Add Dry-Run-First Artifact Cleanup

**Files:**
- Create: `scripts/cleanup_sequence_model_artifacts.py`
- Create: `tests/scripts/test_cleanup_sequence_model_artifacts.py`

**Acceptance criteria:**
- [ ] Script defaults to dry-run mode.
- [ ] Script reads `settings.storage_root` by default and accepts `--storage-root` for tests or emergency use.
- [ ] Script targets only `hmm_sequences`, `masked_transformer_jobs`, and `motif_extractions` under the resolved storage root.
- [ ] Script never targets `continuous_embeddings`.
- [ ] Script refuses symlinks and resolved paths outside the storage root.
- [ ] Script reports candidate directory count, file count, and bytes.
- [ ] Script writes a JSON manifest under `cleanup-manifests/{timestamp}-sequence-models-hmm-mt.json`.
- [ ] Script requires `--apply` before deleting files.

**Tests needed:**
- Unit tests with a temporary storage root covering dry-run, apply, continuous-embedding preservation, missing target dirs, symlink refusal, and out-of-root refusal.

---

### Task 7: Prune Shared Fixtures and Retired Test Helpers

**Files:**
- Modify: `tests/fixtures/sequence_models/__init__.py`
- Modify: `tests/fixtures/sequence_models/surfperch_stub.py`
- Modify: `tests/fixtures/sequence_models/synthetic_sequences.py`
- Delete: `tests/fixtures/sequence_models/classify_binding.py`
- Modify: `tests/services/test_continuous_embedding_service.py`
- Modify: `tests/workers/test_continuous_embedding_worker.py`
- Modify: `tests/sequence_models/test_chunk_projection.py`
- Modify: `tests/sequence_models/test_crnn_features.py`
- Modify: `tests/sequence_models/test_event_overlap_join.py`

**Acceptance criteria:**
- [ ] Shared fixtures support retained Continuous Embedding and CRNN tests only.
- [ ] Retired HMM/MT/motif fixtures and helper factories are removed.
- [ ] Continuous Embedding tests still cover SurfPerch and CRNN source modes.
- [ ] Test collection succeeds without skipped imports of deleted modules.

**Tests needed:**
- Run `uv run pytest tests/services/test_continuous_embedding_service.py tests/workers/test_continuous_embedding_worker.py tests/sequence_models/test_chunk_projection.py tests/sequence_models/test_crnn_features.py tests/sequence_models/test_event_overlap_join.py`.
- Run full `uv run pytest tests/` before session review.

---

### Task 8: Update Active Documentation and ADR Index

**Files:**
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/frontend.md`
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] Sequence Models API reference documents Continuous Embedding only.
- [ ] Frontend reference documents Sequence Models navigation as Continuous Embedding only.
- [ ] Frontend reference documents retained shared motif/state visualization primitives without presenting retired HMM/MT product pages as active.
- [ ] `DECISIONS.md` removes ADR entries for retired HMM Sequence, Masked Transformer, and motif-extraction product surfaces.
- [ ] `DECISIONS.md` preserves or rewrites active Continuous Embedding decisions that remain relevant.
- [ ] Historical specs and plans are left in place.

**Tests needed:**
- Documentation review plus search checks for active-doc references to retired HMM/MT/motif endpoints and routes.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback tests scripts alembic`
2. `uv run ruff check src/humpback tests scripts alembic`
3. `uv run pyright src/humpback scripts tests`
4. `uv run pytest tests/`
5. `uv run alembic upgrade head`
6. `uv run python scripts/cleanup_sequence_model_artifacts.py --dry-run`
7. `cd frontend && npm run build`
8. `cd frontend && npx vitest run`
9. `cd frontend && npx playwright test frontend/e2e/sequence-models/continuous-embedding.spec.ts`
10. `bash -lc '! rg -n "HMMSequenceJob|MaskedTransformerJob|MaskedTransformerJobSource|MotifExtractionJob|hmm_sequence_service|masked_transformer_service|motif_extraction_service|hmm_sequence_worker|masked_transformer_worker|motif_extraction_worker" src tests frontend/src frontend/e2e -g "!**/node_modules/**"'`
11. `bash -lc '! rg -n "hmmlearn" pyproject.toml uv.lock'`
