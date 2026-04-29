# Legacy Workflow Removal Implementation Plan

**Goal:** Retire file-path and embedding-set based legacy workflows while preserving detection-job based classifier, vocalization clustering, call parsing, and sequence-model workflows.
**Spec:** [docs/specs/2026-04-29-legacy-workflow-removal-design.md](../specs/2026-04-29-legacy-workflow-removal-design.md)

---

### Task 1: Retire Removed Top-Level Frontend Routes

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/layout/SideNav.tsx`
- Modify: `frontend/src/components/layout/TopNav.tsx`
- Modify: `frontend/src/components/layout/Breadcrumbs.tsx`
- Delete: `frontend/src/components/audio/AudioTab.tsx`
- Delete: `frontend/src/components/audio/AudioUpload.tsx`
- Delete: `frontend/src/components/audio/AudioList.tsx`
- Delete: `frontend/src/components/audio/AudioDetail.tsx`
- Delete: `frontend/src/components/audio/DeleteFolderDialog.tsx`
- Delete: `frontend/src/components/audio/WindowPlayer.tsx`
- Delete: `frontend/src/components/audio/SimilarityMatrix.tsx`
- Delete: `frontend/src/components/audio/SpectrogramPlot.tsx`
- Delete: `frontend/src/components/processing/ProcessingTab.tsx`
- Delete: `frontend/src/components/processing/QueueJobForm.tsx`
- Delete: `frontend/src/components/processing/ProcessingJobsList.tsx`
- Delete: `frontend/src/components/processing/EmbeddingSetsList.tsx`
- Delete: `frontend/src/components/search/SearchTab.tsx`
- Delete: `frontend/src/components/label-processing/LabelProcessingTab.tsx`
- Delete: `frontend/src/components/label-processing/LabelProcessingJobCard.tsx`
- Delete: `frontend/src/components/label-processing/LabelProcessingPreview.tsx`
- Delete: `frontend/src/components/clustering/ClusteringTab.tsx`
- Delete: `frontend/src/components/clustering/EmbeddingSetSelector.tsx`
- Delete: `frontend/src/components/clustering/ClusteringParamsForm.tsx`
- Delete: `frontend/src/components/clustering/ClusteringJobCard.tsx`
- Delete: `frontend/src/components/clustering/DeleteClusteringJobDialog.tsx`

**Acceptance criteria:**
- [x] `/` redirects to a retained workflow route, preferably `/app/call-parsing/detection`.
- [x] Wildcard route fallback redirects to the same retained default route.
- [x] `/app/audio`, `/app/processing`, `/app/clustering`, `/app/search`, and `/app/label-processing` routes are removed.
- [x] Side navigation no longer renders Audio, Processing, top-level Clustering, Search, or Label Processing.
- [x] Top navigation home link points to the retained default route.
- [x] Breadcrumbs contain no static entries for removed routes.
- [x] Deleted component directories have no remaining imports from retained frontend code.

**Tests needed:**
- Update or add Playwright navigation smoke coverage that verifies removed labels are absent and the default route loads.
- Type-check catches any stale imports from deleted components.

---

### Task 2: Make Classifier Training Detection-Job Only

**Files:**
- Modify: `frontend/src/components/classifier/TrainingTab.tsx`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useClassifier.ts`
- Modify: `src/humpback/schemas/classifier.py`
- Modify: `src/humpback/api/routers/classifier/training.py`
- Modify: `src/humpback/services/classifier_service/training.py`
- Modify: `src/humpback/workers/classifier_worker/training.py`
- Modify: `src/humpback/schemas/converters.py`

**Acceptance criteria:**
- [x] Classifier / Training has no embedding-set source mode, source radio controls, `EmbeddingSetPanel`, `ModelFilter`, or `useEmbeddingSets` dependency.
- [x] The create form always uses `DetectionSourcePicker`.
- [x] Training job creation requires `detection_job_ids` and `embedding_model_version`.
- [x] Payloads containing `positive_embedding_set_ids` or `negative_embedding_set_ids` are rejected with a clear API validation error during compatibility.
- [x] Existing legacy classifier models remain visible and are labeled or represented as legacy when their `training_source_mode` is `embedding_sets`.
- [x] Existing queued/running embedding-set training jobs fail fast with a clear retirement error rather than trying to train from deleted sources.
- [x] Candidate-backed autoresearch promotion still works and is not mistaken for embedding-set input.
- [x] Detection-manifest training still writes manifests and trains from detection-job embeddings.

**Tests needed:**
- Backend schema/API tests for rejecting embedding-set training payloads.
- Backend integration test for detection-job training payload success.
- Worker unit test that legacy embedding-set training jobs fail with the retirement error.
- Frontend type-check and Playwright smoke for detection-job-only training form.

---

### Task 3: Remove Retired API Routers and Public Endpoints

**Files:**
- Modify: `src/humpback/api/app.py`
- Modify: `src/humpback/api/routers/__init__.py`
- Modify: `src/humpback/api/routers/clustering.py`
- Modify: `src/humpback/api/routers/vocalization.py`
- Delete: `src/humpback/api/routers/audio.py`
- Delete: `src/humpback/api/routers/processing.py`
- Delete: `src/humpback/api/routers/search.py`
- Delete: `src/humpback/api/routers/label_processing.py`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/api/types.ts`

**Acceptance criteria:**
- [x] `audio`, `processing`, `search`, and `label_processing` routers are no longer registered.
- [x] Unused top-level `/clustering/*` endpoints are removed rather than kept as read-only compatibility.
- [x] Endpoints needed by Vocalization / Clustering remain available under the retained vocalization route surface.
- [x] Any genuinely retained audio endpoint is moved to a retained domain router before deleting `audio.py`; unused audio endpoints are not preserved for old links.
- [x] Frontend API client no longer exports removed endpoint functions or removed request/response types.
- [x] Removed endpoints return 404 through normal router absence rather than custom compatibility shims.

**Tests needed:**
- API route tests confirm removed route prefixes are absent or 404.
- Vocalization / Clustering API tests confirm retained clustering endpoints still work.
- Type-check confirms no frontend code references deleted client functions.

---

### Task 4: Retire Processing, Search, and Label-Processing Workers

**Files:**
- Modify: `src/humpback/workers/runner.py`
- Modify: `src/humpback/workers/queue.py`
- Delete: `src/humpback/workers/processing_worker.py`
- Delete: `src/humpback/workers/search_worker.py`
- Delete: `src/humpback/workers/label_processing_worker.py`
- Modify: `tests/unit/test_queue.py`

**Acceptance criteria:**
- [x] Worker runner no longer imports or polls `run_processing_job`, `run_search_job`, or `run_label_processing_job`.
- [x] Queue helpers for processing, search, and label-processing jobs are removed after no retained code imports them.
- [x] Stale recovery no longer references `ProcessingJob`, `SearchJob`, or `LabelProcessingJob` once their tables are retired.
- [x] Retained worker polling order for detection, detection embeddings, vocalization, call parsing, hyperparameter, continuous embedding, and HMM sequence jobs remains intact.
- [x] No removed worker module is imported by package entrypoints or tests.

**Tests needed:**
- Queue tests updated to cover retained job types only.
- Targeted worker runner import test or pyright check catches stale imports.

---

### Task 5: Make Vocalization Sources Detection-Job Only

**Files:**
- Modify: `src/humpback/schemas/vocalization.py`
- Modify: `src/humpback/api/routers/vocalization.py`
- Modify: `src/humpback/services/vocalization_service.py`
- Modify: `src/humpback/workers/vocalization_worker.py`
- Modify: `src/humpback/services/training_dataset.py`
- Modify: `frontend/src/components/vocalization/VocalizationTrainForm.tsx`
- Modify: `frontend/src/components/vocalization/SourceSelector.tsx`
- Modify: `frontend/src/components/vocalization/VocalizationLabelingTab.tsx`
- Modify: `frontend/src/components/vocalization/InferencePanel.tsx`
- Modify: `frontend/src/components/vocalization/VocabularyManager.tsx`
- Modify: `frontend/src/components/vocalization/TrainingDataView.tsx`
- Modify: `frontend/src/components/vocalization/RetrainFooter.tsx`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useVocalization.ts`

**Acceptance criteria:**
- [x] `VocalizationTrainingSourceConfig` no longer accepts `embedding_set_ids`.
- [x] Vocalization inference no longer accepts `source_type="embedding_set"`.
- [x] Vocabulary import from embedding sets is removed.
- [x] Local-folder labeling behavior that creates or fetches folder embedding sets is removed.
- [x] Vocalization training form lists completed/labeled detection jobs only.
- [x] Training dataset snapshot and extension logic collect from detection jobs only.
- [x] Training data filters no longer expose `embedding_set` as an active source type.
- [x] Existing legacy vocalization rows can be archived by the cleanup script before the final DB drop.

**Tests needed:**
- Update vocalization API tests to detection-job-only source configs.
- Update vocalization worker training tests to use detection-job embeddings and labels.
- Frontend type-check for removed embedding-set source type.

---

### Task 6: Convert Clustering to Detection-Job Only While Preserving Vocalization / Clustering

**Files:**
- Modify: `src/humpback/models/clustering.py`
- Modify: `src/humpback/schemas/clustering.py`
- Modify: `src/humpback/schemas/converters.py`
- Modify: `src/humpback/services/clustering_service.py`
- Modify: `src/humpback/workers/clustering_worker.py`
- Modify: `src/humpback/api/routers/vocalization.py`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useVocalization.ts`
- Modify: `frontend/src/components/vocalization/VocalizationClusteringPage.tsx`
- Modify: `frontend/src/components/vocalization/VocalizationClusteringDetail.tsx`
- Modify: `frontend/src/components/vocalization/VocalizationClusteringJobTable.tsx`
- Modify: `frontend/src/components/vocalization/VocalizationUmapPlot.tsx`

**Acceptance criteria:**
- [x] New clustering jobs can only be created with `detection_job_ids`.
- [x] `run_clustering_job()` no longer reads `EmbeddingSet.parquet_path` or joins `AudioFile`.
- [x] Detection-job clustering still reads `DetectionEmbeddingJob` parquet and resolves active vocalization inference labels.
- [x] Legacy embedding-set clustering jobs are either archived/deleted or fail clearly before final schema cleanup.
- [x] API and frontend terminology uses neutral `source_id` or detection-job terminology instead of presenting `embedding_set_id` to users.
- [x] `/clusters/{job_id}` artifacts remain supported for retained Vocalization / Clustering jobs.
- [x] Pure clustering algorithm modules remain intact.

**Tests needed:**
- Service and worker tests for detection-job clustering success.
- Tests for rejecting or failing legacy embedding-set clustering jobs during compatibility.
- API tests for retained Vocalization / Clustering list/detail/metrics/visualization endpoints.

---

### Task 7: Add Archive-Backed Legacy Cleanup Script

**Files:**
- Create: `scripts/cleanup_legacy_workflows.py`
- Create: `tests/scripts/test_cleanup_legacy_workflows.py`
- Modify: `src/humpback/storage.py`

**Acceptance criteria:**
- [x] Dry-run is the default and does not modify DB rows or files.
- [x] `--apply` requires `--archive-root`.
- [x] Dry-run prints and writes the computed archive layout.
- [x] Script reads settings through the repo `.env` / `Settings.from_repo_env()` path.
- [x] Script reports counts for direct legacy tables and dependency blockers.
- [x] Script discovers candidates under `/audio/raw`, `/embeddings`, `/label_processing`, and legacy-only `/clusters`.
- [x] Script refuses to archive or delete paths outside `settings.storage_root`.
- [x] Script copies every deletion candidate into the archive location before deleting originals.
- [x] Script writes an archive/deletion manifest to `storage_root/cleanup-manifests/{timestamp}-legacy-workflow-removal.json`.
- [x] Script processes all legacy artifact classes in one apply operation.
- [x] Script verifies each directory class independently for candidate count, total bytes, archive copies, and source deletion.
- [x] Script is idempotent on rerun after a successful apply.
- [x] Script skips `/classifiers/{classifier_model_id}` artifacts so legacy models remain usable.
- [x] Script exits non-zero if retained rows still reference legacy data that would be dropped.

**Tests needed:**
- Unit tests for dry-run manifest generation.
- Unit tests for archive-root requirement.
- Unit tests for outside-storage-root refusal.
- Unit tests for per-directory verification.
- Integration-style temp-directory test for archive then delete idempotency.

---

### Task 8: Add Alembic Migration for Legacy Schema Cleanup

**Files:**
- Create: `alembic/versions/060_legacy_workflow_removal.py`
- Modify: `src/humpback/models/__init__.py`
- Modify: `src/humpback/models/audio.py`
- Modify: `src/humpback/models/processing.py`
- Modify: `src/humpback/models/search.py`
- Modify: `src/humpback/models/label_processing.py`
- Modify: `src/humpback/models/clustering.py`
- Modify: `src/humpback/models/classifier.py`
- Modify: `tests/unit/test_migration_060_legacy_workflow_removal.py`

**Acceptance criteria:**
- [ ] Back up the production database before applying the migration: run `DB_URL=$(grep '^HUMPBACK_DATABASE_URL=' .env | cut -d= -f2-)`, run `DB_PATH=${DB_URL#sqlite+aiosqlite:///}`, run `BACKUP="$DB_PATH.$(date -u +%Y-%m-%d-%H:%M).bak"`, run `cp "$DB_PATH" "$BACKUP"`, and run `test -s "$BACKUP"` before running `uv run alembic upgrade head`.
- [x] Migration refuses to proceed if preflight blockers are present, including queued/running removed job types, legacy clustering rows, or unarchived legacy source references.
- [x] Migration drops `search_jobs`.
- [x] Migration drops `label_processing_jobs`.
- [x] Migration drops `processing_jobs`.
- [x] Migration drops `embedding_sets`.
- [x] Migration drops `audio_metadata`.
- [ ] Migration drops `audio_files` after retained dependencies are removed.
- [x] Migration renames `cluster_assignments.embedding_set_id` to a neutral column such as `source_id` if retained clustering still needs the assignment source.
- [x] Migration drops or replaces `clustering_jobs.embedding_set_ids` after detection-job clustering uses the retained source field.
- [x] Migration preserves enough classifier model/training provenance for legacy model display before dropping `positive_embedding_set_ids` and `negative_embedding_set_ids`.
- [x] Migration removes `embedding_sets` as the default `source_mode` / `training_source_mode`.
- [x] SQLite table rewrites use `op.batch_alter_table()` where needed.
- [x] Downgrade is either implemented safely or explicitly documented as not reconstructing dropped user data.

**Tests needed:**
- Migration test applies to a representative fresh database.
- Migration test blocks when legacy active rows are present.
- Migration test preserves legacy classifier model display provenance.
- Migration test verifies dropped tables are absent after upgrade.

---

### Task 9: Delete Retired Backend Services, Schemas, and Tests

**Files:**
- Delete: `src/humpback/services/audio_service.py`
- Delete: `src/humpback/services/processing_service.py`
- Delete: `src/humpback/services/search_service.py`
- Delete: `src/humpback/services/label_processing_service.py`
- Delete: `src/humpback/schemas/audio.py`
- Delete: `src/humpback/schemas/processing.py`
- Delete: `src/humpback/schemas/search.py`
- Delete: `src/humpback/schemas/label_processing.py`
- Modify: `src/humpback/schemas/converters.py`
- Delete: `tests/integration/test_audio_api.py`
- Delete: `tests/integration/test_audio_window.py`
- Delete: `tests/integration/test_audio_visualization.py`
- Delete: `tests/integration/test_processing_api.py`
- Delete: `tests/integration/test_search_api.py`
- Delete: `tests/integration/test_label_processing_api.py`
- Delete: `tests/integration/test_label_processing_e2e.py`
- Delete: `tests/unit/test_search_service.py`
- Delete: `tests/unit/test_search_worker.py`
- Delete: `tests/unit/test_label_processor.py`
- Delete: `tests/unit/test_models.py`

**Acceptance criteria:**
- [x] Removed services and schemas have no imports from retained code.
- [x] Retained lower-level audio processing utilities remain where detection, call parsing, timeline, and sequence models use them.
- [x] Removed tests are either deleted or replaced with retained workflow coverage.
- [x] `rg` for removed service/schema modules returns no retained-code imports.

**Tests needed:**
- Full backend test run after removals.
- `uv run pyright` catches stale imports.

---

### Task 10: Update Project Instructions and Reference Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/frontend.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/classifier-api.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `docs/reference/testing.md`
- Modify: `docs/reference/signal-processing.md`
- Modify: `docs/workflows/session-review.md`
- Modify: `docs/plans/backlog.md`
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [x] `CLAUDE.md` no longer presents Audio, Processing, top-level Clustering, Search, or Label Processing as active capabilities.
- [x] `CLAUDE.md` table list matches the post-cleanup schema.
- [x] `AGENTS.md` no longer includes the retired embedding-set idempotency constraint and instead names retained idempotency constraints.
- [x] Reference docs describe detection-job based workflows as the active path.
- [x] Storage layout removes `/audio/raw`, `/embeddings`, and `/label_processing` after archive-backed cleanup is implemented, while keeping detection embeddings and vocalization clustering artifacts.
- [x] Classifier API docs describe detection-job-only training.
- [x] Testing docs replace the legacy upload/process/cluster smoke test with a retained detection-job workflow smoke test.
- [x] Session-review docs replace the no-duplicate-embedding-sets review check with current idempotency checks.
- [x] `DECISIONS.md` gets a new ADR only if implementation confirms this cleanup is an architectural direction rather than ordinary pruning.

**Tests needed:**
- Documentation search gates for stale active-workflow references.
- Manual review of CLAUDE/AGENTS instructions for consistency with the implemented app.

---

### Task 11: Final Search Gates and End-to-End Verification

**Files:**
- Modify: `frontend/e2e/sequence-models/continuous-embedding.spec.ts`
- Modify: `frontend/e2e/sequence-models/hmm-sequence.spec.ts`
- Create: `frontend/e2e/navigation-retired-workflows.spec.ts`
- Modify: `frontend/e2e/sequence-models/*` if route defaults affect shared setup
- Modify: `tests/integration/test_classifier_api.py`
- Modify: `tests/integration/test_vocalization_api.py`
- Modify: `tests/integration/test_clustering_api.py`
- Modify: `tests/integration/test_detection_embedding_api.py`

**Acceptance criteria:**
- [ ] `rg "Audio|Processing|Search|Label Processing" frontend/src docs CLAUDE.md AGENTS.md` has only historical or explicitly retired references.
- [ ] `rg "EmbeddingSet|embedding_sets|processing_jobs|search_jobs|label_processing_jobs" src tests docs CLAUDE.md AGENTS.md` has only migration tests, archive manifests, legacy provenance labels, or historical ADR/spec references.
- [x] Retained navigation loads without removed routes.
- [x] Classifier training from detection jobs works.
- [x] Vocalization / Clustering works from detection jobs.
- [ ] Detection embedding, call parsing, sequence models, and timeline-related tests still pass.
- [x] Cleanup script dry-run and apply tests pass.
- [x] Legacy classifier models remain visible as legacy models.

**Tests needed:**
- Full backend test suite.
- Frontend type-check.
- Playwright navigation smoke.
- Focused retained workflow API/integration tests.

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback scripts tests`
2. `uv run ruff check src/humpback scripts tests`
3. `uv run pyright src/humpback scripts`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
7. `rg "Audio|Processing|Search|Label Processing" frontend/src docs CLAUDE.md AGENTS.md`
8. `rg "EmbeddingSet|embedding_sets|processing_jobs|search_jobs|label_processing_jobs" src tests docs CLAUDE.md AGENTS.md`
9. `uv run python scripts/cleanup_legacy_workflows.py --dry-run --archive-root /tmp/humpback-legacy-archive-dry-run`
