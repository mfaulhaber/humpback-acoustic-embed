# Legacy Workflow Removal Design

**Date:** 2026-04-29
**Status:** Design
**Scope:** Remove unused file-path and embedding-set workflows while preserving detection-job based classifier, vocalization, call parsing, and sequence-model workflows.

## Goal

Remove the legacy application flows that create, manage, search, label-process, cluster, or train from file-path based `EmbeddingSet` data:

1. Top-level **Audio** page, including upload/import and the entry point for creating embedding jobs from file paths.
2. Top-level **Processing** page, including `ProcessingJob` management and `EmbeddingSet` artifact management created by Audio/Processing.
3. Top-level **Clustering** page, while keeping **Vocalization / Clustering**.
4. `EmbeddingSet` input mode on **Classifier / Training**. Binary classifier training should use detection jobs only, except for existing candidate-backed promotion workflows.
5. Top-level **Search** page.
6. Top-level **Label Processing** page.
7. Project instructions, memory/status references, and reference docs that still describe these retired workflows.

The implementation should be phaseable across sessions so the app can remain shippable after each phase.

## Non-Goals

- Do not remove detection jobs, detection embedding jobs, hydrophone detection, call parsing, sequence models, or vocalization clustering.
- Do not remove the clustering algorithms. They remain used by Vocalization / Clustering through detection-job embeddings.
- Do not delete existing classifier models merely because they were trained from embedding sets. Existing models can remain usable for inference while new training no longer accepts embedding-set sources.
- Do not remove audio decoding, slicing, spectrogram, PCEN, timeline playback, or `AudioLoader` utilities that are used by detection, call parsing, sequence models, and tests.
- Do not remove `DetectionEmbeddingJob` artifacts under `/detections/{job_id}/embeddings/...`; they are part of the retained detection-job workflow.

## Current-State Analysis

### Frontend

Current top-level routing lives in `frontend/src/App.tsx`.

Routes to remove:

- `/app/audio` and `/app/audio/:audioId`
- `/app/processing`
- `/app/clustering` and `/app/clustering/:jobId`
- `/app/search`
- `/app/label-processing`

Navigation references live in:

- `frontend/src/components/layout/SideNav.tsx`
- `frontend/src/components/layout/TopNav.tsx`
- `frontend/src/components/layout/Breadcrumbs.tsx`
- default redirects in `frontend/src/App.tsx`

Components that become removable once their routes and dependents are gone:

- `frontend/src/components/audio/*`
- `frontend/src/components/processing/*`
- `frontend/src/components/search/SearchTab.tsx`
- `frontend/src/components/label-processing/*`
- legacy top-level clustering components that only create or browse embedding-set clustering jobs:
  - `ClusteringTab.tsx`
  - `EmbeddingSetSelector.tsx`
  - `ClusteringParamsForm.tsx`
  - `ClusteringJobCard.tsx`
  - delete dialog and route-only wrappers, after checking reused pieces

Clustering components to keep or move:

- `ClusterTable.tsx`, `UmapPlot.tsx`, `EvaluationPanel.tsx`, and `ExportReport.tsx` may still be useful as generic visualization pieces. Vocalization currently uses `VocalizationUmapPlot`, but the backend still exposes cluster/metrics endpoints. Remove only after confirming no import from `frontend/src/components/vocalization/*`.

Classifier / Training currently defaults to embedding sets:

- `TrainingTab.tsx` initializes `sourceMode` as `"embedding_sets"`.
- It imports `useEmbeddingSets`, `useAudioFiles`, `ModelFilter`, and `EmbeddingSetPanel`.
- It renders source radio controls for `Embedding sets` and `Detection jobs`.
- It submits either `positive_embedding_set_ids`/`negative_embedding_set_ids` or `detection_job_ids`/`embedding_model_version`.

This page should become detection-job only:

- Remove source mode state and radio controls.
- Remove embedding-set queries, tree-building, mismatch warning, and `EmbeddingSetPanel` dependency.
- Make `DetectionSourcePicker` the only source picker.
- Keep advanced classifier parameters and job/model tables.
- Update row labels from "Positive Sets" to a neutral "Source" or "Detection Jobs".

Remaining frontend embedding-set dependencies outside the requested list:

- `VocalizationTrainForm.tsx` can train from curated embedding-set datasets.
- `SourceSelector.tsx`, `VocalizationLabelingTab.tsx`, and `InferencePanel.tsx` support `embedding_set` and local-folder sources.
- `VocabularyManager.tsx` imports vocabulary from embedding sets.
- `TrainingDataView.tsx` can filter by `source_type="embedding_set"`.

These dependencies mean there are two safe paths:

- If the project wants to remove only the listed surfaces, keep the `embedding_sets` table and some low-level services temporarily for remaining Vocalization legacy sources.
- If the project wants a full DB/artifact purge of `embedding_sets`, the remaining Vocalization embedding-set and local-folder flows must also be removed or migrated to detection-job/training-dataset sources in a later phase.

### API

Routers to retire from public API registration:

- `src/humpback/api/routers/processing.py`
- `src/humpback/api/routers/search.py`
- `src/humpback/api/routers/label_processing.py`
- legacy top-level embedding-set routes in `src/humpback/api/routers/clustering.py`
- Audio upload/import/list/detail routes in `src/humpback/api/routers/audio.py`, if no retained frontend or worker needs them

`src/humpback/api/app.py` currently imports and includes `audio`, `processing`, `clustering`, `search`, and `label_processing`. These include registrations should be removed in the same phase as route retirement.

The audio router has mixed responsibilities. The legacy Audio page uses upload/import/list/detail/metadata/download/window/spectrogram/embedding-similarity endpoints. Some audio utility endpoints may still be test-only or useful for local detection debugging. The design recommendation is to retire the router as a page API, then remove it after an import audit proves no retained UI calls it. If any retained endpoint is needed, move it to a clearer retained router, such as classifier/timeline utilities, before deleting `audio.py`.

Classifier training API:

- `POST /classifier/training-jobs` currently accepts either embedding-set IDs or detection-job IDs.
- `ClassifierTrainingJobCreate` validates both source shapes.
- `create_training_job()` in `classifier_service/training.py` validates `EmbeddingSet` rows.
- `run_training_job()` has an embedding-set branch for jobs whose `source_mode` is neither `autoresearch_candidate` nor `detection_manifest`.

Target behavior:

- New `POST /classifier/training-jobs` requests require `detection_job_ids` and `embedding_model_version`.
- Requests containing `positive_embedding_set_ids` or `negative_embedding_set_ids` return 422 during the compatibility phase, then those fields are removed from the schema in the destructive cleanup phase.
- Existing rows with `source_mode="embedding_sets"` stay readable as legacy history until final schema cleanup. Existing classifier models produced by those jobs stay visible and usable as legacy models; the cleanup must not delete their `/classifiers/{classifier_model_id}` artifacts.
- Worker behavior for existing queued embedding-set training jobs should be handled explicitly:
  - Phase 1 can fail them fast with a clear "embedding-set training retired" error.
  - Phase 3 can enforce a pre-migration check that no queued/running embedding-set training jobs remain before dropping columns.

Vocalization API dependencies:

- `POST /vocalization/types/import` imports type names from embedding sets.
- `VocalizationTrainingSourceConfig` includes `embedding_set_ids`.
- `VocalizationInferenceJobCreate.source_type` allows `embedding_set`.
- Several endpoints resolve embedding-set source metadata.

These must be addressed before dropping `embedding_sets`. The chosen design retires these remaining embedding-set/local-folder entry points as part of the cleanup track, after archiving any local files/artifacts to the script-specified archive location.

### Workers and Queue

Worker runner paths to remove or modify:

- Remove processing job polling and `run_processing_job`.
- Remove search job polling and `run_search_job`.
- Remove label-processing job polling and `run_label_processing_job`.
- Keep clustering worker, but remove its embedding-set branch once top-level clustering and embedding-set clustering jobs are retired.
- Keep detection embedding, vocalization, call parsing, continuous embedding, HMM, hyperparameter, and detection workers.

Queue changes:

- Remove `claim_processing_job`, `complete_processing_job`, `fail_processing_job` once processing is fully retired.
- Remove `claim_search_job`, `complete_search_job`, `fail_search_job` once Search is fully retired.
- Remove `claim_label_processing_job` once Label Processing is fully retired.
- Remove stale recovery for `ProcessingJob`, `SearchJob`, and `LabelProcessingJob`.
- Keep stale recovery for `ClusteringJob`, but optionally filter to detection-job clustering only after legacy rows are purged.

Clustering worker target:

- Current worker supports two inputs:
  - `embedding_set_ids`: reads `EmbeddingSet.parquet_path`, joins `AudioFile`, infers categories from folder paths.
  - `detection_job_ids`: reads `DetectionEmbeddingJob` parquet and resolves active vocalization inference labels.
- The retained worker should require `detection_job_ids` and reject or fail legacy `embedding_set_ids` jobs until DB cleanup deletes or archives them.
- Output files can remain under `/clusters/{clustering_job_id}` because Vocalization / Clustering already uses that layout.

### Database

Tables directly associated with removed workflows:

- `audio_files`
- `audio_metadata`
- `processing_jobs`
- `embedding_sets`
- `search_jobs`
- `label_processing_jobs`

Tables partially associated and retained:

- `clustering_jobs`, `clusters`, `cluster_assignments`
- `classifier_training_jobs`
- `classifier_models`
- `vocalization_training_jobs`
- `vocalization_inference_jobs`
- `training_datasets`
- `model_configs`

Important dependency details:

- `embedding_sets.audio_file_id` references `audio_files.id`.
- `processing_jobs.audio_file_id` references `audio_files.id`.
- `clustering_jobs.embedding_set_ids` stores JSON text, not enforced by FK.
- `cluster_assignments.embedding_set_id` is also reused by detection-job clustering to store detection job IDs, so the column name is legacy but not removable without a rename.
- `classifier_training_jobs.positive_embedding_set_ids` and `negative_embedding_set_ids` are JSON text. Detection-manifest and candidate-backed jobs store empty arrays there today.
- `classifier_training_jobs.source_mode` and `classifier_models.training_source_mode` can contain `"embedding_sets"`.
- `training_datasets.source_config` and parquet `source_type` can contain `embedding_set`.
- `vocalization_training_jobs.source_config` can contain `embedding_set_ids`.
- `vocalization_inference_jobs.source_type` can be `"embedding_set"`.
- `model_registry_service.delete_model()` prevents deleting `ModelConfig` rows when `EmbeddingSet` rows reference the model.

Recommended DB cleanup sequence:

1. Add preflight reporting before destructive migrations:
   - Counts for all direct legacy tables.
   - Count of `clustering_jobs` where `detection_job_ids IS NULL` or empty.
   - Count of `classifier_training_jobs` with `source_mode="embedding_sets"` and active statuses.
   - Count of `classifier_models` with `training_source_mode="embedding_sets"`.
   - Count of vocalization/training-dataset source JSON containing `"embedding_set"` or `embedding_set_ids`.
   - List of artifact paths referenced by legacy rows.
2. Add application-level retirement guards:
   - Prevent creating new processing/search/label-processing jobs.
   - Prevent creating new embedding-set clustering jobs.
   - Prevent creating new classifier training jobs from embedding sets.
3. Archive direct legacy rows and local artifacts before purge:
   - Cleanup tooling requires an explicit archive location supplied by the operator.
   - Local files and artifacts are copied into that archive location before deletion.
   - The archive manifest records original path, archive path, size, checksum when practical, and source DB row IDs.
   - Delete `search_jobs` because they are ephemeral.
   - Delete `label_processing_jobs` after backing up output summaries.
   - Delete `processing_jobs`.
   - Delete `embedding_sets`.
   - Delete `audio_metadata` and `audio_files`.
   - Delete legacy-only clustering jobs and their clusters/assignments.
4. Schema migration:
   - Drop `search_jobs`, `label_processing_jobs`, `processing_jobs`, `embedding_sets`, `audio_metadata`, and `audio_files` after archive verification succeeds.
   - Rename `cluster_assignments.embedding_set_id` to a neutral name such as `source_id`, and update APIs/types to avoid implying embedding-set semantics.
   - Keep `clustering_jobs.embedding_set_ids` only temporarily. In final cleanup, either drop it or replace with a non-null `source_detection_job_ids`/`source_kind` design for detection-job clustering.
   - Keep enough classifier model/training provenance to display legacy models. If `positive_embedding_set_ids` and `negative_embedding_set_ids` are dropped, first preserve a compact legacy provenance snapshot in `training_summary`, `source_comparison_context`, or a dedicated archive manifest.

SQLite migration constraints:

- Use Alembic with `op.batch_alter_table()` for table rewrites/renames.
- Before any production migration, follow the database backup requirement: read `HUMPBACK_DATABASE_URL` from `.env`, copy the SQLite file to a UTC timestamped `.bak`, and verify non-zero size.
- Destructive cleanup should be split into at least two migrations: one compatibility migration for renames/additive columns, and one final drop migration after runtime code no longer references dropped objects.

### Disk Data and Artifacts

Legacy artifact directories to clean:

- `/audio/raw/{audio_file_id}/original.*`
- `/embeddings/{model_version}/{audio_file_id}/{encoding_signature}.parquet`
- `/label_processing/{job_id}/...`
- `/clusters/{clustering_job_id}/...` for legacy top-level clustering jobs only

Artifacts to keep:

- `/detections/{detection_job_id}/...`
- `/detections/{detection_job_id}/embeddings/{model_version}/detection_embeddings.parquet`
- `/clusters/{clustering_job_id}/...` for Vocalization / Clustering jobs with `detection_job_ids`
- `/classifiers/{classifier_model_id}/...`
- `/vocalization_models/...`
- `/vocalization_inference/...`
- `/training_datasets/...`
- `/call_parsing/...`
- `/continuous_embeddings/...`
- `/hmm_sequences/...`
- `/timeline_cache/...`
- `/hyperparameter/...`

Artifact cleanup should be a script, not an Alembic migration:

- The script should run in dry-run mode by default and require `--apply`.
- The script should require an explicit archive location, for example `--archive-root /path/to/humpback-legacy-archive`, before `--apply` can run.
- It should read the DB and produce a manifest of paths to delete.
- It should refuse to delete paths outside `settings.storage_root`.
- It should copy every deletion candidate to the archive location before deleting the original.
- It should write both an archive manifest and a deletion manifest to `storage_root/cleanup-manifests/{timestamp}-legacy-workflow-removal.json`.
- It should process all selected artifact classes in one apply operation: `/audio/raw`, `/embeddings`, `/label_processing`, and legacy-only `/clusters`.
- It should verify each directory class independently during dry-run and apply:
  - candidate count
  - total bytes
  - paths are under the expected storage-root subdirectory
  - archive copies exist before source deletion
  - source paths are absent after deletion
- It should delete empty parent directories under `/audio/raw`, `/embeddings`, and `/label_processing`.
- It should skip classifier model artifacts by default, even if their training source was embedding sets.
- It should be idempotent and safe to rerun.

### Tests

Tests to remove or rewrite:

- Processing API and worker tests.
- Audio page/API tests that only cover upload/import/list/detail and embedding-set visualization.
- Search API/service/worker tests.
- Label Processing API/E2E/worker tests.
- Top-level clustering API/frontend tests that create `embedding_set_ids`.
- Classifier API/service/frontend tests asserting embedding-set training input.
- Queue tests for `ProcessingJob`, `SearchJob`, and `LabelProcessingJob`.

Tests to keep or update:

- Clustering algorithm unit tests under `humpback.clustering.*`.
- Vocalization / Clustering service/API tests using `detection_job_ids`.
- Detection embedding tests.
- Classifier detection-manifest training tests.
- Hyperparameter manifest/search tests, but remove their embedding-set training-job path if detection manifests are now the only supported source.
- Vocalization training/labeling tests should be updated if and when embedding-set sources are retired from those pages too.

New tests needed:

- Frontend route tests verify removed top-level routes redirect to a retained default.
- Side nav and breadcrumbs no longer render removed labels.
- `POST /classifier/training-jobs` rejects embedding-set payloads and accepts detection-job payloads.
- Worker runner no longer claims processing/search/label-processing jobs.
- Clustering worker fails or rejects legacy embedding-set clustering jobs in compatibility phase and processes detection-job clustering jobs.
- Cleanup script dry-run reports direct legacy tables, artifact paths, archive destinations, and per-directory verification without deleting.
- Cleanup script `--apply --archive-root ...` archives then deletes all legacy artifact classes in one shot, verifies each directory class, and is idempotent.
- Alembic migration tests apply and downgrade or document non-downgrade behavior for destructive drops.

### Documentation, Memory Files, and Instructions

Files that need attention:

- `CLAUDE.md`
  - Rename the project summary away from "Embedding & Clustering Platform" if that wording becomes misleading.
  - Remove top-level Audio, Processing, Search, Label Processing from the Web UI summary.
  - Update capability bullets so they emphasize detection-job based workflows.
  - Remove or revise `ProcessingJob` idempotency constraints once `ProcessingJob` is gone.
  - Update the table list after DB migrations.
- `AGENTS.md`
  - Remove the key constraint "Idempotency: never create duplicate embedding sets..." once `EmbeddingSet` creation is gone.
  - Replace it with retained idempotency constraints, such as no duplicate continuous embedding jobs for the same `encoding_signature`.
- `DECISIONS.md`
  - Do not rewrite historical ADRs.
  - Add a new ADR only if the removal changes architectural direction beyond cleanup. A good ADR topic would be "Detection-job sources supersede file-path embedding-set workflows."
- `docs/reference/data-model.md`
  - Remove retired tables and update retained table semantics.
- `docs/reference/frontend.md`
  - Remove deleted components/routes and update default route.
- `docs/reference/storage-layout.md`
  - Remove `/audio/raw`, `/embeddings`, and `/label_processing` only after archive-backed disk cleanup and DB cleanup are implemented.
  - Clarify `/clusters` is retained for vocalization clustering.
- `docs/reference/classifier-api.md`
  - Replace "alternative source mode" language with detection-job-only training.
- `docs/reference/behavioral-constraints.md`
  - Remove ProcessingJob concurrency/idempotency rules after processing is removed.
- `docs/reference/testing.md`
  - Replace the legacy E2E smoke path that uploads audio, queues ProcessingJob, and clusters an EmbeddingSet.
  - New smoke path should use a small completed detection job, detection embeddings, classifier training from detection jobs, and/or Vocalization / Clustering.
- `docs/workflows/session-review.md`
  - Replace the "no duplicate embedding sets" review check with current idempotency checks.
- `docs/plans/backlog.md`
  - Remove backlog items tied to Search by uploaded audio and existing embedding sets.
- Existing specs/plans
  - Historical specs and plans can remain, but any "current state" or "active workflow" doc should be updated if it is referenced by CLAUDE.md or workflow instructions.

Memory-file scan outcome:

- No standalone memory state files are present in the repo root.
- A prior plan, `docs/plans/2026-04-07-remove-legacy-memory-status-doc-refs.md`, indicates old memory/status files had already been retired.
- `docs/workflows/session-begin.md` still says not to read memory files unless needed; that instruction can remain generic unless the team wants to remove memory-file language entirely.

## Design Alternatives

### Approach A: UI-First Retirement, Keep Legacy Tables Temporarily

Remove top-level pages, creation endpoints, and worker polling first. Keep legacy ORM models/tables and read-only converters so historical rows and old models do not break.

Pros:

- Lowest risk and easiest to split across sessions.
- Avoids immediate destructive migrations.
- Keeps existing classifier model history and any vocalization embedding-set sources readable.

Cons:

- Leaves dead tables and code paths behind.
- Requires explicit guards so no new legacy rows are created.
- Does not fully satisfy disk and DB cleanup until later phases.

### Approach B: Big-Bang Removal

Delete pages, routers, services, workers, tests, DB tables, and artifacts in one implementation.

Pros:

- Fastest route to a clean codebase.
- Fewer temporary compatibility branches.

Cons:

- Highest risk in SQLite migrations and artifact deletion.
- Easy to break retained Vocalization / Clustering because it shares `clustering_jobs`, `clusters`, and worker code.
- Existing classifier models or vocalization jobs with embedding-set provenance become difficult to inspect.

### Approach C: Strangler With Final Destructive Cleanup

Phase 1 removes user-facing surfaces and blocks new legacy creation. Phase 2 rewires shared retained modules so they no longer import legacy models. Phase 3 runs explicit archive-backed data/artifact cleanup and destructive migrations after preflight reports are clean.

Pros:

- Gives immediate product cleanup while preserving recoverability.
- Lets each session finish with a working app.
- Makes DB/artifact deletion auditable and reversible through explicit archive copies, database backups, and manifests.
- Handles the shared clustering and vocalization dependencies deliberately.

Cons:

- Takes more sessions.
- Requires temporary compatibility code and documentation that clearly marks legacy paths as retired.

Recommended: **Approach C**.

## Phased Implementation Plan

### Phase 1: User-Facing Retirement and Creation Guards

Objective: remove the unused workflows from the visible app and prevent new legacy data from being created, without destructive DB cleanup.

Frontend:

- Remove nav entries and breadcrumbs for Audio, Processing, top-level Clustering, Search, and Label Processing.
- Change `/` and wildcard fallback from `/app/audio` to `/app/call-parsing/detection` or another retained workflow default.
- Remove route imports and route definitions for the removed pages.
- Update `TopNav` home link to the new default route.
- Change Classifier / Training to detection-job only.
- Remove `useEmbeddingSets` and `useAudioFiles` imports from `TrainingTab.tsx`.
- Remove `EmbeddingSetPanel`, `ModelFilter`, and embedding-set source radio UI from Classifier / Training.

API:

- Stop registering `processing`, `search`, and `label_processing` routers in `api/app.py`.
- Stop registering the `audio` router after moving any genuinely retained endpoint to a retained domain router; do not keep unused audio endpoints for old links.
- For `/classifier/training-jobs`, reject embedding-set payloads with a clear 422/400 message.
- Remove unused top-level clustering API endpoints rather than keeping read-only compatibility endpoints. Retain only the endpoints needed by Vocalization / Clustering, preferably under `/vocalization/clustering-*`.

Workers:

- Remove processing, search, and label-processing polling from `workers/runner.py`.
- Remove stale recovery for those job types or leave it only until their tables are dropped.
- Keep clustering worker intact but add a guard that legacy embedding-set jobs fail clearly once no UI can create them.

Docs:

- Update `CLAUDE.md`, `AGENTS.md`, frontend reference, classifier API reference, and testing reference to reflect visible workflow removal.

Verification:

- `uv run ruff format --check` and `uv run ruff check` on touched backend files.
- `uv run pyright` on touched backend files.
- Focused backend tests for classifier training rejection and retained detection-job training.
- `cd frontend && npx tsc --noEmit`.
- Playwright smoke for navigation and Classifier / Training form.

### Phase 2: Shared-Code Untangling

Objective: remove legacy imports from retained modules and prepare DB cleanup.

Backend:

- Split clustering service/worker API around detection-job clustering:
  - rename internal variables from `embedding_set_id` to neutral `source_id` where they represent detection job IDs.
  - update `ClusteringJobOut` and frontend types if needed while preserving API compatibility for Vocalization / Clustering.
- Remove embedding-set branch from `run_clustering_job()` after legacy queued jobs are handled.
- Remove `create_clustering_job()` for embedding-set jobs or make it private test-only until deletion.
- Remove embedding-set training branch from `run_training_job()`, after existing queued embedding-set jobs are failed or completed.
- Remove `create_training_job()` for embedding-set sources.
- Update `model_registry_service.delete_model()` so it no longer imports `EmbeddingSet`; it should check retained references such as detection embeddings, classifier models, or active model configs instead.

Vocalization dependency decision:

- Fully retire embedding-set and local-folder sources in Vocalization as part of this cleanup.
- Remove `embedding_set_ids` from `VocalizationTrainingSourceConfig`.
- Remove `embedding_set` from `VocalizationInferenceJobCreate.source_type`.
- Remove `types/import`.
- Remove local-folder source behavior that creates or fetches folder embedding sets.
- Update `TrainingDataView` filters and training dataset snapshot logic to detection-job only.
- Archive any local-folder/audio/embedding artifacts through the cleanup script before dropping `embedding_sets` and `audio_files`.

Tests:

- Rewrite tests to use detection-job fixtures where possible.
- Keep pure algorithm tests.
- Remove tests for retired routers and workers.

### Phase 3: Legacy Data Preflight and Cleanup Script

Objective: make deletion auditable before changing schema.

Add a script such as `scripts/cleanup_legacy_workflows.py`:

- Dry-run default.
- Requires `--archive-root` for `--apply`; dry-run prints the computed archive layout.
- Reads repo `.env` through existing Settings.
- Reports DB counts and dependency blockers.
- Builds artifact deletion manifest for `/audio/raw`, `/embeddings`, `/label_processing`, and legacy-only `/clusters` jobs.
- Requires `--apply` to archive and delete files.
- Writes a cleanup manifest under `storage_root/cleanup-manifests/`.
- Copies all candidate artifacts into the archive location before deleting originals.
- Processes all legacy artifact classes in one apply operation instead of per-directory opt-ins.
- Runs per-directory verification for candidate discovery, archive copies, deletion completion, and idempotent rerun behavior.
- Exits non-zero if retained rows still reference legacy data that would be dropped.

Blockers to check:

- queued/running `processing_jobs`, `search_jobs`, `label_processing_jobs`
- queued/running classifier training jobs with `source_mode="embedding_sets"`
- clustering jobs with non-empty `embedding_set_ids` and no `detection_job_ids`
- vocalization/training-dataset source configs containing embedding-set IDs
- any imported/uploaded audio rows still needed by retained local detection workflows

### Phase 4: Destructive DB Migration

Objective: remove legacy schema after preflight is clean and backups exist.

Migration candidates:

- Drop `search_jobs`.
- Drop `label_processing_jobs`.
- Drop `processing_jobs`.
- Drop `embedding_sets`.
- Drop `audio_metadata`.
- Drop `audio_files` after Phase 2 removes retained dependencies and Phase 3 archives local audio files/artifacts.
- Drop or rename `clustering_jobs.embedding_set_ids`.
- Rename `cluster_assignments.embedding_set_id` to `source_id` if retained clustering now uses detection job IDs.
- Drop `classifier_training_jobs.positive_embedding_set_ids` and `negative_embedding_set_ids` only after legacy model provenance is preserved for display/history.
- Remove `"embedding_sets"` as a default `source_mode`/`training_source_mode` in model defaults.

The migration should refuse to proceed if preflight blockers are present.

### Phase 5: Final Code and Documentation Prune

Objective: delete legacy modules and references after DB cleanup.

Delete:

- `src/humpback/models/audio.py` if `audio_files` is dropped.
- `src/humpback/models/processing.py`
- `src/humpback/models/search.py`
- `src/humpback/models/label_processing.py`
- `src/humpback/services/audio_service.py` if no retained endpoints use it.
- `src/humpback/services/processing_service.py`
- `src/humpback/services/search_service.py`
- `src/humpback/services/label_processing_service.py`
- `src/humpback/workers/processing_worker.py`
- `src/humpback/workers/search_worker.py`
- `src/humpback/workers/label_processing_worker.py`
- retired schemas, converters, hooks, TS types, and frontend component directories.

Update:

- `src/humpback/models/__init__.py`
- `src/humpback/api/app.py`
- `src/humpback/workers/queue.py`
- `src/humpback/workers/runner.py`
- all reference docs and workflow instructions listed above.

Final search gates:

- `rg "Audio|Processing|Search|Label Processing" frontend/src docs CLAUDE.md AGENTS.md`
- `rg "EmbeddingSet|embedding_sets|processing_jobs|search_jobs|label_processing_jobs" src tests docs CLAUDE.md AGENTS.md`
- Remaining matches must be historical ADR/spec/plan references, retained detection embedding concepts, or explicit migration tests.

## Resolved Decisions

1. Retire remaining embedding-set/local-folder sources as part of this cleanup, but archive local files and artifacts to a script-specified archive location before deletion.
2. Leave existing embedding-set trained classifier models visible and usable as legacy models.
3. Drop `audio_files` only after backing up local audio files/artifacts to the script-specified archive location and removing retained dependencies.
4. Remove unused API endpoints rather than keeping read-only compatibility endpoints for old links.
5. Process all legacy artifact classes in one shot. The script must support dry-run and must verify each directory class independently before and after apply.

## Acceptance Criteria

- Removed pages no longer appear in navigation, breadcrumbs, default redirects, or route table.
- Classifier / Training offers only detection-job based training.
- No API endpoint can create new `ProcessingJob`, `EmbeddingSet`, `SearchJob`, `LabelProcessingJob`, embedding-set clustering job, or embedding-set classifier training job.
- Worker runner no longer processes removed job types.
- Vocalization / Clustering still creates, runs, displays, and deletes detection-job clustering jobs.
- Cleanup script can dry-run and report DB rows/artifacts before deletion.
- Cleanup script archives local files/artifacts to the operator-specified archive location before deleting originals.
- Destructive migrations are gated by database backup, archive verification, and preflight checks.
- Project instructions and reference docs no longer describe retired workflows as active capability.
- Full verification gates pass at the end of each implementation phase.
