# Vocalization Clustering Implementation Plan

**Goal:** Add a Vocalization Clustering page that lets users cluster detection job embeddings labeled by the active vocalization model, reusing the existing clustering pipeline with a new detection-job input path.
**Spec:** [docs/specs/2026-04-24-vocalization-clustering-design.md](../specs/2026-04-24-vocalization-clustering-design.md)

---

### Task 1: Alembic Migration — Add detection_job_ids to ClusteringJob

**Files:**
- Create: `alembic/versions/056_clustering_detection_job_ids.py`
- Modify: `src/humpback/models/clustering.py`

**Acceptance criteria:**
- [ ] New nullable `detection_job_ids` column (Text) added to `clustering_jobs` table
- [ ] Migration uses `op.batch_alter_table()` for SQLite compatibility
- [ ] `ClusteringJob` model updated with `detection_job_ids: Mapped[Optional[str]]`
- [ ] Migration runs cleanly against the production database (backup first per CLAUDE.md §3.5)
- [ ] Existing clustering jobs unaffected (column is null for them)

**Tests needed:**
- Migration applies and rolls back without error
- Existing clustering job rows still load correctly after migration

---

### Task 2: Backend — Pydantic Schemas and Converters

**Files:**
- Modify: `src/humpback/schemas/clustering.py`
- Modify: `src/humpback/schemas/converters.py`

**Acceptance criteria:**
- [ ] New `VocalizationClusteringJobCreate` schema with `detection_job_ids: list[str]` and `parameters: Optional[dict]`
- [ ] `ClusteringJobOut` extended with `detection_job_ids: Optional[list[str]]` field
- [ ] New `ClusteringEligibleDetectionJobOut` schema with `id`, `hydrophone_name`, `start_timestamp`, `end_timestamp`, `detection_count`
- [ ] `clustering_job_to_out` converter updated to serialize `detection_job_ids` from the model

**Tests needed:**
- Schema validation: VocalizationClusteringJobCreate rejects empty detection_job_ids
- ClusteringJobOut serializes detection_job_ids correctly for both null and populated cases

---

### Task 3: Backend — Service Layer for Vocalization Clustering

**Files:**
- Modify: `src/humpback/services/clustering_service.py`

**Acceptance criteria:**
- [ ] New `create_vocalization_clustering_job(session, detection_job_ids, parameters)` function
- [ ] Validates all detection job IDs exist
- [ ] Validates each detection job has a completed VocalizationInferenceJob from the active model
- [ ] Validates each detection job has a completed DetectionEmbeddingJob with existing parquet
- [ ] Creates ClusteringJob with `detection_job_ids` populated and `embedding_set_ids` set to `"[]"`
- [ ] New `list_clustering_eligible_detection_jobs(session)` function that joins DetectionJob → VocalizationInferenceJob → VocalizationClassifierModel with the active-model and status filters
- [ ] New `list_vocalization_clustering_jobs(session)` that filters to jobs where `detection_job_ids` is not null
- [ ] Existing `create_clustering_job` unchanged

**Tests needed:**
- create_vocalization_clustering_job happy path creates job with correct field values
- Rejects detection job without completed inference job
- Rejects detection job without completed embedding job
- list_clustering_eligible_detection_jobs returns only properly filtered jobs
- list_vocalization_clustering_jobs excludes standard clustering jobs

---

### Task 4: Backend — API Endpoints

**Files:**
- Modify: `src/humpback/api/routers/vocalization.py`

**Acceptance criteria:**
- [ ] `GET /vocalization/clustering-eligible-jobs` returns eligible detection jobs
- [ ] `POST /vocalization/clustering-jobs` creates a vocalization clustering job
- [ ] `GET /vocalization/clustering-jobs` lists vocalization clustering jobs only
- [ ] `GET /vocalization/clustering-jobs/{job_id}` returns single job
- [ ] `DELETE /vocalization/clustering-jobs/{job_id}` deletes job and artifacts
- [ ] `GET /vocalization/clustering-jobs/{job_id}/clusters` returns cluster list
- [ ] `GET /vocalization/clustering-jobs/{job_id}/visualization` returns UMAP data (adapted for detection job source — resolves detection_job_id to hydrophone_name instead of audio filename)
- [ ] `GET /vocalization/clustering-jobs/{job_id}/metrics` returns parsed metrics
- [ ] `GET /vocalization/clustering-jobs/{job_id}/stability` returns stability data
- [ ] All endpoints return appropriate 404/400 errors

**Tests needed:**
- Integration tests for each endpoint (create, list, get, delete)
- Eligible jobs endpoint returns correct filtered results
- Visualization endpoint works with detection-sourced jobs

---

### Task 5: Backend — Worker Detection-Job Embedding Loading

**Files:**
- Modify: `src/humpback/workers/clustering_worker.py`

**Acceptance criteria:**
- [ ] Worker detects detection-job-sourced clustering jobs via `detection_job_ids` field
- [ ] Looks up completed DetectionEmbeddingJob for each detection job to get model_version
- [ ] Loads embeddings from `detection_embeddings_path()` for each detection job
- [ ] Handles row_id (string) from detection parquet, converting to integer index for cluster assignments
- [ ] Stores detection_job_id in `embedding_set_id` field and row index in `embedding_row_index` for cluster assignments
- [ ] HDBSCAN, UMAP, cluster creation, metrics computation all work identically
- [ ] Standard embedding-set clustering path remains unchanged

**Tests needed:**
- Worker loads detection embeddings and produces valid clustering output
- Cluster assignments correctly map back to detection_job_id and row_id
- Standard clustering worker path still works after changes

---

### Task 6: Frontend — Query Hooks

**Files:**
- Modify: `frontend/src/hooks/queries/useVocalization.ts`
- Modify: `frontend/src/api/types.ts` (add response types)

**Acceptance criteria:**
- [ ] `useClusteringEligibleJobs()` hook fetches `GET /vocalization/clustering-eligible-jobs` with polling
- [ ] `useVocalizationClusteringJobs(pollInterval)` hook fetches job list with polling
- [ ] `useCreateVocalizationClusteringJob()` mutation hook for POST
- [ ] `useDeleteVocalizationClusteringJob()` mutation hook for DELETE
- [ ] `useVocalizationClusteringJob(jobId)` hook for single job detail
- [ ] Sub-resource hooks: `useVocClusteringClusters(jobId)`, `useVocClusteringVisualization(jobId)`, `useVocClusteringMetrics(jobId)`, `useVocClusteringStability(jobId)`
- [ ] TypeScript types added for API response shapes

**Tests needed:**
- Type-check passes (`npx tsc --noEmit`)

---

### Task 7: Frontend — Job Creation Form

**Files:**
- Create: `frontend/src/components/vocalization/VocalizationClusteringForm.tsx`

**Acceptance criteria:**
- [ ] Card with "Vocalization Clustering" title
- [ ] Detection job multi-select table: checkbox, hydrophone name, date range (UTC), detection count
- [ ] Fetches eligible jobs from the API hook
- [ ] Select-all / deselect-all toggle in card header
- [ ] HDBSCAN params visible by default (min_cluster_size, min_samples, selection_method)
- [ ] Collapsible "Advanced Settings" with UMAP toggle + params and stability_runs
- [ ] "Start Clustering" button disabled until at least one job selected
- [ ] Calls create mutation on submit, shows success/error toast
- [ ] Follows RegionJobForm styling patterns (Card, grid layout, Collapsible for advanced)

**Tests needed:**
- Type-check passes
- Manual verification: form renders, selection works, submit creates job

---

### Task 8: Frontend — Job Table Component

**Files:**
- Create: `frontend/src/components/vocalization/VocalizationClusteringJobTable.tsx`

**Acceptance criteria:**
- [ ] `VocalizationClusteringJobTable` component with active/previous mode
- [ ] `VocalizationClusteringJobTablePanel` wrapper with title and count badge
- [ ] Active mode: status, created, detection jobs summary, cancel action
- [ ] Previous mode: checkbox, status, created, detection jobs summary, cluster count, error, View button
- [ ] Previous mode toolbar: text filter by hydrophone name, bulk delete, pagination (20/page)
- [ ] Sortable columns in previous mode: status, created, cluster count
- [ ] View button navigates to `/app/vocalization/clustering/:jobId`
- [ ] Follows RegionJobTable styling (border panel, table rows, StatusBadge, pagination controls)

**Tests needed:**
- Type-check passes
- Manual verification: table renders, sorting/filtering/pagination work

---

### Task 9: Frontend — List Page and Detail Page

**Files:**
- Create: `frontend/src/components/vocalization/VocalizationClusteringPage.tsx`
- Create: `frontend/src/components/vocalization/VocalizationClusteringDetail.tsx`

**Acceptance criteria:**
- [ ] `VocalizationClusteringPage` composes form + active table + previous table (DetectionPage pattern)
- [ ] Splits jobs into active (queued/running) and previous (complete/failed)
- [ ] Polls job list at 3000ms
- [ ] `VocalizationClusteringDetail` page with back link, job metadata header, and result sections
- [ ] Reuses `ClusterTable`, `UmapPlot`, `EvaluationPanel`, `ExportReport` from clustering components
- [ ] Only shows result sections when job status is complete
- [ ] Shows error message when job status is failed

**Tests needed:**
- Type-check passes
- Manual verification: list page renders, detail page renders with cluster results

---

### Task 10: Frontend — Navigation and Routing

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/layout/SideNav.tsx`

**Acceptance criteria:**
- [ ] Routes added: `/app/vocalization/clustering` and `/app/vocalization/clustering/:jobId`
- [ ] SideNav updated: "Clustering" child added to Vocalization group
- [ ] Imports for VocalizationClusteringPage and VocalizationClusteringDetail

**Tests needed:**
- Type-check passes
- Manual verification: nav link appears, routes resolve correctly

---

### Task 11: Documentation Updates

**Files:**
- Modify: `CLAUDE.md` (§9.1 capabilities, §9.2 migration number)

**Acceptance criteria:**
- [ ] §9.1 mentions vocalization clustering capability
- [ ] §9.2 latest migration updated to 056
- [ ] §9.2 no new tables (reuses clustering_jobs)

**Tests needed:**
- None (documentation only)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/models/clustering.py src/humpback/schemas/clustering.py src/humpback/schemas/converters.py src/humpback/services/clustering_service.py src/humpback/api/routers/vocalization.py src/humpback/workers/clustering_worker.py`
2. `uv run ruff check src/humpback/models/clustering.py src/humpback/schemas/clustering.py src/humpback/schemas/converters.py src/humpback/services/clustering_service.py src/humpback/api/routers/vocalization.py src/humpback/workers/clustering_worker.py`
3. `uv run pyright src/humpback/models/clustering.py src/humpback/schemas/clustering.py src/humpback/schemas/converters.py src/humpback/services/clustering_service.py src/humpback/api/routers/vocalization.py src/humpback/workers/clustering_worker.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
