# Vocalization Clustering Page — Design Spec

## Overview

New page under Vocalization that enables clustering of detection job embeddings
that have been labeled by the active vocalization model. Reuses the existing
clustering backend pipeline (HDBSCAN, UMAP, stability evaluation) with a new
input path for detection job embeddings. Follows the Call Parsing / Detection
page style for look and feel.

## Navigation

Add "Clustering" as a child of the **Vocalization** nav group in the sidebar:

- Training
- Labeling
- Training Data
- **Clustering** (new)

Routes:

- `/app/vocalization/clustering` — list view (job form + job tables)
- `/app/vocalization/clustering/:jobId` — detail view (cluster results)

## Page Layout — List View

Three vertically stacked sections, following the DetectionPage pattern.

### Section 1: Job Creation Form

Card with title "Vocalization Clustering".

**Detection job selector:**

- Compact table with checkbox column for multi-select.
- Columns: checkbox, hydrophone name, date range (UTC), detection count.
- Only shows detection jobs that have a completed `VocalizationInferenceJob`
  whose `vocalization_model_id` references the active vocalization model
  (`is_active=True`).
- Select-all / deselect-all toggle button in the card header.

**Clustering parameters:**

Visible by default:

- HDBSCAN: `min_cluster_size` (default 5), `min_samples` (optional),
  `cluster_selection_method` (leaf/eom, default leaf).

Collapsible "Advanced Settings" section (collapsed by default):

- UMAP toggle (default on): `umap_cluster_n_components` (default 5),
  `umap_n_neighbors` (default 15), `umap_min_dist` (default 0.1).
- Stability: `stability_runs` (default 0 = off).

**Submit button:** "Start Clustering", disabled until at least one detection
job is selected.

### Section 2: Active Jobs Table

Only rendered when queued/running jobs exist. Panel with "Active Jobs" header
and count badge.

Columns: status badge, created timestamp, detection jobs summary (e.g.
"3 detection jobs"), cancel action.

### Section 3: Previous Jobs Table

Always rendered. Panel with "Previous Jobs" header and count badge.

Toolbar: text filter (by hydrophone name), bulk delete button, pagination
controls (20 per page).

Columns: checkbox (for bulk ops), status badge, created timestamp, detection
jobs summary, cluster count, error (truncated), "View" action button.

Sortable columns: status, created, cluster count.

"View" navigates to `/app/vocalization/clustering/:jobId`.

## Page Layout — Detail View

Route: `/app/vocalization/clustering/:jobId`

Top: back link to list view.

**Job metadata header:** status badge, creation timestamp, source detection
jobs (hydrophone names + date ranges), clustering parameters.

**Result sections** (reuse existing clustering components):

- Cluster table (`ClusterTable`)
- UMAP scatter plot (`UmapPlot`)
- Evaluation panel (`EvaluationPanel`)
- Export report (`ExportReport`)

Components are imported from `frontend/src/components/clustering/` and rendered
with data from the vocalization clustering job endpoints.

## Backend — API

### New endpoints

Mounted under the vocalization router (`/vocalization`):

| Method | Path | Description |
|--------|------|-------------|
| GET | `/vocalization/clustering-eligible-jobs` | Detection jobs eligible for vocalization clustering |
| POST | `/vocalization/clustering-jobs` | Create vocalization clustering job |
| GET | `/vocalization/clustering-jobs` | List all vocalization clustering jobs |
| GET | `/vocalization/clustering-jobs/{job_id}` | Single job detail |
| DELETE | `/vocalization/clustering-jobs/{job_id}` | Delete job + storage artifacts |
| GET | `/vocalization/clustering-jobs/{job_id}/clusters` | Cluster list |
| GET | `/vocalization/clustering-jobs/{job_id}/visualization` | UMAP coordinates |
| GET | `/vocalization/clustering-jobs/{job_id}/metrics` | Parsed metrics JSON |
| GET | `/vocalization/clustering-jobs/{job_id}/stability` | Stability summary |

### Eligible jobs endpoint

`GET /vocalization/clustering-eligible-jobs`

Joins `DetectionJob` -> `VocalizationInferenceJob` -> `VocalizationClassifierModel`
with filters:

- `VocalizationInferenceJob.status == "complete"`
- `VocalizationInferenceJob.source_type == "detection_job"`
- `VocalizationClassifierModel.is_active == True`

Returns detection job summaries: `id`, `hydrophone_name`, `start_timestamp`,
`end_timestamp`, detection count (from `result_summary`).

### Create job request

```json
{
  "detection_job_ids": ["uuid1", "uuid2"],
  "parameters": {
    "min_cluster_size": 5,
    "cluster_selection_method": "leaf",
    "umap_cluster_n_components": 5,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "stability_runs": 0
  }
}
```

Validations:

- `detection_job_ids` is non-empty.
- All detection jobs exist and have a completed inference job from the active
  vocalization model.
- All detection jobs have completed detection embedding jobs (embeddings parquet
  exists).

## Backend — Database

Add nullable `detection_job_ids` column (Text, JSON array) to the `ClusteringJob`
table via Alembic migration.

A vocalization clustering job sets `detection_job_ids` to the JSON array and
`embedding_set_ids` to `"[]"`. A standard clustering job continues to use
`embedding_set_ids` with `detection_job_ids` as `null`.

The worker determines the input type by checking which field is populated.

## Backend — Worker

When `detection_job_ids` is populated (and `embedding_set_ids` is `"[]"`):

1. For each detection job ID, look up the completed `DetectionEmbeddingJob`
   row to get `model_version`, then resolve the parquet path via
   `detection_embeddings_path(storage_root, detection_job_id, model_version)`.
   If multiple embedding jobs exist for a detection job, use the most recent
   completed one.
2. Load embeddings from each parquet file. The detection embeddings parquet is
   keyed by `row_id` (string) rather than integer index — adapt the loading to
   use `row_id` as the row identifier in cluster assignments.
3. Track `(detection_job_id, row_id)` tuples instead of
   `(embedding_set_id, embedding_row_index)` for cluster assignments.
4. The rest of the pipeline (HDBSCAN, UMAP reduction, cluster creation, metrics
   computation) runs identically.

Storage output goes to the same `cluster_dir(storage_root, job.id)` path.

Cluster assignments for detection-sourced jobs store the detection_job_id in
the `embedding_set_id` field and the row_id (as integer index) in
`embedding_row_index` — reusing the existing schema without new columns.

## Frontend — File Structure

New files under `frontend/src/components/vocalization/`:

- `VocalizationClusteringPage.tsx` — list view (form + tables)
- `VocalizationClusteringForm.tsx` — detection job selector + clustering params
- `VocalizationClusteringJobTable.tsx` — active/previous job table
- `VocalizationClusteringDetail.tsx` — detail view (reuses clustering components)

New hooks in `frontend/src/hooks/queries/`:

- Add vocalization clustering queries to `useVocalization.ts` (or new file):
  `useClusteringEligibleJobs`, `useVocalizationClusteringJobs`,
  `useCreateVocalizationClusteringJob`, `useDeleteVocalizationClusteringJob`,
  plus detail sub-resource hooks (clusters, visualization, metrics, stability).

## Scope Exclusions

- **Classifier baseline** and **metric learning** parameters are not included.
  These depend on folder-path category labels which don't apply to detection
  job input.
- **Dendrogram heatmap** and **label dot plot** are excluded from the detail
  view (these require confusion matrix data from folder-path labels).
- **Refinement workflow** (`refined_from_job_id`) is not exposed on this page
  for MVP. Can be added later.
- No changes to the existing Clustering page — it continues to work with
  embedding set input only.
