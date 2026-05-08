# Vocalization Clustering Domain

Load this domain for vocalization labels, vocalization vocabulary, multi-label
vocalization models, training datasets, vocalization inference, clustering, or
vocalization UI.

## Primary Paths

- `src/humpback/api/routers/labeling.py`
- `src/humpback/api/routers/vocalization.py`
- `src/humpback/classifier/vocalization_inference.py`
- `src/humpback/classifier/vocalization_trainer.py`
- `src/humpback/clustering/`
- `src/humpback/services/training_dataset.py`
- `src/humpback/services/vocalization_service.py`
- `src/humpback/workers/clustering_worker.py`
- `src/humpback/workers/vocalization_worker.py`
- `frontend/src/components/vocalization/`
- `frontend/src/components/shared/ClusterProjectionPlot.tsx`
- `frontend/e2e/vocalization-labeling.spec.ts`

## Frontend Scope

- Vocalization / Clustering detail renders persisted UMAP coordinates through
  `VocalizationUmapPlot`, which adapts vocalization clustering data into the
  shared `ClusterProjectionPlot` renderer. Keep vocalization-specific fetching,
  palette/noise semantics, labels, and audio-on-click behavior in the
  vocalization adapter rather than the shared component.

## Artifact Roots

- `training_datasets/{dataset_id}/`
- `clusters/{clustering_job_id}/`
- Detection-job embedding roots owned by `ingest-detection` are common inputs.

## Likely Neighbors

- `ingest-detection` for detection rows, labels, embeddings, and source jobs.
- `call-parsing` for event-level type corrections and classifier feedback.
- `frontend-shell` for query hooks and shared UI.

## Before Editing

1. Identify whether the change affects label semantics, vocabulary behavior, or
   training data assembly.
2. Load `ingest-detection` if row identity or detection-job labels are touched.
