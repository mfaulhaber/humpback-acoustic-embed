# Ingest Detection Domain

Load this domain for hydrophone providers, detection jobs, detection row stores,
classifier training, detection embeddings, hyperparameter tuning, autoresearch,
or classifier UI.

## Primary Paths

- `src/humpback/classifier/`
- `src/humpback/api/routers/classifier/`
- `src/humpback/services/classifier_service/`
- `src/humpback/services/detection_embedding_service.py`
- `src/humpback/services/hyperparameter_service/`
- `src/humpback/workers/classifier_worker/`
- `src/humpback/workers/detection_embedding_worker.py`
- `src/humpback/workers/hyperparameter_worker.py`
- `frontend/src/components/classifier/`
- `frontend/e2e/detection-*.spec.ts`
- `frontend/e2e/hydrophone-*.spec.ts`

## Artifact Roots

- `detections/{detection_job_id}/`
- `detections/{detection_job_id}/embeddings/{model_version}/`
- `classifiers/{classifier_model_id}/`
- `hyperparameter/manifests/{manifest_id}/`
- `hyperparameter/searches/{search_id}/`

## Likely Neighbors

- `signal-timeline` for hydrophone audio, playback, and timeline display.
- `vocalization-clustering` for labels, training datasets, and clustering.
- `call-parsing` when detection jobs feed Pass 1 region detection.
- `core-platform` for model/schema/storage changes.

## Before Editing

1. Identify whether the change affects retained row identity or embedding
   model versioning.
2. Load neighbor context when changing outputs consumed by labels, call
   parsing, or sequence models.
