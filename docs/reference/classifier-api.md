# Classifier API Surface

Classifier training is now detection-job-based. Legacy embedding-set creation
paths are retired, although legacy models and legacy training provenance remain
readable.

## Detection-Job Training

`POST /classifier/training-jobs` accepts `detection_job_ids` plus
`embedding_model_version`. The service validates that model-versioned detection
embeddings exist for every source job and that the selected row stores contain
at least one positive and one negative binary label. If a requested
model-versioned embeddings parquet is missing but the legacy-path detection
embeddings exist and the source classifier model matches, the embeddings are
copied forward automatically.

## Hyperparameter Tuning

UI-driven manifest generation + random search:

- `POST /classifier/hyperparameter/manifests` — create and queue manifest generation
- `GET /classifier/hyperparameter/manifests` — list manifests
- `GET /classifier/hyperparameter/manifests/{id}` — manifest detail
- `DELETE /classifier/hyperparameter/manifests/{id}` — delete (409 if referenced by search)
- `POST /classifier/hyperparameter/searches` — create and queue search job
- `GET /classifier/hyperparameter/searches` — list searches
- `GET /classifier/hyperparameter/searches/{id}` — search detail
- `GET /classifier/hyperparameter/searches/{id}/history` — trial history
- `DELETE /classifier/hyperparameter/searches/{id}` — delete search + artifacts
- `GET /classifier/hyperparameter/search-space-defaults` — default search space
- `POST /classifier/hyperparameter/searches/{id}/import-candidate` — import as candidate

## Autoresearch Candidate Review and Promotion

Relocated under `/classifier/hyperparameter/candidates/*`; old `/classifier/autoresearch-candidates/*` paths still work:

- `POST /classifier/hyperparameter/candidates/import`
- `GET /classifier/hyperparameter/candidates`
- `GET /classifier/hyperparameter/candidates/{candidate_id}`
- `DELETE /classifier/hyperparameter/candidates/{candidate_id}` — delete candidate
- `POST /classifier/hyperparameter/candidates/{candidate_id}/training-jobs`

Candidate-backed promotion imports reviewed autoresearch artifacts, persists a durable `AutoresearchCandidate`, and creates manifest-backed training jobs that keep source candidate and comparison provenance on both the training job and resulting classifier model. After candidate-backed training completes, the training job's `source_comparison_context` and the model's `training_summary` include a `replay_verification` dict with status (`"verified"`/`"mismatch"`), per-split metric comparisons, and effective config. The candidate detail endpoint (`GET /classifier/autoresearch-candidates/{id}`) also exposes `replay_verification` when the linked model exists.

## Retired Embedding-Set Training

`POST /classifier/training-jobs` rejects payloads containing
`positive_embedding_set_ids` or `negative_embedding_set_ids` with a validation
error. Legacy classifier models still surface their retired provenance through
`training_source_mode="embedding_sets"` and `legacy_source_summary`.

## Detection Job Label Counts

- `GET /classifier/detection-jobs/label-counts?detection_job_ids=...` — returns per-job positive/negative label counts by reading row stores. Positive = humpback or orca; negative = ship or background. Returns `0/0` for missing row stores.

## Detection Embedding Jobs

- `GET /classifier/detection-embedding-jobs?detection_job_ids=...&model_version=...` — returns a status row for each `(detection_job_id, model_version)` pair including rows that don't yet exist (`status="not_started"`). Also recognizes legacy-path embeddings (written inline by the hydrophone worker) when the detection job's source classifier model matches the requested model version. Response includes `rows_processed`, `rows_total`, `error_message`.
- `POST /classifier/detection-jobs/{id}/generate-embeddings?mode=full|sync` — enqueue re-embedding for a detection job
- `GET /classifier/detection-jobs/{id}/embedding-generation-status` — most recent embedding generation job

## Legacy Retrain Workflow Surface

- `GET /classifier/models/{id}/retrain-info`
- `POST /classifier/retrain`
- `GET /classifier/retrain-workflows`
- `GET /classifier/retrain-workflows/{id}`

These endpoints are retained only for legacy visibility. New retrain workflow
creation is retired: `GET /retrain-info` returns 404 for current models, and
`POST /classifier/retrain` returns a retirement error. Existing historical
`retrain_workflows` rows remain listable/readable.

## Timeline Export

- `POST /classifier/detection-jobs/{id}/timeline/export` — export a completed detection job's timeline as a self-contained static bundle (tiles, MP3 audio, JSON manifest) for hosting as a readonly viewer on S3. Also available as `scripts/export_timeline.py` CLI.
