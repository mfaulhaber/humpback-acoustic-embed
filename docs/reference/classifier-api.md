# Classifier API Surface

Classifier training currently has three distinct flows:

## Embedding-set Training

`POST /classifier/training-jobs` creates the original positive/negative embedding-set-backed training jobs.

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

## Legacy Retrain Workflow

- `GET /classifier/models/{id}/retrain-info`
- `POST /classifier/retrain`
- `GET /classifier/retrain-workflows`
- `GET /classifier/retrain-workflows/{id}`

## Timeline Export

- `POST /classifier/detection-jobs/{id}/timeline/export` — export a completed detection job's timeline as a self-contained static bundle (tiles, MP3 audio, JSON manifest) for hosting as a readonly viewer on S3. Also available as `scripts/export_timeline.py` CLI.
