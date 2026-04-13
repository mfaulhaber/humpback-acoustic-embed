# Call Parsing Pipeline API Surface

Four-pass pipeline under `/call-parsing/*`. Passes 1–3 are fully functional; Pass 4 still returns HTTP 501 for the sequence export endpoint. Per-pass job status transitions follow the standard `queued → running → complete|failed|canceled` pattern.

## Parent Runs

- `POST /call-parsing/runs` — create a parent run and its queued Pass 1 job in one transaction; accepts the same source + model + config shape as `POST /region-jobs`
- `GET /call-parsing/runs` — list runs with pagination
- `GET /call-parsing/runs/{id}` — parent run with nested Pass 1/2/3 status summaries
- `DELETE /call-parsing/runs/{id}` — cascade deletes all three child jobs and their parquet directories
- `GET /call-parsing/runs/{id}/sequence` — **501** (Pass 4 sequence export)

## Pass 1 — Region Detection

- `POST /call-parsing/region-jobs` — create a queued Pass 1 job from an audio-file source or a hydrophone range; validates every FK and the `source` XOR invariant
- `GET /call-parsing/region-jobs`, `GET /call-parsing/region-jobs/{id}`, `DELETE /call-parsing/region-jobs/{id}` — list / detail / delete
- `GET /call-parsing/region-jobs/{id}/trace` — stream `trace.parquet` as JSON `{time_sec, score}` rows; 409 while the job is not `complete`, 404 if the parquet file is missing
- `GET /call-parsing/region-jobs/{id}/regions` — return `regions.parquet` sorted by `start_sec`; same 409/404 guards

## Pass 2 — Event Segmentation

- `POST /call-parsing/segmentation-jobs` — create a queued Pass 2 job; validates `region_detection_job_id` (404) and `segmentation_model_id` (404); 409 if upstream Pass 1 job is not `complete`
- `GET /call-parsing/segmentation-jobs`, `GET /call-parsing/segmentation-jobs/{id}`, `DELETE /call-parsing/segmentation-jobs/{id}` — list / detail / delete
- `GET /call-parsing/segmentation-jobs/{id}/events` — return `events.parquet` as JSON; 409 while job is not `complete`, 404 if parquet file is missing

## Pass 2 — Segmentation Training Datasets

- `GET /call-parsing/segmentation-training-datasets` — list training datasets with sample counts (used for bootstrap dataset inspection; bootstrap scripts call trainers directly)

## Pass 2 — Segmentation Models

- `GET /call-parsing/segmentation-models` — list models with condensed metrics
- `GET /call-parsing/segmentation-models/{id}` — full detail
- `DELETE /call-parsing/segmentation-models/{id}` — removes row + checkpoint directory on disk; 409 if referenced by an in-flight segmentation job

## Pass 2 — Boundary Corrections

- `POST /call-parsing/segmentation-jobs/{id}/corrections` — batch upsert boundary corrections; validates job exists (404) and is complete (409); returns correction count
- `GET /call-parsing/segmentation-jobs/{id}/corrections` — list all corrections for a segmentation job
- `DELETE /call-parsing/segmentation-jobs/{id}/corrections` — clear all corrections; 204

## Pass 3 — Event Classification

- `POST /call-parsing/classification-jobs` — create a queued Pass 3 job; validates `vocalization_model_id` exists (404) and has `model_family='pytorch_event_cnn'` + `input_mode='segmented_event'` (422); validates `event_segmentation_job_id` exists (404) and is `complete` (409)
- `GET /call-parsing/classification-jobs`, `GET /call-parsing/classification-jobs/{id}`, `DELETE /call-parsing/classification-jobs/{id}` — list / detail / delete
- `GET /call-parsing/classification-jobs/{id}/typed-events` — return `typed_events.parquet` as JSON sorted by `start_sec`; 409 while job is not `complete`, 404 if parquet file is missing

## Pass 3 — Type Corrections

- `POST /call-parsing/classification-jobs/{id}/corrections` — batch upsert type corrections; unique on `(job_id, event_id)`; validates job exists (404) and is complete (409); returns correction count
- `GET /call-parsing/classification-jobs/{id}/corrections` — list all corrections for a classification job
- `DELETE /call-parsing/classification-jobs/{id}/corrections` — clear all corrections; 204

## Pass 2 — Segmentation Feedback Training

- `POST /call-parsing/segmentation-feedback-training-jobs` — create queued job; validates all source segmentation job IDs exist and are complete (404/409); 201
- `GET /call-parsing/segmentation-feedback-training-jobs` — list jobs
- `GET /call-parsing/segmentation-feedback-training-jobs/{id}` — detail; 404 if not found
- `DELETE /call-parsing/segmentation-feedback-training-jobs/{id}` — deletes job row; produced models managed via segmentation model endpoints; 204; 404 if not found

## Pass 3 — Classifier Feedback Training

- `POST /call-parsing/classifier-training-jobs` — create queued job; validates all source classification job IDs exist and are complete (404/409); 201
- `GET /call-parsing/classifier-training-jobs` — list jobs
- `GET /call-parsing/classifier-training-jobs/{id}` — detail; 404 if not found
- `DELETE /call-parsing/classifier-training-jobs/{id}` — deletes job row; produced models managed via classifier model endpoints; 204; 404 if not found

## Pass 3 — Classifier Model Management

- `GET /call-parsing/classifier-models` — list `pytorch_event_cnn` models
- `DELETE /call-parsing/classifier-models/{id}` — deletes model + checkpoint directory; 409 if referenced by in-flight classification or training jobs; 204; 404 if not found

## Pass 3 — Event Classifier Training (Bootstrap Only)

- Bootstrap scripts call `train_event_classifier()` directly; no API endpoint or worker queue for bootstrap training
- `scripts/bootstrap_event_classifier_dataset.py` generates training samples from vocalization-labeled detection windows
