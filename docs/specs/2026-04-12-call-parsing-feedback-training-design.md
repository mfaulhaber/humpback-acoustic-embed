# Call Parsing — Human Feedback Loop for Segmentation & Event Classifier Training

**Date:** 2026-04-12
**Status:** Draft

---

## 1. Overview

Design a clean backend for human-in-the-loop retraining of both Pass 2 (event segmentation) and Pass 3 (event classification) models. The human reviews inference output, corrects errors, and those corrections become training data for the next model iteration.

This spec covers: correction storage, training job tables, feedback training workers, API endpoints, and cleanup of bootstrap training paths from the backend. Frontend UI is deferred to a separate spec (approved layouts saved in session memory).

All audio sources in this workflow are hydrophone-based. File-based audio is not supported.

**Feedback loop:**
```
Bootstrap (CLI) → initial model
       ↓
Run inference → human corrects output → retrain from corrections → better model
       ↓                                                              ↑
       └──────────────────────────────────────────────────────────────┘
```

---

## 2. Data Model — Correction Tables

Human corrections are stored in separate tables, not by amending parquet output. Original inference parquet files (`events.parquet`, `typed_events.parquet`) remain immutable. Corrections reference events by `event_id`.

### `event_boundary_corrections` (Pass 2)

| Column | Type | Nullable | Default | Purpose |
|--------|------|----------|---------|---------|
| `id` | VARCHAR (UUID) | NO | uuid4() | Primary key |
| `event_segmentation_job_id` | VARCHAR FK | NO | — | Which segmentation job's output |
| `event_id` | VARCHAR | NO | — | References event in `events.parquet` (or new UUID for `add`) |
| `region_id` | VARCHAR | NO | — | Denormalized from parquet for query ease |
| `correction_type` | VARCHAR | NO | — | `"adjust"`, `"add"`, `"delete"` |
| `start_sec` | FLOAT | YES | NULL | Corrected start (null if `delete`) |
| `end_sec` | FLOAT | YES | NULL | Corrected end (null if `delete`) |
| `created_at` | DATETIME | NO | utcnow | |
| `updated_at` | DATETIME | NO | utcnow | |

- `"adjust"`: `event_id` matches existing parquet event, `start_sec`/`end_sec` are corrected values.
- `"add"`: `event_id` is a new UUID (no parquet row), `start_sec`/`end_sec` required.
- `"delete"`: `event_id` matches existing parquet event, `start_sec`/`end_sec` null.

### `event_type_corrections` (Pass 3)

| Column | Type | Nullable | Default | Purpose |
|--------|------|----------|---------|---------|
| `id` | VARCHAR (UUID) | NO | uuid4() | Primary key |
| `event_classification_job_id` | VARCHAR FK | NO | — | Which classification job's output |
| `event_id` | VARCHAR | NO | — | References event in `typed_events.parquet` |
| `type_name` | VARCHAR | YES | NULL | Corrected single type (null = negative/no-call) |
| `created_at` | DATETIME | NO | utcnow | |
| `updated_at` | DATETIME | NO | utcnow | |

- Unique constraint on `(event_classification_job_id, event_id)` — one correction per event per job.
- Single-label: each event has exactly one type or null. This differs from the multi-label vocalization labeling system.

---

## 3. Data Model — Training Job Tables

Both tables follow the same pattern: user selects source jobs, worker assembles training data from corrections + uncorrected events, trains in one step.

### `event_segmentation_training_jobs` (Pass 2 feedback path)

| Column | Type | Nullable | Default | Purpose |
|--------|------|----------|---------|---------|
| `id` | VARCHAR (UUID) | NO | uuid4() | Primary key |
| `status` | VARCHAR | NO | "queued" | queued/running/complete/failed |
| `source_job_ids` | TEXT (JSON) | NO | — | JSON array of `event_segmentation_job_id`s |
| `config_json` | TEXT | YES | NULL | Training hyperparameters |
| `segmentation_model_id` | VARCHAR FK | YES | NULL | Produced model, set on completion |
| `result_summary` | TEXT | YES | NULL | JSON: metrics, sample counts |
| `error_message` | TEXT | YES | NULL | |
| `created_at` | DATETIME | NO | utcnow | |
| `updated_at` | DATETIME | NO | utcnow | |
| `started_at` | DATETIME | YES | NULL | |
| `completed_at` | DATETIME | YES | NULL | |

### `event_classifier_training_jobs` (Pass 3 feedback path)

| Column | Type | Nullable | Default | Purpose |
|--------|------|----------|---------|---------|
| `id` | VARCHAR (UUID) | NO | uuid4() | Primary key |
| `status` | VARCHAR | NO | "queued" | queued/running/complete/failed |
| `source_job_ids` | TEXT (JSON) | NO | — | JSON array of `event_classification_job_id`s |
| `config_json` | TEXT | YES | NULL | Training hyperparameters |
| `vocalization_model_id` | VARCHAR FK | YES | NULL | Produced model (`pytorch_event_cnn`), set on completion |
| `result_summary` | TEXT | YES | NULL | JSON: per-type metrics, thresholds |
| `error_message` | TEXT | YES | NULL | |
| `created_at` | DATETIME | NO | utcnow | |
| `updated_at` | DATETIME | NO | utcnow | |
| `started_at` | DATETIME | YES | NULL | |
| `completed_at` | DATETIME | YES | NULL | |

Pass 3 models stored in `vocalization_models` with `model_family='pytorch_event_cnn'`, `input_mode='segmented_event'` (unchanged from current).

---

## 4. Training Workers

### Pass 2 — `event_segmentation_feedback_worker.py`

1. **Claim** — atomic compare-and-set on `event_segmentation_training_jobs`.
2. **Collect regions** — for each source segmentation job, read `events.parquet`, load all `event_boundary_corrections` for that job. Group events by `region_id`.
3. **Apply corrections per region** — for each region: start with original events from parquet, apply `adjust` (overwrite start/end), `delete` (remove event), `add` (insert new event). Result is the corrected ground-truth event set for that region.
4. **Build training crops** — for each region, resolve audio via segmentation job → region detection job → hydrophone chain using `resolve_timeline_audio`. Build framewise binary labels from the corrected event set (same format the existing `SegmentationCRNN` trainer expects).
5. **Include uncorrected regions** — regions in selected jobs that have no corrections are included as-is (implicit approval). Their original events become the framewise labels.
6. **Train model** — reuse `train_model` from `call_parsing/segmentation/trainer.py`. Per-audio-source train/val split. Save checkpoint, create `SegmentationModel` row.
7. **Update job** — set `segmentation_model_id`, `result_summary`, `status='complete'`.

### Pass 3 — `event_classifier_feedback_worker.py`

1. **Claim** — atomic compare-and-set on `event_classifier_training_jobs`.
2. **Collect events** — for each source classification job, read `typed_events.parquet`, load all `event_type_corrections` for that job.
3. **Assemble training samples** — for each event: if a correction exists, use the corrected `type_name` (skip if null/negative); if no correction, use the original above-threshold type from the parquet. Events with no type (neither corrected nor above threshold) are negatives.
4. **Resolve audio** — trace through classification job → segmentation job → region detection job → hydrophone, fetch via `resolve_timeline_audio` with context padding for z-score normalization.
5. **Train model** — reuse `train_event_classifier` from `call_parsing/event_classifier/trainer.py`. Per-audio-source train/val split, per-type threshold optimization. Save checkpoint, create `VocalizationClassifierModel` row with `model_family='pytorch_event_cnn'`, `input_mode='segmented_event'`.
6. **Update job** — set `vocalization_model_id`, `result_summary`, `status='complete'`.

### Both workers

- Crash safety: delete partial artifacts on exception, set `status='failed'` with error message.
- Join stale-job recovery sweep in `queue.py`.
- Worker priority: between existing call parsing workers in the priority order.

---

## 5. API Endpoints

All under the existing `/call-parsing/` router.

### Pass 2 — Boundary Corrections

- **`POST /call-parsing/segmentation-jobs/{id}/corrections`** — Batch upsert boundary corrections. Accepts array of `{event_id, region_id, correction_type, start_sec, end_sec}`. Validates job exists and is complete (404/409). Returns correction count.
- **`GET /call-parsing/segmentation-jobs/{id}/corrections`** — List all corrections for a segmentation job. Returns corrections joined with original events for a complete picture.
- **`DELETE /call-parsing/segmentation-jobs/{id}/corrections`** — Clear all corrections for a job.

### Pass 3 — Type Corrections

- **`POST /call-parsing/classification-jobs/{id}/corrections`** — Batch upsert type corrections. Accepts array of `{event_id, type_name}` (type_name null for negative). Unique on event_id — repeated POST overwrites. Validates job exists and is complete (404/409). Enforces single-label constraint.
- **`GET /call-parsing/classification-jobs/{id}/corrections`** — List all corrections for a classification job.
- **`DELETE /call-parsing/classification-jobs/{id}/corrections`** — Clear all corrections for a job.

### Pass 2 — Feedback Training Jobs

- **`POST /call-parsing/segmentation-feedback-training-jobs`** — Create queued job. Validates all source segmentation job IDs exist and are complete. Returns job summary.
- **`GET /call-parsing/segmentation-feedback-training-jobs`** — List jobs, ordered by `created_at DESC`.
- **`GET /call-parsing/segmentation-feedback-training-jobs/{id}`** — Detail.
- **`DELETE /call-parsing/segmentation-feedback-training-jobs/{id}`** — Delete job + artifacts.

### Pass 3 — Feedback Training Jobs

- **`POST /call-parsing/classifier-training-jobs`** — Create queued job. Validates all source classification job IDs exist and are complete. Returns job summary.
- **`GET /call-parsing/classifier-training-jobs`** — List jobs, ordered by `created_at DESC`.
- **`GET /call-parsing/classifier-training-jobs/{id}`** — Detail.
- **`DELETE /call-parsing/classifier-training-jobs/{id}`** — Delete job + artifacts.

### Model Management

- **`GET /call-parsing/classifier-models`** — List `vocalization_models` filtered to `model_family='pytorch_event_cnn'`.
- **`DELETE /call-parsing/classifier-models/{id}`** — Delete model + checkpoint directory. 409 if referenced by in-flight classification or training jobs.
- Segmentation models: existing `GET /call-parsing/segmentation-models` and `DELETE /call-parsing/segmentation-models/{id}` remain unchanged.

---

## 6. Bootstrap Cleanup

Remove bootstrap training paths from the backend so the intended workflow is unambiguous.

### Removals

- **`vocalization_worker.py`**: Remove `_run_pytorch_event_cnn_training` dispatch. Vocalization worker becomes sklearn-only.
- **`segmentation_training_worker.py`**: Remove entirely.
- **`queue.py`**: Remove `segmentation_training` from worker priority order.
- **API endpoints**: Remove `POST /call-parsing/segmentation-training-jobs` and segmentation training job CRUD (`GET`, `GET/{id}`, `DELETE/{id}`).

### What bootstrap scripts do instead

Bootstrap scripts call the trainer library functions directly — no worker queue:
- `scripts/bootstrap_segmentation_dataset.py` + a training script call `train_model()` from `call_parsing/segmentation/trainer.py`
- `scripts/bootstrap_event_classifier_dataset.py` + a training script call `train_event_classifier()` from `call_parsing/event_classifier/trainer.py`

### What stays

- `segmentation_training_datasets` / `segmentation_training_samples` tables — bootstrap scripts still write to and read from them.
- `SegmentationModel` table — both bootstrap and feedback workers write models here.
- All Pass 2 and Pass 3 inference endpoints unchanged.

---

## 7. Migration Summary

Single migration `046_feedback_training_tables.py`:
- Create `event_boundary_corrections` with index on `event_segmentation_job_id`
- Create `event_type_corrections` with unique constraint on `(event_classification_job_id, event_id)`
- Create `event_segmentation_training_jobs`
- Create `event_classifier_training_jobs`

Uses `op.batch_alter_table()` for SQLite compatibility.

Removal of `segmentation_training_jobs` table is NOT included — it stays for bootstrap scripts that may reference existing rows. The API endpoints are removed but the table persists.

---

## 8. Scope Exclusions

- **Frontend UI** — deferred. Approved Classify/Classify Training page layouts saved in session memory. Segment Training page rework deferred.
- **Pass 4** (sequence export) — still deferred.
- **Dataset inspection/editing UI** — training data assembly is automated within workers.
- **Multi-label events** — Pass 3 is single-label only. Enforced in correction API.
- **File-based audio** — hydrophone only throughout.
