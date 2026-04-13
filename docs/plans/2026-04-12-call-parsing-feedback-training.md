# Call Parsing — Feedback Training Implementation Plan

**Goal:** Add human-in-the-loop correction storage, feedback training workers, and API endpoints for both Pass 2 (segmentation boundaries) and Pass 3 (event type labels), and remove bootstrap training paths from the backend.
**Spec:** [docs/specs/2026-04-12-call-parsing-feedback-training-design.md](../specs/2026-04-12-call-parsing-feedback-training-design.md)

---

### Task 1: Migration and SQLAlchemy Models

**Files:**
- Create: `alembic/versions/046_feedback_training_tables.py`
- Create: `src/humpback/models/feedback_training.py`
- Modify: `src/humpback/database.py` (register new models for metadata)

**Acceptance criteria:**
- [ ] Migration creates `event_boundary_corrections` table with columns: `id` (UUID PK), `event_segmentation_job_id` (VARCHAR NOT NULL), `event_id` (VARCHAR NOT NULL), `region_id` (VARCHAR NOT NULL), `correction_type` (VARCHAR NOT NULL), `start_sec` (FLOAT nullable), `end_sec` (FLOAT nullable), `created_at`, `updated_at`
- [ ] Migration creates `event_type_corrections` table with columns: `id` (UUID PK), `event_classification_job_id` (VARCHAR NOT NULL), `event_id` (VARCHAR NOT NULL), `type_name` (VARCHAR nullable), `created_at`, `updated_at`
- [ ] `event_type_corrections` has unique constraint on `(event_classification_job_id, event_id)`
- [ ] `event_boundary_corrections` has index on `event_segmentation_job_id`
- [ ] Migration creates `event_segmentation_training_jobs` table with columns: `id` (UUID PK), `status` (default "queued"), `source_job_ids` (TEXT NOT NULL), `config_json` (TEXT nullable), `segmentation_model_id` (VARCHAR nullable), `result_summary` (TEXT nullable), `error_message` (TEXT nullable), `created_at`, `updated_at`, `started_at` (nullable), `completed_at` (nullable)
- [ ] Migration creates `event_classifier_training_jobs` table with columns: `id` (UUID PK), `status` (default "queued"), `source_job_ids` (TEXT NOT NULL), `config_json` (TEXT nullable), `vocalization_model_id` (VARCHAR nullable), `result_summary` (TEXT nullable), `error_message` (TEXT nullable), `created_at`, `updated_at`, `started_at` (nullable), `completed_at` (nullable)
- [ ] `downgrade()` drops all four tables
- [ ] Uses `op.batch_alter_table()` for SQLite compatibility
- [ ] SQLAlchemy model classes match migration schema exactly
- [ ] `uv run alembic upgrade head` succeeds against production DB

**Tests needed:**
- Migration up/down round-trips cleanly

---

### Task 2: Pydantic Schemas

**Files:**
- Modify: `src/humpback/schemas/call_parsing.py`

**Acceptance criteria:**
- [ ] `BoundaryCorrection` model: `event_id` (str), `region_id` (str), `correction_type` (Literal["adjust", "add", "delete"]), `start_sec` (Optional[float]), `end_sec` (Optional[float])
- [ ] `BoundaryCorrectionRequest` model: `corrections` (list of `BoundaryCorrection`)
- [ ] `BoundaryCorrectionResponse` model: fields from DB row plus original event data for joined view
- [ ] `TypeCorrection` model: `event_id` (str), `type_name` (Optional[str])
- [ ] `TypeCorrectionRequest` model: `corrections` (list of `TypeCorrection`)
- [ ] `TypeCorrectionResponse` model: fields from DB row
- [ ] `CreateSegmentationFeedbackTrainingJobRequest` model: `source_job_ids` (list[str]), `config` (optional training hyperparameters)
- [ ] `CreateClassifierTrainingJobRequest` model: `source_job_ids` (list[str]), `config` (optional training hyperparameters)
- [ ] `SegmentationFeedbackTrainingJobResponse` model: all DB fields
- [ ] `ClassifierTrainingJobResponse` model: all DB fields
- [ ] Validation: `BoundaryCorrection` with `correction_type="add"` requires `start_sec` and `end_sec`; `correction_type="delete"` forbids them

**Tests needed:**
- Pydantic validation rejects add without start/end, delete with start/end
- Round-trip serialization for all response models

---

### Task 3: Service Layer — Corrections

**Files:**
- Modify: `src/humpback/services/call_parsing.py`

**Acceptance criteria:**
- [ ] `upsert_boundary_corrections(session, job_id, corrections)` — validates segmentation job exists and is complete; batch inserts/updates corrections; returns count
- [ ] `list_boundary_corrections(session, job_id)` — returns corrections for a segmentation job, joined with original event data from parquet for a complete picture
- [ ] `clear_boundary_corrections(session, job_id)` — deletes all corrections for a segmentation job
- [ ] `upsert_type_corrections(session, job_id, corrections)` — validates classification job exists and is complete; upserts by `(job_id, event_id)` unique key; returns count
- [ ] `list_type_corrections(session, job_id)` — returns corrections for a classification job
- [ ] `clear_type_corrections(session, job_id)` — deletes all corrections for a classification job
- [ ] Proper error types: `CallParsingFKError` (404), `CallParsingStateError` (409)

**Tests needed:**
- Upsert creates new corrections, updates existing on repeat call
- Type correction enforces unique (job_id, event_id) — second upsert overwrites
- Validates job exists and is complete; rejects non-existent (404) and non-complete (409)
- Clear removes all corrections for a job
- List returns empty list for job with no corrections

---

### Task 4: Service Layer — Feedback Training Jobs and Model Management

**Files:**
- Modify: `src/humpback/services/call_parsing.py`

**Acceptance criteria:**
- [ ] `create_segmentation_feedback_training_job(session, request)` — validates all source segmentation job IDs exist and are complete; creates queued job row; returns job
- [ ] `list_segmentation_feedback_training_jobs(session)` — ordered by `created_at DESC`
- [ ] `get_segmentation_feedback_training_job(session, job_id)` — single row
- [ ] `delete_segmentation_feedback_training_job(session, job_id)` — deletes job row + artifacts directory
- [ ] `create_classifier_training_job(session, request)` — validates all source classification job IDs exist and are complete; creates queued job row; returns job
- [ ] `list_classifier_training_jobs(session)` — ordered by `created_at DESC`
- [ ] `get_classifier_training_job(session, job_id)` — single row
- [ ] `delete_classifier_training_job(session, job_id)` — deletes job row + artifacts directory
- [ ] `list_classifier_models(session)` — lists `vocalization_models` filtered to `model_family='pytorch_event_cnn'`
- [ ] `delete_classifier_model(session, model_id, settings)` — deletes model + checkpoint directory; raises `CallParsingStateError` (409) if referenced by in-flight classification or training jobs

**Tests needed:**
- Create validates source job existence and completion status
- Create with non-existent source job raises 404
- Create with non-complete source job raises 409
- Delete removes job row and cleans up storage directory
- Classifier model delete raises 409 when referenced by in-flight job
- Classifier model list returns only `pytorch_event_cnn` models

---

### Task 5: API Endpoints — Corrections

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`

**Acceptance criteria:**
- [ ] `POST /call-parsing/segmentation-jobs/{id}/corrections` — calls `upsert_boundary_corrections`, returns correction count, status 200; 404 if job not found; 409 if job not complete
- [ ] `GET /call-parsing/segmentation-jobs/{id}/corrections` — calls `list_boundary_corrections`, returns list of corrections
- [ ] `DELETE /call-parsing/segmentation-jobs/{id}/corrections` — calls `clear_boundary_corrections`, returns 204
- [ ] `POST /call-parsing/classification-jobs/{id}/corrections` — calls `upsert_type_corrections`, returns correction count; 404/409 guards
- [ ] `GET /call-parsing/classification-jobs/{id}/corrections` — calls `list_type_corrections`
- [ ] `DELETE /call-parsing/classification-jobs/{id}/corrections` — calls `clear_type_corrections`, returns 204

**Tests needed:**
- POST with valid corrections returns count
- POST on non-existent job returns 404
- POST on non-complete job returns 409
- GET returns corrections list
- DELETE clears corrections, returns 204
- Repeated POST on same event_id overwrites (type corrections)

---

### Task 6: API Endpoints — Feedback Training Jobs and Model Management

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`

**Acceptance criteria:**
- [ ] `POST /call-parsing/segmentation-feedback-training-jobs` — creates queued job, returns 201
- [ ] `GET /call-parsing/segmentation-feedback-training-jobs` — lists jobs
- [ ] `GET /call-parsing/segmentation-feedback-training-jobs/{id}` — detail; 404 if not found
- [ ] `DELETE /call-parsing/segmentation-feedback-training-jobs/{id}` — deletes; 204; 404 if not found
- [ ] `POST /call-parsing/classifier-training-jobs` — creates queued job, returns 201
- [ ] `GET /call-parsing/classifier-training-jobs` — lists jobs
- [ ] `GET /call-parsing/classifier-training-jobs/{id}` — detail; 404 if not found
- [ ] `DELETE /call-parsing/classifier-training-jobs/{id}` — deletes; 204; 404 if not found
- [ ] `GET /call-parsing/classifier-models` — lists `pytorch_event_cnn` models
- [ ] `DELETE /call-parsing/classifier-models/{id}` — deletes model; 204; 409 if in-flight reference; 404 if not found

**Tests needed:**
- POST with valid source job IDs returns 201
- POST with non-existent source job returns 404
- POST with non-complete source job returns 409
- Full CRUD round-trip for both training job types
- Model delete returns 409 when referenced

---

### Task 7: Pass 2 Feedback Training Worker

**Files:**
- Create: `src/humpback/workers/event_segmentation_feedback_worker.py`

**Acceptance criteria:**
- [ ] `run_event_segmentation_feedback_training(session, job, settings)` implements the full pipeline
- [ ] Reads source segmentation job IDs from `job.source_job_ids` JSON
- [ ] For each source job: reads `events.parquet`, loads `event_boundary_corrections` from DB, groups by `region_id`
- [ ] Applies corrections per region: `adjust` overwrites start/end, `delete` removes event, `add` inserts new event
- [ ] Includes uncorrected regions as-is (implicit approval)
- [ ] Resolves audio via segmentation job → region detection job → hydrophone chain using `resolve_timeline_audio`
- [ ] Builds framewise binary labels from corrected event sets
- [ ] Calls `train_model` from `call_parsing/segmentation/trainer.py` with per-audio-source split
- [ ] Saves checkpoint, creates `SegmentationModel` row
- [ ] Updates job: `segmentation_model_id`, `result_summary`, `status='complete'`
- [ ] Crash safety: deletes partial artifacts on exception, sets `status='failed'` with error message

**Tests needed:**
- Worker processes a job with mocked corrections and synthetic audio
- Correction application: adjust/add/delete produce correct event sets
- Uncorrected regions are included in training data
- Crash leaves no partial artifacts, job is failed
- Audio resolution traces through the job chain correctly

---

### Task 8: Pass 3 Feedback Training Worker

**Files:**
- Create: `src/humpback/workers/event_classifier_feedback_worker.py`

**Acceptance criteria:**
- [ ] `run_event_classifier_feedback_training(session, job, settings)` implements the full pipeline
- [ ] Reads source classification job IDs from `job.source_job_ids` JSON
- [ ] For each source job: reads `typed_events.parquet`, loads `event_type_corrections` from DB
- [ ] Assembles training samples: corrected events use corrected `type_name` (skip if null/negative); uncorrected events use original above-threshold type; events with no type are negatives
- [ ] Resolves audio via classification job → segmentation job → region detection job → hydrophone chain using `resolve_timeline_audio` with context padding for z-score normalization
- [ ] Calls `train_event_classifier` from `call_parsing/event_classifier/trainer.py` with per-audio-source split and per-type threshold optimization
- [ ] Saves checkpoint, creates `VocalizationClassifierModel` row with `model_family='pytorch_event_cnn'`, `input_mode='segmented_event'`
- [ ] Updates job: `vocalization_model_id`, `result_summary`, `status='complete'`
- [ ] Crash safety: deletes partial artifacts on exception, sets `status='failed'` with error message

**Tests needed:**
- Worker processes a job with mocked corrections and synthetic audio
- Type correction overrides inference result
- Null type_name produces negative sample
- Uncorrected above-threshold events used as-is
- Crash leaves no partial artifacts, job is failed
- Audio resolution traces through the full job chain

---

### Task 9: Worker Registration

**Files:**
- Modify: `src/humpback/workers/runner.py`
- Modify: `src/humpback/workers/queue.py`

**Acceptance criteria:**
- [ ] `claim_segmentation_feedback_training_job(session)` in `queue.py` claims from `event_segmentation_training_jobs`
- [ ] `claim_classifier_feedback_training_job(session)` in `queue.py` claims from `event_classifier_training_jobs`
- [ ] Both new job types added to stale-job recovery in `recover_stale_jobs`
- [ ] Worker runner dispatches to new workers in priority order: existing call parsing workers → segmentation feedback training → classifier feedback training (before manifest generation)
- [ ] Runner imports and calls `run_event_segmentation_feedback_training` and `run_event_classifier_feedback_training`

**Tests needed:**
- Claim functions return queued jobs and set status to running
- Stale recovery resets stuck jobs

---

### Task 10: Bootstrap Cleanup

**Files:**
- Modify: `src/humpback/workers/vocalization_worker.py`
- Delete: `src/humpback/workers/segmentation_training_worker.py`
- Modify: `src/humpback/workers/runner.py`
- Modify: `src/humpback/workers/queue.py`
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `src/humpback/services/call_parsing.py`

**Acceptance criteria:**
- [ ] Remove `_run_pytorch_event_cnn_training` function and `pytorch_event_cnn` dispatch branch from `vocalization_worker.py`; vocalization training worker handles only `sklearn_perch_embedding` (and legacy `None`)
- [ ] Delete `segmentation_training_worker.py` entirely
- [ ] Remove `claim_segmentation_training_job` from `queue.py`
- [ ] Remove `segmentation_training` stale-job recovery from `recover_stale_jobs` in `queue.py`
- [ ] Remove segmentation training worker dispatch block from `runner.py`
- [ ] Remove API endpoints: `POST /call-parsing/segmentation-training-jobs`, `GET /call-parsing/segmentation-training-jobs`, `GET /call-parsing/segmentation-training-jobs/{id}`, `DELETE /call-parsing/segmentation-training-jobs/{id}`
- [ ] Remove service methods: `create_segmentation_training_job`, `list_segmentation_training_jobs`, `get_segmentation_training_job`, `delete_segmentation_training_job`
- [ ] Keep `list_segmentation_training_datasets` endpoint and service method (still used for bootstrap dataset inspection)
- [ ] Keep `segmentation_training_datasets` / `segmentation_training_samples` / `segmentation_training_jobs` tables in DB (no migration to drop them)

**Tests needed:**
- Vocalization worker rejects `pytorch_event_cnn` model_family (or raises ValueError)
- Removed API endpoints return 404 (or are simply absent from router)
- Existing tests referencing removed endpoints are updated or removed

---

### Task 11: Documentation Updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] CLAUDE.md §8.7: add behavioral constraints for feedback training (correction tables, implicit approval, single-label Pass 3, hydrophone-only)
- [ ] CLAUDE.md §8.9: add correction and feedback training API endpoints; remove segmentation training job endpoints; note bootstrap scripts call trainers directly
- [ ] CLAUDE.md §9.1: update implemented capabilities with feedback training loop
- [ ] CLAUDE.md §9.2: update latest migration number and table list
- [ ] CLAUDE.md §8.7 worker priority order: remove `segmentation_training`, add `segmentation_feedback_training` and `classifier_feedback_training`
- [ ] DECISIONS.md: ADR for feedback training architecture (correction tables vs parquet amendment, implicit approval, bootstrap cleanup rationale)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/models/feedback_training.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/workers/event_segmentation_feedback_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/workers/vocalization_worker.py src/humpback/workers/runner.py src/humpback/workers/queue.py`
2. `uv run ruff check src/humpback/models/feedback_training.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/workers/event_segmentation_feedback_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/workers/vocalization_worker.py src/humpback/workers/runner.py src/humpback/workers/queue.py`
3. `uv run pyright src/humpback/models/feedback_training.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/workers/event_segmentation_feedback_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/workers/vocalization_worker.py src/humpback/workers/runner.py src/humpback/workers/queue.py`
4. `uv run pytest tests/`
