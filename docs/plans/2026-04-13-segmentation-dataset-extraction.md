# Segmentation Dataset Extraction Implementation Plan

**Goal:** Enable training segmentation models from scratch on human-annotated
call boundaries by extracting corrections into the existing training dataset
infrastructure.

**Spec:** [docs/specs/2026-04-13-segmentation-dataset-extraction-design.md](../specs/2026-04-13-segmentation-dataset-extraction-design.md)

---

### Task 1: Create shared extraction module

**Files:**
- Create: `src/humpback/call_parsing/segmentation/extraction.py`

**Acceptance criteria:**
- [ ] `CorrectedSample` dataclass with fields: `hydrophone_id`, `start_timestamp`, `end_timestamp`, `crop_start_sec`, `crop_end_sec`, `events_json`
- [ ] `apply_corrections(original_events, corrections)` function — applies delete/adjust/add corrections, returns list of event dicts
- [ ] `subdivide_region(...)` function — splits regions into 30s crops with 15s hops, returns `CorrectedSample` list
- [ ] `collect_corrected_samples(session, segmentation_job_id, storage_root)` async function — reads regions/corrections for one job, applies corrections, subdivides, returns samples for corrected regions only
- [ ] Module constants `MAX_CROP_SEC = 30.0` and `CROP_HOP_SEC = 15.0`

**Tests needed:**
- `apply_corrections` with combinations of delete, adjust, and add corrections
- `subdivide_region` with a short region (single crop) and a long region (multiple crops with correct event filtering per window)
- `collect_corrected_samples` with mocked DB session and parquet files

---

### Task 2: Refactor feedback worker to use shared module

**Files:**
- Modify: `src/humpback/workers/event_segmentation_feedback_worker.py`

**Acceptance criteria:**
- [ ] `_apply_corrections` function removed from worker
- [ ] `_subdivide_region` function removed from worker
- [ ] `_FeedbackSample` dataclass removed from worker
- [ ] Imports `apply_corrections`, `subdivide_region`, `CorrectedSample` from `extraction.py`
- [ ] `_collect_samples` delegates per-region correction application and cropping to shared functions
- [ ] Training execution, model saving, and error handling unchanged

**Tests needed:**
- Regression test: verify the worker produces identical samples for the same inputs after the refactor

---

### Task 3: Add service function for dataset extraction

**Files:**
- Modify: `src/humpback/services/call_parsing.py`

**Acceptance criteria:**
- [ ] `create_dataset_from_corrections(session, segmentation_job_id, name, description)` async function
- [ ] Validates segmentation job exists and is complete
- [ ] Resolves upstream `RegionDetectionJob` for hydrophone context
- [ ] Calls `collect_corrected_samples()` from extraction module
- [ ] Raises descriptive error if no corrected regions found
- [ ] Creates `SegmentationTrainingDataset` row with provided or auto-generated name
- [ ] Bulk-inserts `SegmentationTrainingSample` rows with `source="boundary_correction"` and `source_ref=segmentation_job_id`
- [ ] Returns the created dataset

**Tests needed:**
- Service function with a segmentation job that has corrections — verify dataset and sample rows created
- Service function with a segmentation job with no corrections — verify error raised
- Service function with nonexistent or incomplete job — verify error raised

---

### Task 4: Add API endpoint and request/response schemas

**Files:**
- Modify: `src/humpback/schemas/call_parsing.py`
- Modify: `src/humpback/api/routers/call_parsing.py`

**Acceptance criteria:**
- [ ] `CreateDatasetFromCorrectionsRequest` schema with `segmentation_job_id` (required), `name` (optional), `description` (optional)
- [ ] `CreateDatasetFromCorrectionsResponse` schema with `id`, `name`, `sample_count`, `created_at`
- [ ] `POST /call-parsing/segmentation-training-datasets/from-corrections` endpoint
- [ ] Endpoint calls `create_dataset_from_corrections` service function
- [ ] Returns 400 if no corrected regions found
- [ ] Returns 404 if segmentation job not found or not complete

**Tests needed:**
- Integration test: create segmentation job with corrections, call endpoint, verify response and database state
- Error cases: missing job, incomplete job, no corrections

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/call_parsing/segmentation/extraction.py src/humpback/workers/event_segmentation_feedback_worker.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py`
2. `uv run ruff check src/humpback/call_parsing/segmentation/extraction.py src/humpback/workers/event_segmentation_feedback_worker.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py`
3. `uv run pyright src/humpback/call_parsing/segmentation/extraction.py src/humpback/workers/event_segmentation_feedback_worker.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py`
4. `uv run pytest tests/`
