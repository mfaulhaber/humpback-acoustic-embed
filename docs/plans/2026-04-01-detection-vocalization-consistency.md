# Detection–Vocalization Label Consistency Implementation Plan

**Goal:** Prevent data consistency issues between timeline label editing and vocalization labeling by adding version tracking, reconciliation endpoints, and deletion guards.
**Spec:** [docs/specs/2026-04-01-detection-vocalization-consistency-design.md](../specs/2026-04-01-detection-vocalization-consistency-design.md)

---

### Task 1: Alembic Migration — Add Version Columns

**Files:**
- Create: `alembic/versions/033_detection_vocalization_consistency.py`

**Acceptance criteria:**
- [ ] `DetectionJob` gets `row_store_version INTEGER NOT NULL DEFAULT 1`
- [ ] `VocalizationLabel` gets `row_store_version_at_import INTEGER` (nullable)
- [ ] Migration uses `op.batch_alter_table()` for SQLite compatibility
- [ ] `uv run alembic upgrade head` succeeds

**Tests needed:**
- Migration applies and rolls back cleanly

---

### Task 2: Model Updates — Version Fields on DetectionJob and VocalizationLabel

**Files:**
- Modify: `src/humpback/models/classifier.py`
- Modify: `src/humpback/models/labeling.py`

**Acceptance criteria:**
- [ ] `DetectionJob.row_store_version` field added with default `1`
- [ ] `VocalizationLabel.row_store_version_at_import` nullable field added
- [ ] Fields match migration column definitions

**Tests needed:**
- None beyond migration verification — model fields are declarative

---

### Task 3: Version Increment on Timeline Label Edits

**Files:**
- Modify: `src/humpback/api/routers/classifier.py`

**Acceptance criteria:**
- [ ] `batch_edit_labels` endpoint increments `job.row_store_version` after writing the Parquet file
- [ ] Version is committed to DB in the same transaction as `has_positive_labels`

**Tests needed:**
- Unit test: `batch_edit_labels` increments `row_store_version` on each call
- Unit test: version starts at 1 for a new detection job

---

### Task 4: Record Version on Vocalization Label Creation

**Files:**
- Modify: `src/humpback/api/routers/labeling.py`

**Acceptance criteria:**
- [ ] `create_vocalization_label` looks up the detection job's current `row_store_version` and sets `row_store_version_at_import` on the new label
- [ ] Existing labels with NULL version continue to work (no migration backfill needed)

**Tests needed:**
- Unit test: newly created VocalizationLabel has correct `row_store_version_at_import`

---

### Task 5: Refresh Preview Endpoint

**Files:**
- Modify: `src/humpback/api/routers/labeling.py`
- Modify: `src/humpback/schemas/labeling.py`

**Acceptance criteria:**
- [ ] `POST /labeling/vocalization-labels/{detection_job_id}/refresh` returns a preview with no mutations
- [ ] Preview includes: `matched_count`, `orphaned_count`, `orphaned_labels` (list with type, start_utc, end_utc)
- [ ] Reads current row store Parquet to build the set of valid `(start_utc, end_utc)` keys
- [ ] Compares against all VocalizationLabel records for that detection job
- [ ] Returns 404 if detection job not found, 400 if row store not available
- [ ] Response schema added to `schemas/labeling.py`

**Tests needed:**
- Unit test: all labels matched when row store unchanged
- Unit test: orphaned labels detected after row deletion
- Unit test: orphaned labels detected after row move
- Unit test: 404 for nonexistent detection job

---

### Task 6: Refresh Apply Endpoint

**Files:**
- Modify: `src/humpback/api/routers/labeling.py`
- Modify: `src/humpback/schemas/labeling.py`

**Acceptance criteria:**
- [ ] `POST /labeling/vocalization-labels/{detection_job_id}/refresh/apply` deletes all orphaned labels
- [ ] Updates `row_store_version_at_import` on surviving labels to current `row_store_version`
- [ ] Returns summary: `deleted_count`, `surviving_count`
- [ ] Idempotent — calling when already in sync is a no-op

**Tests needed:**
- Unit test: orphaned labels deleted, surviving labels updated
- Unit test: idempotent when already in sync
- Unit test: version field updated on surviving labels

---

### Task 7: Deletion Guards

**Files:**
- Modify: `src/humpback/services/classifier_service.py`
- Modify: `src/humpback/api/routers/classifier.py`

**Acceptance criteria:**
- [ ] `delete_detection_job` checks for VocalizationLabel records referencing the job
- [ ] `delete_detection_job` checks for TrainingDataset `source_config` JSON containing the job ID
- [ ] Returns 409 with dependency detail message when blocked
- [ ] Deletion proceeds when no dependencies exist (existing behavior preserved)
- [ ] `bulk_delete_detection_jobs` deletes safe jobs, returns 409 for blocked ones with per-job details

**Tests needed:**
- Unit test: delete blocked when vocalization labels exist
- Unit test: delete blocked when training dataset references exist
- Unit test: delete succeeds when no dependencies
- Unit test: bulk delete partial success — some blocked, some deleted

---

### Task 8: Frontend — Staleness Badge and Refresh Dialog

**Files:**
- Modify: vocalization labeling workspace component (staleness badge + refresh button)
- Create: refresh review dialog component

**Acceptance criteria:**
- [ ] Staleness check: compare detection job's `row_store_version` against labels' `row_store_version_at_import` (via existing API responses or new staleness field)
- [ ] Warning badge displayed when stale: "Detection labels have changed since last import. Refresh to sync."
- [ ] Refresh button triggers preview endpoint
- [ ] Modal displays matched count, orphaned count, and orphaned label details (type, timestamps)
- [ ] "Discard orphaned labels & sync" button calls apply endpoint and refreshes workspace state
- [ ] Detection job delete shows error toast on 409 with dependency message

**Tests needed:**
- Playwright: staleness badge appears after timeline edit
- Playwright: refresh dialog shows correct counts
- Playwright: apply clears orphaned labels and removes badge

---

### Task 9: Documentation Updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/reference/data-model.md`

**Acceptance criteria:**
- [ ] CLAUDE.md §9.1 updated to mention detection–vocalization consistency (version tracking, reconciliation, deletion guards)
- [ ] CLAUDE.md §9.2 updated with migration 033
- [ ] CLAUDE.md §8.7 updated with behavioral constraint: detection jobs with vocalization labels or training dataset references cannot be deleted
- [ ] `docs/reference/data-model.md` updated with new columns on DetectionJob and VocalizationLabel

**Tests needed:**
- None

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/models/classifier.py src/humpback/models/labeling.py src/humpback/api/routers/classifier.py src/humpback/api/routers/labeling.py src/humpback/services/classifier_service.py src/humpback/schemas/labeling.py`
2. `uv run ruff check src/humpback/models/classifier.py src/humpback/models/labeling.py src/humpback/api/routers/classifier.py src/humpback/api/routers/labeling.py src/humpback/services/classifier_service.py src/humpback/schemas/labeling.py`
3. `uv run pyright src/humpback/models/classifier.py src/humpback/models/labeling.py src/humpback/api/routers/classifier.py src/humpback/api/routers/labeling.py src/humpback/services/classifier_service.py src/humpback/schemas/labeling.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
