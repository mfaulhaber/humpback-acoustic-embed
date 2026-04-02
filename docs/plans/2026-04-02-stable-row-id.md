# Stable Row ID Refactor Implementation Plan

**Goal:** Replace `(start_utc, end_utc)` composite key identity for detection windows with stable UUID row IDs, eliminating timestamp drift between row store, embeddings, and vocalization labels.
**Spec:** [docs/specs/2026-04-02-stable-row-id-refactor-design.md](../specs/2026-04-02-stable-row-id-refactor-design.md)

---

### Task 1: Add row_id to row store schema and creation

**Files:**
- Modify: `src/humpback/classifier/detection_rows.py`
- Modify: `src/humpback/classifier/detector.py`
- Modify: `src/humpback/workers/classifier_worker.py`

**Acceptance criteria:**
- [ ] Row store parquet schema includes a `row_id` column (string, UUID4)
- [ ] `_detection_dicts_to_store_rows()` assigns a UUID to each new row
- [ ] `read_detection_row_store()` handles old parquets without `row_id` (assigns UUIDs on read, does not rewrite)
- [ ] `append_detection_row_store()` preserves existing row_ids when appending
- [ ] Row IDs are unique within a row store file

**Tests needed:**
- Row store round-trip preserves row_id values
- New rows get unique UUIDs assigned
- Reading a legacy parquet without row_id column returns rows with generated IDs

---

### Task 2: Update row store editing to use row_id

**Files:**
- Modify: `src/humpback/classifier/detection_rows.py`
- Modify: `src/humpback/schemas/classifier.py`
- Modify: `src/humpback/api/routers/classifier.py`

**Acceptance criteria:**
- [ ] `_row_key()` returns `row_id` instead of `(start_utc, end_utc)` tuple
- [ ] `apply_label_edits()` matches rows by `row_id` for `move`, `delete`, `change_type` actions
- [ ] `apply_label_edits()` assigns a new UUID for `add` actions
- [ ] `LabelEditItem` schema uses `row_id` field instead of `start_utc`/`end_utc`/`new_start_utc`/`new_end_utc` (move action uses `start_utc`/`end_utc` as the new target values)
- [ ] `row_store_version` increment removed from batch edit endpoint
- [ ] Overlap validation still works using timestamp values from matched rows

**Tests needed:**
- Edit operations (move, delete, change_type, add) work with row_id keys
- Overlap validation rejects conflicting edits
- Unknown row_id in edit raises appropriate error

---

### Task 3: Alembic migration and database model changes

**Files:**
- Create: `alembic/versions/035_stable_row_id.py`
- Modify: `src/humpback/models/labeling.py`
- Modify: `src/humpback/models/classifier.py`

**Acceptance criteria:**
- [ ] `VocalizationLabel` model adds `row_id` (string, required)
- [ ] `VocalizationLabel` model drops `start_utc`, `end_utc`, `row_store_version_at_import`
- [ ] Index on `(detection_job_id, start_utc, end_utc)` replaced by index on `(detection_job_id, row_id)`
- [ ] `DetectionJob` model drops `row_store_version`
- [ ] Alembic migration uses `batch_alter_table()` for SQLite compatibility
- [ ] Migration assigns `row_id` to existing vocalization labels by matching `(start_utc, end_utc)` against detection job row stores using tolerance matching (final use of old logic)
- [ ] Unmatched labels deleted during migration
- [ ] Migration skips detection jobs without row store files

**Tests needed:**
- Migration runs cleanly on a database with existing labels
- Labels matched to row store entries get correct row_id
- Unmatched labels are removed

---

### Task 4: Update detection embedding schema and sync

**Files:**
- Modify: `src/humpback/classifier/detector.py`
- Modify: `src/humpback/workers/detection_embedding_worker.py`

**Acceptance criteria:**
- [ ] Detection embedding parquet schema is `(row_id, embedding, confidence)` — `filename`, `start_sec`, `end_sec` removed
- [ ] `write_detection_embeddings()` writes with new schema
- [ ] `diff_row_store_vs_embeddings()` performs set comparison on row_id — no tolerance matching
- [ ] `_SYNC_TOLERANCE_SEC` removed
- [ ] Sync mode in embedding worker resolves audio for missing rows using row store `(start_utc, end_utc)` and detection job audio source
- [ ] Full mode in embedding worker writes row_id from row store
- [ ] `_rewrite_embeddings()` handles new schema
- [ ] Reading old embedding parquets (with filename/start_sec/end_sec) handled gracefully during migration period

**Tests needed:**
- Embedding sync correctly identifies missing and orphaned rows by row_id
- Embedding generation resolves audio from row store timestamps
- Round-trip write/read of new embedding schema

---

### Task 5: Update vocalization inference pipeline

**Files:**
- Modify: `src/humpback/workers/vocalization_worker.py`
- Modify: `src/humpback/classifier/vocalization_inference.py`

**Acceptance criteria:**
- [ ] Inference output parquet uses `row_id` instead of `start_utc`/`end_utc`
- [ ] Worker loads embeddings with `row_id`, passes through to inference
- [ ] `run_inference()` writes predictions keyed by `row_id`
- [ ] UTC reconstruction from filename parsing removed from inference path
- [ ] `parse_recording_timestamp` no longer called in vocalization worker for embedding loading

**Tests needed:**
- Inference output contains row_id column
- Predictions can be joined back to row store by row_id

---

### Task 6: Update labeling API to use row_id

**Files:**
- Modify: `src/humpback/api/routers/labeling.py`
- Modify: `src/humpback/schemas/labeling.py`

**Acceptance criteria:**
- [ ] `create_vocalization_label()` accepts `row_id` parameter instead of `start_utc`/`end_utc`
- [ ] `list_vocalization_labels()` query filtering uses `row_id` instead of UTC range
- [ ] `VocalizationLabelOut` schema includes `row_id`, drops `start_utc`/`end_utc`/`row_store_version_at_import`
- [ ] `refresh_preview` and `refresh_apply` endpoints removed
- [ ] `_matches_any_row()` and `_LABEL_MATCH_TOLERANCE_SEC` removed
- [ ] `OrphanedLabelDetail`, `RefreshPreviewResponse`, `RefreshApplyResponse` schemas removed
- [ ] `all_vocalization_labels` endpoint resolves UTC from row store by `row_id` for display
- [ ] Deleting a detection row (via batch edit) cascade-deletes vocalization labels referencing that `row_id`

**Tests needed:**
- Label creation with row_id stores correctly
- Cascade delete when detection row removed
- All-labels endpoint returns current UTC from row store

---

### Task 7: Frontend changes

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useLabelEdits.ts`
- Modify: `frontend/src/components/vocalization/LabelingWorkspace.tsx`

**Acceptance criteria:**
- [ ] `VocalizationLabel` type uses `row_id`, drops `start_utc`/`end_utc`/`row_store_version_at_import`
- [ ] `DetectionJob` type drops `row_store_version`
- [ ] `LabelEdit` interface and `utcKey()` updated to use `row_id`
- [ ] Edit payloads sent with `row_id` instead of UTC pairs
- [ ] `fetchRefreshPreview()` and `applyRefresh()` API calls removed
- [ ] `OrphanedLabelDetail`, `RefreshPreviewResponse`, `RefreshApplyResponse` types removed
- [ ] Orphan warning badge and reconciliation dialog removed from `LabelingWorkspace`
- [ ] Row store data sent to frontend includes `row_id` for each detection window
- [ ] Timeline component uses `row_id` in edit dispatch

**Tests needed:**
- TypeScript compilation passes with no errors
- Existing Playwright tests updated for removed UI elements

---

### Task 8: Data migration script

**Files:**
- Create: `scripts/migrate_row_ids.py`

**Acceptance criteria:**
- [ ] Script iterates all completed detection jobs
- [ ] Row store parquets: assigns UUID to each row without one, rewrites atomically
- [ ] Embedding parquets: matches entries to row store (using old filename+offset->UTC->tolerance method), rewrites with `(row_id, embedding, confidence)` schema; unmatched entries dropped
- [ ] Inference output parquets: maps `(start_utc, end_utc)` to `row_id` via row store lookup; unmatched predictions dropped
- [ ] Reports summary: jobs processed, rows assigned IDs, embeddings matched/dropped, labels matched/orphaned
- [ ] Skips detection jobs without row store files
- [ ] Idempotent: safe to run multiple times (rows with existing row_id kept)

**Tests needed:**
- Migration of fixture row store assigns IDs correctly
- Migration of fixture embeddings matches and rewrites correctly
- Idempotency: running twice produces same result

---

### Task 9: Documentation and cleanup

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/storage-layout.md`

**Acceptance criteria:**
- [ ] CLAUDE.md behavioral constraints updated (remove row_store_version references, update detection-vocalization consistency section)
- [ ] CLAUDE.md current state updated (schema version, removed capabilities)
- [ ] Data model reference updated with new row_id fields and removed columns
- [ ] Storage layout updated for new embedding/inference parquet schemas

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/ scripts/ tests/`
2. `uv run ruff check src/humpback/ scripts/ tests/`
3. `uv run pyright src/humpback/ scripts/ tests/`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
