# Detection–Vocalization Label Consistency

**Date**: 2026-04-01
**Status**: Approved

## Problem

The timeline viewer edits detection row store Parquet files (binary labels like humpback/orca/ship/background), while the vocalization labeling workflow stores type labels (whup, moan, etc.) as independent `VocalizationLabel` database records keyed by `(detection_job_id, start_utc, end_utc)`. These two systems share the same detection job windows but have no synchronization.

Concrete failure scenarios:
- **Orphaned labels after delete**: User labels a window as "whup" in vocalization labeling, then deletes that window in the timeline viewer. The VocalizationLabel persists and is picked up by the next training run using data that was intentionally removed.
- **Orphaned labels after move**: A timeline `move` operation changes a row's `(start_utc, end_utc)`. VocalizationLabel records still reference the old timestamps and become permanently unlinked.
- **Conflicting semantics**: A window can be `humpback=0` in the row store (cleared via timeline) while simultaneously having active vocalization type labels.
- **Unguarded deletion**: `delete_detection_job` does `shutil.rmtree` on the detection directory and deletes the DB row without checking for VocalizationLabel records or TrainingDataset references, destroying downstream data silently.

## Approach

Snapshot versioning with user-triggered reconciliation. The user controls when changes flow from timeline editing into vocalization labeling via an explicit refresh action. Deletion is blocked when downstream dependencies exist.

Rejected alternatives:
- **Immutable detection snapshot entity**: Adds a new table and duplicates Parquet files. More schema and storage overhead than justified — can graduate to this later if needed.
- **Sync-on-edit (propagate timeline changes immediately)**: Silently destroys vocalization labeling work. Tightly couples two systems that currently have clean separation.

## Design

### 1. Row Store Version Tracking

Add `row_store_version` integer column to `DetectionJob`, starting at `1` when the detection job completes, incrementing each time `batch_edit_labels` rewrites the Parquet file.

Add `row_store_version_at_import` nullable integer column to `VocalizationLabel`, defaulting to `NULL` for existing labels. When vocalization labels are created via the labeling API, they record the detection job's current `row_store_version`.

A detection job is "stale" from the vocalization perspective when `detection_job.row_store_version > max(vocalization_labels.row_store_version_at_import)` for that job.

### 2. Refresh/Reconciliation

New endpoint: `POST /labeling/vocalization-labels/{detection_job_id}/refresh`

Preview step (no mutations):
1. Read the current detection row store Parquet.
2. Build a set of all current `(start_utc, end_utc)` keys from the row store.
3. Query all VocalizationLabel records for that detection job.
4. Classify each label:
   - **Matched**: `(start_utc, end_utc)` exists in both — no action needed.
   - **Orphaned**: label's `(start_utc, end_utc)` not found in current row store — window was deleted or moved.
5. Return a preview: counts of matched vs orphaned labels, plus orphaned label details (type, timestamps).

Apply step: `POST /labeling/vocalization-labels/{detection_job_id}/refresh/apply`

No request body needed — all orphaned labels are discarded. Then:
1. Delete all orphaned VocalizationLabel records (those whose `(start_utc, end_utc)` no longer exists in the row store).
2. Update `row_store_version_at_import` on all surviving labels to the current version.
3. Return summary of what was deleted and what remains.

No automatic fuzzy matching for moved windows. Moves are rare and the stakes of a wrong match are high. Orphaned labels from moves appear in the review list — the user can manually re-label the moved window.

### 3. Deletion Guards

`delete_detection_job` checks for downstream dependencies before deleting:
1. **VocalizationLabel records**: `SELECT COUNT(*) FROM vocalization_labels WHERE detection_job_id = :id`
2. **TrainingDataset references**: scan `training_datasets.source_config` JSON for the job ID.

If either exists, return HTTP 409 with a message listing what depends on the job, e.g.:
```
Cannot delete detection job: used by 47 vocalization labels and 2 training datasets. Remove these associations first.
```

Bulk delete applies the same check per-job — deletes safe ones, returns 409 for blocked ones with dependency details.

### 4. Staleness Indicators in the UI

**Vocalization labeling workspace header**: When the active detection job has `row_store_version` higher than the labels' `row_store_version_at_import`, show a warning badge: "Detection labels have changed since last import. Refresh to sync." with a Refresh button that triggers the preview endpoint.

**Refresh review dialog**: A modal showing:
- N labels still valid
- N labels orphaned (window deleted/moved)
- List of orphaned labels with type and timestamp
- "Discard orphaned labels & sync" button to apply

No staleness indicator on training dataset views — training datasets are intentionally point-in-time snapshots.

## Schema Changes

### Alembic Migration

**DetectionJob** — add column:
- `row_store_version INTEGER NOT NULL DEFAULT 1`

**VocalizationLabel** — add column:
- `row_store_version_at_import INTEGER` (nullable, NULL for existing labels)

## API Changes

| Method | Path | Description |
|--------|------|-------------|
| POST | `/labeling/vocalization-labels/{detection_job_id}/refresh` | Preview reconciliation (matched/orphaned counts and details) |
| POST | `/labeling/vocalization-labels/{detection_job_id}/refresh/apply` | Apply reconciliation (delete orphans, update version) |
| DELETE | `/classifier/detection-jobs/{job_id}` | Now returns 409 if downstream dependencies exist |
| DELETE | `/classifier/detection-jobs/bulk` | Same per-job guard, partial success possible |

## Files Affected

### Backend
- `src/humpback/models/classifier.py` — `DetectionJob.row_store_version`
- `src/humpback/models/labeling.py` — `VocalizationLabel.row_store_version_at_import`
- `src/humpback/api/routers/classifier.py` — `batch_edit_labels` increments version; delete endpoint adds guard
- `src/humpback/api/routers/labeling.py` — new refresh/apply endpoints; label creation records version
- `src/humpback/services/classifier_service.py` — `delete_detection_job` dependency check
- `alembic/versions/` — new migration for both columns

### Frontend
- Vocalization labeling workspace — staleness badge + refresh button
- Refresh review modal — new component
- Detection job delete — error toast for 409 responses

### Tests
- Reconciliation logic unit tests (matched, orphaned, apply)
- Deletion guard unit tests (blocked, allowed, bulk partial)
- Version increment on `batch_edit_labels`
- Label creation records current version
