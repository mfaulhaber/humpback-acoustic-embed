# Stable Row ID Refactor

**Date:** 2026-04-02
**Status:** Approved

## Problem

Detection windows are identified by their `(start_utc, end_utc)` composite key across three independent stores: the detection row store (Parquet), detection embeddings (Parquet), and vocalization labels (database). When timeline editing shifts a row's timestamps, the other stores retain the original values, causing timestamp drift. This drift requires tolerance-based matching (`_SYNC_TOLERANCE_SEC = 0.5`), version tracking (`row_store_version` / `row_store_version_at_import`), and an orphan reconciliation system (`refresh_preview` / `refresh_apply`) to keep everything consistent.

The fundamental issue: the identity of a detection window is also the mutable data. Moving a window changes its identity, breaking all references.

See `docs/report/2026-04-02-vocalization-label-timestamp-drift.md` for the detailed root cause analysis.

## Design

### Stable Row ID

Each detection row store entry gets a `row_id` field: a UUID4 string generated at creation time. This becomes the stable identity for a detection window. The row store remains the single source of truth for `(start_utc, end_utc)` — these are data fields, no longer identity.

- **Format**: UUID4 string, consistent with the rest of the codebase.
- **Assignment**: Generated in `_detection_dicts_to_store_rows()` (initial detection) and `apply_label_edits()` (for `add` actions).
- **Edit operations**: `move`, `delete`, and `change_type` in the batch edit API reference `row_id` instead of `(start_utc, end_utc)`.

### Embedding Schema

Detection embedding parquet simplifies from `(filename, start_sec, end_sec, embedding, confidence)` to `(row_id, embedding, confidence)`.

- Sync logic (`diff_row_store_vs_embeddings`) becomes a set comparison of row IDs. No tolerance matching, no filename parsing.
- Embedding generation for a missing row resolves audio from the row store's `(start_utc, end_utc)` and the detection job's audio source.
- `_SYNC_TOLERANCE_SEC` and all tolerance-based matching removed.

### Vocalization Label Schema

The `VocalizationLabel` database model changes:

- **Add**: `row_id` (string, required) — references the detection row store entry.
- **Remove**: `start_utc`, `end_utc`, `row_store_version_at_import`.
- Result: `(id, detection_job_id, row_id, label, confidence, source, notes)`.

The `DetectionJob` model:

- **Remove**: `row_store_version` — no longer needed.

**Cascade behavior**: Deleting a detection row (via timeline `delete` action) deletes associated vocalization labels for that `row_id`.

**Removed code**:
- `refresh_preview` / `refresh_apply` endpoints.
- `_matches_any_row` tolerance matching.
- `_LABEL_MATCH_TOLERANCE_SEC` constant.
- `row_store_version` increment logic in batch edit.
- Frontend orphan warning UI and refresh buttons.

### Vocalization Inference

Inference results parquet adopts `row_id` instead of `(start_utc, end_utc)`:

- **Output schema**: `(row_id, type_scores...)`.
- **Worker**: Loads embeddings by `row_id`, runs model, writes predictions keyed by `row_id`.
- **Labeling workspace**: Resolves current `(start_utc, end_utc)` from the row store by `row_id` at display time — always shows current positions.
- **Label creation**: Stores `row_id` directly.

Re-running inference after timeline edits is never necessary for correctness.

### Data Migration

Migrate forward: assign row IDs to all existing data, rewrite parquets and labels.

**Database (Alembic)**:
- Add `row_id` column to `vocalization_labels`.
- Remove `start_utc`, `end_utc`, `row_store_version_at_import` from `vocalization_labels`.
- Remove `row_store_version` from `detection_jobs`.
- For existing labels: match each label's `(start_utc, end_utc)` against its detection job's row store using tolerance matching to assign `row_id`. Unmatched labels deleted as true orphans.

**Row store parquet**:
- Read each detection job's row store, assign UUID to each row, rewrite atomically.

**Embedding parquet**:
- Match existing entries to row store via old filename+offset to UTC to tolerance method. Rewrite with `row_id` replacing `(filename, start_sec, end_sec)`. Unmatched embeddings dropped (regenerated on next sync).

**Inference output parquet**:
- Map existing `(start_utc, end_utc)` to `row_id` via row store lookup. Unmatched predictions dropped.

Old tolerance/matching code retained only for migration, then removed.

Detection jobs without a row store file (failed, canceled, or in-progress) are skipped during migration — they have no rows to assign IDs to, and no downstream references to fix.

### Testing

- **Row store**: Row IDs assigned on creation, preserved through move/change_type edits, new IDs for add actions.
- **Embedding sync**: Set-based diff by row_id — missing, orphaned, matched cases.
- **Vocalization labels**: Labels reference row_id, cascade delete when row removed, no timestamp storage.
- **Inference**: Predictions keyed by row_id, labeling workspace resolves current timestamps from row store.
- **Migration**: One-time script against fixture data — row ID assignment, label matching, embedding rewrite, orphan cleanup.
- **Remove**: All tolerance-matching tests, orphan refresh tests, row_store_version tests.

## What Gets Removed

| Component | Location |
|-----------|----------|
| `_SYNC_TOLERANCE_SEC` | `detector.py` |
| `_LABEL_MATCH_TOLERANCE_SEC` | `labeling.py` |
| `_matches_any_row` | `labeling.py` |
| `refresh_preview` endpoint | `labeling.py` |
| `refresh_apply` endpoint | `labeling.py` |
| `row_store_version` field | `DetectionJob` model |
| `row_store_version_at_import` field | `VocalizationLabel` model |
| `start_utc` / `end_utc` fields | `VocalizationLabel` model |
| `filename` / `start_sec` / `end_sec` fields | Detection embedding parquet |
| `start_utc` / `end_utc` columns | Inference output parquet |
| Tolerance-based matching in embedding sync | `detector.py` |
| UTC recomputation from filename in inference | `vocalization_worker.py` |
| Frontend orphan warning UI + refresh buttons | Frontend labeling components |
| Version increment in batch edit | `classifier.py` |
