# Canonical UTC Detection Identity — Design Spec (Phased)

**Date:** 2026-03-28
**Status:** Approved
**Approach:** Big-bang internal, phased external — all phases on one feature branch, merged together at the end

## Problem

Detection event identity is scattered across multiple coordinate systems:

1. **Storage** stores file-relative `start_sec`/`end_sec` offsets (seconds into a chunk audio file)
2. **API** converts to job-relative offsets for hydrophone jobs (`_to_job_relative_offsets`)
3. **Frontend** converts back to file-relative for playback URLs (`hydrophoneJobRelativeToFileRelativeOffset`)
4. **`detection_filename`** encodes absolute UTC ranges but is treated as derived, not primary
5. **`row_id`** is a SHA-1 hash of `filename|detection_filename|start_sec|end_sec`
6. **Label save endpoints** use two keying strategies: `filename+start_sec+end_sec` (PUT) vs `row_id` (PATCH)

Every layer boundary requires error-prone offset conversions, and hydrophone vs local jobs diverge in identity handling throughout the stack.

## Design

### Canonical Identity

Every detection event is identified by `(start_utc, end_utc)` — float epoch seconds. This pair is unique within a detection job, enforced by existing overlap validation. The full identity is `(job_id, start_utc, end_utc)` where `job_id` is the partition (URL path, Parquet file path).

### Fields Removed from Parquet Row Store

- `row_id` — replaced by the `(start_utc, end_utc)` composite key
- `filename` — chunk anchor, no longer needed for identity or resolution
- `start_sec` / `end_sec` — file-relative offsets, replaced by `start_utc`/`end_utc`
- `detection_filename` / `extract_filename` — derived on-the-fly from UTC pair
- `raw_start_sec` / `raw_end_sec` — replaced by `raw_start_utc`/`raw_end_utc`

### Fields Added

- `start_utc` (float, epoch seconds) — canonical event start
- `end_utc` (float, epoch seconds) — canonical event end
- `raw_start_utc` (float, epoch seconds) — pre-snap audit boundary
- `raw_end_utc` (float, epoch seconds) — pre-snap audit boundary

### Fields Retained (unchanged)

- `avg_confidence`, `peak_confidence`, `n_windows`, `merged_event_count`
- `hydrophone_name`
- Label fields: `humpback`, `orca`, `ship`, `background`
- All `positive_selection_*` and `auto_positive_selection_*` fields (their `_start_sec`/`_end_sec` subfields become `_start_utc`/`_end_utc`)

### Derived Values (computed on-the-fly, never stored)

- `detection_filename`: `format_compact_utc(start_utc) + "_" + format_compact_utc(end_utc) + ".flac"`
- `extract_filename`: same derivation
- Duration: `end_utc - start_utc`

### Composite Key Semantics

- Within a job, `(start_utc, end_utc)` is unique (overlap validation enforces this)
- API endpoints reference rows by `start_utc` + `end_utc` query/body params
- Move edits change the key (the event *is* its time range)
- Frontend React keys: `${start_utc}:${end_utc}`

## Detection Workers

### Local Detection

Workers compute `start_utc`/`end_utc` at detection time:
- For timestamp-named files (e.g., `20240115T123000Z.wav`): `start_utc = chunk_epoch + offset`
- For non-timestamp files: use file modification time as anchor; if unavailable, use `0.0 + offset` as degenerate case
- `raw_start_utc`/`raw_end_utc` computed from pre-snap boundaries using same anchor

### Hydrophone Detection

Direct UTC computation: `start_utc = chunk_epoch + window_start_offset`. The synthetic filename, `_build_detection_filename()`, and anchor-based offset math are eliminated.

### Event Merging

Hysteresis merge shifts from file-relative to UTC epoch arithmetic (same operations, different coordinate system).

## API Layer

### Eliminated Code

- `_to_job_relative_offsets()` — deleted entirely
- `hydrophone_job_relative_to_file_relative_offset()` — deleted entirely
- All hydrophone/local branching for offset conversion in endpoints

### Endpoint Changes

| Endpoint | Old Params | New Params |
|----------|-----------|------------|
| `GET /content` | `filename`, `start_sec`, `end_sec`, `row_id` in response | `start_utc`, `end_utc` in response |
| `GET /audio-slice` | `filename`, `start_sec`, `duration_sec` | `start_utc`, `duration_sec` (or `end_utc`) |
| `GET /spectrogram` | `filename`, `start_sec`, `duration_sec` | `start_utc`, `duration_sec` |
| `GET /embedding` | `filename`, `start_sec`, `end_sec` | `start_utc`, `end_utc` |
| `PUT /labels` | `DetectionLabelRow{filename, start_sec, end_sec}` | `DetectionLabelRow{start_utc, end_utc}` |
| `PATCH /labels` | `LabelEditItem{row_id, start_sec, ...}` | `LabelEditItem{start_utc, end_utc, ...}` |
| `PUT /row-state` | `DetectionRowStateUpdate{row_id, ...}` | `DetectionRowStateUpdate{start_utc, end_utc, ...}` |

### Audio Resolution

- **Local:** API resolves file + offset internally: `offset = start_utc - file_anchor_epoch`. File-relative math is encapsulated in the resolver, never exposed.
- **Hydrophone:** `resolve_audio_slice()` receives absolute timestamps directly.

### TSV Download

Columns: `start_utc`, `end_utc`, `detection_filename` (derived for readability), confidence fields, `raw_start_utc`, `raw_end_utc`, `merged_event_count`, `hydrophone_name`, labels, positive selection fields.

## Frontend

### Type Changes

`DetectionRow` removes `row_id`, `filename`, `start_sec`, `end_sec`, `raw_start_sec`, `raw_end_sec`, `detection_filename`, `extract_filename`. Adds `start_utc`, `end_utc`, `raw_start_utc`, `raw_end_utc`.

### Deleted Code

- `computeUtcRange()` in HydrophoneTab
- `hydrophoneJobRelativeToFileRelativeOffset()` in LabelingTab
- `parseDetectionFilenameRange()`
- `rowKey()` function
- All `if (job.hydrophone_id)` branches for offset conversion

### Unified Patterns

- React keys: `${start_utc}:${end_utc}`
- Audio URLs: `detectionAudioSliceUrl(jobId, start_utc, duration)`
- Label edits: `start_utc`/`end_utc` identify rows (no `row_id`)
- Timeline positioning: `start_utc - job.start_timestamp` for pixel offset

## Legacy Parquet Migration

Lazy migration on first read in `read_detection_row_store()`:

1. Detect old schema by checking for `start_utc` column absence
2. For each row, derive UTC values:
   - Primary: parse `detection_filename` to epoch floats
   - Fallback: parse `filename` timestamp + add offsets
3. Convert positive selection `_start_sec`/`_end_sec` to `_start_utc`/`_end_utc`
4. Rewrite atomically in new schema (one-way migration)

Window diagnostics Parquet files remain unchanged.

## SQL Schema Migration (Alembic 028)

Three SQL tables reference detection row identity and need migration:

### `search_jobs` table
- Remove: `filename` (str), `start_sec` (float), `end_sec` (float)
- Add: `start_utc` (float), `end_utc` (float)

### `vocalization_labels` table
- Remove: `row_id` (str)
- Add: `start_utc` (float), `end_utc` (float)
- Update index: `ix_vocalization_labels_job_row` to cover `(detection_job_id, start_utc, end_utc)`

### `labeling_annotations` table
- Remove: `row_id` (str)
- Add: `start_utc` (float), `end_utc` (float)
- Update index: `ix_labeling_annotations_job_row` to cover `(detection_job_id, start_utc, end_utc)`

Data migration for existing rows:
- `vocalization_labels` and `labeling_annotations`: look up the corresponding detection row store Parquet to map old `row_id` to `(start_utc, end_utc)`. Rows whose `row_id` can't be resolved are deleted (orphaned labels).
- `search_jobs`: derive `start_utc`/`end_utc` from `filename` + `start_sec`/`end_sec` using the same anchor-parsing logic as Parquet migration. Ephemeral jobs (deleted after results returned) are unlikely to need migration, but the columns still change.

### API Route Changes (Labeling)
- `/vocalization-labels/{detection_job_id}/{row_id}` → `/vocalization-labels/{detection_job_id}?start_utc=...&end_utc=...`
- `/annotations/{detection_job_id}/{row_id}` → `/annotations/{detection_job_id}?start_utc=...&end_utc=...`
- `DetectionNeighborsRequest`: `filename`/`start_sec`/`end_sec`/`detection_filename` → `start_utc`/`end_utc`
- `UncertaintyQueueRow`, `PredictionRow`: `row_id` → `start_utc`/`end_utc`

## Phasing Strategy

### Approach

Big-bang internal, phased external: the Parquet schema changes fully in Phase 1, but the API contract and frontend aren't updated until Phases 3-4. The app may not work end-to-end between phases. All phases live on one feature branch and are merged together via a single PR.

### Resume Protocol

Each phase has a checklist in the implementation plan. Each completed phase gets its own commit with a structured prefix (e.g., `phase-1: core schema + detection workers`). To resume after a context clear:

1. Read the implementation plan, find the first unchecked phase
2. Run `uv run pytest tests/` to confirm prior phases' tests pass
3. Pick up from the unchecked phase

### Phase Overview

| Phase | Scope | Verification Gate |
|-------|-------|-------------------|
| 1 | Core schema (`detection_rows.py`), detection workers (`detector.py`, `hydrophone_detector.py`), classifier worker, search worker, s3_stream | Unit tests for all modified modules pass |
| 2 | Extractor (`extractor.py`) | Extractor unit tests pass + Phase 1 tests still pass |
| 3 | API routes (`classifier.py`, `labeling.py`), schemas, SQL models, Alembic 028 | Full `pytest` suite passes, `alembic upgrade head` succeeds |
| 4 | Frontend types, client, components, Playwright tests | `npx tsc --noEmit` passes, Playwright tests pass |
| 5 | CLAUDE.md, DECISIONS.md, dead code grep, cleanup | All linters + full test suite + `tsc --noEmit` clean |

## Testing

Each phase includes unit/integration tests for the code changed in that phase:

- **Phase 1:** `test_detection_rows.py`, `test_detection_spans.py`, `test_classifier_worker.py`, `test_s3_stream.py`, `test_search_worker.py`, `test_hydrophone_resume.py`, new `test_legacy_parquet_migration.py`
- **Phase 2:** `test_extractor.py`
- **Phase 3:** `test_label_batch_endpoint.py`, `test_classifier_api.py`, `test_labeling_api.py`
- **Phase 4:** Playwright e2e tests (hydrophone-utc-timezone, hydrophone-canceled-job, hydrophone-active-queue, detection-spectrogram)
- **Phase 5:** Lint/type-check verification only (no new tests)
