# Canonical UTC Detection Identity — Implementation Plan

**Goal:** Replace all detection row identity with canonical `(start_utc, end_utc)` epoch float pairs, eliminating file-relative offsets, derived filenames, and coordinate-system conversions across the entire stack.
**Spec:** [docs/specs/2026-03-28-canonical-utc-detection-identity-design.md](../specs/2026-03-28-canonical-utc-detection-identity-design.md)

---

### Task 1: Core Row Store Schema and Identity

**Files:**
- Modify: `src/humpback/classifier/detection_rows.py`

**Acceptance criteria:**
- [ ] `ROW_STORE_FIELDNAMES` updated: `start_utc`, `end_utc`, `raw_start_utc`, `raw_end_utc` replace `row_id`, `filename`, `start_sec`, `end_sec`, `raw_start_sec`, `raw_end_sec`, `detection_filename`, `extract_filename`
- [ ] `ROW_STORE_SCHEMA` rebuilt from updated fieldnames
- [ ] All `positive_selection_*` and `auto_positive_selection_*` `_start_sec`/`_end_sec` subfields renamed to `_start_utc`/`_end_utc` in fieldname lists and throughout the module
- [ ] `POSITIVE_SELECTION_FIELDNAMES`, `AUTO_POSITIVE_SELECTION_FIELDNAMES`, `MANUAL_POSITIVE_SELECTION_FIELDNAMES` updated
- [ ] `PositiveSelectionResult` dataclass fields renamed from `start_sec`/`end_sec` to `start_utc`/`end_utc`
- [ ] `build_detection_row_id()` deleted
- [ ] `derive_detection_filename()` retained as a utility but renamed to clarify it formats UTC epochs to compact filename strings; signature changes to accept `(start_utc: float, end_utc: float) -> str`
- [ ] `hydrophone_job_relative_to_file_relative_offset()` deleted
- [ ] `build_hydrophone_anchor_filename()` deleted
- [ ] `normalize_detection_row()` rewritten: accepts row with `start_utc`/`end_utc`, no longer derives `detection_filename`/`extract_filename`/`row_id`; validates and parses UTC epoch floats
- [ ] `apply_label_edits()` rewritten: row lookup by `(start_utc, end_utc)` composite key instead of `row_id`; add/move/delete/change_type all use UTC pairs; move edits update `start_utc`/`end_utc` directly
- [ ] `backfill_hydrophone_row_metadata()` deleted or simplified (no longer needs to derive filenames/anchors)
- [ ] `resolve_clip_bounds()` rewritten: reads `start_utc`/`end_utc` directly instead of parsing `detection_filename`
- [ ] `merge_detection_row_store_state()` uses `(start_utc, end_utc)` for matching instead of `row_id`
- [ ] `build_detection_row_store_rows()` updated to produce new-schema rows
- [ ] `read_detection_row_store()` includes lazy migration: detects old schema, derives `start_utc`/`end_utc` from `detection_filename` (primary) or `filename` + offsets (fallback), rewrites atomically
- [ ] `write_detection_row_store()` writes new schema only
- [ ] `stream_detection_rows_as_tsv()` / `iter_detection_rows_as_tsv()` export `start_utc`, `end_utc`, plus derived `detection_filename` for readability
- [ ] `select_positive_window()` and related selection functions use UTC-based offset fields
- [ ] `apply_effective_positive_selection()` uses `_start_utc`/`_end_utc` fields; derives `positive_extract_filename` from UTC pair on-the-fly
- [ ] `compute_auto_selection_update()` produces `_start_utc`/`_end_utc` keyed results
- [ ] Helper functions (`selection_result_to_row_update`, `prefixed_selection_result_to_row_update`) output `_utc` keys

**Tests needed:**
- `read_detection_row_store` lazy migration from old-schema Parquet (hydrophone and local fixtures)
- `apply_label_edits` add/move/delete/change_type with UTC composite keys
- `normalize_detection_row` with new UTC fields
- `merge_detection_row_store_state` matching by UTC pair
- Round-trip: write → read → verify field integrity

---

### Task 2: Detection Workers — Emit UTC Events

**Files:**
- Modify: `src/humpback/classifier/detector.py`
- Modify: `src/humpback/classifier/hydrophone_detector.py`
- Modify: `src/humpback/workers/classifier_worker.py`

**Acceptance criteria:**
- [ ] `detector.py`: `TSV_FIELDNAMES` updated to use `start_utc`/`end_utc` etc.
- [ ] `detector.py`: `run_detection()` computes `start_utc`/`end_utc` for each event by adding offsets to parsed filename timestamp (for timestamp-named files) or file modification time (fallback)
- [ ] `detector.py`: `merge_detection_events()` / `snap_and_merge_detection_events()` produce `start_utc`/`end_utc` and `raw_start_utc`/`raw_end_utc` in UTC epoch seconds
- [ ] `detector.py`: No more `filename` field on events (the worker knows the file but doesn't pass it as row identity)
- [ ] `hydrophone_detector.py`: `run_hydrophone_detection()` computes `start_utc = chunk_epoch + offset` directly; no synthetic filename generation
- [ ] `hydrophone_detector.py`: `_build_detection_filename()` deleted
- [ ] `classifier_worker.py`: `_detection_dicts_to_store_rows()` maps new UTC fields to store rows; no `row_id` generation, no `detection_filename`/`extract_filename` storage

**Tests needed:**
- Local detector emits `start_utc`/`end_utc` for timestamp-named files
- Local detector handles non-timestamp filenames (modification time fallback)
- Hydrophone detector emits `start_utc`/`end_utc` directly from chunk epoch
- `_detection_dicts_to_store_rows` produces valid new-schema rows

---

### Task 3: Extraction and Audio Resolution

**Files:**
- Modify: `src/humpback/classifier/extractor.py`
- Modify: `src/humpback/classifier/s3_stream.py`
- Modify: `src/humpback/workers/search_worker.py`

**Acceptance criteria:**
- [ ] `extractor.py`: `extract_labeled_samples()` and `extract_hydrophone_labeled_samples()` resolve audio from `start_utc`/`end_utc` instead of `detection_filename`/`filename` + offsets
- [ ] `extractor.py`: Output filenames for extracted clips derived from `start_utc`/`end_utc` on-the-fly
- [ ] `extractor.py`: Remove `_parse_compact_range_filename` usage for identity (may keep utility if still needed for legacy file discovery)
- [ ] `s3_stream.py`: `resolve_audio_slice()` accepts `start_utc` (absolute epoch) and `duration_sec` instead of `filename` + `row_start_sec`; no anchor-filename parsing internally
- [ ] `search_worker.py`: `_resolve_audio()` accepts `start_utc`/`end_utc` instead of `filename`/`start_sec`/`end_sec`; no `hydrophone_job_relative_to_file_relative_offset` calls
- [ ] `search_worker.py`: `run_search_job()` reads `job.start_utc`/`job.end_utc` instead of `job.filename`/`job.start_sec`/`job.end_sec`

**Tests needed:**
- Extraction produces correct output filenames from UTC pair
- `resolve_audio_slice` works with absolute epoch input
- Search worker audio resolution with UTC params

---

### Task 4: API Endpoints

**Files:**
- Modify: `src/humpback/api/routers/classifier.py`
- Modify: `src/humpback/api/routers/labeling.py`
- Modify: `src/humpback/schemas/classifier.py`
- Modify: `src/humpback/schemas/labeling.py`

**Acceptance criteria:**
- [ ] `classifier.py`: `_to_job_relative_offsets()` deleted
- [ ] `classifier.py`: `get_detection_content()` returns rows with `start_utc`/`end_utc`, derives `detection_filename` on-the-fly for each row; no offset conversion for hydrophone vs local
- [ ] `classifier.py`: `get_detection_audio_slice()` query params change to `start_utc` + `duration_sec`; internal resolution computes file-relative offset from UTC when needed (encapsulated in resolver)
- [ ] `classifier.py`: `get_detection_spectrogram()` query params change to `start_utc` + `duration_sec`
- [ ] `classifier.py`: `get_detection_embedding()` query params change to `start_utc` + `end_utc`
- [ ] `classifier.py`: `save_detection_labels()` (PUT) request body uses `start_utc`/`end_utc` for row matching
- [ ] `classifier.py`: `batch_edit_labels()` (PATCH) uses `start_utc`/`end_utc` instead of `row_id` in edit items
- [ ] `classifier.py`: `save_detection_row_state()` (PUT) uses `start_utc`/`end_utc` instead of `row_id`
- [ ] `classifier.py`: `_resolve_detection_audio()` accepts `start_utc` + `duration_sec`; resolves file path internally for local jobs
- [ ] `classifier.py`: Audio decode cache key changes to `(job_id, start_utc, duration_sec)`
- [ ] `labeling.py`: `_resolve_detection_embedding_lookup()` uses `start_utc`/`end_utc`; no `hydrophone_job_relative_to_file_relative_offset` calls
- [ ] `labeling.py`: Vocalization label endpoints use `start_utc`/`end_utc` query params instead of `row_id` path param
- [ ] `labeling.py`: Annotation endpoints use `start_utc`/`end_utc` query params instead of `row_id` path param
- [ ] `schemas/classifier.py`: `LabelEditItem` uses `start_utc`/`end_utc` and `new_start_utc`/`new_end_utc`; no `row_id`
- [ ] `schemas/classifier.py`: `DetectionLabelRow` (if defined as Pydantic) uses `start_utc`/`end_utc`
- [ ] `schemas/classifier.py`: `DetectionRowStateUpdate` (if defined as Pydantic) uses `start_utc`/`end_utc`; no `row_id`; manual bounds become `_start_utc`/`_end_utc`
- [ ] `schemas/labeling.py`: `DetectionNeighborsRequest` uses `start_utc`/`end_utc` instead of `filename`/`start_sec`/`end_sec`/`detection_filename`
- [ ] `schemas/labeling.py`: `UncertaintyQueueRow` uses `start_utc`/`end_utc` instead of `row_id`/`filename`/`start_sec`/`end_sec`
- [ ] `schemas/labeling.py`: `PredictionRow` uses `start_utc`/`end_utc` instead of `row_id`
- [ ] `schemas/labeling.py`: `VocalizationLabelOut`, `AnnotationOut` use `start_utc`/`end_utc` instead of `row_id`

**Tests needed:**
- Content endpoint returns `start_utc`/`end_utc`, no `filename`/`start_sec`/`end_sec`
- Audio slice resolves from `start_utc` for both local and hydrophone
- Label PUT matches on `(start_utc, end_utc)`
- Label PATCH add/move/delete/change_type with UTC pairs
- Row-state PUT identifies row by `(start_utc, end_utc)`
- Vocalization label CRUD via `start_utc`/`end_utc` query params
- Annotation CRUD via `start_utc`/`end_utc` query params

---

### Task 5: SQL Schema Migration

**Files:**
- Create: `alembic/versions/028_canonical_utc_detection_identity.py`
- Modify: `src/humpback/models/labeling.py`
- Modify: `src/humpback/models/search.py`

**Acceptance criteria:**
- [ ] Alembic migration `028` uses `op.batch_alter_table()` for SQLite compatibility
- [ ] `search_jobs`: adds `start_utc` (Float), `end_utc` (Float); drops `filename`, `start_sec`, `end_sec`
- [ ] `vocalization_labels`: adds `start_utc` (Float), `end_utc` (Float); drops `row_id`; index updated to `(detection_job_id, start_utc, end_utc)`
- [ ] `labeling_annotations`: adds `start_utc` (Float), `end_utc` (Float); drops `row_id`; index updated to `(detection_job_id, start_utc, end_utc)`
- [ ] Data migration populates new columns from old values where possible (search_jobs from filename parsing; labels/annotations from Parquet row store lookup by row_id)
- [ ] `models/labeling.py`: `VocalizationLabel` and `LabelingAnnotation` models use `start_utc`/`end_utc` instead of `row_id`
- [ ] `models/search.py`: `SearchJob` model uses `start_utc`/`end_utc` instead of `filename`/`start_sec`/`end_sec`
- [ ] Migration runs cleanly on empty and populated databases

**Tests needed:**
- Migration upgrade and downgrade on empty database
- Migration with existing vocalization labels resolves row_id to UTC pair

---

### Task 6: Frontend Types, Client, and Utilities

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/utils/format.ts` (or create utility)

**Acceptance criteria:**
- [ ] `types.ts`: `DetectionRow` removes `row_id`, `filename`, `start_sec`, `end_sec`, `raw_start_sec`, `raw_end_sec`, `detection_filename`, `extract_filename`; adds `start_utc`, `end_utc`, `raw_start_utc`, `raw_end_utc`
- [ ] `types.ts`: `DetectionLabelRow` uses `start_utc`/`end_utc` instead of `filename`/`start_sec`/`end_sec`
- [ ] `types.ts`: `DetectionRowStateUpdate` uses `start_utc`/`end_utc` instead of `row_id`; manual bounds become `_start_utc`/`_end_utc`
- [ ] `types.ts`: `LabelEditItem` uses `start_utc`/`end_utc` and `new_start_utc`/`new_end_utc`; no `row_id`
- [ ] `types.ts`: Vocalization/annotation response types use `start_utc`/`end_utc` instead of `row_id`
- [ ] `client.ts`: `detectionAudioSliceUrl()` takes `(jobId, startUtc, durationSec)` — no `filename` param
- [ ] `client.ts`: `detectionSpectrogramUrl()` takes `(jobId, startUtc, durationSec)` — no `filename` param
- [ ] `client.ts`: `saveDetectionLabels()` sends `start_utc`/`end_utc` per row
- [ ] `client.ts`: `saveDetectionRowState()` sends `start_utc`/`end_utc` instead of `row_id`
- [ ] `client.ts`: `fetchDetectionEmbedding()` sends `start_utc`/`end_utc` instead of `filename`/`start_sec`/`end_sec`
- [ ] `client.ts`: `patchDetectionLabels()` sends UTC-based edit items
- [ ] `client.ts`: `fetchDetectionNeighbors()` sends `start_utc`/`end_utc` instead of `filename`/`start_sec`/`end_sec`/`detection_filename`
- [ ] `client.ts`: Vocalization label and annotation endpoints use `start_utc`/`end_utc` query params instead of `row_id` path param
- [ ] Add `formatDetectionFilename(startUtc: number, endUtc: number): string` utility for derived display name

**Tests needed:**
- TypeScript compilation passes with no errors (`npx tsc --noEmit`)

---

### Task 7: Frontend Components — Classifier Tabs

**Files:**
- Modify: `frontend/src/components/classifier/HydrophoneTab.tsx`
- Modify: `frontend/src/components/classifier/LabelingTab.tsx`
- Modify: `frontend/src/components/search/SearchTab.tsx`

**Acceptance criteria:**
- [ ] `HydrophoneTab.tsx`: Delete `computeUtcRange()` — no longer needed, display derived from `formatDetectionFilename(row.start_utc, row.end_utc)`
- [ ] `HydrophoneTab.tsx`: Delete `rowKey()` — replaced by `${row.start_utc}:${row.end_utc}`
- [ ] `HydrophoneTab.tsx`: `resolveClipTiming()` simplified — reads `start_utc`/`end_utc` directly, no filename parsing
- [ ] `HydrophoneTab.tsx`: `resolvePositiveSelectionMarkerBounds()` and `resolveManualSelectionMarkerBounds()` use `_start_utc`/`_end_utc` fields
- [ ] `HydrophoneTab.tsx`: `handleApplySpectrogramEdit()` sends `start_utc`/`end_utc` instead of `row_id`
- [ ] `HydrophoneTab.tsx`: Audio playback uses `detectionAudioSliceUrl(jobId, row.start_utc, duration)`
- [ ] `HydrophoneTab.tsx`: Label edit buffer keyed by `${start_utc}:${end_utc}` instead of `filename:start_sec:end_sec`
- [ ] `LabelingTab.tsx`: Delete `hydrophoneJobRelativeToFileRelativeOffset()` entirely
- [ ] `LabelingTab.tsx`: Clip timing reads `start_utc`/`end_utc` directly — no offset conversion, no hydrophone/local branching
- [ ] `LabelingTab.tsx`: Vocalization labels queried by `start_utc`/`end_utc` instead of `row_id`
- [ ] `LabelingTab.tsx`: Detection neighbor request sends `start_utc`/`end_utc`
- [ ] `SearchTab.tsx`: Detection-sourced search sends `start_utc`/`end_utc` instead of `filename`/`start_sec`/`end_sec`

**Tests needed:**
- Frontend compiles (`npx tsc --noEmit`)

---

### Task 8: Frontend Components — Timeline Viewer

**Files:**
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`
- Modify: `frontend/src/components/timeline/LabelEditor.tsx`
- Modify: `frontend/src/components/timeline/DetectionOverlay.tsx`
- Modify: `frontend/src/hooks/queries/useLabelEdits.ts`
- Modify: `frontend/src/hooks/queries/useLabeling.ts`

**Acceptance criteria:**
- [ ] `useLabelEdits.ts`: `LabelEdit` interface uses `start_utc`/`end_utc` and `new_start_utc`/`new_end_utc` instead of `row_id`/`start_sec`/`end_sec`
- [ ] `useLabelEdits.ts`: Reducer actions dispatch/match by `(start_utc, end_utc)` instead of `row_id` or UUID `id`
- [ ] `useLabelEdits.ts`: `mergedRows` computation applies edits by matching `(start_utc, end_utc)` on original rows
- [ ] `useLabelEdits.ts`: New adds generate `(start_utc, end_utc)` directly — no client-side UUID
- [ ] `LabelEditor.tsx`: Row keying, drag handling, ghost overlap detection all use `(start_utc, end_utc)`
- [ ] `LabelEditor.tsx`: Bar rendering positions from `start_utc`/`end_utc`
- [ ] `DetectionOverlay.tsx`: React keys use `${row.start_utc}:${row.end_utc}` instead of `row.row_id ?? idx`
- [ ] `DetectionOverlay.tsx`: Tooltip times derived from `start_utc`/`end_utc` directly
- [ ] `TimelineViewer.tsx`: Save handler maps edits with `start_utc`/`end_utc` fields
- [ ] `TimelineViewer.tsx`: Skip navigation uses `row.start_utc`/`row.end_utc` directly
- [ ] `useLabeling.ts`: Query keys use `start_utc`/`end_utc` instead of `rowId`
- [ ] `useLabeling.ts`: API calls pass `start_utc`/`end_utc` query params instead of `rowId` path param

**Tests needed:**
- Frontend compiles (`npx tsc --noEmit`)

---

### Task 9: Update Existing Tests

**Files:**
- Modify: `tests/unit/test_detection_rows.py`
- Modify: `tests/unit/test_detection_spans.py`
- Modify: `tests/unit/test_classifier_worker.py`
- Modify: `tests/unit/test_extractor.py`
- Modify: `tests/unit/test_hydrophone_resume.py`
- Modify: `tests/unit/test_s3_stream.py`
- Modify: `tests/unit/test_search_worker.py`
- Modify: `tests/unit/test_label_batch_endpoint.py`
- Modify: `tests/integration/test_classifier_api.py`
- Modify: `tests/integration/test_hydrophone_api.py`
- Modify: `tests/integration/test_labeling_api.py`

**Acceptance criteria:**
- [ ] All existing detection-related tests updated to use `start_utc`/`end_utc` instead of `filename`/`start_sec`/`end_sec`/`row_id`
- [ ] No references to `row_id`, `filename` (as detection identity), or file-relative `start_sec`/`end_sec` in test assertions
- [ ] Test fixtures produce new-schema rows
- [ ] All tests pass: `uv run pytest tests/`

**Tests needed:**
- This task IS the test update — ensure full suite passes

---

### Task 10: Legacy Migration Tests

**Files:**
- Create: `tests/unit/test_legacy_parquet_migration.py`

**Acceptance criteria:**
- [ ] Test fixture: old-schema Parquet with hydrophone rows (synthetic filename, file-relative offsets, detection_filename present)
- [ ] Test fixture: old-schema Parquet with local rows (timestamp-named files, detection_filename present)
- [ ] Test fixture: old-schema Parquet with local rows (non-timestamp filenames, detection_filename derived)
- [ ] Verify `read_detection_row_store` migrates old schema to new, producing correct `start_utc`/`end_utc`
- [ ] Verify migrated Parquet file is rewritten in new schema on disk
- [ ] Verify second read does NOT re-migrate (already in new schema)
- [ ] Verify rows that cannot derive UTC (no parseable timestamp, no detection_filename) are logged and skipped

**Tests needed:**
- This task IS the test creation

---

### Task 11: Playwright Tests

**Files:**
- Modify: `frontend/e2e/hydrophone-utc-timezone.spec.ts`
- Modify: `frontend/e2e/hydrophone-canceled-job.spec.ts`
- Modify: `frontend/e2e/hydrophone-active-queue.spec.ts`
- Modify: `frontend/e2e/detection-spectrogram.spec.ts`

**Acceptance criteria:**
- [ ] Playwright tests updated to match new API response shapes (no `row_id`/`filename`/`start_sec` in assertions)
- [ ] Any URL assertions for audio-slice or spectrogram endpoints use `start_utc` param instead of `filename`+`start_sec`
- [ ] Tests pass: `cd frontend && npx playwright test`

**Tests needed:**
- This task IS the Playwright test update

---

### Task 12: Documentation Updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `DECISIONS.md`
- Modify: `README.md`

**Acceptance criteria:**
- [ ] `CLAUDE.md` §4.6 (Hydrophone Detection TSV Metadata): updated field names
- [ ] `CLAUDE.md` §8.3 (Data Model Summary): VocalizationLabel, LabelingAnnotation, SearchJob descriptions updated
- [ ] `CLAUDE.md` §8.5 (Storage Layout): detection file descriptions updated
- [ ] `CLAUDE.md` §9.2: latest migration updated to 028
- [ ] `DECISIONS.md`: new ADR for canonical UTC detection identity
- [ ] `README.md`: API endpoint documentation updated if detection endpoints are listed

**Tests needed:**
- None (documentation only)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/ tests/ scripts/`
2. `uv run ruff check src/humpback/ tests/ scripts/`
3. `uv run pyright src/humpback/ tests/ scripts/`
4. `uv run alembic upgrade head`
5. `uv run pytest tests/`
6. `cd frontend && npx tsc --noEmit`
7. `cd frontend && npx playwright test` (if backend running with test data)
