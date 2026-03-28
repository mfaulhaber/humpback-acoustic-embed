# Canonical UTC Detection Identity — Phased Implementation Plan

**Goal:** Replace all detection row identity with canonical `(start_utc, end_utc)` epoch float pairs, eliminating file-relative offsets, derived filenames, and coordinate-system conversions across the entire stack.
**Spec:** [docs/specs/2026-03-28-canonical-utc-detection-identity-design.md](../specs/2026-03-28-canonical-utc-detection-identity-design.md)
**Approach:** Big-bang internal, phased external. All phases on one feature branch, merged together at the end. The app may not work end-to-end between phases.

---

## Resume Protocol

Each completed phase gets a commit with prefix `phase-N: <description>`. To resume after a context clear:

1. Read this plan — find the first unchecked phase
2. Run `uv run pytest tests/` to confirm prior phases' tests pass
3. Read the spec for design reference
4. Read `memory/project_utc_refactor_report.md` for the dependency graph and complexity notes
5. Pick up from the first unchecked phase

---

## Phase 1: Core Schema + Detection Workers

**Commit:** `phase-1: core schema + detection workers`

### Task 1.1: Row Store Schema and Identity (`detection_rows.py`)

**Files:**
- Modify: `src/humpback/classifier/detection_rows.py`

**Acceptance criteria:**
- [ ] `ROW_STORE_FIELDNAMES` updated: `start_utc`, `end_utc`, `raw_start_utc`, `raw_end_utc` replace `row_id`, `filename`, `start_sec`, `end_sec`, `raw_start_sec`, `raw_end_sec`, `detection_filename`, `extract_filename`
- [ ] `ROW_STORE_SCHEMA` rebuilt from updated fieldnames
- [ ] `PositiveSelectionResult` dataclass: `start_sec`/`end_sec` → `start_utc`/`end_utc`
- [ ] `POSITIVE_SELECTION_FIELDNAMES`: `positive_selection_start_sec`/`_end_sec` → `_start_utc`/`_end_utc`
- [ ] `AUTO_POSITIVE_SELECTION_FIELDNAMES`: `auto_positive_selection_start_sec`/`_end_sec` → `_start_utc`/`_end_utc`
- [ ] `MANUAL_POSITIVE_SELECTION_FIELDNAMES`: `manual_positive_selection_start_sec`/`_end_sec` → `_start_utc`/`_end_utc`
- [ ] `build_detection_row_id()` deleted
- [ ] `derive_detection_filename()` signature changed to `(start_utc: float, end_utc: float) -> str`
- [ ] `hydrophone_job_relative_to_file_relative_offset()` deleted
- [ ] `build_hydrophone_anchor_filename()` deleted
- [ ] `normalize_detection_row()` rewritten: accepts row with `start_utc`/`end_utc`, validates UTC epoch floats, no longer derives `detection_filename`/`extract_filename`/`row_id`
- [ ] `apply_label_edits()` rewritten: row lookup by `(start_utc, end_utc)` composite key; add/move/delete/change_type all use UTC pairs; move edits update `start_utc`/`end_utc` directly
- [ ] `backfill_hydrophone_row_metadata()` deleted or simplified (no filename/anchor derivation)
- [ ] `resolve_clip_bounds()` rewritten: reads `start_utc`/`end_utc` directly
- [ ] `merge_detection_row_store_state()` uses `(start_utc, end_utc)` for matching
- [ ] `build_detection_row_store_rows()` produces new-schema rows
- [ ] `read_detection_row_store()` includes lazy migration: detects old schema (absence of `start_utc` column), derives UTC from `detection_filename` (primary) or `filename` + offsets (fallback), rewrites atomically
- [ ] `write_detection_row_store()` writes new schema only
- [ ] `stream_detection_rows_as_tsv()` / `iter_detection_rows_as_tsv()` export `start_utc`, `end_utc`, plus derived `detection_filename` for readability
- [ ] `select_positive_window()` and related selection functions use `_start_utc`/`_end_utc` offset fields
- [ ] `apply_effective_positive_selection()` uses `_start_utc`/`_end_utc` fields; derives `positive_extract_filename` from UTC pair on-the-fly
- [ ] `compute_auto_selection_update()` produces `_start_utc`/`_end_utc` keyed results
- [ ] `selection_result_to_row_update()` and `prefixed_selection_result_to_row_update()` output `_utc` keys
- [ ] `append_detection_row_store()` writes new schema
- [ ] `ensure_detection_row_store()` creates new schema header

**Tests needed:**
- `read_detection_row_store` lazy migration from old-schema Parquet (hydrophone fixture with `detection_filename`, local fixture with `filename` + offsets)
- `apply_label_edits` add/move/delete/change_type with UTC composite keys
- `normalize_detection_row` with new UTC fields
- `merge_detection_row_store_state` matching by UTC pair
- Round-trip: write → read → verify field integrity
- `derive_detection_filename(start_utc, end_utc)` produces correct compact UTC format

### Task 1.2: Local Detector UTC Emission (`detector.py`)

**Files:**
- Modify: `src/humpback/classifier/detector.py`

**Acceptance criteria:**
- [ ] `_file_base_epoch(filepath)` inline helper added: parses timestamp-named files to epoch float; falls back to file mtime; last resort `0.0`
- [ ] `merge_detection_events()` produces `start_utc`/`end_utc` and `raw_start_utc`/`raw_end_utc` by adding offsets to base epoch
- [ ] `snap_and_merge_detection_events()` produces same UTC fields
- [ ] `merge_detection_spans()` internal function keeps file-relative `start_sec`/`end_sec` (not exposed externally, only consumed by `merge_detection_events`)
- [ ] `run_detection()` passes filepath to `_file_base_epoch()` for UTC computation
- [ ] `select_peak_windows_from_events()` updated if it references old fields

**Tests needed:**
- `merge_detection_events` output contains `start_utc`/`end_utc`, no `filename`
- `snap_and_merge_detection_events` output contains UTC fields + `raw_start_utc`/`raw_end_utc`
- `_file_base_epoch` with timestamp-named file, non-timestamp file, missing file

### Task 1.3: Hydrophone Detector UTC Emission (`hydrophone_detector.py`)

**Files:**
- Modify: `src/humpback/classifier/hydrophone_detector.py`

**Acceptance criteria:**
- [ ] `_build_detection_filename()` deleted
- [ ] `run_hydrophone_detection()` computes `start_utc = chunk_epoch + offset` directly for each event
- [ ] Events yield `start_utc`/`end_utc` and `raw_start_utc`/`raw_end_utc` in UTC epoch seconds
- [ ] No synthetic filename generation; `hydrophone_name` still populated

**Tests needed:**
- `run_hydrophone_detection` output events contain UTC fields, no `detection_filename`
- Verify `hydrophone_name` still present in output

### Task 1.4: Classifier Worker Store Row Conversion (`classifier_worker.py`)

**Files:**
- Modify: `src/humpback/workers/classifier_worker.py`

**Acceptance criteria:**
- [ ] `_detection_dicts_to_store_rows()` maps new UTC fields to store rows; no `row_id` generation, no `detection_filename`/`extract_filename` storage
- [ ] Remove `build_detection_row_id` import
- [ ] Remove `is_hydrophone` branching in store row conversion (both paths now produce same UTC fields)
- [ ] Any remaining `format_optional_float`/`format_optional_int` calls updated for new field names

**Tests needed:**
- `_detection_dicts_to_store_rows` produces rows with `start_utc`/`end_utc`, no `row_id`
- Round-trip: detection event → store row → write → read back

### Task 1.5: Search Worker + S3 Stream (`search_worker.py`, `s3_stream.py`)

**Files:**
- Modify: `src/humpback/workers/search_worker.py`
- Modify: `src/humpback/classifier/s3_stream.py`

**Acceptance criteria:**
- [ ] `search_worker.py`: uses `start_utc`/`end_utc` from job, no offset conversion via `hydrophone_job_relative_to_file_relative_offset`
- [ ] `s3_stream.py`: `resolve_audio_slice()` accepts `start_utc` directly (parameter rename, no coordinate conversion needed)

**Tests needed:**
- `test_search_worker.py` updated for new job field names
- `test_s3_stream.py` updated for `resolve_audio_slice` signature

### Task 1.6: Phase 1 Unit Tests

**Files:**
- Modify: `tests/unit/test_detection_rows.py`
- Modify: `tests/unit/test_detection_spans.py`
- Modify: `tests/unit/test_classifier_worker.py`
- Modify: `tests/unit/test_s3_stream.py`
- Modify: `tests/unit/test_search_worker.py`
- Modify: `tests/unit/test_hydrophone_resume.py`
- Create: `tests/unit/test_legacy_parquet_migration.py`

**Acceptance criteria:**
- [ ] All existing test assertions updated to use `start_utc`/`end_utc` instead of `row_id`/`filename`/`start_sec`/`end_sec`
- [ ] All fixture rows updated to new schema
- [ ] `test_legacy_parquet_migration.py` covers: old hydrophone schema → new, old local schema → new, round-trip integrity, positive selection field migration
- [ ] All Phase 1 tests pass: `uv run pytest tests/unit/test_detection_rows.py tests/unit/test_detection_spans.py tests/unit/test_classifier_worker.py tests/unit/test_s3_stream.py tests/unit/test_search_worker.py tests/unit/test_hydrophone_resume.py tests/unit/test_legacy_parquet_migration.py`

### Phase 1 Verification

```bash
uv run pytest tests/unit/test_detection_rows.py tests/unit/test_detection_spans.py tests/unit/test_classifier_worker.py tests/unit/test_s3_stream.py tests/unit/test_search_worker.py tests/unit/test_hydrophone_resume.py tests/unit/test_legacy_parquet_migration.py -v
```

---

## Phase 2: Extractor

**Commit:** `phase-2: extractor UTC conversion`

### Task 2.1: Extractor UTC Conversion (`extractor.py`)

**Files:**
- Modify: `src/humpback/classifier/extractor.py`

**Acceptance criteria:**
- [ ] `PositiveSelectionResult` dataclass: `start_sec`/`end_sec` → `start_utc`/`end_utc`
- [ ] `POSITIVE_SELECTION_FIELDNAMES` local copy: `_start_sec`/`_end_sec` → `_start_utc`/`_end_utc`
- [ ] Row grouping for hydrophone extraction uses `start_utc` ranges against timeline metadata instead of grouping by `filename`
- [ ] `_resolve_local_audio_for_row()` — new or refactored internal helper that maps `start_utc` back to `(file_path, offset_sec)` for local extraction using anchor parsing (same logic as `_file_base_epoch` in `detector.py`)
- [ ] Hydrophone extraction path accepts `start_utc`/`end_utc` directly from rows
- [ ] `backfill_hydrophone_row_metadata()` usage removed or simplified (no filename/anchor derivation)
- [ ] Extract filename derivation: on-the-fly from `(start_utc, end_utc)` via `derive_detection_filename()`
- [ ] All positive selection field access uses `_start_utc`/`_end_utc` suffixes
- [ ] `_build_absolute_window_records()` in classifier_worker.py (if referenced by extractor) still converts diagnostics to absolute UTC correctly

### Task 2.2: Phase 2 Unit Tests

**Files:**
- Modify: `tests/unit/test_extractor.py`

**Acceptance criteria:**
- [ ] All fixture rows updated to new schema (no `filename`, `start_sec`, `end_sec`, `row_id`)
- [ ] Positive selection field names use `_start_utc`/`_end_utc`
- [ ] Extract filename derivation tests use UTC pair input
- [ ] Local audio resolution tests verify `start_utc` → `(file_path, offset)` mapping
- [ ] All Phase 2 tests pass
- [ ] Phase 1 tests still pass (regression check)

### Phase 2 Verification

```bash
uv run pytest tests/unit/test_extractor.py tests/unit/test_detection_rows.py tests/unit/test_detection_spans.py tests/unit/test_classifier_worker.py tests/unit/test_s3_stream.py tests/unit/test_search_worker.py tests/unit/test_hydrophone_resume.py tests/unit/test_legacy_parquet_migration.py -v
```

---

## Phase 3: API Layer + SQL Migration

**Commit:** `phase-3: API layer + SQL migration`

### Task 3.1: Classifier API Routes (`classifier.py`)

**Files:**
- Modify: `src/humpback/api/routers/classifier.py`

**Acceptance criteria:**
- [ ] `_to_job_relative_offsets()` deleted entirely
- [ ] `build_hydrophone_anchor_filename` and `hydrophone_job_relative_to_file_relative_offset` imports removed
- [ ] Content endpoint (`GET /content`): returns `start_utc`/`end_utc` directly, no hydrophone/local branching for offset conversion
- [ ] Audio-slice endpoint: accepts `start_utc` query param instead of `filename` + `start_sec`
- [ ] Spectrogram endpoint: accepts `start_utc` query param instead of `filename` + `start_sec`
- [ ] Embedding endpoint: accepts `start_utc`/`end_utc` instead of `filename`/`start_sec`/`end_sec`
- [ ] Label PUT: `DetectionLabelRow` keyed by `start_utc`/`end_utc`
- [ ] Label PATCH (batch): `LabelEditItem` uses `start_utc`/`end_utc` (no `row_id`)
- [ ] Row-state PUT: `DetectionRowStateUpdate` keyed by `start_utc`/`end_utc`
- [ ] `_resolve_detection_audio()` accepts `start_utc` — for local jobs, resolves file + offset internally; for hydrophone jobs, passes to `resolve_audio_slice()` directly
- [ ] TSV download: columns include `start_utc`, `end_utc`, derived `detection_filename`

### Task 3.2: Labeling API Routes (`labeling.py`)

**Files:**
- Modify: `src/humpback/api/routers/labeling.py`

**Acceptance criteria:**
- [ ] `hydrophone_job_relative_to_file_relative_offset` import removed
- [ ] Vocalization label CRUD routes: `/{row_id}` path param → `?start_utc=&end_utc=` query params
- [ ] Annotation CRUD routes: `/{row_id}` path param → `?start_utc=&end_utc=` query params
- [ ] `_resolve_detection_embedding_lookup` offset conversion deleted
- [ ] Prediction/uncertainty endpoints return `start_utc`/`end_utc` instead of `row_id`
- [ ] Neighbors request uses `start_utc`/`end_utc`

### Task 3.3: Pydantic Schemas

**Files:**
- Modify: `src/humpback/schemas/classifier.py`
- Modify: `src/humpback/schemas/labeling.py`

**Acceptance criteria:**
- [ ] `LabelEditItem`: `row_id`/`start_sec`/`end_sec` → `start_utc`/`end_utc`
- [ ] `DetectionLabelRow`: `filename`/`start_sec`/`end_sec` → `start_utc`/`end_utc`
- [ ] `DetectionRowStateUpdate`: `row_id` → `start_utc`/`end_utc`
- [ ] `VocalizationLabelOut`: `row_id` → `start_utc`/`end_utc`
- [ ] `AnnotationOut`: `row_id` → `start_utc`/`end_utc`
- [ ] `DetectionNeighborsRequest`: `filename`/`start_sec`/`end_sec`/`detection_filename` → `start_utc`/`end_utc`
- [ ] `UncertaintyQueueRow`, `PredictionRow`: `row_id` → `start_utc`/`end_utc`

### Task 3.4: SQL Models + Alembic Migration

**Files:**
- Modify: `src/humpback/models/labeling.py`
- Modify: `src/humpback/models/search.py`
- Create: `alembic/versions/028_canonical_utc_detection_identity.py`

**Acceptance criteria:**
- [ ] `VocalizationLabel`: `row_id` column → `start_utc` (Float) + `end_utc` (Float); index updated to `(detection_job_id, start_utc, end_utc)`
- [ ] `LabelingAnnotation`: `row_id` column → `start_utc` (Float) + `end_utc` (Float); index updated to `(detection_job_id, start_utc, end_utc)`
- [ ] `SearchJob`: `filename`/`start_sec`/`end_sec` → `start_utc`/`end_utc`
- [ ] Alembic 028: uses `batch_alter_table()` for SQLite compatibility
- [ ] Alembic 028: data migration for `vocalization_labels` and `labeling_annotations` maps old `row_id` to `(start_utc, end_utc)` via Parquet row store lookup; orphaned rows deleted
- [ ] Alembic 028: data migration for `search_jobs` derives UTC from `filename` + `start_sec`/`end_sec`
- [ ] `uv run alembic upgrade head` succeeds on a fresh database and on an existing database with data

### Task 3.5: Search Service

**Files:**
- Modify: `src/humpback/services/search_service.py`

**Acceptance criteria:**
- [ ] Search hit building uses `start_utc`/`end_utc` instead of `filename`/`start_sec`/`end_sec`

### Task 3.6: Phase 3 Integration Tests

**Files:**
- Modify: `tests/unit/test_label_batch_endpoint.py`
- Modify: `tests/integration/test_classifier_api.py`
- Modify: `tests/integration/test_labeling_api.py`

**Acceptance criteria:**
- [ ] `test_label_batch_endpoint.py`: edit items use `start_utc`/`end_utc`, no `row_id`
- [ ] `test_classifier_api.py`: response shape assertions updated (no `row_id`/`filename`/`start_sec`/`end_sec` in content responses), request bodies use UTC params, URL params use `start_utc`
- [ ] `test_labeling_api.py`: route params use query `start_utc`/`end_utc` instead of path `row_id`, response shapes updated
- [ ] Full test suite passes: `uv run pytest tests/`

### Phase 3 Verification

```bash
uv run alembic upgrade head
uv run pytest tests/ -v
uv run ruff check src/humpback/api/ src/humpback/schemas/ src/humpback/models/ src/humpback/services/
uv run pyright src/humpback/api/ src/humpback/schemas/ src/humpback/models/ src/humpback/services/
```

---

## Phase 4: Frontend

**Commit:** `phase-4: frontend UTC identity`

### Task 4.1: Types and API Client

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`

**Acceptance criteria:**
- [ ] `DetectionRow` type: remove `row_id`, `filename`, `start_sec`, `end_sec`, `raw_start_sec`, `raw_end_sec`, `detection_filename`, `extract_filename`; add `start_utc`, `end_utc`, `raw_start_utc`, `raw_end_utc`
- [ ] `DetectionLabelRow`, `DetectionRowStateUpdate`, `LabelEditItem` types updated to use UTC fields
- [ ] All detection API functions in `client.ts` updated: audio-slice URLs, spectrogram URLs, label save, row-state, embedding, neighbors, vocalization/annotation endpoints use UTC params
- [ ] `detectionAudioSliceUrl` and `detectionSpectrogramUrl` accept `start_utc` instead of `filename`/`start_sec`

### Task 4.2: Components

**Files:**
- Modify: `frontend/src/components/classifier/HydrophoneTab.tsx`
- Modify: `frontend/src/components/classifier/LabelingTab.tsx`
- Modify: `frontend/src/components/search/SearchTab.tsx`
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`
- Modify: `frontend/src/components/timeline/LabelEditor.tsx`
- Modify: `frontend/src/components/timeline/DetectionOverlay.tsx`
- Modify: `frontend/src/hooks/queries/useLabelEdits.ts`
- Modify: `frontend/src/hooks/queries/useLabeling.ts`

**Acceptance criteria:**
- [ ] `HydrophoneTab.tsx`: delete `computeUtcRange`, `rowKey`, offset conversion. Simplify `resolveClipTiming` to use `start_utc`/`end_utc` directly.
- [ ] `LabelingTab.tsx`: delete `hydrophoneJobRelativeToFileRelativeOffset`. Direct UTC usage.
- [ ] `SearchTab.tsx`: detection-sourced search sends `start_utc`/`end_utc` fields
- [ ] `TimelineViewer.tsx`: save handler and skip navigation use UTC
- [ ] `LabelEditor.tsx`: row keying and drag handling use `(start_utc, end_utc)` pair
- [ ] `DetectionOverlay.tsx`: React keys use `${start_utc}:${end_utc}`
- [ ] `useLabelEdits.ts`: edit dispatch/match by UTC pair
- [ ] `useLabeling.ts`: query keys and API calls use UTC
- [ ] All `if (job.hydrophone_id)` branches for offset conversion deleted throughout

### Task 4.3: Playwright Tests

**Files:**
- Modify: `frontend/e2e/hydrophone-utc-timezone.spec.ts`
- Modify: `frontend/e2e/hydrophone-canceled-job.spec.ts`
- Modify: `frontend/e2e/hydrophone-active-queue.spec.ts`
- Modify: `frontend/e2e/detection-spectrogram.spec.ts`

**Acceptance criteria:**
- [ ] Response shape assertions updated (no `row_id`, fields use `start_utc`/`end_utc`)
- [ ] URL param assertions use `start_utc` instead of `filename`/`start_sec`
- [ ] `npx tsc --noEmit` passes
- [ ] `npx playwright test` passes

### Phase 4 Verification

```bash
cd frontend && npx tsc --noEmit
cd frontend && npx playwright test
```

---

## Phase 5: Docs + Cleanup

**Commit:** `phase-5: docs + cleanup`

### Task 5.1: Documentation Updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] `CLAUDE.md` §4.6: TSV metadata fields updated to `start_utc`/`end_utc`, remove `detection_filename`/`extract_filename` from stored fields, add as derived
- [ ] `CLAUDE.md` §8.3: data model summary updated for `VocalizationLabel`, `LabelingAnnotation`, `SearchJob` — `row_id`/`filename` → `start_utc`/`end_utc`
- [ ] `CLAUDE.md` §8.5: storage layout references updated if needed
- [ ] `CLAUDE.md` §9.2: migration number → `028_canonical_utc_detection_identity.py`
- [ ] `DECISIONS.md`: new ADR for canonical UTC detection identity (date, context, decision, consequences)

### Task 5.2: Dead Code Cleanup

**Acceptance criteria:**
- [ ] Grep for `row_id`, `start_sec`, `end_sec`, `detection_filename`, `extract_filename`, `raw_start_sec`, `raw_end_sec` across production code — no stale references remain (test fixtures and comments excepted)
- [ ] Grep for `_to_job_relative_offsets`, `build_detection_row_id`, `build_hydrophone_anchor_filename`, `hydrophoneJobRelativeToFileRelativeOffset`, `computeUtcRange`, `parseDetectionFilenameRange` — all deleted
- [ ] No orphaned imports

### Phase 5 Verification

```bash
uv run ruff format --check src/humpback/ scripts/
uv run ruff check src/humpback/ scripts/
uv run pyright src/humpback/ scripts/ tests/
uv run pytest tests/ -v
cd frontend && npx tsc --noEmit
```

---

## Final Verification (All Phases)

Run in order after all phases complete:

1. `uv run ruff format --check src/humpback/ scripts/ tests/`
2. `uv run ruff check src/humpback/ scripts/ tests/`
3. `uv run pyright src/humpback/ scripts/ tests/`
4. `uv run pytest tests/ -v`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
7. `uv run alembic upgrade head` (on fresh DB)
