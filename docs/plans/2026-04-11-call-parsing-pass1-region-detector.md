# Call Parsing — Pass 1: Region Detector Implementation Plan

**Goal:** Turn the Phase 0 Pass 1 stub into a real region-detection worker that handles both uploaded audio files and 24-hour hydrophone ranges, producing `trace.parquet` + `regions.parquet` via a streaming scoring loop.
**Spec:** [docs/specs/2026-04-11-call-parsing-pass1-region-detector-design.md](../specs/2026-04-11-call-parsing-pass1-region-detector-design.md)
**Branch:** `feature/call-parsing-pass1-region-detector`

---

## Task ordering

Tasks 1–3 are additive and independent — they can be built in any order. Tasks 4 and 5 depend on tasks 1–3. Task 6 (API router) depends on task 2. Tasks 7–9 are documentation and can run after everything else is green.

---

### Task 1: Migration 043 — Pass 1 source columns

**Files:**
- Create: `alembic/versions/043_call_parsing_pass1_source_columns.py`

**Acceptance criteria:**
- [ ] Alembic migration uses `op.batch_alter_table()` for SQLite compatibility
- [ ] `call_parsing_runs`: drops `audio_source_id`, adds `audio_file_id` (String, nullable), `hydrophone_id` (String, nullable), `start_timestamp` (Float, nullable), `end_timestamp` (Float, nullable)
- [ ] `region_detection_jobs`: same set of column changes
- [ ] `downgrade()` restores `audio_source_id` as nullable String (not NOT NULL, to avoid requiring a backfill) and drops the four new source columns
- [ ] SQLAlchemy models in `src/humpback/models/call_parsing.py` updated to match (drop the `audio_source_id` attribute, add the four new attributes)
- [ ] `uv run alembic upgrade head` applies cleanly on a fresh DB and on a snapshot DB that already has Phase 0 migration 042 applied

**Tests needed:**
- Migration round-trip test (fresh upgrade + downgrade + upgrade) in `tests/test_migrations.py` (or equivalent) covering both tables.

---

### Task 2: Pydantic schemas and service layer

**Files:**
- Modify: `src/humpback/schemas/call_parsing.py`
- Modify: `src/humpback/services/call_parsing.py`

**Acceptance criteria:**
- [ ] New `RegionDetectionConfig` Pydantic model with fields `window_size_seconds=5.0`, `hop_seconds=1.0`, `high_threshold=0.70`, `low_threshold=0.45`, `padding_sec=1.0`, `min_region_duration_sec=0.0`, `stream_chunk_sec=1800.0`
- [ ] New `CreateRegionJobRequest` Pydantic model with optional `audio_file_id` / `hydrophone_id` / `start_timestamp` / `end_timestamp`, required `model_config_id` / `classifier_model_id`, optional `parent_run_id`, and nested `config: RegionDetectionConfig`
- [ ] `@model_validator(mode="after")` on `CreateRegionJobRequest` enforces: exactly one of (`audio_file_id`) or (all of `hydrophone_id`, `start_timestamp`, `end_timestamp`), and `end_timestamp > start_timestamp` when the hydrophone branch is active
- [ ] New service method `create_region_job(session, request) -> RegionDetectionJob` — validates the four FK lookups (404 on missing), serializes `request.config.model_dump_json()` into `config_json`, inserts a queued row, commits, returns the model
- [ ] `create_parent_run` updated to accept the same source + model + config fields; inserts the parent row, then calls `create_region_job` with `parent_run_id` set, all in one transaction

**Tests needed:**
- `RegionDetectionConfig` default values
- `CreateRegionJobRequest` validator: both sources present rejected, neither present rejected, `end_timestamp <= start_timestamp` rejected, valid file-source accepted, valid hydrophone-source accepted
- `create_region_job` happy path + 404-on-missing-FK cases (audio_file_id, hydrophone_id, model_config_id, classifier_model_id)
- `create_parent_run` creates parent + Pass 1 child atomically with `parent_run_id` linked

---

### Task 3: Region decoder module

**Files:**
- Create: `src/humpback/call_parsing/regions.py`
- Create: `tests/unit/test_call_parsing_regions.py`

**Acceptance criteria:**
- [ ] `decode_regions(events, audio_duration_sec, config) -> list[Region]` is a pure function: no I/O, no models, no audio
- [ ] Accepts the dict-shaped events `merge_detection_events` returns: keys `start_sec`, `end_sec`, `avg_confidence`, `peak_confidence`, `n_windows`
- [ ] Applies the algorithm specified in the spec: sort → compute padded bounds → left-to-right merge on padded-bounds overlap → filter by `min_region_duration_sec` → assign `region_id = uuid4().hex` → return sorted by `start_sec`
- [ ] Merge semantics: `max_score` = max of inputs, `mean_score` = window-weighted average `sum(mean_i * n_windows_i) / sum(n_windows_i)`, `n_windows` = sum, raw bounds = min/max of raw bounds, padded bounds = min/max of padded bounds
- [ ] Padded bounds always clamped to `[0.0, audio_duration_sec]`
- [ ] Handles empty input and single-event input without errors

**Tests needed:**
- Empty input returns empty list
- Single event returns one region, no merging, bounds clamped correctly
- Two adjacent events whose padded bounds exactly touch fuse (inclusive boundary)
- Two events whose padded bounds do NOT overlap stay separate
- Event at `start_sec=0` produces `padded_start=0.0`
- Event at `end_sec=audio_duration_sec` produces `padded_end=audio_duration_sec`
- `min_region_duration_sec > 0` correctly drops short regions
- Three events where (1,2) merge but (3) stays standalone — merge cascades correctly
- Weighted-mean-score correctness on a known input (e.g. events with `n_windows=3,7` and `mean_score=0.6,0.9` → merged `mean_score = (0.6*3 + 0.9*7) / 10 = 0.81`)
- Output `region_id` values are unique UUID4 hex strings

---

### Task 4: Detector refactor — expose `score_audio_windows`

**Files:**
- Modify: `src/humpback/classifier/detector.py`
- Modify: `tests/unit/test_detector_refactor.py` (or wherever the Phase 0 snapshot test lives)
- Create: `tests/unit/test_score_audio_windows_chunking.py`

**Acceptance criteria:**
- [ ] New public function `score_audio_windows(audio, sample_rate, perch_model, classifier, config, time_offset_sec=0.0) -> list[dict[str, Any]]` in `detector.py`
- [ ] Returned window records shift `offset_sec` and `end_sec` by `time_offset_sec` so callers streaming audio can concatenate records into a single absolute-time trace
- [ ] `compute_hysteresis_events` is re-implemented as a two-call composition: `score_audio_windows(...)` + `merge_detection_events(window_records, high, low)` — its public signature and return shape are unchanged
- [ ] The Phase 0 `tests/fixtures/detector_refactor_snapshot.json` snapshot assertion still passes
- [ ] A new assertion verifies `compute_hysteresis_events(...)` equals the explicit `score_audio_windows` + `merge_detection_events` composition to float64 precision on the same fixture
- [ ] Public-API surface: `score_audio_windows` appears in `detector.__all__` (if the module uses one) and is importable as `from humpback.classifier.detector import score_audio_windows`

**Tests needed:**
- Snapshot equivalence test (existing, extended) proves the refactor is bit-identical
- New chunking test: split a fixture audio buffer in half at a whole-window boundary, call `score_audio_windows` twice with appropriate `time_offset_sec`, concatenate the results, and assert the concatenated list equals a single `score_audio_windows` call on the whole buffer to float64 precision

---

### Task 5: Region detection worker

**Files:**
- Modify: `src/humpback/workers/region_detection_worker.py`
- Modify: `src/humpback/workers/queue.py`
- Create: `tests/integration/test_region_detection_worker.py`

**Acceptance criteria:**
- [ ] Worker claims a queued `RegionDetectionJob` via the atomic compare-and-set pattern already used elsewhere
- [ ] Deserializes `job.config_json` into `RegionDetectionConfig`
- [ ] Resolves the audio source: file path via the existing `AudioLoader` for `audio_file_id`, archive playback provider for hydrophone ranges (`hydrophone_id` + `start_timestamp` + `end_timestamp`)
- [ ] Loads Perch via `get_model_by_version(session, cm.model_version, settings)` and the classifier via `joblib.load(cm.model_path)`, matching the existing detection worker
- [ ] File path: one call to `score_audio_windows` on the full buffer
- [ ] Hydrophone path: streaming loop over chunks aligned to multiples of `window_size_seconds`, each chunk at most `stream_chunk_sec` wide, per-chunk `score_audio_windows` call with `time_offset_sec = chunk_start - range_start`, per-chunk extend of the in-memory trace list, explicit `del audio_buf` after each chunk to release memory
- [ ] `_aligned_chunk_edges` helper computes chunk boundaries so no Perch window ever straddles two chunks
- [ ] Runs `merge_detection_events` once on the full concatenated trace
- [ ] Runs `decode_regions` once on the events
- [ ] Writes `trace.parquet` and `regions.parquet` via `call_parsing.storage` atomic helpers
- [ ] Updates `trace_row_count`, `region_count`, `completed_at`, `status='complete'`
- [ ] On exception: deletes partial `trace.parquet`, `regions.parquet`, and any `.tmp` sidecars under the job directory; sets `status='failed'` and populates `error_message`
- [ ] `RegionDetectionJob` is added to the stale-job recovery sweep in `queue.py` so a killed worker's row resets to `queued` after the existing timeout

**Tests needed:**
- Worker integration test with a short fixture audio file (~60 s), a mock Perch embedding model (seeded-random output), and a mock binary classifier (deterministic `predict_proba`): create an `audio_file_id`-source job, run one worker iteration, assert `status='complete'`, both parquet files exist, row counts match, bounds clamped, at least one region exists
- Failure path: stub the classifier to raise, run one worker iteration, assert `status='failed'`, `error_message` populated, both parquet files absent
- Stale recovery: insert a `RegionDetectionJob` row with `status='running'` and a stale `updated_at`, run the recovery sweep, assert `status='queued'`
- Hydrophone-path integration test deferred — add a TODO in `docs/plans/backlog.md`

---

### Task 6: API router — unstub Pass 1 endpoints

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `tests/api/test_call_parsing_router.py`

**Acceptance criteria:**
- [ ] `POST /call-parsing/region-jobs` accepts `CreateRegionJobRequest`, calls `create_region_job`, returns the new row. The 501 stub is removed.
- [ ] `GET /call-parsing/region-jobs/{id}/trace` reads `trace.parquet` from the job directory and streams its rows as JSON (`{time_sec, score}` objects). Returns 409 when the job is not `complete`, 404 when the parquet file is missing.
- [ ] `GET /call-parsing/region-jobs/{id}/regions` reads `regions.parquet` and returns `list[Region]` sorted by `start_sec`. Returns 409 when the job is not `complete`, 404 when missing.
- [ ] `POST /call-parsing/runs` accepts the same source + model + config fields as `CreateRegionJobRequest` (either by reusing the model or a parallel `CreateParentRunRequest`), creates the parent row + Pass 1 child in one transaction via the updated `create_parent_run`

**Tests needed:**
- `POST /region-jobs` happy path with valid `audio_file_id`
- `422` on bad source payloads: both source kinds present, neither present, `end_timestamp <= start_timestamp`
- `404` on unknown `audio_file_id` / `classifier_model_id`
- `409` on `GET /trace` and `GET /regions` while job status is `queued` / `running`
- `GET /trace` and `GET /regions` happy path after running the worker synchronously in-test
- `DELETE /{id}` cascades the parquet directory cleanup (assert directory gone)
- `POST /runs` with the new source + config shape creates the parent + Pass 1 child atomically

---

### Task 7: ADR-049

**Files:**
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] New ADR-049 entry appended to `DECISIONS.md` following the existing format (Date, Status, Context, Decision, Consequences)
- [ ] Captures, with rationale: symmetric `padding_sec=1.0`, padded-overlap-only merge rule (no `merge_gap_sec`), no temporal smoothing, `min_region_duration_sec=0.0` default, dense raw trace no decimation, delete-and-restart crash semantics, `score_audio_windows`/`merge_detection_events` split and chunk-aligned streaming

---

### Task 8: Documentation updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/reference/data-model.md`
- Modify: `README.md` (if it lists endpoints — otherwise no-op)
- Modify: `docs/plans/backlog.md` (if present — otherwise create it with the deferred test entry)

**Acceptance criteria:**
- [ ] CLAUDE.md §8.9 — `POST /call-parsing/region-jobs`, `GET /region-jobs/{id}/trace`, `GET /region-jobs/{id}/regions` marked as functional; 501 callouts removed
- [ ] CLAUDE.md §8.7 — new behavioral-constraint bullets for (a) Pass 1 source contract (`audio_file_id` XOR hydrophone range), (b) chunk-alignment streaming rule, (c) delete-and-restart on crash
- [ ] CLAUDE.md §9.1 — "Pass 1 region detection" appended to the Implemented Capabilities list, and the call-parsing scaffold line updated
- [ ] CLAUDE.md §9.2 — latest migration bumped to `043_call_parsing_pass1_source_columns.py`
- [ ] `docs/reference/data-model.md` — column lists for `call_parsing_runs` and `region_detection_jobs` updated to reflect the dropped `audio_source_id` and the four new source columns
- [ ] Deferred hydrophone integration test noted in `docs/plans/backlog.md`

---

### Task 9: Smoke test

**Files:**
- Modify: whichever existing smoke test file covers the end-to-end API flow (or create a new one under `tests/api/`)

**Acceptance criteria:**
- [ ] New smoke-test scenario: create an `audio_file` record, create a binary classifier model record (mock paths), create a Perch model config record, `POST /call-parsing/region-jobs` with the `audio_file_id` source, drive the worker one iteration, `GET /{id}/trace` and `GET /{id}/regions`, `DELETE /{id}` and assert cleanup
- [ ] Test uses the same mock Perch + mock classifier setup from the worker integration test

---

## Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/ tests/ alembic/versions/043_call_parsing_pass1_source_columns.py`
2. `uv run ruff check src/humpback/ tests/ alembic/versions/043_call_parsing_pass1_source_columns.py`
3. `uv run pyright src/humpback/call_parsing src/humpback/classifier/detector.py src/humpback/workers/region_detection_worker.py src/humpback/workers/queue.py src/humpback/services/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/api/routers/call_parsing.py tests/`
4. `uv run alembic upgrade head`
5. `uv run pytest tests/`
6. Manual: `POST /call-parsing/region-jobs` with a real uploaded `audio_file_id`, confirm the worker completes it, `GET /region-jobs/{id}/regions` returns a non-empty list, and `trace.parquet`/`regions.parquet` exist under the job directory
