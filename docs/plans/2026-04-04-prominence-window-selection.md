# Prominence Window Selection Implementation Plan

**Goal:** Add a `"prominence"` window selection mode to windowed detection that finds distinct vocalizations via score-peak prominence, allowing overlapping detection windows.
**Spec:** [docs/specs/2026-04-04-prominence-window-selection-design.md](../specs/2026-04-04-prominence-window-selection-design.md)

---

### Task 1: Alembic Migration — Add window_selection and min_prominence columns

**Files:**
- Create: `alembic/versions/038_window_selection_columns.py`

**Acceptance criteria:**
- [ ] Adds nullable `window_selection` (String) column to `detection_jobs`
- [ ] Adds nullable `min_prominence` (Float) column to `detection_jobs`
- [ ] Uses `op.batch_alter_table()` for SQLite compatibility
- [ ] Migration runs cleanly on existing database (`uv run alembic upgrade head`)
- [ ] Existing rows remain unaffected (NULL values)

**Tests needed:**
- Migration up/down verified manually

---

### Task 2: SQLAlchemy Model — Add columns to DetectionJob

**Files:**
- Modify: `src/humpback/models/classifier.py`

**Acceptance criteria:**
- [ ] `window_selection` mapped column (Optional[str], default=None)
- [ ] `min_prominence` mapped column (Optional[float], default=None)
- [ ] Column order follows existing pattern (after `detection_mode`)

**Tests needed:**
- Covered by existing model instantiation tests and migration

---

### Task 3: Pydantic Schemas — Add fields to create/output schemas

**Files:**
- Modify: `src/humpback/schemas/classifier.py`

**Acceptance criteria:**
- [ ] `DetectionJobCreate`: add optional `window_selection` (literal "nms"/"prominence", default None) and `min_prominence` (float, default None)
- [ ] `HydrophoneDetectionJobCreate`: same two fields
- [ ] `DetectionJobOut`: add `window_selection` (Optional[str]) and `min_prominence` (Optional[float])
- [ ] Validation: `min_prominence` must be > 0 when provided
- [ ] Validation: `window_selection` must be one of "nms", "prominence" when provided

**Tests needed:**
- Schema validation tests for valid and invalid values

---

### Task 4: Prominence Peak Selection Algorithm

**Files:**
- Modify: `src/humpback/classifier/detector_utils.py`

**Acceptance criteria:**
- [ ] New function `select_prominent_peaks_from_events()` alongside existing `select_peak_windows_from_events()`
- [ ] Accepts same signature as NMS version plus `min_prominence` parameter
- [ ] Smooths scores with 3-window moving average for peak finding
- [ ] Finds local maxima in smoothed scores (>= both neighbors, or >= one neighbor at edges)
- [ ] Computes prominence using raw (unsmoothed) scores — peak raw score minus highest valley between peak and nearest higher neighbor
- [ ] Filters by `min_prominence` (default 0.03) and `min_score`
- [ ] Emits a 5-second window dict for each surviving peak (same dict shape as NMS output)
- [ ] Allows overlapping windows
- [ ] Preserves audit fields from parent event (raw_start_sec, raw_end_sec, merged_event_count, filename, etc.)
- [ ] Includes deduplication for shared peaks across adjacent events (same as NMS)

**Tests needed:**
- Clear peaks separated by deep valley — both detected
- Subtle peak with prominence just above threshold — detected
- Subtle peak with prominence just below threshold — filtered
- Plateau region (constant high scores) — single peak emitted
- Single peak in event — one window emitted
- No peaks above min_score — empty result
- Overlapping windows emitted when peaks are < 5 seconds apart
- Edge peaks (first/last window in event) handled correctly
- Raw scores used for prominence, not smoothed scores (test with values where smoothing would blur a dip below threshold)

---

### Task 5: Wire Prominence Mode into Detectors

**Files:**
- Modify: `src/humpback/classifier/detector.py`
- Modify: `src/humpback/classifier/hydrophone_detector.py`

**Acceptance criteria:**
- [ ] `run_detection()` accepts `window_selection` and `min_prominence` parameters
- [ ] When `detection_mode="windowed"` and `window_selection="prominence"`, calls `select_prominent_peaks_from_events()` instead of `select_peak_windows_from_events()`
- [ ] When `window_selection` is None or `"nms"`, existing NMS behavior unchanged
- [ ] `min_prominence` defaults to 0.03 when not provided and prominence mode is active
- [ ] `run_hydrophone_detection()` accepts and forwards same parameters
- [ ] Both local and hydrophone detection paths tested

**Tests needed:**
- Integration test: run_detection with window_selection="prominence" produces overlapping windows from synthetic score data
- Integration test: run_detection with window_selection="nms" (or None) produces same output as before

---

### Task 6: Service and Worker Layer — Pass Parameters Through

**Files:**
- Modify: `src/humpback/services/classifier_service/detection.py`
- Modify: `src/humpback/services/classifier_service/hydrophone.py`
- Modify: `src/humpback/workers/classifier_worker/detection.py`
- Modify: `src/humpback/workers/classifier_worker/hydrophone.py`

**Acceptance criteria:**
- [ ] Detection service creates jobs with `window_selection` and `min_prominence` from request
- [ ] Hydrophone service creates jobs with same fields
- [ ] Detection worker reads `window_selection` and `min_prominence` from job record and passes to `run_detection()`
- [ ] Hydrophone worker reads same fields and passes to `run_hydrophone_detection()`
- [ ] Fields included in run summary metadata

**Tests needed:**
- Covered by end-to-end detection job tests

---

### Task 7: API Router — Accept New Parameters

**Files:**
- Modify: `src/humpback/api/routers/classifier/detection.py`

**Acceptance criteria:**
- [ ] Local detection creation endpoint passes `window_selection` and `min_prominence` from request body to service
- [ ] Hydrophone detection creation endpoint passes same fields
- [ ] Job detail and list endpoints return the new fields via `DetectionJobOut`

**Tests needed:**
- API test: create detection job with window_selection="prominence" and min_prominence=0.03
- API test: create without window_selection defaults to nms behavior

---

### Task 8: Frontend — Window Selection Toggle

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/components/classifier/HydrophoneTab.tsx`

**Acceptance criteria:**
- [ ] `DetectionJob` type gains `window_selection` and `min_prominence` fields
- [ ] Hydrophone creation form adds window selection toggle (NMS / Prominence)
- [ ] `min_prominence` input appears when prominence is selected, with default 0.03
- [ ] Toggle and input values included in job creation API call
- [ ] Job detail display shows window_selection and min_prominence when set

**Tests needed:**
- TypeScript compiles cleanly (`npx tsc --noEmit`)

---

### Task 9: Unit Tests for Overlapping Windows in Downstream Systems

**Files:**
- Modify: `tests/unit/test_detection_spans.py`
- Modify: `tests/unit/test_detection_rows.py`

**Acceptance criteria:**
- [ ] Test that overlapping detection rows (sharing time range) can be written to and read from Parquet row store
- [ ] Test that overlapping rows each get unique row_id
- [ ] Test that labeling operations work on overlapping rows independently
- [ ] Test prominence vs NMS side-by-side on same synthetic score curve, verify prominence finds more peaks in dense regions

**Tests needed:**
- All acceptance criteria above are tests

---

### Task 10: Documentation Updates

**Files:**
- Modify: `CLAUDE.md` (§8.7 behavioral constraints, §9.1 capabilities, §9.2 schema)
- Modify: `DECISIONS.md` (new ADR for prominence window selection)
- Modify: `docs/reference/data-model.md`

**Acceptance criteria:**
- [ ] ADR documents the design rationale: why prominence over NMS reduction, raw scores for prominence, 0.03 default
- [ ] CLAUDE.md §8.7 notes window_selection parameter and its effect on detection behavior
- [ ] CLAUDE.md §9.1 mentions prominence window selection as a capability
- [ ] CLAUDE.md §9.2 lists migration 038
- [ ] Data model reference updated with new columns

**Tests needed:**
- None (documentation only)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/classifier/detector_utils.py src/humpback/classifier/detector.py src/humpback/classifier/hydrophone_detector.py src/humpback/models/classifier.py src/humpback/schemas/classifier.py src/humpback/services/classifier_service/detection.py src/humpback/services/classifier_service/hydrophone.py src/humpback/workers/classifier_worker/detection.py src/humpback/workers/classifier_worker/hydrophone.py src/humpback/api/routers/classifier/detection.py`
2. `uv run ruff check src/humpback/classifier/detector_utils.py src/humpback/classifier/detector.py src/humpback/classifier/hydrophone_detector.py src/humpback/models/classifier.py src/humpback/schemas/classifier.py src/humpback/services/classifier_service/detection.py src/humpback/services/classifier_service/hydrophone.py src/humpback/workers/classifier_worker/detection.py src/humpback/workers/classifier_worker/hydrophone.py src/humpback/api/routers/classifier/detection.py`
3. `uv run pyright src/humpback/classifier/detector_utils.py src/humpback/classifier/detector.py src/humpback/classifier/hydrophone_detector.py src/humpback/models/classifier.py src/humpback/schemas/classifier.py src/humpback/services/classifier_service/detection.py src/humpback/services/classifier_service/hydrophone.py src/humpback/workers/classifier_worker/detection.py src/humpback/workers/classifier_worker/hydrophone.py src/humpback/api/routers/classifier/detection.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
