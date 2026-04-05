# Tiling Window Selection Implementation Plan

**Goal:** Add a `"tiling"` window selection mode that covers high-scoring regions by tiling outward from peaks until a logit-space drop threshold is exceeded.
**Spec:** [docs/specs/2026-04-04-tiling-window-selection-design.md](../specs/2026-04-04-tiling-window-selection-design.md)

---

### Task 1: Alembic migration for `max_logit_drop` column

**Files:**
- Create: `alembic/versions/039_max_logit_drop_column.py`

**Acceptance criteria:**
- [ ] Adds `max_logit_drop` Float nullable column to `detection_jobs`
- [ ] Uses `op.batch_alter_table()` for SQLite compatibility
- [ ] `uv run alembic upgrade head` succeeds

---

### Task 2: Database model and schema updates

**Files:**
- Modify: `src/humpback/models/classifier.py`
- Modify: `src/humpback/schemas/classifier.py`

**Acceptance criteria:**
- [ ] `DetectionJob` model has `max_logit_drop = Column(Float, nullable=True)`
- [ ] `DetectionJobCreate` extends `window_selection` Literal to include `"tiling"`
- [ ] `DetectionJobCreate` adds `max_logit_drop: Optional[float] = None` with `> 0` validator
- [ ] `HydrophoneDetectionCreate` has matching fields

---

### Task 3: Core tiling algorithm

**Files:**
- Modify: `src/humpback/classifier/detector_utils.py`

**Acceptance criteria:**
- [ ] New `select_tiled_windows_from_events()` function following the greedy peak-then-tile algorithm from the spec
- [ ] Reuses `_to_logit()` for score transformation
- [ ] Multi-pass: seeds from highest uncovered logit, tiles left/right until `max_logit_drop` exceeded
- [ ] Deduplication across adjacent events (same pattern as prominence)
- [ ] Returns detection dicts with same shape as existing modes (start_sec, end_sec, avg_confidence, peak_confidence, audit fields)

---

### Task 4: Wire tiling into detection pipeline

**Files:**
- Modify: `src/humpback/classifier/detector.py`
- Modify: `src/humpback/classifier/hydrophone_detector.py`
- Modify: `src/humpback/services/classifier_service/detection.py`
- Modify: `src/humpback/services/classifier_service/hydrophone.py`
- Modify: `src/humpback/workers/classifier_worker/detection.py`
- Modify: `src/humpback/workers/classifier_worker/hydrophone.py`
- Modify: `src/humpback/api/routers/classifier/detection.py`
- Modify: `src/humpback/api/routers/classifier/hydrophone.py`

**Acceptance criteria:**
- [ ] `window_selection="tiling"` dispatches to `select_tiled_windows_from_events()`
- [ ] `max_logit_drop` flows from API request through service/worker to the selection function
- [ ] Follows the same dispatch pattern as prominence (if/elif on `window_selection`)

---

### Task 5: Unit tests for tiling algorithm

**Files:**
- Modify: `tests/unit/test_detection_spans.py`

**Acceptance criteria:**
- [ ] Basic tiling: single peak with symmetric drop-off, verify correct span
- [ ] Multi-pass: two peaks separated by deep dip, verify two tile groups
- [ ] Plateau: flat high scores, all windows emitted
- [ ] Edge clipping: tiling stops at event boundary
- [ ] Boundary condition: drop exactly at threshold is included (`<=`)
- [ ] No qualifying windows: returns empty
- [ ] Deduplication across adjacent events

**Tests needed:**
- All tests use `select_tiled_windows_from_events()` directly with synthetic window records and events

---

### Task 6: Frontend — tiling option in HydrophoneTab

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/components/classifier/HydrophoneTab.tsx`

**Acceptance criteria:**
- [ ] `"tiling"` added to window selection type/dropdown
- [ ] `max_logit_drop` number input appears when tiling selected
- [ ] `max_logit_drop` included in detection job creation request
- [ ] Same conditional display pattern as `min_prominence` for prominence

---

### Task 7: Extend validation script

**Files:**
- Modify: `scripts/validate_gap_filling.py`

**Acceptance criteria:**
- [ ] Accepts optional `--mode` and `--max-logit-drop` CLI args
- [ ] When `--mode tiling`, re-runs with `select_tiled_windows_from_events()` instead of prominence
- [ ] Side-by-side output includes tiling column when applicable
- [ ] Default behavior (no args) unchanged

---

### Task 8: Documentation updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `DECISIONS.md`
- Modify: `docs/reference/data-model.md`

**Acceptance criteria:**
- [ ] CLAUDE.md 8.7 behavioral constraints: add tiling mode description to window selection modes section
- [ ] CLAUDE.md 9.2: migration count updated to 039
- [ ] DECISIONS.md: new ADR-045 for tiling window selection
- [ ] Data model reference: `max_logit_drop` field documented

---

### Verification

Run in order after all tasks:
1. `uv run alembic upgrade head`
2. `uv run ruff format --check src/humpback/classifier/detector_utils.py src/humpback/classifier/detector.py src/humpback/classifier/hydrophone_detector.py src/humpback/models/classifier.py src/humpback/schemas/classifier.py src/humpback/services/classifier_service/detection.py src/humpback/services/classifier_service/hydrophone.py src/humpback/workers/classifier_worker/detection.py src/humpback/workers/classifier_worker/hydrophone.py src/humpback/api/routers/classifier/detection.py src/humpback/api/routers/classifier/hydrophone.py scripts/validate_gap_filling.py`
3. `uv run ruff check src/humpback/ scripts/validate_gap_filling.py`
4. `uv run pyright src/humpback/ scripts/validate_gap_filling.py`
5. `uv run pytest tests/`
6. `cd frontend && npx tsc --noEmit`
