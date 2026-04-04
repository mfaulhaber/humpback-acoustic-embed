# Logit-Space Prominence Implementation Plan

**Goal:** Transform prominence computation from probability space to logit space so that inter-vocalization dips in high-confidence regions produce sufficient prominence for detection.
**Spec:** [docs/specs/2026-04-04-logit-prominence-design.md](../specs/2026-04-04-logit-prominence-design.md)

---

### Task 1: Add logit transform and update prominence computation in detector_utils.py

**Files:**
- Modify: `src/humpback/classifier/detector_utils.py`

**Acceptance criteria:**
- [ ] Add a `_to_logit` helper that converts a probability to log-odds with epsilon clamping (1e-7)
- [ ] `select_prominent_peaks_from_events()` transforms raw confidence scores to logit space before passing to `_find_prominent_peaks()`
- [ ] `select_prominent_peaks_from_events()` transforms `min_score` to logit space before passing to `_find_prominent_peaks()`
- [ ] Default `min_prominence` parameter changes from 0.03 to 1.0
- [ ] `_find_prominent_peaks()` is unchanged — it still receives scores and thresholds generically

**Tests needed:**
- Logit helper: confirm transform correctness, clamping at 0 and 1
- High-confidence plateau: scores [0.99, 0.998, 0.983, 0.997, 0.999] with min_prominence=2.0 should detect peaks separated by the 0.983 dip
- Existing prominence test cases updated for logit-scale thresholds

---

### Task 2: Update default min_prominence in detector.py and hydrophone_detector.py

**Files:**
- Modify: `src/humpback/classifier/detector.py`
- Modify: `src/humpback/classifier/hydrophone_detector.py`

**Acceptance criteria:**
- [ ] `run_detection()` default for `min_prominence` when None changes from 0.03 to 1.0
- [ ] `run_hydrophone_detection()` default for `min_prominence` when None changes from 0.03 to 1.0

**Tests needed:**
- Covered by integration through Task 4 tests

---

### Task 3: Update frontend slider range for min_prominence

**Files:**
- Modify: `frontend/src/components/classifier/HydrophoneTab.tsx`

**Acceptance criteria:**
- [ ] Default state value changes from 0.03 to 1.0
- [ ] Slider range changes from [0.01, 0.20] step 0.01 to [0.5, 5.0] step 0.1
- [ ] Display label shows one decimal place (e.g., "2.0" not "2.00")

**Tests needed:**
- Frontend type check (`npx tsc --noEmit`)

---

### Task 4: Update tests for logit-space prominence

**Files:**
- Modify: `tests/unit/test_detection_spans.py`

**Acceptance criteria:**
- [ ] All existing `TestSelectProminentPeaksFromEvents` tests updated with logit-appropriate `min_prominence` values
- [ ] New test: high-confidence plateau (all scores > 0.95) with distinct dips — verifies peaks are detected that would be missed in probability space
- [ ] New test: verifies that noise-level wobbles (e.g., 0.998 vs 0.995) do not produce peaks at reasonable threshold
- [ ] Existing `_find_prominent_peaks` tests remain valid (they test the function directly with arbitrary score sequences, not probability-specific)

**Tests needed:**
- All tests in `TestSelectProminentPeaksFromEvents` pass
- New plateau/noise tests pass

---

### Task 5: Update documentation

**Files:**
- Modify: `CLAUDE.md` (§8.7 Behavioral Constraints — update min_prominence default description)
- Modify: `DECISIONS.md` (update ADR-044 to note logit-space change)

**Acceptance criteria:**
- [ ] CLAUDE.md §8.7 `min_prominence` default documented as 2.0 with logit-space note
- [ ] ADR-044 updated to reflect logit-space prominence computation

**Tests needed:**
- None (documentation only)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/classifier/detector_utils.py src/humpback/classifier/detector.py src/humpback/classifier/hydrophone_detector.py`
2. `uv run ruff check src/humpback/classifier/detector_utils.py src/humpback/classifier/detector.py src/humpback/classifier/hydrophone_detector.py`
3. `uv run pyright src/humpback/classifier/detector_utils.py src/humpback/classifier/detector.py src/humpback/classifier/hydrophone_detector.py`
4. `uv run pytest tests/unit/test_detection_spans.py -v`
5. `uv run pytest tests/`
6. `cd frontend && npx tsc --noEmit`
