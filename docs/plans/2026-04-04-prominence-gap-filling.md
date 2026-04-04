# Prominence Gap-Filling Implementation Plan

**Goal:** Add recursive gap-filling to prominence-based window selection so that strong vocalizations in flat score regions between peaks are not missed.
**Spec:** [docs/specs/2026-04-04-prominence-gap-filling-design.md](../specs/2026-04-04-prominence-gap-filling-design.md)

---

### Task 1: Implement `_fill_gaps_recursive` helper

**Files:**
- Modify: `src/humpback/classifier/detector_utils.py`

**Acceptance criteria:**
- [ ] New private function `_fill_gaps_recursive` added after `_find_prominent_peaks`
- [ ] Takes candidates, raw_scores, selected_offsets, left/right boundaries, min_gap_fill, min_score
- [ ] Returns list of indices into candidates for windows to emit
- [ ] Uses raw probability scores (not logit) for selecting the best fill window
- [ ] Candidates are filtered to `(left, right)` exclusive of boundaries
- [ ] Recursion terminates when gap <= min_gap_fill or no candidate above min_score exists

---

### Task 2: Integrate gap-filling into `select_prominent_peaks_from_events`

**Files:**
- Modify: `src/humpback/classifier/detector_utils.py`

**Acceptance criteria:**
- [ ] After existing zero-peaks fallback and before the detection-building loop, gap-fill logic is inserted
- [ ] Sorted peak offsets collected from `peak_indices`
- [ ] Gap list built from event `start_sec`/`end_sec` boundaries and peak offsets
- [ ] `_fill_gaps_recursive` called for each gap; returned indices merged into `peak_indices`
- [ ] `min_gap_fill` hardcoded at 3.0 seconds
- [ ] Existing fallback behavior unchanged — gap-filling runs after it
- [ ] No changes to function signature or callers

---

### Task 3: Add unit tests

**Files:**
- Modify: `tests/unit/test_detection_spans.py`

**Acceptance criteria:**
- [ ] Basic gap fill: two peaks 10s apart, fill window appears between them
- [ ] Recursive fill: two peaks 15s apart, multiple fill windows emitted
- [ ] Edge gap fill: single peak in middle of long event, fills toward event boundaries
- [ ] No fill when gap is small: peaks 2s apart, no extra windows
- [ ] No fill when gap scores below threshold: gap region low scores, no fill
- [ ] Fallback + gap fill compose: zero prominent peaks on 15s plateau, fallback plus gap fills
- [ ] Recursion terminates: flat high scores across long event, bounded output count

**Tests needed:**
- All tests within existing `TestSelectProminentPeaksFromEvents` class
- Use existing `_make_window_records` and `_make_event` helpers

---

### Task 4: Update documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] CLAUDE.md §8.7 window selection modes bullet updated to describe gap-filling as part of prominence mode with 3-second default
- [ ] DECISIONS.md: existing prominence ADR appended with gap-fill addition and rationale

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/classifier/detector_utils.py tests/unit/test_detection_spans.py`
2. `uv run ruff check src/humpback/classifier/detector_utils.py tests/unit/test_detection_spans.py`
3. `uv run pyright src/humpback/classifier/detector_utils.py tests/unit/test_detection_spans.py`
4. `uv run pytest tests/`
