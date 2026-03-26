# Timeline Viewer Enhancements Implementation Plan

**Goal:** Fix tile brightness inconsistency, confidence strip scroll desync, page overflow, and clarify workflow docs.
**Spec:** `docs/specs/2026-03-26-timeline-viewer-enhancements-design.md`

---

### Task 1: Fix tile brightness with fixed reference level

**Files:**
- Modify: `src/humpback/processing/timeline_tiles.py`

**Acceptance criteria:**
- [ ] `generate_timeline_tile()` accepts a `ref_db` parameter (default `-20.0`)
- [ ] `vmax` is set to `ref_db` instead of `power_db.max()`
- [ ] `vmin` is computed as `ref_db - dynamic_range_db`
- [ ] The two per-tile normalization lines are removed
- [ ] Existing callers (timeline cache pipeline) work without changes (default parameter)

**Tests needed:**
- Verify that `generate_timeline_tile()` returns valid PNG bytes with the new default
- Verify that two calls with different audio amplitudes produce tiles with visually different brightness (not normalized to the same range)

---

### Task 2: Fix confidence strip scroll sync

**Files:**
- Modify: `frontend/src/components/timeline/SpectrogramViewport.tsx`

**Acceptance criteria:**
- [ ] `confidenceStrip` useMemo computes bar x-position as `(windowStart - centerTimestamp) * pxPerSec + canvasWidth / 2`
- [ ] Bar width is `windowSec * pxPerSec`
- [ ] `windowStart` for score `i` is `jobStart + i * windowSec`
- [ ] Confidence strip scrolls in sync with spectrogram tiles and detection overlays when panning
- [ ] Confidence strip renders correctly at all zoom levels

**Tests needed:**
- Visual verification: pan the spectrogram and confirm the confidence bars move in lockstep with detection overlays

---

### Task 3: Remove Minimap and pin footer controls

**Files:**
- Delete: `frontend/src/components/timeline/Minimap.tsx`
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`

**Acceptance criteria:**
- [ ] `Minimap.tsx` is deleted
- [ ] `Minimap` import and `<Minimap>` element removed from `TimelineViewer.tsx`
- [ ] `useTimelineConfidence` query remains (feeds `SpectrogramViewport` confidence strip)
- [ ] `ZoomSelector`, `PlaybackControls`, and `LabelToolbar` are wrapped in a footer div
- [ ] Footer is pinned to viewport bottom with `COLORS.headerBg` background and top border
- [ ] Spectrogram viewport fills space between header and footer (`flex-1`)
- [ ] Outer container uses `h-screen` with `overflow: hidden`
- [ ] No window-level scrollbar at any zoom level or label mode state
- [ ] All controls (zoom, playback, labels) remain visible and functional

**Tests needed:**
- Visual verification: no scrollbar, controls visible at all zoom levels and in label mode
- Existing Playwright tests pass

---

### Task 4: Clarify brainstorming workflow docs

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/workflows/session-plan.md`

**Acceptance criteria:**
- [ ] CLAUDE.md §10.1 brainstorming overrides says spec is NOT written to disk during brainstorming
- [ ] `session-plan.md` preconditions clarify the spec exists as an approved design in conversation context, not as a file on main
- [ ] Wording is unambiguous — no reading that implies a file is created on main

**Tests needed:**
- Read the updated docs and confirm the language is clear

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/processing/timeline_tiles.py`
2. `uv run ruff check src/humpback/processing/timeline_tiles.py`
3. `uv run pyright src/humpback/processing/timeline_tiles.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
