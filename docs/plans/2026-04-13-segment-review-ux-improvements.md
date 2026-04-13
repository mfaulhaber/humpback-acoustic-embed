# Segment Review UX Improvements — Implementation Plan

**Goal:** Improve the Segment Review tab with finer zoom levels, region boundary indicators, a playback position indicator, auto-event selection, and correct Call Parsing breadcrumbs.
**Spec:** [docs/specs/2026-04-13-segment-review-ux-improvements-design.md](../specs/2026-04-13-segment-review-ux-improvements-design.md)

---

### Task 1: Add 30s and 10s zoom presets

**Files:**
- Modify: `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx`

**Acceptance criteria:**
- [ ] `ZOOM_PRESETS` expanded with `"30s"` (viewportSpan 30, tileDuration 5) and `"10s"` (viewportSpan 10, tileDuration 2)
- [ ] `SHORT_REGION_THRESHOLD_SEC` replaced by a cascade function that selects zoom based on region duration: >=300s → 5m, >=30s → 1m, >=10s → 30s, <10s → 10s
- [ ] Time axis tick interval adjusts per zoom level: 30s for 5m/1m, 5s for 30s, 2s for 10s
- [ ] Spectrogram tiles request the correct zoom_level and tile_index for new presets

**Tests needed:**
- Playwright test: load a region shorter than 30s and verify the 30s zoom preset is active (viewport span matches)
- Playwright test: load a region shorter than 10s and verify the 10s zoom preset is active

---

### Task 2: Add region boundary indicators

**Files:**
- Modify: `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx`

**Acceptance criteria:**
- [ ] Dashed amber vertical lines rendered at region `start_sec` and `end_sec` positions using `secToX`
- [ ] Each line has a small label ("R start" / "R end") at the top with semi-transparent background pill
- [ ] Semi-transparent dark overlay covers area outside region boundaries (padded_start to start, end to padded_end)
- [ ] Boundary elements respect pan state — reposition correctly when viewport is dragged
- [ ] Z-index layering correct: tiles < dimmed overlay < boundary lines < event bars < playhead

**Tests needed:**
- Playwright test: verify boundary line elements are present in the DOM when a region is selected
- Playwright test: verify dimmed overlay elements are present with correct positioning relative to region bounds

---

### Task 3: Consolidate audio into shared playback state

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`
- Modify: `frontend/src/components/call-parsing/ReviewToolbar.tsx`
- Modify: `frontend/src/components/call-parsing/EventDetailPanel.tsx`

**Acceptance criteria:**
- [ ] Single `<audio>` element owned by SegmentReviewWorkspace, passed down via `audioRef`
- [ ] Remove duplicate `<audio>` elements from ReviewToolbar and EventDetailPanel
- [ ] `isPlaying` and `playbackOriginSec` state managed in SegmentReviewWorkspace
- [ ] ReviewToolbar play sets `playbackOriginSec` to viewport start, loads and plays via shared audioRef
- [ ] EventDetailPanel play slice sets `playbackOriginSec` to `event.startSec`, loads and plays via shared audioRef
- [ ] Audio `ended` event resets `isPlaying` to false
- [ ] Starting playback from one source stops any in-progress playback from the other

**Tests needed:**
- Playwright test: clicking toolbar play triggers audio element (verify src attribute set correctly)
- Playwright test: clicking event play slice triggers audio element with correct start/duration

---

### Task 4: Add playhead indicator to spectrogram viewer

**Files:**
- Modify: `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx`

**Acceptance criteria:**
- [ ] Teal vertical line (1.5px, #70e0c0) with downward triangle rendered as a positioned div with `pointer-events: none`
- [ ] Playhead visible only when `isPlaying` is true
- [ ] rAF loop runs while playing, updates playhead `style.left` via DOM ref using `secToX(playbackOriginSec + audio.currentTime)`
- [ ] rAF loop cancels on unmount and when playback stops
- [ ] Playhead position is correct relative to panned viewport (uses same coordinate system as event bars)

**Tests needed:**
- Playwright test: verify playhead element appears in DOM during playback and disappears when stopped
- Playwright test: verify playhead has `pointer-events: none` style

---

### Task 5: Auto-select first event on region load

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] When `selectedRegionId` changes, first event (sorted by startSec) in that region is auto-selected
- [ ] Auto-selection fires on initial render when first region is auto-selected
- [ ] Regions with no events leave `selectedEventId` as null
- [ ] User clicking a different event is not overridden by auto-selection
- [ ] Switching regions resets selection to first event of new region

**Tests needed:**
- Playwright test: on initial load, first event in first region is selected (ring styling visible on first event bar)
- Playwright test: switching regions selects first event of new region
- Playwright test: region with no events shows placeholder in detail panel

---

### Task 6: Fix Call Parsing breadcrumbs

**Files:**
- Modify: `frontend/src/components/layout/Breadcrumbs.tsx`

**Acceptance criteria:**
- [ ] `/app/call-parsing/detection` shows "Call Parsing > Detection"
- [ ] `/app/call-parsing/segment` shows "Call Parsing > Segment"
- [ ] `/app/call-parsing/segment-training` shows "Call Parsing > Segment Training"
- [ ] "Call Parsing" crumb links to `/app/call-parsing`
- [ ] Terminal crumb (current page) has no link
- [ ] No other breadcrumb routes affected

**Tests needed:**
- Playwright test: navigate to each Call Parsing page, verify breadcrumb text and link targets

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test e2e/call-parsing-segment.spec.ts`
3. Manual browser test: open Segment Review tab, verify zoom, boundaries, playhead, auto-select, and breadcrumbs on a real region
