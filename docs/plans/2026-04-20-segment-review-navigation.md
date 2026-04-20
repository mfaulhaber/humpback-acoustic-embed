# Segment Review Navigation Enhancements — Implementation Plan

**Goal:** Fix spacebar playback, add event-level keyboard navigation, enable full timeline scrolling, and show region indicator bands when zoomed out.
**Spec:** [docs/specs/2026-04-20-segment-review-navigation-design.md](../specs/2026-04-20-segment-review-navigation-design.md)

---

### Task 1: Spacebar Playback Fix

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Remove the button-focus bail-out (`if (tag === "BUTTON" || el.closest("button")) return;`)
- [ ] Move `e.preventDefault()` to fire immediately after the INPUT/TEXTAREA/SELECT guard (before any playback logic)
- [ ] Spacebar always toggles playback regardless of which non-input element has focus

**Tests needed:**
- Manual: Tab to a toolbar button, press Space — should play audio, not activate button

---

### Task 2: Flat navigableEvents List and Navigation State

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Compute `navigableEvents` as a useMemo — all events across all regions for the job, sorted by `startSec`, excluding events with saved or pending delete corrections, including pending "add" corrections
- [ ] The workspace needs access to all events across regions (may need to query all region events, not just the selected region's)
- [ ] Add `currentEventIndex` state (number, default 0)
- [ ] Derive `currentEvent` from `navigableEvents[currentEventIndex]`
- [ ] When `currentEventIndex` changes and the event's `regionId` differs from `selectedRegionId`, auto-switch region
- [ ] Clamp `currentEventIndex` when `navigableEvents` list shrinks (same pattern as classify)
- [ ] `selectedEventId` state stays in sync with `currentEvent.eventId`

**Tests needed:**
- Playwright: navigate events with buttons, verify counter updates and region switches

---

### Task 3: Event Navigation Keyboard Shortcuts

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] `KeyA` calls goPrev (decrement currentEventIndex, clamped to 0)
- [ ] `KeyS` calls goNext (increment currentEventIndex, clamped to length-1)
- [ ] Both fire `e.preventDefault()`
- [ ] INPUT/TEXTAREA/SELECT guard prevents shortcuts while typing
- [ ] Spacebar handler uses the unified switch-case pattern (matching classify)

**Tests needed:**
- Manual: press a/s to step through events, verify cross-region navigation works

---

### Task 4: Event Navigation Buttons in Toolbar

**Files:**
- Modify: `frontend/src/components/call-parsing/ReviewToolbar.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Add new props to ReviewToolbar: `onPrevEvent`, `onNextEvent`, `currentEventIndex`, `totalEventCount`
- [ ] Render "Prev Event" / "Next Event" buttons with ChevronLeft/ChevronRight icons (distinct styling from region buttons)
- [ ] Show counter label: "Event {n} of {total}"
- [ ] Buttons disabled at list boundaries (first/last)
- [ ] Existing region prev/next buttons remain unchanged (no keyboard shortcut labels)

**Tests needed:**
- Playwright: click event navigation buttons, verify event selection changes and counter reflects position

---

### Task 5: Auto-Scroll on Event Navigation

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] When `currentEvent` changes, check if event is fully visible within 10% viewport padding
- [ ] If not visible, fire `scrollToCenter` to position the event in view
- [ ] Works correctly after cross-region switch (region switch fires first, then scroll)

**Tests needed:**
- Manual: navigate to an event that's off-screen, verify spectrogram scrolls to show it

---

### Task 6: Full Timeline Scrolling (Relaxed Clamping)

**Files:**
- Modify: `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] `RegionSpectrogramViewer` accepts new optional prop `timelineExtent: { start: number; end: number }`
- [ ] When `timelineExtent` is provided, `clampCenter` uses its bounds instead of the active region bounds
- [ ] When `timelineExtent` is not provided, behavior is unchanged (backwards compatible for classify usage)
- [ ] Workspace computes `timelineExtent` from `Math.min(...regions.map(r => r.padded_start_sec))` to `Math.max(...regions.map(r => r.padded_end_sec))`
- [ ] Drag-to-pan works seamlessly across the full extent
- [ ] Scrolling beyond active region does NOT auto-switch the active region

**Tests needed:**
- Manual: zoom out, drag timeline past region boundary, verify scrolling continues smoothly

---

### Task 7: Region Band Overlay Component

**Files:**
- Create: `frontend/src/components/call-parsing/RegionBandOverlay.tsx`
- Modify: `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx`

**Acceptance criteria:**
- [ ] New `RegionBandOverlay` component renders semi-transparent colored bands for each region
- [ ] Each band spans `[padded_start_sec, padded_end_sec]` horizontally, full spectrogram height
- [ ] Active region has distinct highlight (brighter fill or accent border)
- [ ] Inactive regions have muted semi-transparent fill
- [ ] Each band shows a region number label at top
- [ ] Clicking an inactive band calls `onSelectRegion(regionId)` callback
- [ ] Uses the same `OverlayContext` (or equivalent viewport info) as EventBarOverlay for coordinate transforms
- [ ] Only rendered at 1m and 5m zoom presets; hidden at 30s and 10s
- [ ] Rendered behind EventBarOverlay (lower z-order)

**Tests needed:**
- Manual: zoom to 1m or 5m, verify region bands appear; click one to switch active region and center spectrogram
- Verify bands disappear at 30s/10s zoom

---

### Task 8: Region Band Click Behavior

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`
- Modify: `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx`

**Acceptance criteria:**
- [ ] `RegionSpectrogramViewer` accepts props for region band interaction: `regions`, `activeRegionId`, `onSelectRegion`
- [ ] On band click: sets the clicked region as active (`setSelectedRegionId`)
- [ ] Centers spectrogram on the clicked region's midpoint via `scrollToCenter`
- [ ] Auto-selects the first event in the new region (sets `currentEventIndex` to that event's position in navigableEvents), or clears event selection if region is empty
- [ ] Region bands and RegionSpectrogramViewer pass region list down to overlay

**Tests needed:**
- Manual: click an inactive region band at wide zoom, verify region switch, centering, and event selection

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test`
