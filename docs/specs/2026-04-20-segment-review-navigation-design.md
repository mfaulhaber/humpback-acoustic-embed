# Segment Review Navigation Enhancements

**Date:** 2026-04-20
**Status:** Approved

## Problem

The Call Parsing Segment review page has several UX issues that make efficient event review difficult:

1. Spacebar sometimes triggers UI button navigation instead of playing the selected event
2. No event-level next/prev navigation — only region-level navigation exists
3. Timeline scrolling is clamped to the current region bounds
4. No visual context for neighboring regions when zoomed out

The Classify review page has already solved problems 1 and 2 with patterns we can reuse.

## Design

### 1. Spacebar Playback Fix

**Current behavior:** The spacebar handler in `SegmentReviewWorkspace` (line 354-355) bails when a button has focus: `if (tag === "BUTTON" || el.closest("button")) return;`. Tab-navigating to a toolbar button and pressing Space triggers the button's click action instead of playback.

**Fix:** Remove the button-focus check. Always `e.preventDefault()` when Space is pressed (still skipping INPUT/TEXTAREA/SELECT). This matches the classify page pattern where spacebar unconditionally means play/stop.

### 2. Event Navigation with `a`/`s` Shortcuts

**Flat event list:** Build a `navigableEvents` list in `SegmentReviewWorkspace` — all events across all regions for the current job, sorted by `startSec`, excluding events with saved or pending delete corrections. Includes pending "add" corrections as navigable entries. Each event carries its `regionId`.

**Navigation state:** Add `currentEventIndex` state as index into the flat list. `goPrev` decrements, `goNext` increments (clamped to bounds).

**Cross-region navigation:** When `navigableEvents[newIndex].regionId !== selectedRegionId`, automatically switch to that region. The spectrogram re-centers on the new region, then `scrollToCenter` positions the event in view.

**Keyboard shortcuts:**
- `KeyA` = previous event
- `KeyS` = next event
- Same input-field guard as classify (skip INPUT/TEXTAREA/SELECT)
- Both fire `e.preventDefault()`

**UI buttons:** Add "Prev Event" / "Next Event" buttons to the toolbar with event counter ("3 of 12"). Existing region prev/next buttons remain but have no keyboard shortcuts.

**Auto-scroll on navigate:** If the event isn't fully visible within 10% viewport padding, fire `scrollToCenter` to bring it into view (same logic as classify lines 542-553).

### 3. Full Timeline Scrolling

**New prop:** `RegionSpectrogramViewer` accepts a `timelineExtent: { start: number; end: number }` prop representing the full time range of the run (min `padded_start_sec` to max `padded_end_sec` across all regions). The workspace computes this from the regions list.

**Relaxed clamping:** `clampCenter` uses `timelineExtent` bounds instead of the active region bounds. The viewport can now pan freely across the entire run's time range.

**Tile loading:** Tiles are fetched by visible time range already — panning beyond the current region shows spectrogram tiles for adjacent time areas.

**Active region unchanged by scrolling:** Panning beyond a region does NOT auto-switch the active region. Only explicit actions (clicking a region band, event navigation) change the active region.

**Drag-to-pan:** Existing mousedown/mousemove/mouseup logic unchanged; only the clamp bounds change.

### 4. Region Indicator Bands

**Visibility condition:** Region bands appear at the 1m and 5m zoom presets (where the viewport is wide enough to show neighboring regions). Hidden at 30s and 10s zoom.

**Band rendering:** An overlay layer (sibling to `EventBarOverlay`) renders a semi-transparent colored band for each region in the run. Each band spans `[padded_start_sec, padded_end_sec]` horizontally, full spectrogram height. The active region gets a distinct highlight (brighter fill or accent border); inactive regions get a muted semi-transparent fill.

**Region label:** Each band shows a small label (region number) at the top.

**Click behavior:** Clicking an inactive region band:
- Sets it as the active region (`setSelectedRegionId`)
- Centers the spectrogram on that region's midpoint
- Auto-selects the first event in that region (or clears selection if empty)

**Layering:** Region bands render behind event bars. Events are always visible on top.

## Files Modified

- `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx` — spacebar fix, event navigation state, keyboard shortcuts, auto-scroll, timeline extent computation
- `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx` — `timelineExtent` prop, relaxed clamping, region band visibility logic
- `frontend/src/components/call-parsing/ReviewToolbar.tsx` — event prev/next buttons with counter
- `frontend/src/components/call-parsing/RegionBandOverlay.tsx` (new) — region indicator band rendering and click handling

## Non-Goals

- Changing the classify page (it already works correctly)
- Adding region bands to the classify view
- Changing how boundary editing or add-mode works
- Any backend changes
