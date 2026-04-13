# Segment Review UX Improvements — Design Spec

**Date:** 2026-04-13
**Scope:** Five targeted improvements to the Segment Review tab and Call Parsing breadcrumbs.

---

## 1. New Zoom Levels (30s, 10s)

### Current State
Two zoom presets in `RegionSpectrogramViewer.tsx`:
- "5m": 300s viewport, 50s tiles (regions >= 300s)
- "1m": 60s viewport, 10s tiles (regions < 300s)

### Change
Add two finer zoom presets for short regions:

| Region Duration | Preset | Viewport Span | Tile Duration |
|----------------|--------|---------------|---------------|
| >= 300s        | 5m     | 300s          | 50s           |
| >= 30s         | 1m     | 60s           | 10s           |
| >= 10s         | 30s    | 30s           | 5s            |
| < 10s          | 10s    | 10s           | 2s            |

### Auto-selection
Zoom preset is chosen automatically based on region duration (`end_sec - start_sec`). The threshold cascade selects the finest zoom level whose viewport span is >= the region duration, falling through to the coarsest level that fits. The existing `SHORT_REGION_THRESHOLD_SEC` constant is replaced by the cascade.

### Backend
The tile endpoint validates zoom levels against `ZOOM_LEVELS` in `timeline_tiles.py`. The new "30s" and "10s" levels must be added to `ZOOM_LEVELS` and `_TILE_DURATIONS` so the backend accepts and renders tiles at these zoom levels.

### Time Axis Labels
Adjust tick interval per zoom level:
- 5m/1m: every 30s (existing)
- 30s: every 5s
- 10s: every 2s

---

## 2. Region Boundary Indicators

### Current State
The spectrogram shows the padded region (`padded_start_sec` to `padded_end_sec`) with no indication of where the actual region boundaries (`start_sec`, `end_sec`) fall.

### Change
Add two visual elements to `RegionSpectrogramViewer`:

**Boundary lines:** Dashed amber vertical lines (`rgba(251, 191, 36, 0.8)`, 2px) at the region's unpadded `start_sec` and `end_sec`. Each line has a small label at the top ("R start" / "R end") with a dark semi-transparent background pill for legibility.

**Dimmed overlay:** Semi-transparent dark overlay (`rgba(0, 0, 0, 0.55)`) covering the area outside the region boundaries (from padded start to region start, and from region end to padded end). This visually de-emphasizes the padding area while keeping it visible for context.

### Implementation
Both elements are absolutely-positioned divs in the spectrogram container, positioned using the existing `secToX` coordinate transform. Z-index layering: spectrogram tiles (base) < dimmed overlay (z-3) < boundary lines (z-4) < event bars (existing) < playhead (z-6).

### Data
The `Region` type already provides `start_sec`, `end_sec`, `padded_start_sec`, and `padded_end_sec` — no additional data fetching needed.

---

## 3. Audio Playback Indicator (Playhead)

### Current State
The toolbar "Play" button and event detail "Play Slice" button play audio via hidden `<audio>` elements. No visual indicator shows the current playback position on the spectrogram.

### Change
Add a sweeping vertical playhead line to the spectrogram viewer.

### Visual Design
- 1.5px solid teal line (`#70e0c0`), matching the detection timeline's accent color
- Small downward-pointing triangle at the top of the line
- `pointer-events: none` so it doesn't interfere with event bar clicks/drags
- Visible only during active playback

### Playback Sync
- rAF loop polls `audio.currentTime` every frame while playing
- Playhead X position computed as `secToX(playbackOriginSec + audio.currentTime)`
- `playbackOriginSec` is the absolute start time of the audio slice being played (the `start_sec` parameter passed to the audio endpoint)

### Audio Source Integration
Both playback triggers feed the same playhead:
- **Toolbar "Play"**: plays from viewport start, `playbackOriginSec = viewStart ?? region.padded_start_sec`
- **Event detail "Play Slice"**: plays the selected event, `playbackOriginSec = event.startSec`

### State
New state in `SegmentReviewWorkspace` (or lifted to a shared ref):
- `isPlaying: boolean` — whether audio is actively playing
- `playbackOriginSec: number` — absolute start time of current audio slice
- `audioRef: RefObject<HTMLAudioElement>` — shared audio element ref

The rAF loop lives in `RegionSpectrogramViewer` and directly manipulates the playhead div's `style.left` via a DOM ref — no React state updates per frame, avoiding re-render overhead. A `playheadRef` on the playhead div enables this: `playheadRef.current.style.left = secToX(...) + "px"`.

### Cleanup
When audio ends (`ended` event) or is stopped, `isPlaying` is set to false and the playhead disappears.

---

## 4. Auto-Select First Event on Region Load

### Current State
When a region is selected, `selectedEventId` remains `null`. The EventDetailPanel shows "Click an event bar to view details" until the user clicks an event.

### Change
When a region is selected (including the initial auto-select of the first region on job load), automatically select the first event in that region sorted by `startSec`.

### Implementation
Add an effect in `SegmentReviewWorkspace` that watches `selectedRegionId` and the events list. When both are available and no event is currently selected for this region, set `selectedEventId` to the first event's ID.

### Edge Cases
- Region has no events: `selectedEventId` stays `null`, detail panel shows placeholder
- Events load asynchronously after region select: effect fires when events arrive
- User manually selects a different event: not overridden (effect only fires on region change)

---

## 5. Call Parsing Breadcrumbs

### Current State
`Breadcrumbs.tsx` has a `staticRoutes` map with entries for Classifier, Vocalization, etc. Call Parsing pages have no entries and fall through to the default "Audio" breadcrumb.

### Change
Add three entries to `staticRoutes`:

```
/app/call-parsing/detection       → Call Parsing > Detection
/app/call-parsing/segment         → Call Parsing > Segment
/app/call-parsing/segment-training → Call Parsing > Segment Training
```

The "Call Parsing" crumb links to `/app/call-parsing` (which redirects to detection via existing `<Navigate>` route). The terminal crumb has no link (current page).

### Pattern
Follows the same structure as existing Classifier entries:
```ts
"/app/classifier/training": [
  { label: "Classifier", to: "/app/classifier/training" },
  { label: "Training" },
],
```

---

## Files Changed

| File | Change |
|------|--------|
| `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx` | New zoom presets, region boundary overlays, playhead rendering, rAF loop |
| `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx` | Playback state management, shared audio ref, auto-select first event effect |
| `frontend/src/components/call-parsing/ReviewToolbar.tsx` | Wire shared audio ref and playback origin state |
| `frontend/src/components/call-parsing/EventDetailPanel.tsx` | Wire shared audio ref and playback origin state |
| `frontend/src/components/layout/Breadcrumbs.tsx` | Add Call Parsing routes to `staticRoutes` |

No new files. No new dependencies.
