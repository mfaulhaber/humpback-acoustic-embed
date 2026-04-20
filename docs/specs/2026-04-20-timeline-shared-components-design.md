# Unified Timeline Composition System — Design Spec

**Date:** 2026-04-20
**Status:** Approved

## Problem

Four timeline viewers share 80% of their functionality but are built on two separate rendering engines with significant code duplication:

| Viewer | Engine | Lines |
|--------|--------|-------|
| TimelineViewer (classifier) | SpectrogramViewport | 657 |
| RegionTimelineViewer (call-parsing detection) | SpectrogramViewport | 477 |
| SegmentReviewWorkspace | RegionSpectrogramViewer | 757 |
| ClassifyReviewWorkspace | RegionSpectrogramViewer | 1108 |

The RegionTimelineViewer has three bugs caused by architectural divergence:
- (a) Regions not editable in edit mode — coordinate space mismatch between externally-positioned RegionEditOverlay and SpectrogramViewport internals
- (b) Zoom controls shift when edit mode toggled — edit toolbar shares a flex row with ZoomSelector
- (c) Drag scroll and cursor modes break on edit toggle — `disablePan` sets `pointerEvents: "none"` on the entire canvas div

## Solution

Converge all four viewers onto a shared compound-component architecture using composable building blocks.

## Core Decisions

1. **UTC epoch timestamps** as the universal internal coordinate system. Review pages adapt job-relative data at the boundary via hooks.
2. **Canvas LRU tile rendering** for all viewers (retire `<img>`-based approach).
3. **Context + children pattern** for overlays. The Spectrogram component provides OverlayContext; overlays consume it via hook.
4. **Compound component architecture** — TimelineProvider owns shared state; child components compose freely.
5. **Playback modes** — `gapless` (double-buffered A/B swap) for long-form timelines, `slice` (single element) for review pages. Same interface.
6. **Composable footer** — ZoomSelector, PlaybackControls, EditToolbar, etc. are independent components. No monolithic footer with 15+ props.

## Architecture

### TimelineProvider

Manages shared state via React context:

**State:** centerTimestamp (epoch), zoomLevel, isPlaying, speed, viewportDimensions  
**Derived:** viewStart, viewEnd, pxPerSec, viewportSpan  
**Actions:** pan(center), setZoomLevel(level), play(), pause(), togglePlay(), seekTo(epoch)

Props:
- `jobStart: number` — epoch seconds
- `jobEnd: number` — epoch seconds
- `zoomLevels: ZoomPreset[]` — configurable per page
- `defaultZoom: string` — initial zoom level key
- `playback: "gapless" | "slice"` — audio strategy
- `audioUrlBuilder: (startEpoch, durationSec) => string` — URL factory

Registers keyboard shortcuts (space, +/-, arrows). Renders hidden audio elements internally.

### Spectrogram

Renders the viewport by consuming TimelineProvider context:

**Renders (top to bottom):**
1. Frequency axis (left, 44px)
2. TileCanvas with LRU cache + crossfade
3. Overlay container (absolute-positioned, provides OverlayContext to children)
4. Playhead (static center line)
5. Confidence strip (optional, 20px, only if `scores` provided)
6. Time axis (UTC-formatted labels, 20px)

**Drag-to-pan** is built in. Edit overlays intercept events via `stopPropagation` on edges they handle; unclaimed areas keep grab cursor and pan behavior. No `disablePan` prop.

Props:
- `tileUrlBuilder` — tile URL factory
- `freqRange` — frequency range (default [0, 3000])
- `scores` / `windowSec` — optional confidence data
- `children` — overlay components

### OverlayContext

Provided inside Spectrogram's overlay container:
- `viewStart`, `viewEnd`, `pxPerSec`, `canvasWidth`, `canvasHeight`
- `epochToX(epoch)`, `xToEpoch(x)` — coordinate transform helpers

All overlay components (DetectionOverlay, EventBarOverlay, RegionOverlay, RegionEditOverlay, LabelEditor, etc.) consume via `useOverlayContext()`.

### Zoom Presets (data-driven)

```
FULL_ZOOM:   24h (86400s), 6h (21600s), 1h (3600s), 15m (900s), 5m (300s), 1m (60s)
REVIEW_ZOOM: 5m (300s), 1m (60s), 30s (30s), 10s (10s)
```

Each preset specifies: key, span (seconds), tileDuration (seconds per tile).

### Footer Controls

Independent composable components:
- `ZoomSelector` — reads levels + active from context
- `PlaybackControls` — play/pause, skip, speed, zoom +/-, accepts children for right-side slots
- `EditToggle` — generic edit mode button
- `EditToolbar` — save/cancel/count, accepts children
- `OverlayToggles` — visibility buttons for overlay layers
- `EventNav` — prev/next with counter
- `TimelineFooter` — dark wrapper with top border

ZoomSelector is always its own full-width row to prevent layout shift.

### Epoch Adapters

Hooks for review pages that receive job-relative data:
- `useEpochRegions(regions, jobStartEpoch)` — adds offset to region timestamps
- `useEpochEvents(events, jobStartEpoch)` — adds offset to event timestamps

Correction callbacks subtract the offset before persisting.

## Page Compositions

### Classifier Timeline
TimelineProvider (FULL_ZOOM, gapless) → TimelineHeader → Spectrogram (DetectionOverlay | VocalizationOverlay | LabelEditor) → TimelineFooter (LabelToolbar, ZoomSelector, PlaybackControls with EditToggle + OverlayToggles)

### Region Detection Timeline
TimelineProvider (FULL_ZOOM, gapless) → Header → Spectrogram (RegionOverlay | RegionEditOverlay) → TimelineFooter (EditToolbar, ZoomSelector, PlaybackControls with EditToggle + OverlayToggles)

### Segment Review
TimelineProvider (REVIEW_ZOOM, slice) + epoch adapters → ReviewToolbar (EventNav, PlaybackControls compact, EditToolbar) → Spectrogram (RegionBoundaryMarkers, RegionBandOverlay, EventBarOverlay) → ZoomSelector → EventDetailPanel → RegionTable

### Classify Review
TimelineProvider (REVIEW_ZOOM, slice) + epoch adapters → ClassifyToolbar (EventNav, PlaybackControls compact, EditToolbar) → Spectrogram (EventBarOverlay) → ZoomSelector → TypePalette → ClassifyDetailPanel

## Bug Fixes (Structural)

| Bug | Root cause | Fixed by |
|-----|-----------|----------|
| (a) Regions not editable | RegionEditOverlay positioned externally with coordinate mismatch | Overlay is a child inside Spectrogram, uses shared OverlayContext |
| (b) Zoom controls shift | Edit toolbar and ZoomSelector share a flex row | ZoomSelector is always its own full-width row |
| (c) Cursor/drag broken | `disablePan` sets `pointerEvents: "none"` on canvas div | Pan always active; overlays intercept via stopPropagation |

## Scope Boundary

**In scope:** Spectrogram rendering, coordinate system, playback, controls, overlay hosting, epoch adapters.

**Out of scope (stays in page components):** Correction merging logic, retrain workflows, job selectors, LabelEditor internals, VocLabelEditor, TypePalette, ClassifyDetailPanel, EventDetailPanel, RegionTable, data hooks, API client, backend.

## Migration Strategy

Incremental — build shared module alongside existing code, migrate one viewer per task, delete old code last. Each task produces a standalone PR. At no point is a viewer broken.
