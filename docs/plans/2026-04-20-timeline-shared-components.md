# Timeline Shared Components — Implementation Plan

**Goal:** Converge four timeline viewers onto a shared compound-component architecture, fixing RegionTimelineViewer bugs and reducing duplication by ~40%.
**Spec:** [docs/specs/2026-04-20-timeline-shared-components-design.md](../specs/2026-04-20-timeline-shared-components-design.md)

---

### Task 1: TimelineProvider + Core Hooks

Build the shared state provider and playback hooks without consuming them yet.

**Files:**
- Create: `frontend/src/components/timeline/provider/TimelineProvider.tsx`
- Create: `frontend/src/components/timeline/provider/useTimelineContext.ts`
- Create: `frontend/src/components/timeline/provider/usePlayback.ts`
- Create: `frontend/src/components/timeline/provider/types.ts`

**Acceptance criteria:**
- [ ] TimelineProvider manages centerTimestamp, zoomLevel, isPlaying, speed state
- [ ] Exposes derived values: viewStart, viewEnd, pxPerSec, viewportSpan
- [ ] Exposes actions: pan, setZoomLevel, play, pause, togglePlay, seekTo
- [ ] Accepts configurable zoomLevels (ZoomPreset array) and defaultZoom
- [ ] usePlayback hook supports "gapless" mode with double-buffered audio elements
- [ ] usePlayback hook supports "slice" mode with single audio element
- [ ] Both modes expose same interface: play(startEpoch, duration?), pause(), isPlaying, currentTime
- [ ] Keyboard shortcuts registered: space (toggle play), +/- (zoom), arrows (pan 10% of viewport)
- [ ] Audio elements rendered hidden inside provider
- [ ] useTimelineContext throws if used outside provider
- [ ] FULL_ZOOM and REVIEW_ZOOM presets defined in types.ts

**Tests needed:**
- Unit tests for zoom preset validation (span > 0, tileDuration > 0)
- Unit test for pxPerSec derivation from viewportSpan and canvasWidth
- Unit test for pan clamping to jobStart/jobEnd bounds
- Unit test for usePlayback slice mode: play/pause state transitions

---

### Task 2: Spectrogram Component + OverlayContext

Build the unified spectrogram renderer that replaces both SpectrogramViewport and RegionSpectrogramViewer.

**Files:**
- Create: `frontend/src/components/timeline/spectrogram/Spectrogram.tsx`
- Create: `frontend/src/components/timeline/spectrogram/FrequencyAxis.tsx`
- Create: `frontend/src/components/timeline/spectrogram/TimeAxis.tsx`
- Create: `frontend/src/components/timeline/spectrogram/ConfidenceStrip.tsx`
- Create: `frontend/src/components/timeline/spectrogram/Playhead.tsx`
- Create: `frontend/src/components/timeline/overlays/OverlayContext.tsx`
- Modify: `frontend/src/components/timeline/TileCanvas.tsx` (minor: accept ZoomPreset tileDuration instead of hardcoded lookup)

**Acceptance criteria:**
- [ ] Spectrogram consumes TimelineProvider context for all coordinate/state info
- [ ] Renders frequency axis (44px left), tile canvas, overlay container, playhead, time axis
- [ ] Confidence strip renders only when scores prop provided
- [ ] OverlayContext provides: viewStart, viewEnd, pxPerSec, canvasWidth, canvasHeight, epochToX, xToEpoch
- [ ] Children (overlays) rendered inside overlay container with OverlayContext.Provider
- [ ] Built-in drag-to-pan: mousedown/move/up on canvas area calls provider's pan() action
- [ ] Cursor: grab (default) → grabbing (during drag). No disablePan prop.
- [ ] ResizeObserver tracks canvas dimensions, updates viewportDimensions in provider
- [ ] TileCanvas accepts tileDuration from current zoom preset
- [ ] Time axis formats labels as UTC (HH:MM:SS for narrow zooms, MM-DD HH:MM for wide)
- [ ] Playhead renders as static center line with triangle indicator

**Tests needed:**
- Unit test for FrequencyAxis label generation at different freq ranges
- Unit test for TimeAxis label formatting per zoom level
- Unit test for OverlayContext epochToX/xToEpoch round-trip accuracy
- Playwright test: Spectrogram renders tiles at default zoom without errors

---

### Task 3: Footer Controls — ZoomSelector, PlaybackControls, EditToolbar

Refactor existing controls into independent, composable components that read from TimelineProvider context.

**Files:**
- Modify: `frontend/src/components/timeline/ZoomSelector.tsx` (read from context instead of props)
- Create: `frontend/src/components/timeline/controls/PlaybackControls.tsx` (new composable version)
- Create: `frontend/src/components/timeline/controls/EditToggle.tsx`
- Create: `frontend/src/components/timeline/controls/EditToolbar.tsx`
- Create: `frontend/src/components/timeline/controls/OverlayToggles.tsx`
- Create: `frontend/src/components/timeline/controls/EventNav.tsx`
- Create: `frontend/src/components/timeline/controls/TimelineFooter.tsx`

**Acceptance criteria:**
- [ ] ZoomSelector reads zoomLevels and active level from context, calls setZoomLevel
- [ ] ZoomSelector renders as its own full-width row (never shares horizontal space with other controls)
- [ ] PlaybackControls renders: timestamp, skip back/forward, play/pause circle, speed toggle, zoom +/-, children slot for right-side content
- [ ] PlaybackControls reads isPlaying/speed/centerTimestamp from context
- [ ] PlaybackControls accepts `variant="compact"` for review toolbars (smaller, no skip buttons)
- [ ] EditToggle accepts: active, enabled, label, onToggle — generic for both "Label" and "Edit Regions" use cases
- [ ] EditToolbar accepts: pendingCount, onSave, onCancel, isSaving, children (for AddModeButton etc.)
- [ ] OverlayToggles accepts array of {key, label, active} options + onToggle callback
- [ ] EventNav accepts: currentIndex, totalCount, onPrev, onNext
- [ ] TimelineFooter provides dark background, top border styling

**Tests needed:**
- Unit test: ZoomSelector renders all preset levels from context
- Unit test: PlaybackControls compact variant hides skip buttons
- Unit test: EditToolbar Save button disabled when pendingCount === 0
- Unit test: EventNav Prev disabled at index 0, Next disabled at last index

---

### Task 4: Epoch Adapters

Build hooks that convert job-relative region/event data to epoch coordinates and convert corrections back.

**Files:**
- Create: `frontend/src/components/timeline/adapters/useEpochRegions.ts`
- Create: `frontend/src/components/timeline/adapters/useEpochEvents.ts`

**Acceptance criteria:**
- [ ] useEpochRegions adds jobStartEpoch to padded_start_sec, padded_end_sec, start_sec, end_sec
- [ ] useEpochEvents adds jobStartEpoch to startSec, endSec, originalStartSec, originalEndSec
- [ ] Both memoize output (stable reference when inputs unchanged)
- [ ] Both handle empty arrays without error
- [ ] Export a utility `epochToJobRelative(epoch, jobStart)` for correction callbacks

**Tests needed:**
- Unit test: useEpochRegions round-trip (add offset, subtract offset = original)
- Unit test: useEpochEvents handles mixed EffectiveEvent correction types
- Unit test: empty input returns empty array with stable reference

---

### Task 5: Migrate Overlays to OverlayContext

Refactor existing overlays to consume OverlayContext instead of receiving coordinate props. Move call-parsing overlays into the shared overlays directory.

**Files:**
- Modify: `frontend/src/components/timeline/overlays/DetectionOverlay.tsx` (use useOverlayContext)
- Modify: `frontend/src/components/timeline/overlays/VocalizationOverlay.tsx` (use useOverlayContext)
- Modify: `frontend/src/components/timeline/overlays/RegionOverlay.tsx` (use useOverlayContext)
- Move + Modify: `frontend/src/components/call-parsing/RegionEditOverlay.tsx` → `frontend/src/components/timeline/overlays/RegionEditOverlay.tsx`
- Move + Modify: `frontend/src/components/call-parsing/EventBarOverlay.tsx` → `frontend/src/components/timeline/overlays/EventBarOverlay.tsx`
- Move + Modify: `frontend/src/components/call-parsing/RegionBandOverlay.tsx` → `frontend/src/components/timeline/overlays/RegionBandOverlay.tsx`
- Create: `frontend/src/components/timeline/overlays/RegionBoundaryMarkers.tsx`

**Acceptance criteria:**
- [ ] All overlays use useOverlayContext() for epochToX, xToEpoch, canvasWidth, canvasHeight
- [ ] No overlay receives centerTimestamp, zoomLevel, width, height, jobStart as props
- [ ] RegionEditOverlay works as a child of Spectrogram (no external positioning, no leftOffset prop)
- [ ] EventBarOverlay uses useOverlayContext() instead of the old RegionSpectrogramViewer context
- [ ] Edit overlays intercept mouse events via stopPropagation (pan continues on unclaimed areas)
- [ ] RegionBoundaryMarkers renders dimmed areas + dashed boundary lines for a single region
- [ ] DetectionOverlay click handler still triggers popover positioning correctly

**Tests needed:**
- Unit test: overlay positioning accuracy (known epochToX values produce expected pixel positions)
- Playwright test: clicking a detection bar opens popover at correct position
- Playwright test: dragging an event boundary in EventBarOverlay updates coordinates

---

### Task 6: Migrate RegionDetectionTimeline (fixes all three bugs)

Replace RegionTimelineViewer with a new thin composition using the shared components.

**Files:**
- Create: `frontend/src/components/call-parsing/RegionDetectionTimeline.tsx` (~100 lines)
- Modify: `frontend/src/App.tsx` or router config (update route to use new component)

**Acceptance criteria:**
- [ ] Uses TimelineProvider with FULL_ZOOM, gapless playback
- [ ] Uses Spectrogram with regionTileUrl builder
- [ ] RegionOverlay displayed in view mode, RegionEditOverlay in edit mode (both as children)
- [ ] Edit mode: regions are editable (drag boundaries) — bug (a) fixed
- [ ] ZoomSelector stays centered regardless of edit mode state — bug (b) fixed
- [ ] Drag scroll works in both view and edit mode; cursor shows grab/grabbing — bug (c) fixed
- [ ] Edit toolbar shows pending count, save, cancel, add mode button
- [ ] All existing functionality preserved: region click enters edit, add mode, save corrections
- [ ] Keyboard shortcuts work: space, +/-, arrows

**Tests needed:**
- Playwright test: enter edit mode, drag a region boundary, verify correction persists
- Playwright test: toggle edit mode, verify zoom controls do not shift position
- Playwright test: in edit mode, drag on empty area pans the viewport
- Playwright test: region overlay shows/hides via toggle button

---

### Task 7: Migrate ClassifierTimeline

Replace TimelineViewer with a new thin composition using the shared components.

**Files:**
- Create: `frontend/src/components/timeline/ClassifierTimeline.tsx` (~120 lines)
- Modify: `frontend/src/App.tsx` or router config (update route to use new component)

**Acceptance criteria:**
- [ ] Uses TimelineProvider with FULL_ZOOM, gapless playback
- [ ] Uses Spectrogram with classifierTileUrl builder, confidence scores
- [ ] DetectionOverlay and VocalizationOverlay switch via overlay mode toggle
- [ ] LabelEditor and VocLabelEditor render as overlay children in edit mode
- [ ] Label mode constraints preserved: only when stopped AND zoomed to 5m/1m
- [ ] LabelToolbar / VocLabelToolbar render conditionally in TimelineFooter
- [ ] TimelineHeader with hydrophone name, time range, embedding sync button
- [ ] Playback: gapless double-buffered with speed control (0.5x/1x/2x)
- [ ] All keyboard shortcuts work (space, +/-, arrows, u/h/o/s/b label keys)
- [ ] Dirty state warning on navigation (beforeunload)
- [ ] Extract dialog accessible from label toolbar

**Tests needed:**
- Playwright test: toggle detection overlay on/off, labels appear/disappear
- Playwright test: enter label mode at 1m zoom, make edit, save succeeds
- Playwright test: playback starts/stops, playhead visible at center
- Playwright test: zoom selector functional, zoom in/out buttons work

---

### Task 8: Migrate SegmentReviewWorkspace

Refactor SegmentReviewWorkspace to use TimelineProvider + Spectrogram + shared controls instead of RegionSpectrogramViewer.

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Uses TimelineProvider with REVIEW_ZOOM, slice playback
- [ ] Uses epoch adapters (useEpochRegions, useEpochEvents) for coordinate conversion
- [ ] Spectrogram renders with RegionBoundaryMarkers, RegionBandOverlay, EventBarOverlay as children
- [ ] ReviewToolbar uses shared EventNav, compact PlaybackControls, EditToolbar
- [ ] ZoomSelector from shared components (renders below spectrogram)
- [ ] Event navigation scrolls spectrogram to center on active event
- [ ] Drag-to-pan works (region band overlay at wide zoom, dim regions at narrow zoom)
- [ ] Boundary editing: drag event edges produces corrections
- [ ] Add mode: click to create new event
- [ ] Save/Cancel/Retrain/Re-segment functionality preserved
- [ ] Cross-region event navigation switches active region

**Tests needed:**
- Playwright test: navigate between events, spectrogram scrolls appropriately
- Playwright test: drag event boundary, verify adjustment recorded in pending corrections
- Playwright test: add mode creates new event, save persists it
- Playwright test: zoom selector changes viewport span

---

### Task 9: Migrate ClassifyReviewWorkspace

Refactor ClassifyReviewWorkspace to use TimelineProvider + Spectrogram + shared controls instead of RegionSpectrogramViewer.

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Uses TimelineProvider with REVIEW_ZOOM, slice playback
- [ ] Uses epoch adapters for coordinate conversion
- [ ] Spectrogram renders with EventBarOverlay as child
- [ ] Toolbar uses shared EventNav, compact PlaybackControls, EditToolbar
- [ ] ZoomSelector from shared components
- [ ] TypePalette and ClassifyDetailPanel render below spectrogram (unchanged)
- [ ] Event navigation + keyboard shortcuts (arrows, space, delete) preserved
- [ ] Type correction via palette click works
- [ ] Boundary correction via drag works
- [ ] Right-click context menu for adding events works
- [ ] Save persists both type and boundary corrections in parallel
- [ ] Retrain/Reclassify workflow preserved

**Tests needed:**
- Playwright test: navigate events, assign type via palette, save
- Playwright test: drag event boundary, verify boundary correction saved
- Playwright test: right-click → "Add event" creates pending event
- Playwright test: delete event via keyboard removes from display

---

### Task 10: Delete Old Components + Barrel Export

Remove superseded components and create clean barrel export.

**Files:**
- Delete: `frontend/src/components/timeline/SpectrogramViewport.tsx`
- Delete: `frontend/src/components/timeline/PlaybackControls.tsx` (old version)
- Delete: `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx`
- Delete: `frontend/src/components/call-parsing/RegionTimelineViewer.tsx`
- Delete: `frontend/src/components/timeline/TimelineViewer.tsx`
- Create: `frontend/src/components/timeline/index.ts` (barrel export)
- Modify: any remaining imports that reference deleted files

**Acceptance criteria:**
- [ ] No imports reference deleted files
- [ ] TypeScript compiles without errors (`npx tsc --noEmit`)
- [ ] All Playwright tests pass
- [ ] Barrel export provides clean public API: TimelineProvider, Spectrogram, all controls, all overlays, adapters, types
- [ ] No dead code remains (verify with grep for old component names)

**Tests needed:**
- Full Playwright suite passes
- TypeScript compilation passes
- Grep for old component names (SpectrogramViewport, RegionSpectrogramViewer, RegionTimelineViewer, TimelineViewer) returns zero hits in source files

---

### Verification

Run in order after all tasks complete:

1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test`
3. Manual verification: open each of the four timeline views in browser, confirm rendering, playback, edit mode, zoom, and pan all work correctly

---

### Task Dependencies

```
Task 1 (Provider)
  └→ Task 2 (Spectrogram)
       └→ Task 3 (Controls)
       └→ Task 5 (Overlays)
            └→ Task 6 (Migrate Region Detection) ← fixes all 3 bugs
            └→ Task 7 (Migrate Classifier)
            └→ Task 8 (Migrate Segment Review) ← requires Task 4
            └→ Task 9 (Migrate Classify Review) ← requires Task 4
                 └→ Task 10 (Cleanup)

Task 4 (Adapters) — independent, can be done any time before Tasks 8/9
```

Each task is a standalone PR. Tasks 1–5 build the shared infrastructure. Tasks 6–9 migrate one viewer each. Task 10 cleans up.
