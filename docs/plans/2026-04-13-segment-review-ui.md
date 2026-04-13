# Segment Review UI — Implementation Plan

**Goal:** Add an interactive Review tab to the Segment page where users can view, play, and correct event boundaries on a spectrogram, supporting the Pass 2 human-in-the-loop feedback workflow.
**Spec:** [docs/specs/2026-04-13-segment-review-ui-design.md](../specs/2026-04-13-segment-review-ui-design.md)

---

### Task 1: Backend — Tile Endpoint for Region Detection Jobs

Add a spectrogram tile endpoint for region detection jobs, reusing the existing tile generation infrastructure from the timeline viewer.

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `src/humpback/processing/timeline_tiles.py` (if tile generation needs to accept region job metadata)
- Modify: `src/humpback/services/call_parsing.py` (helper to resolve audio source from region job)

**Acceptance criteria:**
- [x] `GET /call-parsing/region-jobs/{job_id}/tile?zoom_level=5m&tile_index={n}` returns a PCEN spectrogram PNG
- [x] Tile generation resolves the audio source from the region detection job (hydrophone or file)
- [x] Returns 404 if job does not exist, 409 if job is not complete
- [x] Supports all existing zoom levels (frontend will fix to 5m)

**Tests needed:**
- Unit test: tile endpoint returns 404 for missing job, 409 for non-complete job
- Integration test: tile endpoint returns a valid PNG for a completed region detection job with a mock audio source

---

### Task 2: Frontend API Client and React Query Hooks

Add client functions and hooks for tile URLs, fetching regions, and correction CRUD.

**Files:**
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useCallParsing.ts`

**Acceptance criteria:**
- [x] `regionTileUrl(jobId, zoomLevel, tileIndex)` helper returns the tile URL
- [x] `fetchRegionJobRegions(jobId)` fetches regions for a region detection job
- [x] `fetchBoundaryCorrections(jobId)` fetches existing corrections for a segmentation job
- [x] `saveBoundaryCorrections(jobId, corrections)` POSTs batch corrections
- [x] `clearBoundaryCorrections(jobId)` DELETEs all corrections for a job
- [x] React Query hooks: `useRegionJobRegions`, `useBoundaryCorrections`, `useSaveBoundaryCorrections`, `useClearBoundaryCorrections`
- [x] `useBoundaryCorrections` only enabled when a job is selected
- [x] Mutation hooks invalidate the corrections query key on success

**Tests needed:**
- TypeScript type checking (`npx tsc --noEmit`) confirms no type errors

---

### Task 3: Tab Wrapper — Refactor SegmentPage into Jobs/Review Tabs

Wrap the existing Segment page content in a two-tab layout.

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentPage.tsx`

**Acceptance criteria:**
- [x] Page renders two tabs: "Jobs" and "Review"
- [x] Jobs tab contains the existing SegmentPage content (form + job tables) unchanged
- [x] Review tab renders the new `SegmentReviewWorkspace` component (placeholder initially)
- [x] Tab selection persists via URL query parameter (e.g., `?tab=review`)
- [x] Default tab is "Jobs"

**Tests needed:**
- Playwright: Segment page loads with two tabs, clicking each shows the correct content

---

### Task 4: SegmentReviewWorkspace and RegionSidebar

Build the Review tab orchestrator with job selection and region navigation.

**Files:**
- Create: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`
- Create: `frontend/src/components/call-parsing/RegionSidebar.tsx`

**Acceptance criteria:**
- [x] Job selector dropdown lists completed segmentation jobs with metadata (source, model, event/region counts)
- [x] Selecting a job fetches its regions (via the parent region detection job) and events
- [x] RegionSidebar renders a scrollable list of regions with: time range, event count, correction progress indicator
- [x] Progress indicators: pending (no corrections), partial (some events corrected), reviewed (all events have corrections)
- [x] Clicking a region selects it as the active region for the spectrogram panel
- [x] First region auto-selected on job load

**Tests needed:**
- Playwright: selecting a job populates the region sidebar, clicking a region highlights it

---

### Task 5: RegionSpectrogramViewer

Build the pannable spectrogram viewer scoped to a region's bounds, reusing TileCanvas.

**Files:**
- Create: `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx`

**Acceptance criteria:**
- [x] Renders PCEN spectrogram tiles at fixed 5-min zoom using `TileCanvas` (or equivalent tile rendering)
- [x] Viewport is scoped to the selected region's padded bounds (padded_start_sec to padded_end_sec)
- [x] Pan by dragging left/right, clamped to region bounds
- [x] Frequency axis on the left, time axis on the bottom (UTC labels)
- [x] Accepts children for overlay rendering (EventBarOverlay)

**Tests needed:**
- Playwright: selecting a region renders the spectrogram viewer, dragging pans the viewport

---

### Task 6: EventBarOverlay with Drag Interaction

Render event bars on the spectrogram with selectable, draggable boundary markers.

**Files:**
- Create: `frontend/src/components/call-parsing/EventBarOverlay.tsx`

**Acceptance criteria:**
- [x] Renders a horizontal bar per event, positioned by start_sec/end_sec converted to pixel coordinates
- [x] Click bar to select (one at a time), click empty space to deselect
- [x] Hover near left/right edge of selected bar shows `col-resize` cursor
- [x] Drag edge to adjust boundary with ghost preview during drag
- [x] Snap to 0.1s increments
- [x] Edges cannot cross each other or overlap adjacent events
- [x] On drag-end, records an `adjust` correction in pending edits (calls parent callback)
- [x] Visual states: original (solid fill), adjusted (dashed outline + solid fill), added (green outline), pending delete (red strikethrough at 30% opacity)
- [x] Add mode: crosshair cursor, click to place new event (1s default width), records `add` correction

**Tests needed:**
- Playwright: events render as bars on the spectrogram, clicking selects, visual states change for corrections

---

### Task 7: EventDetailPanel, ReviewToolbar, and Audio Playback

Build the event detail panel and toolbar with Save/Cancel/Add/Play actions.

**Files:**
- Create: `frontend/src/components/call-parsing/EventDetailPanel.tsx`
- Create: `frontend/src/components/call-parsing/ReviewToolbar.tsx`

**Acceptance criteria:**
- [x] EventDetailPanel shows: event timestamps, duration, confidence, correction status
- [x] When event is adjusted, shows original vs. adjusted values
- [x] "Play Slice" button plays the event's audio range (adjusted if modified)
- [x] "Delete Event" button marks event for deletion, toggles visual state
- [x] Clicking a deleted event in the detail panel undoes the delete
- [x] ReviewToolbar shows: region summary (time range, event count, correction count), Play, + Add, Save, Cancel buttons
- [x] Save button disabled when not dirty, shows unsaved change count badge when dirty
- [x] Cancel confirms if dirty ("Discard N unsaved changes?")
- [x] Retrain button present but disabled (placeholder)

**Tests needed:**
- Playwright: selecting an event populates detail panel, Save/Cancel buttons respond to dirty state

---

### Task 8: State Management — Accumulated Edits and Save Flow

Wire up the accumulated edits state, save/cancel flow, and dirty-checking across the workspace.

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [x] `pendingCorrections` Map accumulates adjust/add/delete corrections keyed by event_id (or `add-{uuid}` for adds)
- [x] `effectiveEvents` computed via useMemo: merges original events with pending corrections for rendering
- [x] Save serializes pending corrections into `BoundaryCorrectionRequest` and POSTs via `useSaveBoundaryCorrections`
- [x] On save success: clears pending state, refetches corrections
- [x] Cancel clears pending state after confirmation if dirty
- [x] Switching regions preserves pending edits (corrections accumulate across regions within a job)
- [x] Existing saved corrections fetched on job load and displayed as already-applied (not pending)
- [x] New edits layer on top of existing corrections

**Tests needed:**
- Playwright: make an adjustment, verify Save sends the correction, verify Cancel discards, verify dirty prompt on region switch

---

### Task 9: Jobs Tab "Review" Link

Add a review action to completed jobs in the existing job table.

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentJobTable.tsx`

**Acceptance criteria:**
- [x] Completed segmentation jobs show a "Review" button/link in their row
- [x] Clicking "Review" switches to the Review tab with that job pre-selected (via query parameter)
- [x] Only shown for jobs with status "complete"

**Tests needed:**
- Playwright: completed job row shows Review link, clicking it navigates to Review tab with job selected

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/api/routers/call_parsing.py src/humpback/processing/timeline_tiles.py src/humpback/services/call_parsing.py`
2. `uv run ruff check src/humpback/api/routers/call_parsing.py src/humpback/processing/timeline_tiles.py src/humpback/services/call_parsing.py`
3. `uv run pyright src/humpback/api/routers/call_parsing.py src/humpback/processing/timeline_tiles.py src/humpback/services/call_parsing.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
