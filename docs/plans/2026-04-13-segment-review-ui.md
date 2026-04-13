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
- [ ] `GET /call-parsing/region-jobs/{job_id}/tile?zoom_level=5m&tile_index={n}` returns a PCEN spectrogram PNG
- [ ] Tile generation resolves the audio source from the region detection job (hydrophone or file)
- [ ] Returns 404 if job does not exist, 409 if job is not complete
- [ ] Supports all existing zoom levels (frontend will fix to 5m)

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
- [ ] `regionTileUrl(jobId, zoomLevel, tileIndex)` helper returns the tile URL
- [ ] `fetchRegionJobRegions(jobId)` fetches regions for a region detection job
- [ ] `fetchBoundaryCorrections(jobId)` fetches existing corrections for a segmentation job
- [ ] `saveBoundaryCorrections(jobId, corrections)` POSTs batch corrections
- [ ] `clearBoundaryCorrections(jobId)` DELETEs all corrections for a job
- [ ] React Query hooks: `useRegionJobRegions`, `useBoundaryCorrections`, `useSaveBoundaryCorrections`, `useClearBoundaryCorrections`
- [ ] `useBoundaryCorrections` only enabled when a job is selected
- [ ] Mutation hooks invalidate the corrections query key on success

**Tests needed:**
- TypeScript type checking (`npx tsc --noEmit`) confirms no type errors

---

### Task 3: Tab Wrapper — Refactor SegmentPage into Jobs/Review Tabs

Wrap the existing Segment page content in a two-tab layout.

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentPage.tsx`

**Acceptance criteria:**
- [ ] Page renders two tabs: "Jobs" and "Review"
- [ ] Jobs tab contains the existing SegmentPage content (form + job tables) unchanged
- [ ] Review tab renders the new `SegmentReviewWorkspace` component (placeholder initially)
- [ ] Tab selection persists via URL query parameter (e.g., `?tab=review`)
- [ ] Default tab is "Jobs"

**Tests needed:**
- Playwright: Segment page loads with two tabs, clicking each shows the correct content

---

### Task 4: SegmentReviewWorkspace and RegionSidebar

Build the Review tab orchestrator with job selection and region navigation.

**Files:**
- Create: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`
- Create: `frontend/src/components/call-parsing/RegionSidebar.tsx`

**Acceptance criteria:**
- [ ] Job selector dropdown lists completed segmentation jobs with metadata (source, model, event/region counts)
- [ ] Selecting a job fetches its regions (via the parent region detection job) and events
- [ ] RegionSidebar renders a scrollable list of regions with: time range, event count, correction progress indicator
- [ ] Progress indicators: pending (no corrections), partial (some events corrected), reviewed (all events have corrections)
- [ ] Clicking a region selects it as the active region for the spectrogram panel
- [ ] First region auto-selected on job load

**Tests needed:**
- Playwright: selecting a job populates the region sidebar, clicking a region highlights it

---

### Task 5: RegionSpectrogramViewer

Build the pannable spectrogram viewer scoped to a region's bounds, reusing TileCanvas.

**Files:**
- Create: `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx`

**Acceptance criteria:**
- [ ] Renders PCEN spectrogram tiles at fixed 5-min zoom using `TileCanvas` (or equivalent tile rendering)
- [ ] Viewport is scoped to the selected region's padded bounds (padded_start_sec to padded_end_sec)
- [ ] Pan by dragging left/right, clamped to region bounds
- [ ] Frequency axis on the left, time axis on the bottom (UTC labels)
- [ ] Accepts children for overlay rendering (EventBarOverlay)

**Tests needed:**
- Playwright: selecting a region renders the spectrogram viewer, dragging pans the viewport

---

### Task 6: EventBarOverlay with Drag Interaction

Render event bars on the spectrogram with selectable, draggable boundary markers.

**Files:**
- Create: `frontend/src/components/call-parsing/EventBarOverlay.tsx`

**Acceptance criteria:**
- [ ] Renders a horizontal bar per event, positioned by start_sec/end_sec converted to pixel coordinates
- [ ] Click bar to select (one at a time), click empty space to deselect
- [ ] Hover near left/right edge of selected bar shows `col-resize` cursor
- [ ] Drag edge to adjust boundary with ghost preview during drag
- [ ] Snap to 0.1s increments
- [ ] Edges cannot cross each other or overlap adjacent events
- [ ] On drag-end, records an `adjust` correction in pending edits (calls parent callback)
- [ ] Visual states: original (solid fill), adjusted (dashed outline + solid fill), added (green outline), pending delete (red strikethrough at 30% opacity)
- [ ] Add mode: crosshair cursor, click to place new event (1s default width), records `add` correction

**Tests needed:**
- Playwright: events render as bars on the spectrogram, clicking selects, visual states change for corrections

---

### Task 7: EventDetailPanel, ReviewToolbar, and Audio Playback

Build the event detail panel and toolbar with Save/Cancel/Add/Play actions.

**Files:**
- Create: `frontend/src/components/call-parsing/EventDetailPanel.tsx`
- Create: `frontend/src/components/call-parsing/ReviewToolbar.tsx`

**Acceptance criteria:**
- [ ] EventDetailPanel shows: event timestamps, duration, confidence, correction status
- [ ] When event is adjusted, shows original vs. adjusted values
- [ ] "Play Slice" button plays the event's audio range (adjusted if modified)
- [ ] "Delete Event" button marks event for deletion, toggles visual state
- [ ] Clicking a deleted event in the detail panel undoes the delete
- [ ] ReviewToolbar shows: region summary (time range, event count, correction count), Play, + Add, Save, Cancel buttons
- [ ] Save button disabled when not dirty, shows unsaved change count badge when dirty
- [ ] Cancel confirms if dirty ("Discard N unsaved changes?")
- [ ] Retrain button present but disabled (placeholder)

**Tests needed:**
- Playwright: selecting an event populates detail panel, Save/Cancel buttons respond to dirty state

---

### Task 8: State Management — Accumulated Edits and Save Flow

Wire up the accumulated edits state, save/cancel flow, and dirty-checking across the workspace.

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] `pendingCorrections` Map accumulates adjust/add/delete corrections keyed by event_id (or `add-{uuid}` for adds)
- [ ] `effectiveEvents` computed via useMemo: merges original events with pending corrections for rendering
- [ ] Save serializes pending corrections into `BoundaryCorrectionRequest` and POSTs via `useSaveBoundaryCorrections`
- [ ] On save success: clears pending state, refetches corrections
- [ ] Cancel clears pending state after confirmation if dirty
- [ ] Switching regions with unsaved changes prompts save/discard
- [ ] Existing saved corrections fetched on job load and displayed as already-applied (not pending)
- [ ] New edits layer on top of existing corrections

**Tests needed:**
- Playwright: make an adjustment, verify Save sends the correction, verify Cancel discards, verify dirty prompt on region switch

---

### Task 9: Jobs Tab "Review" Link

Add a review action to completed jobs in the existing job table.

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentJobTable.tsx`

**Acceptance criteria:**
- [ ] Completed segmentation jobs show a "Review" button/link in their row
- [ ] Clicking "Review" switches to the Review tab with that job pre-selected (via query parameter)
- [ ] Only shown for jobs with status "complete"

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
