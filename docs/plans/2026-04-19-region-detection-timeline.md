# Region Detection Timeline Viewer Implementation Plan

**Goal:** Add a read-only timeline viewer for Call Parsing Detection jobs showing regions overlaid on PCEN spectrograms with trace score heatmap and audio playback.
**Spec:** [docs/specs/2026-04-19-region-detection-timeline-design.md](../specs/2026-04-19-region-detection-timeline-design.md)

---

### Task 1: Fix confidence heatmap alignment bug

**Files:**
- Modify: `src/humpback/api/routers/timeline.py`
- Modify: `frontend/src/components/timeline/SpectrogramViewport.tsx`

**Acceptance criteria:**
- [ ] Backend bucket count uses ceiling division so detections in the final partial window are captured
- [ ] Backend returns the effective `window_sec` (based on actual bucketing) rather than the raw model parameter
- [ ] Frontend uses the backend's returned `window_sec` for heatmap bar positioning instead of recalculating from `totalDuration / scores.length`
- [ ] Existing classifier detection timeline heatmap bars align correctly with detection overlay bars

**Tests needed:**
- Backend unit test for confidence endpoint bucket count with non-integer division (e.g., 86401s / 5.0s)
- Verify no regressions in existing timeline confidence endpoint response shape

---

### Task 2: Add region job confidence endpoint

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`

**Acceptance criteria:**
- [ ] `GET /call-parsing/region-jobs/{job_id}/confidence` reads `trace.parquet` and returns bucketed scores in the same response format as the existing timeline confidence endpoint (array of scores + `window_sec`)
- [ ] Endpoint returns 404 for non-existent jobs and 409 for incomplete jobs
- [ ] Bucketing uses the same ceiling-division logic from Task 1

**Tests needed:**
- Unit test for confidence endpoint with mock trace data
- Test error responses for missing/incomplete jobs

---

### Task 3: Refactor SpectrogramViewport URL abstraction

**Files:**
- Modify: `frontend/src/components/timeline/SpectrogramViewport.tsx`
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`

**Acceptance criteria:**
- [ ] `SpectrogramViewport` accepts `tileUrlBuilder` and `audioSliceUrlBuilder` props instead of hardcoding classifier detection URL patterns
- [ ] `TimelineViewer` passes classifier detection URL builders to `SpectrogramViewport`
- [ ] Existing classifier detection timeline continues to work identically after refactor

**Tests needed:**
- Frontend type check (`npx tsc --noEmit`) confirms no type errors
- Manual verification that existing timeline still loads tiles and plays audio

---

### Task 4: Create RegionOverlay component

**Files:**
- Create: `frontend/src/components/timeline/RegionOverlay.tsx`
- Modify: `frontend/src/components/timeline/SpectrogramViewport.tsx`

**Acceptance criteria:**
- [ ] `RegionOverlay` renders full-height semi-transparent rectangles for each region
- [ ] Regions positioned using absolute epoch seconds (job-relative `start_sec`/`end_sec` + `jobStart`)
- [ ] Opacity derived from region `max_score` (higher = more visible)
- [ ] `SpectrogramViewport` accepts `overlayMode="region"` and renders `RegionOverlay` when set
- [ ] No click handling, labels, or tooltips

**Tests needed:**
- Frontend type check confirms no type errors
- Manual verification that regions render at correct positions on spectrogram

---

### Task 5: Create RegionTimelineViewer and wire routing

**Files:**
- Create: `frontend/src/components/call-parsing/RegionTimelineViewer.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/call-parsing/RegionJobTable.tsx`
- Modify: `frontend/src/hooks/queries/useCallParsing.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/api/types.ts`

**Acceptance criteria:**
- [ ] `RegionTimelineViewer` fetches region job, regions, and confidence data
- [ ] Renders `SpectrogramViewport` with `overlayMode="region"`, read-only props, and region job URL builders
- [ ] `audioSliceUrlBuilder` subtracts `jobStart` from absolute timestamps to produce job-relative `start_sec`
- [ ] Header shows job metadata (hydrophone, time range, region count), zoom controls, and playback controls
- [ ] No overlay toggles, label mode buttons, or editing UI
- [ ] Route `/app/call-parsing/region-timeline/:jobId` renders the component
- [ ] "Timeline" button in `RegionJobTable` enabled and navigates to the route
- [ ] `useRegionJobConfidence` hook and `fetchRegionJobConfidence` client function added

**Tests needed:**
- Frontend type check confirms no type errors
- Manual verification: navigate from detection page → timeline → spectrogram renders with region overlay and heatmap

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/api/routers/timeline.py src/humpback/api/routers/call_parsing.py`
2. `uv run ruff check src/humpback/api/routers/timeline.py src/humpback/api/routers/call_parsing.py`
3. `uv run pyright src/humpback/api/routers/timeline.py src/humpback/api/routers/call_parsing.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
