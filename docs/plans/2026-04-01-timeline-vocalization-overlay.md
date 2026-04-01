# Timeline Vocalization Overlay Implementation Plan

**Goal:** Add a toggleable vocalization type overlay to the timeline viewer, showing inference suggestions and human-applied labels as colored bars with type badges.
**Spec:** [docs/specs/2026-04-01-timeline-vocalization-overlay-design.md](../specs/2026-04-01-timeline-vocalization-overlay-design.md)

---

### Task 1: Backend — Add bulk vocalization labels endpoint

**Files:**
- Modify: `src/humpback/api/routers/labeling.py`
- Modify: `src/humpback/schemas/labeling.py` (if response schema needed)

**Acceptance criteria:**
- [ ] `GET /labeling/vocalization-labels/{detection_job_id}/all` returns all vocalization labels for the job without requiring start/end range params
- [ ] Response is a list of `VocalizationLabel` objects (same schema as existing endpoint)
- [ ] Empty list returned when no labels exist (not 404)

**Tests needed:**
- Test endpoint returns all labels for a detection job
- Test endpoint returns empty list when no labels exist
- Test endpoint returns 404 for nonexistent detection job

---

### Task 2: Frontend — API client and types

**Files:**
- Modify: `frontend/src/api/client.ts`

**Acceptance criteria:**
- [ ] `fetchAllVocalizationLabels(detectionJobId)` calls the new `/all` endpoint
- [ ] Function returns `VocalizationLabel[]` (type already exists)

**Tests needed:**
- Type-check only (npx tsc --noEmit)

---

### Task 3: Frontend — useVocalizationOverlay hook

**Files:**
- Create: `frontend/src/hooks/queries/useVocalizationOverlay.ts`

**Acceptance criteria:**
- [ ] Hook accepts `jobId: string`
- [ ] Fetches all vocalization labels via the new client function
- [ ] Fetches inference jobs and filters for completed jobs with `source_type === "detection_job"` and `source_id === jobId`
- [ ] Returns `{ labels: VocalizationLabel[], hasVocalizationData: boolean, isLoading: boolean }`
- [ ] `hasVocalizationData` is true when at least one completed inference job exists OR manual labels exist
- [ ] Both queries use appropriate TanStack Query stale times (labels can be stale for 30s, inference job list for 60s)

**Tests needed:**
- Type-check only (npx tsc --noEmit)

---

### Task 4: Frontend — VocalizationOverlay component

**Files:**
- Create: `frontend/src/components/timeline/VocalizationOverlay.tsx`
- Modify: `frontend/src/components/timeline/constants.ts`

**Acceptance criteria:**
- [ ] Component accepts `labels: VocalizationLabel[]`, `centerTimestamp`, `zoomLevel`, `width`, `height`, `visible`
- [ ] Groups labels by `(start_utc, end_utc)` to render one bar per window
- [ ] Bar color is muted purple (`rgba(168, 130, 220, 0.40)`), hover brightens
- [ ] Each bar displays colored badge chips for each vocalization type on that window
- [ ] Badge colors assigned dynamically from a palette keyed by type name
- [ ] Inference-sourced labels render outlined badges; manual labels render filled badges
- [ ] Tooltip on hover shows: type name(s), source (inference/manual), confidence if present, time range
- [ ] Returns null when `visible` is false
- [ ] No click/edit handlers — read-only

**Tests needed:**
- Type-check (npx tsc --noEmit)
- Visual verification in browser

---

### Task 5: Frontend — Integrate overlay mode into TimelineViewer

**Files:**
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`
- Modify: `frontend/src/components/timeline/SpectrogramViewport.tsx`
- Modify: `frontend/src/components/timeline/PlaybackControls.tsx`

**Acceptance criteria:**
- [ ] New state `overlayMode: "detection" | "vocalization"` in TimelineViewer (default `"detection"`)
- [ ] `useVocalizationOverlay` hook called in TimelineViewer
- [ ] SpectrogramViewport renders VocalizationOverlay when `overlayMode === "vocalization"`, DetectionOverlay when `"detection"`
- [ ] Existing `showLabels` toggle controls visibility of whichever overlay is active
- [ ] New "Vocalizations" button in PlaybackControls right-side group, next to "Labels: ON/OFF"
- [ ] Button disabled with tooltip when `hasVocalizationData` is false
- [ ] Button highlighted when `overlayMode === "vocalization"`
- [ ] Toggling vocalization mode while in label edit mode exits label mode first
- [ ] PlaybackControls props extended for `overlayMode`, `onToggleOverlayMode`, `hasVocalizationData`

**Tests needed:**
- Type-check (npx tsc --noEmit)
- Playwright test: verify button appears disabled when no vocalization data
- Playwright test: verify button toggles overlay mode when vocalization data exists
- Visual verification in browser

---

### Task 6: Backend test for bulk labels endpoint

**Files:**
- Modify: `tests/test_labeling_api.py` (or appropriate existing test file)

**Acceptance criteria:**
- [ ] Test creates a detection job with vocalization labels and verifies `/all` returns them
- [ ] Test verifies empty list for job with no labels
- [ ] All tests pass with `uv run pytest tests/`

**Tests needed:**
- Described above

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/api/routers/labeling.py src/humpback/schemas/labeling.py`
2. `uv run ruff check src/humpback/api/routers/labeling.py src/humpback/schemas/labeling.py`
3. `uv run pyright src/humpback/api/routers/labeling.py src/humpback/schemas/labeling.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
