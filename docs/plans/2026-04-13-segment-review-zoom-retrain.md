# Segment Review: Zoom Buttons & Retrain Feedback Loop — Implementation Plan

**Goal:** Add manual zoom controls to the Segment Review spectrogram and wire the existing feedback training backend into the Review UI to close the human-in-the-loop segmentation model improvement loop.
**Spec:** [docs/specs/2026-04-13-segment-review-zoom-retrain-design.md](../specs/2026-04-13-segment-review-zoom-retrain-design.md)

---

### Task 1: Add zoom buttons to RegionSpectrogramViewer

**Files:**
- Modify: `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx`

**Acceptance criteria:**
- [ ] `zoomLevel` is lifted from a derived const to `useState`, initialized by `selectZoomLevel(regionDuration)`
- [ ] Zoom resets to auto-selected level when `region.region_id` changes
- [ ] A zoom button bar renders below the time axis with four buttons: `10s`, `30s`, `1m`, `5m`
- [ ] Active zoom level button is visually highlighted (blue border/background)
- [ ] Clicking a zoom button updates `zoomLevel` state, which drives `TILE_DURATION_SEC`, `VIEWPORT_SPAN_SEC`, `TICK_INTERVAL_SEC`, and `pxPerSec`
- [ ] `centerTimestamp` is preserved on zoom change; clamping keeps the viewport within region bounds
- [ ] All four zoom levels are always enabled regardless of region duration

**Tests needed:**
- Playwright test: zoom bar renders with four buttons when a region is loaded
- Playwright test: clicking a zoom button changes the active button styling

---

### Task 2: Add feedback training types, client functions, and hooks

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useCallParsing.ts`

**Acceptance criteria:**
- [ ] `SegmentationFeedbackTrainingJob` type added to `types.ts` with fields: id, status, source_job_ids, config_json, segmentation_model_id, result_summary, error_message, created_at, updated_at, started_at, completed_at
- [ ] `CreateSegmentationFeedbackTrainingJobRequest` type added with `source_job_ids: string[]` and optional config
- [ ] Client functions added to `client.ts`: `fetchSegmentationFeedbackTrainingJobs`, `createSegmentationFeedbackTrainingJob`, `deleteSegmentationFeedbackTrainingJob`
- [ ] Hook `useSegmentationFeedbackTrainingJobs(refetchInterval?)` added — GET with optional polling
- [ ] Hook `useCreateSegmentationFeedbackTrainingJob()` added — POST mutation, invalidates feedback training jobs and segmentation models queries
- [ ] Hook `useDeleteSegmentationFeedbackTrainingJob()` added — DELETE mutation, invalidates feedback training jobs query
- [ ] TypeScript compiles cleanly (`npx tsc --noEmit`)

**Tests needed:**
- None (hooks follow existing patterns; tested via integration in Tasks 3–4)

---

### Task 3: Wire retrain and re-segment into ReviewToolbar and SegmentReviewWorkspace

**Files:**
- Modify: `frontend/src/components/call-parsing/ReviewToolbar.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] `ReviewToolbar` accepts new props: `hasCorrections`, `onRetrain`, `retrainStatus`, `onResegment`
- [ ] Retrain button is enabled when `hasCorrections` is true and `isDirty` is false; disabled otherwise (no longer permanently disabled)
- [ ] Clicking Retrain opens a `window.confirm` dialog; on confirm calls `onRetrain`
- [ ] Training status displays in toolbar: spinner + "Training..." for queued/running, "Model ready" + green Re-segment button for complete, red "Training failed" (with error in title attribute) for failed
- [ ] Retrain button is hidden while a job is queued/running; visible for complete (allows another round) and failed (allows retry)
- [ ] Re-segment button click opens a `window.confirm` dialog; on confirm calls `onResegment`
- [ ] `SegmentReviewWorkspace` manages retrain state: uses `useSegmentationFeedbackTrainingJobs` to find jobs for the current segmentation job, uses `useCreateSegmentationFeedbackTrainingJob` for retrain action, uses `useCreateSegmentationJob` for re-segment action
- [ ] Polling at 3-second interval activates only when a matching job is queued or running; no polling otherwise
- [ ] Re-segment creates a new `EventSegmentationJob` with the same `region_detection_job_id` and the trained model's `segmentation_model_id`
- [ ] Toast notifications on retrain start and re-segment creation

**Tests needed:**
- Playwright test: Retrain button is disabled when no corrections exist for the job
- Playwright test: Retrain button is enabled when corrections exist and no unsaved changes
- Playwright test: clicking Retrain shows a confirmation dialog

---

### Task 4: Replace bootstrap UI with FeedbackTrainingJobTable on Segment Training page

**Files:**
- Create: `frontend/src/components/call-parsing/FeedbackTrainingJobTable.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentTrainingPage.tsx`
- Delete: `frontend/src/components/call-parsing/SegmentTrainingForm.tsx`
- Delete: `frontend/src/components/call-parsing/SegmentTrainingJobTable.tsx`

**Acceptance criteria:**
- [ ] `FeedbackTrainingJobTable` renders feedback training jobs with columns: Status, Created, Source Jobs (truncated UUIDs), Config (summary), Model (link if complete), Metrics (F1 scores), Delete button
- [ ] Table uses `useSegmentationFeedbackTrainingJobs(3000)` for 3-second polling
- [ ] Delete button uses `useDeleteSegmentationFeedbackTrainingJob` with confirmation
- [ ] `SegmentTrainingPage` renders `SegmentModelTable` + `FeedbackTrainingJobTable` (no create form)
- [ ] `SegmentTrainingForm.tsx` deleted
- [ ] `SegmentTrainingJobTable.tsx` deleted

**Tests needed:**
- Playwright test: Segment Training page renders models table and feedback training jobs table
- Playwright test: feedback training jobs table shows "No training jobs yet" when empty

---

### Task 5: Remove bootstrap training hooks and client functions

**Files:**
- Modify: `frontend/src/hooks/queries/useCallParsing.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/api/types.ts`

**Acceptance criteria:**
- [ ] `useCreateSegmentationTrainingJob`, `useSegmentationTrainingJobs`, `useDeleteSegmentationTrainingJob`, `useSegmentationTrainingDatasets` hooks removed
- [ ] Corresponding client functions removed from `client.ts` (`fetchSegmentationTrainingJobs`, `createSegmentationTrainingJob`, `deleteSegmentationTrainingJob`, `fetchSegmentationTrainingDatasets`)
- [ ] `SegmentationTrainingJob` and `CreateSegmentationTrainingJobRequest` types removed from `types.ts` (if only used by deleted code)
- [ ] No remaining imports reference deleted hooks, functions, or types
- [ ] TypeScript compiles cleanly (`npx tsc --noEmit`)

**Tests needed:**
- None (removal verified by TypeScript compilation)

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test`
3. `uv run pytest tests/`
