# Event Classifier UI Implementation Plan

**Goal:** Build the frontend UI for Pass 3 event classification — job management, type label review/correction workspace, and model training — plus two backend endpoint additions.
**Spec:** [docs/specs/2026-04-14-event-classifier-ui-design.md](../specs/2026-04-14-event-classifier-ui-design.md)

---

### Task 1: Backend — classification jobs with correction counts endpoint

Add the `GET /call-parsing/classification-jobs/with-correction-counts` endpoint mirroring the segmentation equivalent. Traces through segmentation job → region detection job to include hydrophone info.

**Files:**
- Modify: `src/humpback/services/call_parsing.py`
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `src/humpback/schemas/call_parsing.py`

**Acceptance criteria:**
- [ ] New service function `list_classification_jobs_with_correction_counts` uses LEFT JOIN subquery on `event_type_corrections` to count corrections per classification job
- [ ] Joins through `EventSegmentationJob` → `RegionDetectionJob` to include `hydrophone_id`, `start_timestamp`, `end_timestamp`
- [ ] Filters to completed classification jobs, ordered by `created_at DESC`
- [ ] New schema `ClassificationJobWithCorrectionCount` extends `EventClassificationJobSummary` with `correction_count`, `hydrophone_id`, `start_timestamp`, `end_timestamp`
- [ ] New router endpoint at `/classification-jobs/with-correction-counts` returns the schema list

**Tests needed:**
- Unit test creating classification jobs with and without type corrections, verifying counts and hydrophone info in response

---

### Task 2: Backend — add region_id to typed events endpoint

Modify the `GET /classification-jobs/{job_id}/typed-events` endpoint to include `region_id` by joining `typed_events.parquet` with `events.parquet` on `event_id`.

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`

**Acceptance criteria:**
- [ ] `get_classification_typed_events` loads both `typed_events.parquet` and `events.parquet` (from the segmentation job directory) for the classification job
- [ ] Builds an `event_id → region_id` lookup from `events.parquet`
- [ ] Each typed event response row includes `region_id` field
- [ ] Falls back gracefully if events.parquet is unavailable (404)

**Tests needed:**
- Unit test verifying `region_id` appears in typed events response for a completed classification job

---

### Task 3: Frontend — API client functions and types for Pass 3

Add TypeScript types and API client functions for classification jobs, typed events, type corrections, classifier training jobs, and classifier models.

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`

**Acceptance criteria:**
- [ ] Types added: `EventClassificationJob`, `TypedEventRow`, `TypeCorrection`, `TypeCorrectionResponse`, `EventClassifierTrainingJobCreate`, `EventClassifierTrainingJob`, `EventClassifierModel`, `ClassificationJobWithCorrectionCount`
- [ ] Client functions: `fetchClassificationJobs`, `createClassificationJob`, `deleteClassificationJob`, `fetchTypedEvents`, `fetchTypeCorrections`, `saveTypeCorrections`, `clearTypeCorrections`, `fetchClassificationJobsWithCorrectionCounts`, `fetchClassifierTrainingJobs`, `createClassifierTrainingJob`, `deleteClassifierTrainingJob`, `fetchClassifierModels`, `deleteClassifierModel`
- [ ] All functions point to the correct `/call-parsing/` API paths

**Tests needed:**
- TypeScript compilation verifies type correctness (`npx tsc --noEmit`)

---

### Task 4: Frontend — TanStack Query hooks for Pass 3

Create hooks wrapping all Pass 3 API client functions with proper query keys and cache invalidation.

**Files:**
- Modify: `frontend/src/hooks/queries/useCallParsing.ts`

**Acceptance criteria:**
- [ ] Hooks added for all 13 operations listed in the spec (§8 New Hooks table)
- [ ] Query hooks use consistent query key naming (`classification-jobs`, `typed-events`, `type-corrections`, `classifier-training-jobs`, `classifier-models`, `classification-jobs-correction-counts`)
- [ ] Mutation hooks invalidate relevant query keys on success
- [ ] `useClassificationJobs` accepts optional `refetchInterval` parameter (default 3000)
- [ ] `useTypedEvents` and `useTypeCorrections` accept `jobId: string | null` with `enabled: !!jobId`

**Tests needed:**
- TypeScript compilation verifies hook signatures

---

### Task 5: Frontend — ClassifyPage with Jobs tab

Build the ClassifyPage shell with Jobs and Review tabs, and the Jobs tab content: job creation form, active jobs table, previous jobs table with expandable type summary rows.

**Files:**
- Create: `frontend/src/components/call-parsing/ClassifyPage.tsx`
- Create: `frontend/src/components/call-parsing/ClassifyJobForm.tsx`
- Create: `frontend/src/components/call-parsing/ClassifyJobTable.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/layout/SideNav.tsx`

**Acceptance criteria:**
- [ ] ClassifyPage has Jobs and Review tabs using `useSearchParams` (mirrors SegmentPage pattern)
- [ ] Route registered at `/app/call-parsing/classify` in App.tsx
- [ ] SideNav adds "Classify" and "Classify Training" entries under Call Parsing group
- [ ] ClassifyJobForm has segmentation job selector (completed only) and event classifier model selector (filtered to `pytorch_event_cnn`)
- [ ] Supports `?segmentJobId={id}` query param to pre-fill segmentation job selector
- [ ] ClassifyJobTable shows active jobs (queued/running) with 3s auto-refresh, and previous jobs (completed/failed) with Review/Delete actions
- [ ] Previous job rows are expandable showing type summary: type name, count, mean score, % of events
- [ ] Bulk delete via checkbox selection on previous jobs

**Tests needed:**
- Playwright test verifying page loads, tabs switch, and job form renders

---

### Task 6: Frontend — ClassifyReviewWorkspace

Build the review workspace: job selector, toolbar with event navigation and playback, spectrogram viewer reusing RegionSpectrogramViewer, zoom bar, type palette, detail panel. Wire up correction state management and save/cancel flow.

**Files:**
- Create: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`
- Create: `frontend/src/components/call-parsing/TypePalette.tsx`
- Create: `frontend/src/components/call-parsing/ClassifyDetailPanel.tsx`

**Acceptance criteria:**
- [ ] Job selector lists completed classification jobs, auto-selects `reviewJobId` from query params
- [ ] Traces job chain: classification job → segmentation job → region detection job to get spectrogram tiles and audio
- [ ] Loads typed events, aggregates per event_id (highest above-threshold score = predicted type), deduplicates and sorts by start_sec
- [ ] Loads existing type corrections and merges with predicted types to show current state
- [ ] Event navigation with Prev/Next buttons and keyboard shortcuts (←/→ or [/])
- [ ] Spectrogram viewer shows region tiles with event bars color-coded by type, current event centered and highlighted
- [ ] Automatic region switching when navigating across region boundaries
- [ ] TypePalette shows all vocalization types from DB + (Negative) + "Add Type" button. Click to select active type, Enter to stamp onto current event
- [ ] ClassifyDetailPanel shows predicted type, confidence, correction status, time range, all model scores sorted descending
- [ ] Playback via Space key, audio slice from region detection job
- [ ] Backspace/Delete marks current event as negative
- [ ] Escape deselects active type
- [ ] Save batches pending corrections via `POST /classification-jobs/{id}/corrections`
- [ ] Cancel discards unsaved changes with confirmation dialog
- [ ] Unsaved changes trigger confirmation on navigation away
- [ ] Retrain button calls `POST /classifier-training-jobs` with current job as sole source, shows spinner, offers "Re-classify" on completion

**Tests needed:**
- Playwright test verifying workspace loads with a classification job, event navigation works, type palette renders

---

### Task 7: Frontend — ClassifyTrainingPage

Build the training page with model table, classification job picker with correction counts, and training job table.

**Files:**
- Create: `frontend/src/components/call-parsing/ClassifyTrainingPage.tsx`
- Create: `frontend/src/components/call-parsing/ClassifyModelTable.tsx`
- Create: `frontend/src/components/call-parsing/ClassificationJobPicker.tsx`
- Create: `frontend/src/components/call-parsing/ClassifyTrainingJobTable.tsx`
- Modify: `frontend/src/App.tsx`

**Acceptance criteria:**
- [ ] Route registered at `/app/call-parsing/classify-training` in App.tsx
- [ ] ClassifyModelTable lists `pytorch_event_cnn` models with name, type count, avg F1 (color-coded), created date, delete action
- [ ] Model rows expandable showing per-type metrics: type name, precision, recall, F1, threshold, sample count
- [ ] ClassificationJobPicker shows completed classification jobs with correction counts (checkbox, hydrophone, date range, correction count)
- [ ] "Train Model" button creates a training job with selected job IDs
- [ ] ClassifyTrainingJobTable lists training jobs with status, source job count, created time, resulting model link, error message, delete action
- [ ] Auto-refresh every 3s for in-flight training jobs

**Tests needed:**
- Playwright test verifying training page loads with all three sections

---

### Task 8: Cross-page navigation wiring

Add "Classify →" action to Segment page for completed segmentation jobs, linking to the Classify page with pre-fill.

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentJobTable.tsx`

**Acceptance criteria:**
- [ ] Completed segmentation jobs in the previous jobs table show a "Classify →" action button
- [ ] Button navigates to `/app/call-parsing/classify?tab=jobs&segmentJobId={id}`

**Tests needed:**
- Verify navigation link renders on completed segmentation job rows

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py`
2. `uv run ruff check src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py`
3. `uv run pyright src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
