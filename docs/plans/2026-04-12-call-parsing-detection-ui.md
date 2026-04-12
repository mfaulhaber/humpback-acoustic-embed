# Call Parsing Detection UI — Implementation Plan

**Goal:** Add a frontend page for creating and managing Call Parsing Pass 1 region detection jobs, under a new "Call Parsing" nav group.
**Spec:** [docs/specs/2026-04-12-call-parsing-detection-ui-design.md](../specs/2026-04-12-call-parsing-detection-ui-design.md)

---

### Task 1: TypeScript Types and API Hooks

**Files:**
- Modify: `frontend/src/api/types.ts`
- Create: `frontend/src/hooks/queries/useCallParsing.ts`

**Acceptance criteria:**
- [ ] `RegionDetectionJob` interface in types.ts matches the `RegionDetectionJobSummary` Pydantic schema (id, status, hydrophone_id, start_timestamp, end_timestamp, model_config_id, classifier_model_id, parent_run_id, error_message, trace_row_count, region_count, created_at, updated_at, started_at, completed_at)
- [ ] `RegionDetectionConfig` interface for advanced settings (window_size_seconds, hop_seconds, high_threshold, low_threshold, padding_sec, min_region_duration_sec, stream_chunk_sec)
- [ ] `CreateRegionJobRequest` interface matching the Pydantic request schema
- [ ] `useRegionDetectionJobs()` query hook fetches `GET /call-parsing/region-jobs` with configurable refetch interval
- [ ] `useCreateRegionJob()` mutation hook posts to `POST /call-parsing/region-jobs` and invalidates the jobs query
- [ ] `useDeleteRegionJob()` mutation hook calls `DELETE /call-parsing/region-jobs/{id}` and invalidates the jobs query
- [ ] Model config resolution helper: given a classifier model's `model_version`, look up the matching `model_config_id` from the model configs list

**Tests needed:**
- No unit tests for hooks (covered by E2E)

---

### Task 2: RegionJobForm Component

**Files:**
- Create: `frontend/src/components/call-parsing/RegionJobForm.tsx`

**Acceptance criteria:**
- [ ] Hydrophone dropdown using `useHydrophones()`, showing Orcasound + NOAA sources
- [ ] `DateRangePickerUtc` for date range selection
- [ ] Classifier model dropdown using `useClassifierModels()`, showing name + version
- [ ] High threshold slider (0–1, default 0.90) with real-time value display
- [ ] Low threshold slider (0–1, default 0.80) with real-time value display
- [ ] Collapsible "Advanced Settings" section (collapsed by default) with: hop size (default 1.0), padding (default 1.0), min region duration (default 0.0), stream chunk size (default 1800)
- [ ] "Start Detection" button disabled when required fields missing or mutation pending
- [ ] On submit, resolves `model_config_id` from selected classifier's `model_version` and calls `useCreateRegionJob()` with the full request payload
- [ ] Error message display for failed submissions

**Tests needed:**
- Playwright test: form renders with all fields, submit button disabled until required fields filled

---

### Task 3: RegionJobTable Component

**Files:**
- Create: `frontend/src/components/call-parsing/RegionJobTable.tsx`
- Create: `frontend/src/components/call-parsing/RegionJobSummary.tsx`

**Acceptance criteria:**
- [ ] Accepts a `mode` prop: `"active"` or `"previous"` to control column set and features
- [ ] Active mode columns: Status, Created, Hydrophone, Date Range, Thresholds, Actions (Cancel button)
- [ ] Previous mode columns: Checkbox, Status, Created, Hydrophone, Date Range, Thresholds, Regions, Timeline (disabled), Error
- [ ] Cancel button in active mode calls `useDeleteRegionJob()` (delete = cancel for in-flight jobs)
- [ ] Previous mode: checkbox selection for bulk delete, "Delete (N)" button with `BulkDeleteDialog` confirmation
- [ ] Previous mode: "Filter by hydrophone..." search input
- [ ] Previous mode: pagination controls with configurable page size (10/20/50/100)
- [ ] Previous mode: column sorting via header click, default sort Created descending
- [ ] Timeline column shows a disabled button as placeholder
- [ ] `RegionJobSummary` renders region count and threshold values in compact form
- [ ] Hydrophone name resolved from hydrophone_id using the hydrophones list

**Tests needed:**
- Playwright test: active table shows running job, cancel button works
- Playwright test: previous table shows completed job with region count, sorting and pagination work

---

### Task 4: DetectionPage Shell and Routing

**Files:**
- Create: `frontend/src/components/call-parsing/DetectionPage.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/layout/SideNav.tsx`

**Acceptance criteria:**
- [ ] `DetectionPage` composes `RegionJobForm` (top) and two `RegionJobTable` instances (active panel, previous panel)
- [ ] Active panel: titled "Active Jobs" with badge count, auto-refreshes at 3s interval, filters to queued/running jobs
- [ ] Previous panel: titled "Previous Jobs" with badge count, filters to complete/failed/canceled jobs
- [ ] Route `/app/call-parsing/detection` added to App.tsx
- [ ] "Call Parsing" collapsible nav group added to SideNav after "Classifier", with a suitable lucide-react icon
- [ ] "Detection" sub-item links to `/app/call-parsing/detection`
- [ ] Nav group highlights correctly when on the call-parsing route

**Tests needed:**
- Playwright test: navigate to `/app/call-parsing/detection`, page renders with form and both panels
- Playwright test: nav group appears in sidebar, clicking "Detection" navigates to correct route

---

### Task 5: Playwright E2E Tests

**Files:**
- Create: `frontend/tests/call-parsing-detection.spec.ts`

**Acceptance criteria:**
- [ ] Test: page loads at `/app/call-parsing/detection` with form visible
- [ ] Test: nav group "Call Parsing" visible in sidebar with "Detection" link
- [ ] Test: form fields render (hydrophone dropdown, date picker, model dropdown, threshold sliders)
- [ ] Test: "Start Detection" button disabled when form incomplete
- [ ] Test: advanced settings section is collapsible
- [ ] Test: previous jobs table renders with correct columns

**Tests needed:**
- This task IS the test task

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test tests/call-parsing-detection.spec.ts`
3. Manual browser check: navigate to Call Parsing > Detection, verify form renders, create a job (if backend running)
