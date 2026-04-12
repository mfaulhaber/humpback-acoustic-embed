# Call Parsing — Segment & Segment Training UI Implementation Plan

**Goal:** Add two new UI pages (Segment inference and Segment Training) under Call Parsing, plus a "Segment →" shortcut on the Detection page and one new backend endpoint for listing training datasets.
**Spec:** [docs/specs/2026-04-12-call-parsing-segment-ui-design.md](../specs/2026-04-12-call-parsing-segment-ui-design.md)
**Branch:** `feature/call-parsing-segment-ui`

---

## Task ordering

Task 1 (backend endpoint) is independent and can land first or in parallel with frontend work. Tasks 2–3 (types + hooks) are foundational — all frontend components depend on them. Task 4 (routes + nav + page shells) wires up the pages so subsequent tasks can be tested in the browser. Tasks 5–7 build out the Segment page components. Tasks 8–9 build the Segment Training page. Task 10 modifies the Detection page. Task 11 adds Playwright tests. Task 12 updates documentation.

---

### Task 1: Backend — training datasets list endpoint

**Files:**
- Modify: `src/humpback/schemas/call_parsing.py`
- Modify: `src/humpback/services/call_parsing.py`
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `tests/api/test_call_parsing_router.py`

**Acceptance criteria:**
- [ ] New `SegmentationTrainingDatasetSummary` Pydantic response model with fields: `id` (str), `name` (str), `sample_count` (int), `created_at` (datetime)
- [ ] New `list_segmentation_training_datasets(session) -> list[SegmentationTrainingDatasetSummary]` service method that queries `segmentation_training_datasets` with a `COUNT(*)` subquery on `segmentation_training_samples` grouped by `training_dataset_id`
- [ ] New `GET /call-parsing/segmentation-training-datasets` route handler that returns the list, no pagination
- [ ] Returns empty list when no datasets exist
- [ ] Sample count is accurate (reflects actual row count in `segmentation_training_samples` for each dataset)

**Tests needed:**
- `GET /call-parsing/segmentation-training-datasets` returns empty list on clean DB
- Create a dataset with known sample rows, assert `sample_count` matches
- Create two datasets with different sample counts, assert both returned with correct counts
- Response shape matches `SegmentationTrainingDatasetSummary` fields

---

### Task 2: TypeScript types and API client methods

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`

**Acceptance criteria:**
- [ ] New `EventSegmentationJob` interface mirroring the backend response: `id`, `status` (queued/running/complete/failed), `region_detection_job_id`, `segmentation_model_id`, `parent_run_id`, `config_json`, `event_count`, `error_message`, `created_at`, `updated_at`, `started_at`, `completed_at`
- [ ] New `SegmentationDecoderConfig` interface: `high_threshold`, `low_threshold`, `min_event_sec`, `merge_gap_sec`
- [ ] New `CreateSegmentationJobRequest` interface: `region_detection_job_id`, `segmentation_model_id`, `parent_run_id?`, `config?` (partial `SegmentationDecoderConfig`)
- [ ] New `SegmentationModel` interface: `id`, `name`, `model_family`, `model_path`, `config_json`, `training_job_id`, `created_at`
- [ ] New `SegmentationTrainingJob` interface: `id`, `status`, `training_dataset_id`, `config_json`, `segmentation_model_id`, `result_summary`, `error_message`, `created_at`, `updated_at`, `started_at`, `completed_at`
- [ ] New `SegmentationTrainingConfig` interface: `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `early_stopping_patience`, `grad_clip`, `seed`
- [ ] New `CreateSegmentationTrainingJobRequest` interface: `training_dataset_id`, `config?` (partial `SegmentationTrainingConfig`)
- [ ] New `SegmentationTrainingDatasetSummary` interface: `id`, `name`, `sample_count`, `created_at`
- [ ] New `SegmentationEvent` interface: `event_id`, `region_id`, `start_sec`, `end_sec`, `center_sec`, `segmentation_confidence`
- [ ] API client methods following existing pattern (`api<T>` for GET, `post<T>` for POST, `api<{status:string}>` with `method: "DELETE"` for DELETE):
  - `fetchSegmentationJobs()` → GET `/call-parsing/segmentation-jobs`
  - `createSegmentationJob(body)` → POST `/call-parsing/segmentation-jobs`
  - `deleteSegmentationJob(jobId)` → DELETE `/call-parsing/segmentation-jobs/{jobId}`
  - `fetchSegmentationJobEvents(jobId)` → GET `/call-parsing/segmentation-jobs/{jobId}/events`
  - `fetchSegmentationModels()` → GET `/call-parsing/segmentation-models`
  - `deleteSegmentationModel(modelId)` → DELETE `/call-parsing/segmentation-models/{modelId}`
  - `fetchSegmentationTrainingJobs()` → GET `/call-parsing/segmentation-training-jobs`
  - `createSegmentationTrainingJob(body)` → POST `/call-parsing/segmentation-training-jobs`
  - `deleteSegmentationTrainingJob(jobId)` → DELETE `/call-parsing/segmentation-training-jobs/{jobId}`
  - `fetchSegmentationTrainingDatasets()` → GET `/call-parsing/segmentation-training-datasets`

**Tests needed:**
- Type-check passes: `npx tsc --noEmit`

---

### Task 3: TanStack Query hooks

**Files:**
- Modify: `frontend/src/hooks/queries/useCallParsing.ts`

**Acceptance criteria:**
- [ ] `useSegmentationJobs()` — `useQuery` wrapping `fetchSegmentationJobs`, refetchInterval 3000ms, queryKey `["segmentation-jobs"]`
- [ ] `useCreateSegmentationJob()` — `useMutation` wrapping `createSegmentationJob`, invalidates `["segmentation-jobs"]` on success
- [ ] `useDeleteSegmentationJob()` — `useMutation` wrapping `deleteSegmentationJob`, invalidates `["segmentation-jobs"]` on success
- [ ] `useSegmentationJobEvents(jobId)` — `useQuery` wrapping `fetchSegmentationJobEvents`, queryKey `["segmentation-job-events", jobId]`, `enabled: !!jobId`
- [ ] `useSegmentationModels()` — `useQuery` wrapping `fetchSegmentationModels`, queryKey `["segmentation-models"]`
- [ ] `useDeleteSegmentationModel()` — `useMutation` wrapping `deleteSegmentationModel`, invalidates `["segmentation-models"]` on success
- [ ] `useSegmentationTrainingJobs()` — `useQuery` wrapping `fetchSegmentationTrainingJobs`, refetchInterval 3000ms, queryKey `["segmentation-training-jobs"]`
- [ ] `useCreateSegmentationTrainingJob()` — `useMutation` wrapping `createSegmentationTrainingJob`, invalidates `["segmentation-training-jobs"]` and `["segmentation-models"]` on success
- [ ] `useDeleteSegmentationTrainingJob()` — `useMutation` wrapping `deleteSegmentationTrainingJob`, invalidates `["segmentation-training-jobs"]` on success
- [ ] `useSegmentationTrainingDatasets()` — `useQuery` wrapping `fetchSegmentationTrainingDatasets`, queryKey `["segmentation-training-datasets"]`
- [ ] All hooks follow the existing pattern in `useCallParsing.ts` (useQuery/useMutation from `@tanstack/react-query`, typed with the interfaces from Task 2)

**Tests needed:**
- Type-check passes: `npx tsc --noEmit`

---

### Task 4: Routes, navigation, and page shell components

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/layout/SideNav.tsx`
- Create: `frontend/src/components/call-parsing/SegmentPage.tsx`
- Create: `frontend/src/components/call-parsing/SegmentTrainingPage.tsx`

**Acceptance criteria:**
- [ ] Two new routes in `App.tsx`: `/app/call-parsing/segment` → `SegmentPage`, `/app/call-parsing/segment-training` → `SegmentTrainingPage`
- [ ] Call Parsing index redirect remains at `/app/call-parsing/detection`
- [ ] SideNav Call Parsing children updated to three items: Detection (`/app/call-parsing/detection`), Segment (`/app/call-parsing/segment`), Segment Training (`/app/call-parsing/segment-training`)
- [ ] `SegmentPage.tsx` renders a placeholder shell with the page title "Segmentation" — subsequent tasks fill in the form and tables
- [ ] `SegmentTrainingPage.tsx` renders a placeholder shell with the page title "Segment Training" — subsequent tasks fill in the sections
- [ ] Both pages render and are navigable from the side nav
- [ ] `SegmentPage` reads `regionJobId` from URL search params (via `useSearchParams`) and stores it in state for the form to consume

**Tests needed:**
- Manual verification: navigate to both routes, see placeholder content, nav highlights correctly

---

### Task 5: SegmentJobForm component

**Files:**
- Create: `frontend/src/components/call-parsing/SegmentJobForm.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentPage.tsx`

**Acceptance criteria:**
- [ ] Two dropdowns side-by-side in a bordered card, matching the RegionJobForm layout pattern
- [ ] **Region Detection Job dropdown**: lists jobs from `useRegionDetectionJobs()` filtered to `status === "complete"`. Display format: `<hydrophone name> · <date range UTC> · <region_count> regions`. Hydrophone name resolved from `useHydrophones()`. If `initialRegionJobId` prop is set, pre-selects that job
- [ ] **Segmentation Model dropdown**: lists models from `useSegmentationModels()`. Display format: `<name> (F1: <event_f1>)`. Event F1 parsed from `config_json`
- [ ] **Collapsible "Advanced Settings"** section with four number inputs: High Threshold (default 0.5), Low Threshold (default 0.3), Min Event Duration (default 0.2), Merge Gap (default 0.1). Uses HTML `<details>/<summary>` or the same collapsible pattern as `RegionJobForm`
- [ ] **"Start Segmentation" button**: disabled when either dropdown is empty or mutation is pending. On click, calls `useCreateSegmentationJob()` with the selected values
- [ ] Form resets model dropdown after successful submission (region job stays selected for re-runs with different models)
- [ ] `SegmentPage` mounts this component and passes `initialRegionJobId` from URL search params

**Tests needed:**
- Type-check passes: `npx tsc --noEmit`
- Manual verification: dropdowns populate, form submits, job appears in active list

---

### Task 6: SegmentJobTable and SegmentJobTablePanel components

**Files:**
- Create: `frontend/src/components/call-parsing/SegmentJobTable.tsx`
- Create: `frontend/src/components/call-parsing/SegmentJobTablePanel.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentPage.tsx`

**Acceptance criteria:**
- [ ] `SegmentJobTablePanel` wraps the table with a title, count badge, and bordered card — matching `RegionJobTablePanel` pattern
- [ ] `SegmentJobTable` accepts a `mode` prop: `"active"` or `"previous"`
- [ ] **Active mode**: filters to queued/running jobs. Columns: Status (badge) | Created (relative time) | Source (hydrophone + date range) | Model name | Events (dash) | Cancel action (delete button). Returns `null` when no active jobs
- [ ] **Previous mode**: filters to completed/failed jobs. Columns: Status | Created | Source (linked, blue underline) | Model (linked, blue underline) | Events count | Thresholds (parsed from `config_json` as "high / low") | Actions (expand toggle + Delete button)
- [ ] Source column resolves hydrophone name from the upstream `region_detection_job_id` by looking up the region job from `useRegionDetectionJobs()`, then resolving the hydrophone name. Source link navigates to `/app/call-parsing/detection`
- [ ] Model column links to `/app/call-parsing/segment-training`
- [ ] Previous mode includes: search/filter input for hydrophone name, pagination (20 per page with prev/next), bulk delete with checkbox selection and confirmation dialog
- [ ] Clicking a row in previous mode toggles expanded state (handled via row state, detail rendered in Task 7)
- [ ] Delete action calls `useDeleteSegmentationJob()` with confirmation
- [ ] `SegmentPage` mounts Active panel (hidden when empty) and Previous panel below the form

**Tests needed:**
- Type-check passes: `npx tsc --noEmit`
- Manual verification: tables render, status badges correct, links navigate, delete works, pagination works

---

### Task 7: SegmentJobDetail component (expandable row)

**Files:**
- Create: `frontend/src/components/call-parsing/SegmentJobDetail.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentJobTable.tsx`

**Acceptance criteria:**
- [ ] Renders inside an expanded `<tr>` spanning all columns when a previous job row is toggled open
- [ ] Fetches events lazily via `useSegmentationJobEvents(jobId)` only when expanded (enabled by expand state)
- [ ] Shows loading spinner while events are being fetched
- [ ] **Summary stats row**: five stat cards in a grid — Event Count, Mean Duration, Median Duration, Min Confidence, Max Confidence. All computed client-side from the events array
- [ ] **Events table**: columns Region (truncated ID, first 4 chars + "…"), Start (seconds), End (seconds), Duration (computed), Confidence. Sortable by Start, Duration, and Confidence columns. Paginated at 20 rows per page with prev/next controls
- [ ] Handles empty events list gracefully (shows "No events" message)
- [ ] Duration and confidence values formatted to 2 decimal places
- [ ] `SegmentJobTable` integrates this component: renders it as a `<tr colspan>` immediately after the parent row when expanded

**Tests needed:**
- Type-check passes: `npx tsc --noEmit`
- Manual verification: expand/collapse works, stats compute correctly, events table paginates and sorts

---

### Task 8: SegmentModelTable component

**Files:**
- Create: `frontend/src/components/call-parsing/SegmentModelTable.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentTrainingPage.tsx`

**Acceptance criteria:**
- [ ] Bordered card with title "Segmentation Models" and count badge
- [ ] Table columns: Name (font-weight medium) | Family | Framewise F1 | Event F1 (IoU≥0.3) | Created (formatted date) | Delete action
- [ ] Metrics parsed from model's `config_json` — the condensed metrics snapshot written by the training worker. Handles missing/malformed JSON gracefully (show "—")
- [ ] F1 values color-coded: green (≥0.7), amber (≥0.5), default otherwise
- [ ] Delete action calls `useDeleteSegmentationModel()`. Shows confirmation dialog before deletion. On 409 response (model in use), shows toast with the error message
- [ ] Fetched from `useSegmentationModels()`
- [ ] Shows "No models yet" empty state when list is empty
- [ ] `SegmentTrainingPage` mounts this as the first section

**Tests needed:**
- Type-check passes: `npx tsc --noEmit`
- Manual verification: models display with metrics, delete works, 409 shows toast

---

### Task 9: SegmentTrainingForm and SegmentTrainingJobTable components

**Files:**
- Create: `frontend/src/components/call-parsing/SegmentTrainingForm.tsx`
- Create: `frontend/src/components/call-parsing/SegmentTrainingJobTable.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentTrainingPage.tsx`

**Acceptance criteria:**
- [ ] `SegmentTrainingForm`: bordered card with background, containing a dataset picker dropdown and "Start Training" button on the same row, plus collapsible "Advanced Settings" below
- [ ] Dataset dropdown lists datasets from `useSegmentationTrainingDatasets()`. Display format: `<name> (<sample_count> samples)`. Disabled with "No datasets available" text when list is empty
- [ ] Collapsible "Advanced Settings" with seven number inputs: Epochs (30), Batch Size (16), Learning Rate (0.001), Weight Decay (0.0001), Early Stop Patience (5), Grad Clip (1.0), Seed (42)
- [ ] "Start Training" button disabled when no dataset selected or mutation pending. Calls `useCreateSegmentationTrainingJob()` with selected dataset and config values
- [ ] `SegmentTrainingJobTable`: table inside a bordered card titled "Training Jobs" with count badge
- [ ] Table columns: Status (badge) | Created | Dataset name | Config summary (formatted as "N ep · lr=X") | Model (linked to models section via anchor/scroll, blue underline) | Metrics (F1_frame and F1_event inline) | Delete action
- [ ] Config summary parsed from `config_json`: shows epochs and learning rate
- [ ] Metrics parsed from `result_summary` JSON when present, dash when absent (job not yet complete)
- [ ] Model column shows model name when `segmentation_model_id` is set (link to models section), dash when null
- [ ] Delete calls `useDeleteSegmentationTrainingJob()` with confirmation. 409 toast on conflict
- [ ] Polls via the 3s refetchInterval from the `useSegmentationTrainingJobs()` hook
- [ ] `SegmentTrainingPage` mounts form and table as the second section, below models

**Tests needed:**
- Type-check passes: `npx tsc --noEmit`
- Manual verification: form submits, training jobs appear and update status, metrics display after completion

---

### Task 10: Detection page "Segment →" button

**Files:**
- Modify: `frontend/src/components/call-parsing/RegionJobTable.tsx`

**Acceptance criteria:**
- [ ] In previous mode, completed jobs show a "Segment →" button in the Actions column alongside the existing Timeline button
- [ ] Button uses the same `Button` component with `variant="outline"` and `size="sm"`, styled with a blue accent (matching the mockup's `dbeafe`/`1d4ed8` colors or using the existing outline variant)
- [ ] On click, navigates to `/app/call-parsing/segment?regionJobId=<job.id>` using `useNavigate()` from react-router-dom
- [ ] Button only renders for jobs with `status === "complete"`. Failed/canceled jobs do not show it
- [ ] Does not interfere with existing bulk delete or other table functionality

**Tests needed:**
- Type-check passes: `npx tsc --noEmit`
- Manual verification: button appears on complete jobs only, navigates correctly, Segment page pre-selects the job

---

### Task 11: Playwright tests

**Files:**
- Create: `frontend/e2e/call-parsing-segment.spec.ts`

**Acceptance criteria:**
- [ ] Test: Segment page loads at `/app/call-parsing/segment` and shows the form with both dropdowns
- [ ] Test: Segment Training page loads at `/app/call-parsing/segment-training` and shows both sections (models + training)
- [ ] Test: SideNav shows all three Call Parsing items (Detection, Segment, Segment Training) and navigation between them works
- [ ] Test: Detection page shows "Segment" button on completed region jobs (requires a completed region job fixture or skips gracefully)
- [ ] Tests follow existing Playwright patterns in `frontend/e2e/` (page fixtures, API mocking if needed, reasonable timeouts)

**Tests needed:**
- All tests pass: `cd frontend && npx playwright test call-parsing-segment`

---

### Task 12: Documentation updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/reference/frontend.md`

**Acceptance criteria:**
- [ ] CLAUDE.md §8.9 — add `GET /call-parsing/segmentation-training-datasets` to the Call Parsing Pipeline API Surface under a new bullet
- [ ] CLAUDE.md §9.1 — append "Call parsing Segment and Segment Training UI pages" to Implemented Capabilities list
- [ ] `docs/reference/frontend.md` — update Navigation description to include Segment and Segment Training sub-routes under Call Parsing; update the file structure tree to include the new component files under `call-parsing/`

---

## Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py`
2. `uv run ruff check src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py`
3. `uv run pyright src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py`
4. `uv run pytest tests/api/test_call_parsing_router.py`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test call-parsing-segment`
