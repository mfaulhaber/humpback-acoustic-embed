# Segment Review: Zoom Buttons & Retrain Feedback Loop

**Date:** 2026-04-13
**Status:** Approved

## Overview

Two features for the Call Parsing / Segment Review workspace:

1. **Zoom buttons** — manual zoom control for the region spectrogram viewer (10s, 30s, 1m, 5m).
2. **Retrain feedback loop** — wire the existing feedback training backend into the Review UI so users can retrain a segmentation model from their boundary corrections, then re-run inference with the new model. The Segment Training page becomes a read-only dashboard.

## Feature A: Zoom Buttons

### Current State

`RegionSpectrogramViewer` defines four zoom presets (`10s`, `30s`, `1m`, `5m`) in `ZOOM_PRESETS` and auto-selects one based on region duration via `selectZoomLevel()`. The zoom level is computed, not stateful — the user cannot change it.

### Design

- Lift `zoomLevel` from a derived value to `useState`, initialized by `selectZoomLevel(regionDuration)`.
- Reset zoom to auto-selected level when the region changes (same as current behavior on region switch).
- Add a zoom button bar below the time axis: four buttons labeled `10s`, `30s`, `1m`, `5m`. Active level is visually highlighted (blue border/background). A "Zoom" label precedes the buttons.
- All four levels are always available regardless of region size. The user can freely zoom in and out.
- On zoom change, preserve the current `centerTimestamp` so the user doesn't lose their position. Clamping logic already handles out-of-bounds viewports.
- No other changes needed — tile loading, overlay context, pxPerSec, and drag-to-pan all derive from `ZOOM_PRESETS[zoomLevel]` and work unchanged.

### Files Modified

- `frontend/src/components/call-parsing/RegionSpectrogramViewer.tsx` — lift zoom to state, add zoom bar UI.

## Feature B: Retrain Feedback Loop

### Current State

**Backend (complete):**
- `POST /call-parsing/segmentation-feedback-training-jobs` — creates a feedback training job from `source_job_ids` (segmentation jobs with corrections).
- `GET /call-parsing/segmentation-feedback-training-jobs` — lists all feedback training jobs.
- `GET /call-parsing/segmentation-feedback-training-jobs/{id}` — single job.
- `DELETE /call-parsing/segmentation-feedback-training-jobs/{id}` — delete job.
- `event_segmentation_feedback_worker.py` — claims queued jobs, collects `EventBoundaryCorrection` rows, trains `SegmentationCRNN`, registers `SegmentationModel`.
- `POST /call-parsing/segmentation-jobs` — creates a new segmentation job with `region_detection_job_id` and `segmentation_model_id`.

**Frontend (missing):**
- No hooks for the feedback training API.
- `ReviewToolbar` has a disabled "Retrain" button placeholder.
- `SegmentTrainingPage` shows bootstrap training UI (dataset selector, bootstrap job table) — bootstrap endpoints are unimplemented.

### Design

#### B1: Frontend API Hooks

New hooks in `frontend/src/hooks/queries/useCallParsing.ts`:

- `useSegmentationFeedbackTrainingJobs(pollInterval?)` — GET list, optional polling.
- `useCreateSegmentationFeedbackTrainingJob()` — POST mutation, invalidates feedback training jobs and segmentation models queries on success.
- `useDeleteSegmentationFeedbackTrainingJob()` — DELETE mutation, invalidates feedback training jobs query.

New client functions in `frontend/src/api/client.ts`:

- `fetchSegmentationFeedbackTrainingJobs()` — GET.
- `createSegmentationFeedbackTrainingJob(body)` — POST with `{ source_job_ids: string[], config?: SegmentationTrainingConfig }`.
- `deleteSegmentationFeedbackTrainingJob(id)` — DELETE.

New types in `frontend/src/api/types.ts`:

- `SegmentationFeedbackTrainingJob` — id, status, source_job_ids (JSON string), config_json, segmentation_model_id, result_summary, error_message, created_at, updated_at, started_at, completed_at.
- `CreateSegmentationFeedbackTrainingJobRequest` — source_job_ids: string[], config?: partial training config.

#### B2: Retrain Button in ReviewToolbar

**Preconditions to enable:**
- The current job has saved corrections (`useBoundaryCorrections(jobId)` returns non-empty).
- No unsaved pending changes (`isDirty === false`).

**Retrain button behavior:**
- Click opens a confirmation dialog: "Train a new segmentation model from corrections on this job?"
- On confirm: calls `createSegmentationFeedbackTrainingJob({ source_job_ids: [currentJobId] })` with default config (no config UI in the dialog).
- On success: toast "Training job started."

**Training status in toolbar:**
- The toolbar polls for feedback training jobs and filters to jobs whose `source_job_ids` includes the current job ID.
- Displays the most recent matching job's status:
  - `queued` / `running`: spinner + "Training..." text. Retrain button hidden.
  - `complete`: "Model ready" label + "Re-segment" button (green). Retrain button remains visible for additional retraining rounds.
  - `failed`: "Training failed" text (red, with error on hover/title). Retrain button remains visible for retry.
- Polling interval: 3 seconds while a job is `queued` or `running`, no polling otherwise.

**Props added to ReviewToolbar:**
- `hasCorrections: boolean`
- `onRetrain: () => void`
- `retrainStatus: null | { status: string; modelId?: string; modelName?: string; error?: string }`
- `onResegment: () => void`

#### B3: Re-segment with New Model

**Trigger:** "Re-segment" button in toolbar (visible when training completes successfully).

**Click behavior:**
- Confirmation dialog: "Create a new segmentation job using model {modelName} on the same regions?"
- On confirm: calls `POST /call-parsing/segmentation-jobs` with:
  - `region_detection_job_id`: same as current job's region detection job.
  - `segmentation_model_id`: the model produced by the feedback training job.
- On success: toast "Segmentation job created. It will appear in the job selector when complete."
- The new job uses the existing `useCreateSegmentationJob` hook (already exists for the standard segmentation job creation flow).

**After re-segmentation completes:**
- The new job appears in the job selector dropdown (populated by `useSegmentationJobs()`).
- User manually switches to the new job to review results.
- Old job remains in the dropdown for comparison.

#### B4: Segment Training Page Overhaul

**Remove:**
- `SegmentTrainingForm` component (bootstrap dataset selector + config form).
- `SegmentTrainingJobTable` component (bootstrap training jobs).
- Bootstrap-specific hooks: `useCreateSegmentationTrainingJob`, `useSegmentationTrainingJobs`, `useDeleteSegmentationTrainingJob`.
- Keep `useSegmentationTrainingDatasets` only if referenced elsewhere; remove if not.

**Keep:**
- `SegmentModelTable` — unchanged, shows all segmentation models.

**Add:**
- `FeedbackTrainingJobTable` component — replaces `SegmentTrainingJobTable`. Columns:
  - Status (badge)
  - Created (timestamp)
  - Source Jobs (truncated UUIDs of source segmentation job IDs)
  - Config (summary: "30 ep · lr=0.001")
  - Model (link if complete, "—" otherwise)
  - Metrics (F1 scores: "F1^f=X.XX · F1^e=Y.YY")
  - Delete button
- Uses `useSegmentationFeedbackTrainingJobs(3000)` for 3-second polling.
- Uses `useDeleteSegmentationFeedbackTrainingJob()` for deletion.
- No "create" form — training is initiated from the Review workspace.

**Updated `SegmentTrainingPage` layout:**
```
SegmentModelTable
FeedbackTrainingJobTable
```

### Files Modified

- `frontend/src/api/client.ts` — add feedback training client functions.
- `frontend/src/api/types.ts` — add feedback training types.
- `frontend/src/hooks/queries/useCallParsing.ts` — add feedback training hooks, remove bootstrap training hooks.
- `frontend/src/components/call-parsing/ReviewToolbar.tsx` — wire retrain button, training status, re-segment action.
- `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx` — manage retrain/re-segment state, pass props to toolbar.
- `frontend/src/components/call-parsing/SegmentTrainingPage.tsx` — replace bootstrap components with feedback table.
- `frontend/src/components/call-parsing/SegmentTrainingForm.tsx` — delete file.
- `frontend/src/components/call-parsing/SegmentTrainingJobTable.tsx` — delete file (replaced by FeedbackTrainingJobTable).
- `frontend/src/components/call-parsing/FeedbackTrainingJobTable.tsx` — new file.

### No Backend Changes

All backend endpoints, services, workers, and database tables already exist. This feature is frontend-only.

## Testing

- **Zoom buttons:** Playwright test verifying zoom bar renders, clicking a zoom button changes the active level, spectrogram viewport updates.
- **Retrain button:** Playwright test verifying button is disabled without corrections, enabled with corrections, confirmation dialog appears on click.
- **Feedback training job table:** Playwright test verifying the table renders on the training page.
- **Unit tests:** None needed — no new backend code.

## Out of Scope

- Multi-source retrain (selecting multiple source jobs) — future enhancement.
- Training config UI in the retrain dialog — use defaults for now.
- Constraining zoom levels to region duration — all levels always available.
- Bootstrap training pipeline (dataset-based) — removed from UI, backend stubs remain.
- Automated retrain triggers (e.g., after N corrections) — future enhancement.
