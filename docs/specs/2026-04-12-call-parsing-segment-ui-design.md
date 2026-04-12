# Call Parsing — Segment & Segment Training UI

**Date:** 2026-04-12
**Status:** Approved
**Inherits from:** [Pass 2 spec](2026-04-11-call-parsing-pass2-segmentation-design.md) (ADR-050), [Pass 2 implementation plan](../plans/2026-04-11-call-parsing-pass2-segmentation.md)

---

## Problem

Pass 2 (event segmentation) is fully functional on the backend — training jobs,
inference jobs, models, and all API endpoints are live. But there is no frontend
to drive it. Users must use curl or scripts to create segmentation jobs, start
training, or inspect results. Two new pages give Pass 2 a proper UI surface.

---

## Inherited from Pass 2 Backend (NOT re-derived here)

- `event_segmentation_jobs` table with standard queue columns, upstream
  `region_detection_job_id` FK, `segmentation_model_id` FK, `event_count`,
  `config_json`.
- `segmentation_models` table with `id`, `name`, `model_family`, `model_path`,
  `config_json` (includes condensed metrics snapshot), `training_job_id`,
  `created_at`.
- `segmentation_training_datasets`, `segmentation_training_samples`,
  `segmentation_training_jobs` tables.
- Functional API endpoints: `POST/GET/DELETE /call-parsing/segmentation-jobs`,
  `GET /segmentation-jobs/{id}/events`, `POST/GET/DELETE
  /call-parsing/segmentation-training-jobs`, `GET/DELETE
  /call-parsing/segmentation-models`.
- Pydantic schemas: `CreateSegmentationJobRequest`,
  `CreateSegmentationTrainingJobRequest`, `SegmentationDecoderConfig`,
  `SegmentationTrainingConfig`, and response models.
- Workers: `segmentation_training_worker.py`,
  `event_segmentation_worker.py`.

---

## Scope

**Ships:**

- **Segment page** (`/app/call-parsing/segment`) — create segmentation inference
  jobs from completed Pass 1 region jobs, monitor active jobs, browse previous
  jobs with expandable event detail and summary statistics.
- **Segment Training page** (`/app/call-parsing/segment-training`) — segmentation
  models table with metrics, training job creation form with dataset picker and
  advanced hyperparameter config, training jobs table.
- **Detection page modification** — "Segment →" action button on completed
  region detection jobs linking to the Segment page with pre-selection.
- **Navigation update** — three items under Call Parsing: Detection, Segment,
  Segment Training.
- **One new backend endpoint** — `GET /call-parsing/segmentation-training-datasets`
  for the training form's dataset picker.
- Frontend API client methods, TypeScript interfaces, and TanStack Query hooks
  for all segmentation endpoints.

**Does NOT ship:**

- Training dataset CRUD UI (datasets created by bootstrap script; future UI
  editor is out of scope).
- Event visualization beyond table and stats (no spectrogram, no timeline
  integration).
- Pass 3 (event classification) UI.

---

## Page 1: Segment (Inference)

**Route:** `/app/call-parsing/segment`
**Query params:** `?regionJobId=<id>` (optional, pre-selects region job picker)

Three stacked panels matching the Detection page pattern:

### 1. New Segmentation Job Form

- **Region Detection Job dropdown** — lists completed Pass 1 jobs. Display
  format: `<hydrophone name> · <date range> · <region_count> regions`. Fetched
  from `GET /call-parsing/region-jobs`, filtered client-side to
  `status="complete"`. Pre-selected when `regionJobId` query param is present.
- **Segmentation Model dropdown** — lists available models. Display format:
  `<name> (F1: <event_f1>)`. Fetched from `GET /call-parsing/segmentation-models`.
- **Collapsible "Advanced Settings"** with decoder config fields:
  - High Threshold (default 0.5, range 0–1)
  - Low Threshold (default 0.3, range 0–1)
  - Min Event Duration seconds (default 0.2)
  - Merge Gap seconds (default 0.1)
- **"Start Segmentation" button** — disabled until both dropdowns have
  selections. POSTs `CreateSegmentationJobRequest` to
  `POST /call-parsing/segmentation-jobs`.

### 2. Active Jobs Panel

Queued and running jobs. Hidden when empty.

Columns: Status | Created | Source (hydrophone + date range from upstream
Pass 1 job) | Model | Events (dash while running) | Cancel action (delete).

Polls via `GET /call-parsing/segmentation-jobs` with 3-second refetchInterval.

### 3. Previous Jobs Panel

Completed and failed jobs.

Columns: Status | Created | Source (linked to Detection page) | Model (linked
to Segment Training page) | Events count | Thresholds (parsed from
`config_json`) | Actions (expand toggle + Delete).

**Expandable row detail:** Summary statistics computed client-side from events
response (event count, mean duration, median duration, min confidence, max
confidence), plus a paginated events table (Region ID truncated, Start, End,
Duration, Confidence). Events fetched lazily on first expand from
`GET /call-parsing/segmentation-jobs/{id}/events`.

Search/filter by source hydrophone name, pagination (20 per page), bulk delete
with checkbox selection.

**Source column resolution:** Each segmentation job references a
`region_detection_job_id`. The frontend fetches the region job to get
`hydrophone_id` + timestamps, then resolves the hydrophone name from the
hydrophone list. The Source link navigates to `/app/call-parsing/detection`.

**Model column link:** Navigates to `/app/call-parsing/segment-training`.

---

## Page 2: Segment Training (Models + Training Jobs)

**Route:** `/app/call-parsing/segment-training`

Two stacked sections matching the Tuning page pattern:

### 1. Segmentation Models Section

Table columns: Name | Family | Framewise F1 | Event F1 (IoU≥0.3) | Created |
Delete action.

Metrics parsed from the model's `config_json` condensed metrics snapshot. Delete
shows confirmation dialog. Returns 409 toast if model is referenced by an
in-flight segmentation job. Fetched from `GET /call-parsing/segmentation-models`.

### 2. Training Jobs Section

**Training form:** Dataset picker dropdown showing `<name> (<sample_count>
samples)`, fetched from new `GET /call-parsing/segmentation-training-datasets`
endpoint. "Start Training" button. Collapsible "Advanced Settings" with:
Epochs (30), Batch Size (16), Learning Rate (0.001), Weight Decay (0.0001),
Early Stop Patience (5), Grad Clip (1.0), Seed (42).

**Training jobs table:** All jobs. Columns: Status | Created | Dataset name |
Config summary (e.g., "30 ep · lr=0.001") | Model (linked to models section) |
Metrics (framewise F1 + event F1 inline) | Delete action.

Polls with 3-second refetchInterval for active jobs. Delete returns 409 toast
if the resulting model is referenced by in-flight jobs.

---

## Detection Page Modification

Add "Segment →" button in the Previous Jobs table's Actions column for rows
with `status="complete"`. Navigates to
`/app/call-parsing/segment?regionJobId=<job.id>`. Failed and canceled jobs do
not show the button.

---

## Navigation

SideNav Call Parsing group gets two new children:

- **Detection** → `/app/call-parsing/detection` (existing)
- **Segment** → `/app/call-parsing/segment` (new)
- **Segment Training** → `/app/call-parsing/segment-training` (new)

Call Parsing index redirect remains `/app/call-parsing/detection`.

---

## Missing Backend

One new endpoint:

**`GET /call-parsing/segmentation-training-datasets`**

- Returns: `[{id, name, sample_count, created_at}]`
- `sample_count` via `COUNT(*)` subquery join on `segmentation_training_samples`
- No pagination (small cardinality — bootstrap-created datasets only)
- Added to `call_parsing.py` router and service layer
- Pydantic response model: `SegmentationTrainingDatasetSummary`

---

## Testing

- Playwright tests: page load for both routes, form submission, job table
  rendering, expand/collapse detail, navigation links.
- Backend test: `GET /call-parsing/segmentation-training-datasets` returns
  correct sample counts.
- Type-check: `npx tsc --noEmit` passes.
