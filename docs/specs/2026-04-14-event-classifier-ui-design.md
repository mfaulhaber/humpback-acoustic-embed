# Event Classifier UI Design

**Date:** 2026-04-14
**Status:** Approved

---

## 1. Overview

Frontend UI for the Pass 3 event classifier workflow: running classification
jobs, reviewing/correcting typed event labels, and training new models from
corrections. Follows the established Pass 2 (segmentation) page architecture.

Two new pages:
- **ClassifyPage** — job management + review workspace (tabbed)
- **ClassifyTrainingPage** — model management + training from corrections

Two backend additions to support the UI.

---

## 2. Page Architecture

Mirrors Pass 2: ClassifyPage with Jobs + Review tabs, separate
ClassifyTrainingPage. Quick-retrain button in the review workspace for rapid
iteration; full training page for assembling models from multiple jobs'
corrections.

### Routes

| Route | Component | Purpose |
|-------|-----------|---------|
| `/app/call-parsing/classify` | ClassifyPage | Redirects to `?tab=jobs` |
| `/app/call-parsing/classify?tab=jobs` | ClassifyPage (Jobs tab) | Create/manage classification jobs |
| `/app/call-parsing/classify?tab=review&reviewJobId={id}` | ClassifyPage (Review tab) | Correct typed event labels |
| `/app/call-parsing/classify-training` | ClassifyTrainingPage | Train models from corrections |

### SideNav

Two entries added under the Call Parsing group:
- "Classify" → `/app/call-parsing/classify`
- "Classify Training" → `/app/call-parsing/classify-training`

---

## 3. ClassifyPage — Jobs Tab

Mirrors the Segment page Jobs tab.

### Job Creation Form

Card with two selectors:
- **Segmentation job** — dropdown of completed `EventSegmentationJob` rows.
  Pre-filled via `?segmentJobId={id}` query param (linked from Segment page).
- **Event classifier model** — dropdown of `vocalization_models` filtered to
  `model_family='pytorch_event_cnn'`.

"Run Classification" button creates an `EventClassificationJob`.

### Active Jobs Table

Jobs with status `queued` or `running`. Columns: ID (short), segmentation job,
model, status badge, created time, cancel action. Auto-refreshes every 3
seconds.

### Previous Jobs Table

Completed and failed jobs. Columns: ID (short), segmentation job, model,
status, typed event count, created time, actions (Review, Delete).

Expandable rows show a type summary table: one row per vocalization type with
count, mean score, and percentage of total events.

Bulk delete via checkbox selection.

"Review" action navigates to `?tab=review&reviewJobId={id}`.

---

## 4. ClassifyPage — Review Workspace

Matches the Segment Review workspace layout: no cards, continuous flow within
a bordered container.

### Layout (top to bottom)

1. **Job selector** — plain `<select>` outside the bordered container, listing
   completed classification jobs.
2. **Bordered container** wrapping:
   - **Review toolbar** — event navigation (Prev/Next with counter "Event 3 of
     47"), playback button, Save/Cancel buttons, Retrain button.
   - **Spectrogram** — `RegionSpectrogramViewer` reused from segment review.
     Shows region spectrogram tiles with event bars overlaid. Current event
     centered and highlighted. Neighbor events visible with type labels
     color-coded by vocalization type. Corrected events get a visual indicator
     (e.g., checkmark or border change).
   - **Zoom bar** — left-justified below spectrogram, offset by frequency axis
     width. Zoom presets: 10s / 30s / 1m / 5m. Identical to segment review.
   - **Type palette** — persistent horizontal strip of pill buttons below the
     zoom bar. Shows all vocalization types from the database + a "(Negative)"
     option + an "Add Type" button. Selected type has a bold border/highlight.
     Click a type to select it as active; click again or press Enter to stamp
     it onto the current event. Click-only for type selection (no keyboard
     shortcuts for types — the vocabulary exceeds single-digit counts).
   - **Detail panel** — predicted type + confidence score, correction status
     (original/corrected/negative), time range and duration, all model scores
     for the event sorted by score descending.

### Typed Event Aggregation

The typed events endpoint returns multiple rows per event (one per type in the
model vocabulary). The frontend aggregates per `event_id`:
- **Predicted type**: highest-scoring row where `above_threshold` is true. If
  no type is above threshold, the event has no predicted type.
- **All scores**: shown in the detail panel, sorted descending.
- **Event list**: deduplicated by `event_id`, sorted by `start_sec`.

### Event Navigation

Events are sorted by `start_sec` across all regions. Prev/Next buttons (and
`←`/`→` or `[`/`]` keys) move through the global event list. When crossing a
region boundary, the spectrogram viewer switches to the new region's tiles
automatically.

### Labeling Flow

1. Select a type in the palette (click).
2. Navigate events with Prev/Next.
3. Press Enter to stamp the active type onto the current event.
4. Press Space to play the event's audio for verification.
5. Press Backspace/Delete to mark as negative (null type).
6. Press Escape to deselect the active type.
7. Skip events whose predictions look correct.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` or `[` / `]` | Prev/next event |
| `Enter` | Stamp active type onto current event |
| `Space` | Toggle playback |
| `Backspace` / `Delete` | Mark current event as negative |
| `Escape` | Deselect active type |

### Saving Corrections

Save button batches all pending corrections and calls
`POST /classification-jobs/{id}/corrections`. Cancel discards unsaved changes.
Unsaved changes trigger a confirmation dialog on navigation away.

### Quick Retrain

Retrain button in the toolbar calls
`POST /classifier-training-jobs` with the current classification job ID as the
sole source. Shows a spinner/toast while training runs. On completion, offers
"Re-classify" to run a new classification job with the freshly trained model
against the same segmentation job.

---

## 5. ClassifyTrainingPage

Three sections, following the Segment Training page pattern.

### Section 1: Model Table

Lists `vocalization_models` filtered to `model_family='pytorch_event_cnn'`.

Columns: name, type count, avg F1 (color-coded: green ≥0.8, yellow ≥0.6,
red <0.6), created date, actions (Delete).

Expandable rows show per-type metrics: type name, precision, recall, F1,
threshold, sample count.

### Section 2: Training Form

Multi-select table of completed classification jobs that have type corrections.

Powered by `GET /classification-jobs/with-correction-counts`. Columns:
checkbox, hydrophone/source, date range, correction count.

Below the table: "Train Model" button. Calls
`POST /classifier-training-jobs` with checked job IDs.

### Section 3: Training Job Table

Lists `EventClassifierTrainingJob` rows. Columns: ID (short), status badge,
source job count, created time, resulting model link (on complete), error
message (on failed), actions (Delete).

Auto-refreshes every 3 seconds for in-flight jobs. Completed rows link to the
model in Section 1.

---

## 6. Backend Additions

### New Endpoint

`GET /call-parsing/classification-jobs/with-correction-counts`

Returns all completed classification jobs annotated with their type correction
count. Mirrors the existing
`GET /call-parsing/segmentation-jobs/with-correction-counts` pattern.

Response: list of objects with existing `EventClassificationJobSummary` fields
plus `correction_count: int`, `hydrophone_id: Optional[str]`,
`start_timestamp: Optional[float]`, `end_timestamp: Optional[float]` (traced
through segmentation job → region detection job → run).

### Modified Endpoint

`GET /call-parsing/classification-jobs/{job_id}/typed-events`

Add `region_id` to each typed event response row. Resolved by joining
`typed_events.parquet` with `events.parquet` on `event_id` to pull the
region association. The frontend groups events by region client-side for
spectrogram navigation.

Updated response shape per row:
```json
{
  "event_id": "...",
  "region_id": "...",
  "start_sec": 0.0,
  "end_sec": 0.0,
  "type_name": "...",
  "score": 0.0,
  "above_threshold": true
}
```

### No Other Changes

All other endpoints already exist and are sufficient:
- Type corrections CRUD (POST/GET/DELETE per job)
- Classifier training jobs CRUD
- Classifier models CRUD (GET/DELETE)
- Vocalization types CRUD (existing vocalization module)
- Region tile and audio-slice endpoints (reused from Pass 1)

---

## 7. Cross-Page Navigation

- **Segment page → Classify page**: "Classify →" action on completed
  segmentation jobs navigates to
  `/app/call-parsing/classify?tab=jobs&segmentJobId={id}` to pre-fill the
  segmentation job selector.
- **Classify Jobs → Classify Review**: "Review" action on completed
  classification jobs switches to `?tab=review&reviewJobId={id}`.
- **Classify Training → Model**: Completed training job rows link to the
  resulting model row in the model table (scroll-to or highlight).

---

## 8. Frontend Components

### New Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `ClassifyPage` | `components/call-parsing/` | Top-level page with Jobs + Review tabs |
| `ClassifyJobForm` | `components/call-parsing/` | Job creation form |
| `ClassifyJobTable` | `components/call-parsing/` | Active + previous job tables |
| `ClassifyReviewWorkspace` | `components/call-parsing/` | Type correction workspace |
| `TypePalette` | `components/call-parsing/` | Persistent type selector strip |
| `ClassifyDetailPanel` | `components/call-parsing/` | Event detail + scores panel |
| `ClassifyTrainingPage` | `components/call-parsing/` | Training page (3 sections) |
| `ClassifyModelTable` | `components/call-parsing/` | Model table with per-type metrics |
| `ClassificationJobPicker` | `components/call-parsing/` | Multi-select job picker for training |
| `ClassifyTrainingJobTable` | `components/call-parsing/` | Training job status table |

### Reused Components

| Component | From | Usage |
|-----------|------|-------|
| `RegionSpectrogramViewer` | Segment Review | Spectrogram rendering + zoom |
| `EventBarOverlay` | Segment Review | Event bars on spectrogram (adapted for type colors) |
| `ReviewToolbar` | Segment Review | Toolbar layout (adapted for type-specific actions) |

### New Hooks

| Hook | Purpose |
|------|---------|
| `useClassificationJobs` | List classification jobs |
| `useCreateClassificationJob` | Create classification job |
| `useDeleteClassificationJob` | Delete classification job |
| `useTypedEvents` | Fetch typed events for a job |
| `useTypeCorrections` | Fetch existing corrections for a job |
| `useUpsertTypeCorrections` | Save type corrections |
| `useClearTypeCorrections` | Clear all corrections |
| `useClassificationJobsWithCorrectionCounts` | Job picker for training page |
| `useClassifierTrainingJobs` | List training jobs |
| `useCreateClassifierTrainingJob` | Create training job |
| `useDeleteClassifierTrainingJob` | Delete training job |
| `useClassifierModels` | List event classifier models |
| `useDeleteClassifierModel` | Delete model |

---

## 9. Spectrogram & Audio Resolution

Classification job → `event_segmentation_job_id` → segmentation job →
`region_detection_job_id` → region detection job → tiles + audio slices.

The frontend traces this chain:
1. Load classification job to get `event_segmentation_job_id`.
2. Load segmentation job to get `region_detection_job_id`.
3. Load regions from the region detection job.
4. Use `regionTileUrl()` and `regionAudioSliceUrl()` for spectrogram rendering
   and playback, same as segment review.

Typed events include `region_id`, so the frontend knows which region's
spectrogram to display for each event.

---

## 10. Bootstrap (Initial Dataset)

Manual process — no special UI tooling.

1. Take events from a segmentation training dataset with good boundaries.
2. Assign random vocalization type labels.
3. Train an event classifier model (will be poor quality).
4. Run Pass 3 inference with the synthetic model.
5. Correct predictions in the review workspace (human-in-the-loop).
6. Feedback train to produce a real model from corrections.

Documented in project notes. The UI supports this workflow naturally — it just
runs inference, reviews, and retrains like any other iteration.

---

## 11. Scope Exclusions

- **Pass 4** (sequence export) — still deferred.
- **Multi-label corrections** — Pass 3 corrections are single-label only.
- **Dataset extraction step** — unlike Pass 2 segmentation, Pass 3 training
  assembles samples directly from corrections in the worker. No intermediate
  dataset model.
- **Batch auto-accept** — no "accept all predictions above threshold" action.
  Each event is reviewed individually or skipped.
- **Bootstrap UI** — the synthetic dataset creation is a manual/scripted
  process, not a UI feature.
