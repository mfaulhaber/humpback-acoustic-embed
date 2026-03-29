# Vocalization Labeling Workspace Design

**Date:** 2026-03-29
**Status:** Approved

## Problem

The current Vocalization/Labeling tab is inference-only: select a completed detection
job, queue a vocalization inference job, browse scored results. It has several
limitations:

1. No manual labeling UI — vocalization labels can only be added via the timeline
   viewer or API, not from the labeling page itself.
2. The implicit assumption that detection jobs must be binary labeled or extracted
   before vocalization labeling is unnecessary — any completed detection job should
   be a candidate.
3. Detection jobs without stored embeddings cannot participate in inference, and
   there is no way to provision embeddings from this page.
4. The source selector shows opaque IDs + folder paths — hydrophone jobs lack date
   range context, local jobs lack window counts.
5. No retrain loop — after labeling, users must switch to the Training tab to feed
   new labels back into model retraining.

## Solution

Replace the current tab with a **progressive pipeline** page that guides users
through: source selection -> embedding provisioning -> inference -> manual labeling
(sorted by model uncertainty) -> one-click retrain.

## Design

### Page Layout

The page is a vertical stack of collapsible cards that progressively unlock:

```
┌─────────────────────────────────────────────────┐
│ Vocalization Labeling                           │
├─────────────────────────────────────────────────┤
│ ▼ Source   [NOAA SanctSound (OC01) 2019-06...] │
├─────────────────────────────────────────────────┤
│ ▶ Embeddings  ✓ Ready (847 vectors)            │
├─────────────────────────────────────────────────┤
│ ▶ Inference   ✓ Complete (model: humpback-v3)  │
├─────────────────────────────────────────────────┤
│                                                 │
│  Labeling Workspace                             │
│  ┌─────────────────────────────────────────┐   │
│  │ [spectrogram] [▶] score: 0.42           │   │
│  │  Binary: humpback  Voc: [whup ✕] [+add] │   │
│  ├─────────────────────────────────────────┤   │
│  │ [spectrogram] [▶] score: 0.38           │   │
│  │  Binary: —         Voc: [ ] [+add]       │   │
│  └─────────────────────────────────────────┘   │
│  Page 1 of 17   ◀ ▶                            │
│                                                 │
├─────────────────────────────────────────────────┤
│ 12 new labels since last training  [Retrain]    │
└─────────────────────────────────────────────────┘
```

Steps 2-3 (Embeddings, Inference) collapse once satisfied. The labeling workspace
dominates the page.

### Component Architecture

All within `frontend/src/components/vocalization/`:

- `VocalizationLabelingTab` — orchestrator, owns selected detection job + pipeline state
- `DetectionJobPicker` — rich source selector card
- `EmbeddingStatusPanel` — checks/provisions detection embeddings
- `InferencePanel` — select/queue inference, shows status
- `LabelingWorkspace` — paginated rows with label controls
- `RetrainFooter` — label count + retrain button

### Source Selector (DetectionJobPicker)

Shows **all completed detection jobs** with no binary label or extraction filter.

Display format:

- **Hydrophone jobs** (`start_timestamp` + `end_timestamp` present):
  `NOAA SanctSound (Olympic Coast OC01)    2019-06-21 00:00 UTC — 2019-06-22 00:00 UTC`

- **Local jobs** (`audio_folder` + `result_summary` present):
  `recordings/2024-june-fieldwork    847 windows`

Hydrophone and local jobs shown in optgroups. No embedding-set source type — this
page is detection-job-only. Embedding-set-based inference stays on the Training tab.

### Embedding Status Panel (EmbeddingStatusPanel)

Checks whether the selected detection job has `detection_embeddings.parquet`.

States:

1. **Exists** — collapsed: `✓ Ready (N vectors)`. Pipeline continues.
2. **Missing** — expanded with message + `[Generate Embeddings]` button.
3. **Generating** — spinner + progress: `Generating embeddings... 234 / 847 windows`.
4. **Failed** — error message + retry button.

Detection embeddings are created during the detection job run (not via EmbeddingSet
processing jobs). For jobs that lack them, a new backend task generates them post-hoc
by reading the row store, slicing the audio windows, and running them through the
classifier model's embedding layer.

### Inference Panel (InferencePanel)

States:

1. **No inference yet** — model selector (defaults to active) + `[Run Inference]` button.
2. **Running** — spinner, polls status.
3. **Complete** — collapsed: `✓ Complete (model: humpback-v3, 847 scored)`.
4. **Previous inference exists** — auto-selects matching completed inference job
   and collapses. Offers `[Rescore]` if model has changed.

Auto-detection: when a detection job is selected, checks existing inference jobs for
a match (`source_type === "detection_job"` and `source_id === selectedJobId`).

Uses existing `POST /vocalization/inference-jobs` endpoint. No backend changes.

### Labeling Workspace (LabelingWorkspace)

Paginated detection rows sorted by inference score uncertainty (scores closest to
threshold midpoint), surfacing the rows where human labels add the most training
value. Sort toggleable to score descending or chronological.

Row layout:

```
┌──────────────────────────────────────────────────────────────┐
│ [spectrogram thumb]  ▶ Play   Score: 0.42                   │
│                                                              │
│ Binary: humpback          Voc: [whup ✕] [moan ✕] [+ add ▾] │
│ 2019-06-21 03:14:15 UTC — 2019-06-21 03:14:20 UTC           │
└──────────────────────────────────────────────────────────────┘
```

- **Binary label**: read-only badge (humpback, orca, ship, background, or `—` for
  unlabeled). Displayed when available, not required.
- **Vocalization labels**: removable chips. `[+ add]` opens a dropdown populated
  from vocabulary (active model's `vocabulary_snapshot` merged with
  `GET /labeling/label-vocabulary`).
- **Add/remove**: immediate calls to `POST/DELETE /labeling/vocalization-labels/{detection_job_id}`.
  Optimistic UI update.
- **Spectrogram + audio**: via existing `detectionSpectrogramUrl()` and
  `detectionAudioSliceUrl()` endpoints.
- **Pagination**: 50 rows per page.

Row data sourced from:
- Inference results (scores, filename, start/end times)
- Existing vocalization labels per row

### Retrain Footer (RetrainFooter)

Sticky bar at the bottom of the page.

Display: `12 new labels since last training    [Retrain Model]`

"Since last training" logic:
- Finds the most recent completed vocalization training job whose
  `source_config.detection_job_ids` includes this detection job.
- Counts vocalization labels on this job with `created_at` after that training
  job's `completed_at`.
- If no prior training included this job: `12 labels (not yet used in training)`.

Retrain button:
1. Creates `POST /vocalization/training-jobs` with the active model's original
   `source_config` extended to include the current detection job (if not already).
2. Parameters inherited from the active model's last training job.
3. Shows inline status (queued -> running -> complete).
4. On completion, offers `[Activate new model]` button.

## Backend Changes

### New Endpoints

1. **`GET /classifier/detection-jobs/{id}/embedding-status`**
   Returns `{ "has_embeddings": bool, "count": int | null }`.
   Checks for `detection_embeddings.parquet` on disk.

2. **`POST /classifier/detection-jobs/{id}/generate-embeddings`**
   Queues embedding generation task. Returns 202 with status object for polling.

3. **`GET /vocalization/models/{id}/training-source`**
   Returns the `source_config` and `parameters` from the training job that
   produced this model.

### New Worker Task

**`generate_detection_embeddings`** — reads the detection row store, slices audio
windows, runs through classifier model embedding layer, writes
`detection_embeddings.parquet`. Added to worker priority queue after detection,
before vocalization training.

### No Changes To

- Vocalization label CRUD endpoints
- Vocalization inference job creation
- Vocalization training job creation
- Detection job listing endpoint

## Data Flow

```
Select detection job
        │
        ▼
Check embedding status ──── missing ──→ Generate embeddings (worker)
        │                                        │
        │ exists                                 │ complete
        ▼                                        ▼
Run inference (or reuse existing) ◄──────────────┘
        │
        │ complete
        ▼
Load rows: inference scores + existing vocalization labels
        │
        ▼
User labels rows (add/remove vocalization types)
        │
        ▼
Retrain model (extends active model's source_config with this job)
        │
        ▼
Activate new model → rescore with updated model
```
