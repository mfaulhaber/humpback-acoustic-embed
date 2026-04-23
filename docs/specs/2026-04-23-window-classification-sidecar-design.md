# Window Classification Sidecar — Design Spec

**Date:** 2026-04-23
**Status:** Draft

## Overview

Window Classification is a standalone sidecar enrichment for the call parsing
pipeline. It slides 5-second windows across Pass 1 regions, scores cached Perch
embeddings through existing multi-label vocalization classifiers, and produces
dense per-window probability vectors. Output is consumed downstream via
compute-on-read aggregation onto segmented events.

This pass is not numbered — it runs independently against a completed region
detection job and has no FK on `CallParsingRun`.

## Data Flow

```
Pass 1 (RegionDetectionJob)
  ├── trace.parquet          (existing — scalar scores)
  ├── embeddings.parquet     (NEW — 1536-d Perch vectors, 1s hop)
  └── regions.parquet        (existing)
          │
          ▼
WindowClassificationJob (standalone sidecar)
  ├── reads embeddings.parquet + regions.parquet from Pass 1
  ├── selects windows whose center falls within padded region bounds
  ├── scores embeddings through VocalizationClassifierModel pipelines
  └── writes window_scores.parquet (wide format)
          │
          ▼
Downstream consumers (compute-on-read)
  ├── load window_scores.parquet + events.parquet
  └── compute overlap-weighted event priors on the fly
```

## Pass 1 Modification: Embedding Cache

The region detection worker already computes 1536-d Perch embedding vectors
during inference (the scalar confidence score comes from a classifier running on
them). Currently only the scalar score is persisted to `trace.parquet`.

**Change:** After writing `trace.parquet`, also write `embeddings.parquet` with
the full vectors. This is persisting what is already in memory — no additional
model inference.

**`embeddings.parquet` schema:**

| Column | Type | Nullable |
|--------|------|----------|
| time_sec | float64 | no |
| embedding | list_[float32, 1536] | no |

Written to the existing region job directory:
`call_parsing/regions/<job_id>/embeddings.parquet`.

Hop size matches Pass 1's default (1.0s). No separate hop configuration.

## Database

### New table: `window_classification_jobs`

| Column | Type | Notes |
|--------|------|-------|
| id | str (UUID, PK) | UUIDMixin |
| status | str | queued → running → complete / failed |
| region_detection_job_id | str | Completed Pass 1 job (required) |
| vocalization_model_id | str | sklearn_perch_embedding model (required) |
| config_json | text, nullable | Threshold overrides, future knobs |
| window_count | int, nullable | Total windows scored |
| vocabulary_snapshot | text, nullable | JSON array of type names at inference time |
| error_message | text, nullable | |
| started_at | datetime, nullable | |
| completed_at | datetime, nullable | |
| created_at | datetime | TimestampMixin |
| updated_at | datetime | TimestampMixin |

### New table: `window_score_corrections`

| Column | Type | Notes |
|--------|------|-------|
| id | str (UUID, PK) | UUIDMixin |
| window_classification_job_id | str | FK to parent job (required) |
| time_sec | float | Window start time |
| region_id | str | Region the window belongs to |
| correction_type | str | "add" or "remove" |
| type_name | str | Vocalization type being corrected |
| created_at | datetime | TimestampMixin |
| updated_at | datetime | TimestampMixin |

A correction row means "this type is present (add) or absent (remove) for this
window, overriding inference." Multiple correction rows per window are valid
(multi-label).

## Storage Layout

```
call_parsing/
  regions/<job_id>/
    trace.parquet
    embeddings.parquet        ← NEW (Pass 1 writes this)
    regions.parquet
  window_classification/<job_id>/
    window_scores.parquet     ← wide format
```

### `window_scores.parquet` schema (wide format)

| Column | Type | Nullable |
|--------|------|----------|
| time_sec | float64 | no |
| region_id | string | no |
| *<type_name>* | float64 | no |

One float64 column per vocalization type in the model's vocabulary at inference
time. Column set depends on the model. Example columns: `time_sec`, `region_id`,
`whup`, `moan`, `growl`, `shriek`.

## Worker

**File:** `src/humpback/workers/window_classification_worker.py`

Follows the existing worker pattern (crash-safe, atomic writes):

1. Claim a queued job via `UPDATE ... WHERE status = 'queued'`
2. Load upstream Pass 1 artifacts: `regions.parquet` + `embeddings.parquet`
3. Load the `VocalizationClassifierModel` via `load_vocalization_model()` from
   `vocalization_inference.py`
4. For each region, select embeddings where the window center
   (`time_sec + 2.5`) falls within `[padded_start_sec, padded_end_sec]`
5. Score all selected embeddings through `score_embeddings()` — produces
   `{type_name: ndarray}` dict
6. Write `window_scores.parquet` atomically via temp-file rename
7. Update job row: `status=complete`, `window_count`, `vocabulary_snapshot`
8. On failure: clean up partial artifacts, set `status=failed` + `error_message`

No GPU needed — sklearn pipelines run on CPU.

## API Endpoints

Added to the existing `call_parsing` router.

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/window-classification-jobs` | Create job (region_job_id + vocalization_model_id) |
| GET | `/window-classification-jobs` | List all jobs |
| GET | `/window-classification-jobs/{id}` | Get single job |
| DELETE | `/window-classification-jobs/{id}` | Delete job + artifacts |
| GET | `/window-classification-jobs/{id}/scores` | Read window_scores.parquet, return JSON |
| POST | `/window-classification-jobs/{id}/corrections` | Upsert corrections (batch) |
| GET | `/window-classification-jobs/{id}/corrections` | List corrections |
| DELETE | `/window-classification-jobs/{id}/corrections` | Clear all corrections |

The `/scores` endpoint supports query params for filtering:
`?region_id=...&min_score=0.5&type_name=whup`.

## Window Inclusion Rule

A window is included in a region's score set when its center falls within the
region's padded bounds:

```
padded_start_sec <= time_sec + 2.5 <= padded_end_sec
```

This ensures full coverage without requiring audio padding or re-extraction.

## Event-Level Aggregation (Compute-on-Read)

When a downstream consumer (Pass 3, UI) needs event-level priors, it loads both
`events.parquet` and `window_scores.parquet` and computes overlap-weighted
aggregation on the fly:

```
For each event [e_start, e_end]:
  1. Find all windows overlapping the event
  2. Compute overlap weight: overlap = duration(window ∩ event)
  3. Aggregate per class: weighted = sum(p_i * overlap_i) / sum(overlap_i)
```

No materialized aggregation — always reflects current event boundaries (including
corrections).

## Frontend

### Navigation

"Window Classify" added at the bottom of the Call Parsing sub-nav in the left
sidebar.

**Route:** `/app/call-parsing/window-classify`

### Page Structure

Two tabs — Jobs and Review — matching the existing `ClassifyPage` pattern.

### Jobs Tab

- **Job form:** Two selectors — completed region detection job +
  vocalization model (sklearn_perch_embedding family only). Create button.
- **Active Jobs table:** Source, region job, model, status. Polls at 3s.
- **Previous Jobs table:** Adds window count column + Review button.

### Review Tab

**Job selector** — dropdown of completed window classification jobs, labeled
with hydrophone/source name + short ID + window count.

**Region navigator** — prev/next buttons stepping through regions. Shows
"Region N of M" plus the region's time range.

**Spectrogram** — PCEN spectrogram from the upstream region detection job, using
existing `TimelineProvider` + `Spectrogram` + `RegionBoundaryMarkers`. Clicking
on the spectrogram selects the window at that position. Selected window
highlighted with a vertical band overlay.

**Confidence strip** — single `ConfidenceStrip` below the spectrogram with two
controls above it:

- **Type selector:** dropdown — "All types (max)" shows `max(scores)` per
  window; individual type name shows that type's score.
- **Threshold input:** numeric value (default from model's per-type thresholds).

**Zoom selector** — standard `ZoomSelector` component.

**Detail panel** — appears when a window is selected:

- Window time range + region ID + play button
- Vocalization badges (labeling workspace pattern):
  - Above-threshold types as colored badges with score percentages
  - Below-threshold types as dimmed badges
  - Pending corrections with ring highlight + dot indicator
  - Plus popover button listing available types to add
  - Click existing badge to toggle removal
- Corrections accumulated locally, batch-saved via Save/Cancel toolbar

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `[` / `A` | Previous region |
| `]` / `D` | Next region |
| `Space` | Play selected window |
| `+` / `-` | Zoom in/out |
| `←` / `→` | Pan |

**Unsaved state:** Dirty indicator in toolbar, `beforeunload` warning.

## Correction Flow

Corrections are stored in the `window_score_corrections` table and loaded by the
review workspace alongside inference scores. Effective labels per window are
computed by applying corrections on top of inference: "add" corrections mark a
type as present regardless of score; "remove" corrections mark a type as absent.

How corrections flow back into model training is deferred to a future session.

## Not In Scope

- Numbered pipeline pass or FK on `CallParsingRun`
- New model training infrastructure (reuses existing vocalization models)
- Materialized event-level aggregation (compute-on-read only)
- Correction-driven retraining (deferred)
- Custom hop size (matches Pass 1's 1.0s hop)
