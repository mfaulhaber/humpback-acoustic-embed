# Classifier Training Detection Job Table Improvements

**Date:** 2026-04-19
**Status:** Draft

## Problem

The Classifier/Training page's detection job picker lacks key information needed for informed training job setup:

1. No visibility into label distribution (positive/negative counts) per detection job before training
2. No clear indication when a detection job is missing embeddings for the selected model
3. The "Re-embed" button and status column are confusing — status shows for all states including complete, and the button label implies re-doing work rather than initial embedding

## Design

### New Backend Endpoint — Label Counts

`GET /classifier/detection-jobs/label-counts?detection_job_ids[]=abc&detection_job_ids[]=def`

Returns per-job positive/negative counts by reading row stores. Reuses existing `detection_rows.py` counting logic: humpback=1 or orca=1 counts as positive; ship=1 or background=1 counts as negative.

Response schema:
```json
[
  { "detection_job_id": "abc", "positive": 142, "negative": 1208 },
  { "detection_job_id": "def", "positive": 87, "negative": 943 }
]
```

Called once on page load for all labeled detection jobs. No caching, no database migration.

### Frontend — Detection Job Table Upgrade

Replace the compact checkbox list in `DetectionSourcePicker` with a flat table. Remove `ReembeddingStatusTable` entirely — its functionality merges into the new table.

#### Table Columns

| Column | Source | Behavior |
|--------|--------|----------|
| Checkbox | local state | Toggle selection |
| Detection Job | existing `fmtDetectionJobLabel()` | Hydrophone name + date range |
| Pos | new label counts endpoint | Green text, always visible for all listed jobs |
| Neg | new label counts endpoint | Gray text, always visible for all listed jobs |
| Embedding | existing `useReembeddingStatus()` | Dash until model selected. Blank when embeddings exist (`complete`). "Missing" indicator when `not_started`. Status badge (Queued/Running with progress %) when queued or running. "Failed" badge with error popover on failure. |
| Action | existing `useEnqueueReembedding()` | "Embed" button only when status is `not_started`. "Retry" button when `failed`. Hidden otherwise. |

#### Key Behaviors

- Label counts fetched upfront for all listed jobs via a single batch call on mount
- Embedding status queried only when a model is selected (existing `useReembeddingStatus` hook, unchanged)
- Table lives inside a scrollable container (same `max-h-40` as current picker) to keep the component compact
- `isReady` logic unchanged — all selected jobs must have complete embeddings before training is enabled

### File Changes

| File | Change |
|------|--------|
| `src/humpback/api/routers/classifier/training.py` | New `GET /classifier/detection-jobs/label-counts` endpoint |
| `src/humpback/schemas/classifier.py` | New `DetectionJobLabelCount` response schema |
| `frontend/src/api/client.ts` | New `fetchDetectionJobLabelCounts()` function |
| `frontend/src/api/types.ts` | New `DetectionJobLabelCount` type |
| `frontend/src/hooks/queries/useClassifier.ts` | New `useDetectionJobLabelCounts()` hook |
| `frontend/src/components/classifier/DetectionSourcePicker.tsx` | Replace checkbox list with table, integrate embedding status + label counts + embed button inline |
| `frontend/src/components/classifier/ReembeddingStatusTable.tsx` | Delete — functionality absorbed into DetectionSourcePicker |
| `tests/` | New test for label counts endpoint |

### What Doesn't Change

- No database migration
- No changes to the embedding status API or re-embedding API
- No changes to the training job creation flow
- `TrainingTab.tsx` unchanged — it consumes `DetectionSourcePickerValue` which keeps the same interface
