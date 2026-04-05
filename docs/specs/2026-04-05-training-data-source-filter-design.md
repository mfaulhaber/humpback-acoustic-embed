# Training Data Source Type Filter

**Date**: 2026-04-05
**Status**: Approved

## Problem

The Training Data Review page filters rows by vocalization type and positive/negative group, but has no way to filter by data source (detection job vs embedding set). Users working with mixed-source training datasets cannot isolate rows from one source type.

## Design

### Backend

Add an optional `source_type` query parameter to `GET /vocalization/training-datasets/{dataset_id}/rows`:

- **Parameter**: `source_type: str | None = Query(None)` — accepts `"detection_job"` or `"embedding_set"`
- **Behavior**: When present, skip parquet rows whose `source_type` column doesn't match. Applied before the existing type/group label filter, before pagination.
- **No migration needed** — `source_type` is already stored in the parquet row data.

### Frontend

Add a segmented button group to the filter bar in `TrainingDataView.tsx`:

- **Control**: All | Detection | Embedding — three buttons in a bordered group, matching the style of the existing positive/negative toggle.
- **State**: `sourceType: "detection_job" | "embedding_set" | null`, default `null` (All).
- **Placement**: Left side of the filter bar, before the type filter buttons. Source type is a broader filter so it appears first.
- **Interactions**:
  - Clicking a source filter resets page to 0.
  - Composes with existing type and group filters (AND logic).
  - Passed as `source_type` query param to `useTrainingDatasetRows`.

### Query parameter mapping

Frontend state `sourceType` maps to the API `source_type` param:
- `null` → omitted (all rows)
- `"detection_job"` → `source_type=detection_job`
- `"embedding_set"` → `source_type=embedding_set`

## Scope

- One backend change: add `source_type` filter to rows endpoint
- One frontend change: add segmented button group and state to `TrainingDataView.tsx`
- One frontend change: pass `source_type` param through query hook and API client
- No database migration, no new endpoints, no new components
