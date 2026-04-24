# Unified Event Boundary Corrections

## Problem

Event boundary editing currently lives in two review surfaces (Pass 2 Segment Review, Pass 3 Classify Review) backed by an `event_boundary_corrections` table keyed on `event_segmentation_job_id`. This FK anchors corrections to a specific Pass 2 run, preventing sharing across review surfaces and coupling corrections to ephemeral segmentation output rather than the stable Pass 1 detection job.

Window Classify review shows segmentation events as overlays but has no boundary editing capability.

The vocalization corrections unification in PR #139 proved the pattern: reanchor corrections to `region_detection_job_id` (Pass 1), share one table across all review surfaces, and let downstream consumers read corrections via a single overlay function.

## Design

### Database Schema

Drop the existing `event_boundary_corrections` table and create a replacement with `region_detection_job_id` as the FK anchor.

| Column | Type | Constraints | Purpose |
|---|---|---|---|
| `id` | String | PK, UUID | Row identity |
| `region_detection_job_id` | String | NOT NULL, indexed | FK anchor to Pass 1 |
| `region_id` | String | NOT NULL | Scopes correction to a region |
| `correction_type` | String | NOT NULL | `"adjust"`, `"add"`, or `"delete"` |
| `original_start_sec` | Float | nullable | Original event start (null for `"add"`) |
| `original_end_sec` | Float | nullable | Original event end (null for `"add"`) |
| `corrected_start_sec` | Float | nullable | Corrected event start (null for `"delete"`) |
| `corrected_end_sec` | Float | nullable | Corrected event end (null for `"delete"`) |
| `created_at` | DateTime | auto | |
| `updated_at` | DateTime | auto | |

**Index:** `ix_event_boundary_corrections_detection_job` on `region_detection_job_id`.

**No DB-level unique constraint.** Upsert deduplication is handled in the service layer:
- For `adjust` / `delete`: match on `(region_detection_job_id, region_id, original_start_sec, original_end_sec)`
- For `add`: match on `(region_detection_job_id, region_id, corrected_start_sec, corrected_end_sec)`

**Model location:** `models/call_parsing.py`, co-located with `VocalizationCorrection`.

### API Surface

Unified endpoint set paralleling vocalization corrections:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/call-parsing/event-boundary-corrections` | Upsert corrections |
| `GET` | `/call-parsing/event-boundary-corrections?region_detection_job_id={id}` | List corrections |
| `DELETE` | `/call-parsing/event-boundary-corrections?region_detection_job_id={id}` | Clear corrections |

**POST request body:**

```json
{
  "region_detection_job_id": "...",
  "corrections": [
    {
      "region_id": "...",
      "correction_type": "adjust | add | delete",
      "original_start_sec": "float | null",
      "original_end_sec": "float | null",
      "corrected_start_sec": "float | null",
      "corrected_end_sec": "float | null"
    }
  ]
}
```

Old endpoints removed: `POST/GET/DELETE /call-parsing/segmentation-jobs/{job_id}/corrections`.

### Correction Overlay Logic

`load_corrected_events()` in `extraction.py` changes signature:

```python
# Before
async def load_corrected_events(session, segmentation_job_id, storage_root)

# After
async def load_corrected_events(session, region_detection_job_id, segmentation_job_id, storage_root)
```

`segmentation_job_id` locates the parquet file. `region_detection_job_id` queries corrections.

**Overlay algorithm:**

1. Read events from segmentation parquet (`events.parquet`)
2. Query `event_boundary_corrections` by `region_detection_job_id`
3. Apply corrections (scoped by `region_id`):
   - **Delete:** Remove events matching `(region_id, original_start_sec, original_end_sec)`
   - **Adjust:** Replace matching event times with `(corrected_start_sec, corrected_end_sec)`
   - **Add:** Insert new event with `(corrected_start_sec, corrected_end_sec)` into the specified `region_id`
4. Return corrected event list

**Matching:** Exact float equality on `(region_id, start_sec, end_sec)` — times originate from the same parquet or from prior corrections, so exact equality is reliable.

**Consumers:**
- Classification inference worker — corrected boundaries for Pass 3
- Classifier feedback training worker — corrected boundaries + vocalization label resolution
- Segmentation training (future) — corrections as ground truth targets
- All three review UIs — merge saved + pending corrections over parquet events

### Frontend Changes

**Shared hooks** (in `useCallParsing.ts`):
- `useEventBoundaryCorrections(regionDetectionJobId)` — replaces `useBoundaryCorrections(segJobId)`
- `useUpsertEventBoundaryCorrections()` — POST to unified endpoint
- `useClearEventBoundaryCorrections()` — DELETE to unified endpoint

**Pass 2 — SegmentReviewWorkspace:**
- Look up `region_detection_job_id` from the segmentation job's parent chain
- Swap save/load calls to new hooks
- Overlay logic unchanged (merge saved + pending over parquet events)

**Pass 3 — ClassifyReviewWorkspace:**
- Already has `region_detection_job_id` (used for vocalization corrections)
- Swap boundary correction hooks to new unified ones
- Dual-correction display (boundary + vocalization) unchanged

**Window Classify — WindowClassifyReviewWorkspace:**
- Already has `region_detection_job_id`
- Add `EventBarOverlay` component with draggable handles, add mode, and delete — same interactive boundary editing as Pass 2 and Pass 3
- Add pending boundary correction state (`Map<correctionKey, BoundaryCorrection>`)
- Save both boundary and vocalization corrections on submit (parallel POSTs, same pattern as Pass 3)

### Migration & Cleanup

**Alembic migration (055):**
1. Drop the existing `event_boundary_corrections` table
2. Create the new table with the schema above
3. Use `op.batch_alter_table()` for SQLite compatibility

Clean drop-and-replace, no data migration.

**Dead code removal:**
- Old model: `EventBoundaryCorrection` in `models/feedback_training.py`
- Old Pydantic schemas: `BoundaryCorrectionRequest`, `BoundaryCorrectionResponse`
- Old service functions: `upsert_boundary_corrections(segmentation_job_id, ...)`, `list_boundary_corrections(segmentation_job_id, ...)`, `clear_boundary_corrections(segmentation_job_id, ...)`
- Old API endpoints: `POST/GET/DELETE /call-parsing/segmentation-jobs/{job_id}/corrections`
- Old frontend hooks: `useBoundaryCorrections(segJobId)`, `useSaveBoundaryCorrections()`

### Re-running Pass 2

Corrections survive Pass 2 re-runs because they are anchored to the detection job, not the segmentation job. Corrections are silently applied to all downstream consumers. This is intentional — corrections represent ground truth about what events exist in the audio.

## Scope

**In scope:**
- New unified `event_boundary_corrections` table with `region_detection_job_id` FK
- Unified API endpoints
- Updated overlay logic in `extraction.py`
- All three review surfaces reading/writing via unified table
- Full boundary editing (adjust, add, delete) added to Window Classify review
- Migration dropping old table
- Dead code cleanup
- Updated downstream workers

**Not in scope:**
- Segmentation model training from corrections (future work)
- Stale correction handling after re-running Pass 2 (silent application)
- Changes to vocalization corrections (already unified)
- Changes to Pass 1 or Pass 4

## Testing

- Unit tests for the new service layer (upsert matching logic for each correction type)
- Unit tests for the updated overlay logic (time-range matching instead of event_id)
- Update existing tests referencing old table/endpoints
- Playwright tests for Window Classify boundary editing
