# Event-Level Vocalization Corrections

## Problem

Window classification scores individual 5s windows at 1s hop, producing per-window
per-type probabilities. Due to Perch embedding position sensitivity, the same
vocalization scores very differently across overlapping windows (e.g., 97% → 9% → 1.8%
for a single Whup). The current per-window correction system requires correcting N
overlapping windows per vocalization, which is not tractable.

Additionally, two review surfaces — Window Classify review and Classify review — both
produce label corrections but write to separate tables (`window_score_corrections` and
`event_type_corrections`), creating duplicate effort and no shared source of truth.

## Solution

Replace both per-window and per-event correction tables with a single unified
`vocalization_corrections` table keyed by time range and detection job. Corrections are
made at the Pass 2 event level — one correction per vocalization occurrence — and are
accessible from both review UIs. The Window Classify review workspace gains event
boundary overlays and event-level navigation.

## Data Model

### New table: `vocalization_corrections`

| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| region_detection_job_id | string | FK → region_detection_jobs |
| start_sec | float | Event start time (snapshot from Pass 2 event) |
| end_sec | float | Event end time (snapshot) |
| type_name | string | Vocalization type |
| correction_type | string | "add" or "remove" |
| created_at | datetime | |
| updated_at | datetime | |

**Unique constraint:** `(region_detection_job_id, start_sec, end_sec, type_name)`.
Upserting the same key updates `correction_type`.

**Index:** `region_detection_job_id`.

**Cascade:** Deleting a region detection job cascades to its vocalization corrections.

### Tables retired

- `window_score_corrections` — dropped, no data migration
- `event_type_corrections` — dropped, no data migration

Both tables are lightly used review artifacts. Existing rows are discarded.

### Key design: detection job as anchor

Corrections are keyed by `region_detection_job_id` rather than run or audio source:

- Corrections **survive** Pass 2 and Pass 3 re-runs (the most common re-runs)
- Corrections **don't carry** if Pass 1 is re-run (correct, since new detection means
  different windows and context)
- Works regardless of whether a parent `call_parsing_run` exists

### Correction semantics

- One row per `(detection_job, start_sec, end_sec, type_name)` — binary add/remove
- Correcting multiple types on one event produces multiple rows with the same time range
- Time range is a snapshot of Pass 2 event boundaries at correction time, not a live
  reference

## Correction Resolution in Prior Vector Construction

When Pass 3 (future work) builds the aggregated prior vector for an event:

1. Collect all windows overlapping the event `[e_start, e_end]`
2. Query `vocalization_corrections` for the same detection job where the correction's
   time range overlaps the event (`c_start < e_end AND c_end > e_start`)
3. For each matching correction:
   - **"add"**: override that type's score to **1.0** for all overlapping windows before
     aggregation
   - **"remove"**: override that type's score to **0.0** for all overlapping windows
     before aggregation
4. Aggregate per type using the corrected scores (weighted mean by overlap duration,
   plus max)

A correction on one event can influence an adjacent event if their time ranges overlap.
This is intentional — if a reviewer says "Whup is present at 94–99s", a neighboring
event at 97–103s should benefit for the overlapping portion.

## API

### New endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/call-parsing/vocalization-corrections` | Upsert corrections (batch) |
| GET | `/call-parsing/vocalization-corrections?region_detection_job_id={id}` | List corrections for a detection job |
| DELETE | `/call-parsing/vocalization-corrections?region_detection_job_id={id}` | Clear all corrections for a detection job |

### POST request body

```json
{
  "region_detection_job_id": "...",
  "corrections": [
    {
      "start_sec": 94.0,
      "end_sec": 99.0,
      "type_name": "Whup",
      "correction_type": "add"
    }
  ]
}
```

### Retired endpoints

- `POST/GET/DELETE /call-parsing/window-classification-jobs/{job_id}/corrections`
- `POST/GET/DELETE /call-parsing/event-classification-jobs/{job_id}/type-corrections`

## UI: Window Classify Review

### Prerequisite gate

The review workspace requires a completed Pass 2 event segmentation job for the same
detection job. If none exists, show a message indicating event segmentation must run
first.

### Visual changes

- **Event boundaries overlaid** on the existing per-window score timeline — vertical
  lines or shaded regions marking each Pass 2 event
- **Per-event correction controls** — clicking an event region shows type badges
  displaying the max score from overlapping windows, with toggleable add/remove
  corrections
- **Per-window detail preserved** — raw window-level scores remain visible underneath
  for diagnostic context
- Per-window badge toggle interaction removed

### Navigation

| Action | Keyboard | Notes |
|--------|----------|-------|
| Next event | `→` or `D` | Next Pass 2 event within current region |
| Previous event | `←` or `A` | |
| Next region | `Shift+→` or `Shift+D` | First event of next region |
| Previous region | `Shift+←` or `Shift+A` | Last event of previous region |
| Play/pause | `Space` | Selected event's time range |

### Selection state

- Always exactly one event selected (highlighted in the timeline)
- Timeline viewport centers on the selected event's region
- Event correction panel shows the selected event's type badges
- Entering the workspace selects the first event of the first region

### Playback

- Bounded to the selected event's time range
- Playhead shown on the timeline over the per-window score detail

### Correction flow

1. Reviewer sees event boundaries overlaid on window scores
2. Clicks an event → sees type badges with max scores from overlapping windows
3. Toggles a badge (add/remove)
4. Pending corrections tracked in local state, dirty indicator shown
5. Save → POST to `/call-parsing/vocalization-corrections`
6. Corrections stored with the event's `(start_sec, end_sec)` as time anchor

## UI: Classify Review

The Classify review workspace switches from `event_type_corrections` to the unified
`vocalization_corrections` endpoint:

- Correction read/write uses `/call-parsing/vocalization-corrections`
- Stores the event's `(start_sec, end_sec)` when saving
- Corrections made in Window Classify review appear automatically
- No navigation or layout changes

## Migration & Cleanup

### Alembic migration

1. Create `vocalization_corrections` table with unique constraint and index
2. Drop `window_score_corrections` table
3. Drop `event_type_corrections` table

### Code cleanup

- Remove `WindowScoreCorrection` model, service functions, API endpoints, schemas
- Remove `EventTypeCorrection` model, service functions, API endpoints, schemas
- Remove corresponding frontend hooks, API client functions, component code
- Add `VocalizationCorrection` model, service functions, API endpoints, schemas,
  frontend hooks, API client

### No changes to

- Window classification worker (still writes raw per-window scores)
- Region detection worker
- Pass 2 event segmentation

## Scope Boundaries

### In scope

- New `vocalization_corrections` table + migration
- API endpoints (CRUD)
- Window Classify review: event boundary overlay, event-level correction controls,
  region/event navigation, event playback
- Classify review: swap to unified correction endpoints
- Retire `window_score_corrections` and `event_type_corrections`

### Out of scope (future work)

- Prior vector aggregation logic in Pass 3 worker
- Manual time-range adjustment of corrections
- Aggregated score display (max-pooled) in event panel
- Multi-reviewer conflict resolution
