# Effective Event Identity Across Call Parsing and Sequence Models

**Date:** 2026-05-03
**Status:** Draft

## Problem

The Call Parsing and Sequence Models workflows use the word "event" for two
different contracts:

- `events.parquet` rows produced by a specific `EventSegmentationJob`
- human-reviewed effective events produced by overlaying boundary corrections
  and vocalization corrections on top of segmentation output

This ambiguity caused stale or duplicated events to appear in review and creates
unclear downstream semantics. One visible example was a Segment Review job where
the raw segmentation event appeared next to a saved `add` correction that looked
like the user's adjusted replacement. The deeper issue is not just display
logic: boundary and vocalization corrections are currently anchored to
`region_detection_job_id`, while the event rows they edit are emitted by a
specific `event_segmentation_job_id`.

The cleaned-up current production state for the motivating source is simpler:
one retained segmentation job (`49e94a4a-61bc-4a20-98e8-b1675286c985`) and one
retained classification job (`2070352e-e459-40ee-955f-0b013563dd41`) remain for
region detection job `8aced89b-32e7-4582-a82e-c81d7ec8ef26`, and no Sequence
Models consumer rows remain for that source. Existing correction data will be
cleaned up manually; this design only needs to prevent the ambiguity from
recurring.

A second observed symptom in the retained classify job: navigating to Event 57
shows "No events to display" in the spectrogram area. The typed event row exists
at `8288.960-8291.584`, and the raw segmentation event has a valid `region_id`,
but a later boundary adjustment changed the effective bounds to
`8288.5-8291.7`. The classify typed-events endpoint builds its `event_id ->
region_id` map from the current corrected event list only; because the adjusted
event no longer has the original bounds in that corrected list, the endpoint
returns an empty `region_id`, and the frontend cannot resolve `currentRegion`.

## Goals

- Make event correction ownership explicit.
- Provide one backend canonical effective-event loader used by all consumers
  that want reviewed events.
- Prevent saved `add` corrections from multiplying when a user adjusts them.
- Preserve immutable inference artifacts: do not rewrite `events.parquet`.
- Let Sequence Models choose whether to consume raw segmentation output or
  reviewed effective events, and make that choice visible in job metadata.
- Keep existing review workflows ergonomic: Segment Review, Classify Review, and
  Window Classify should keep sharing reviewed event/vocal labels when they are
  truly reviewing the same segmentation output.

## Non-Goals

- No wholesale replacement of parquet artifacts with SQL event rows.
- No automatic semantic deduplication of overlapping events from different
  segmentation models.
- No attempt to infer stale correction intent across deleted historical
  segmentation jobs. Existing data cleanup is manual.
- No mutation of existing `events.parquet` or `typed_events.parquet` files.

## Approaches Considered

### Approach A: Keep Region-Scoped Corrections, Filter More Carefully

Keep `event_boundary_corrections` and `vocalization_corrections` anchored only
to `region_detection_job_id`. Add UI and loader heuristics that ignore `add`
corrections if they overlap a raw segmentation event, or only show corrections
near matching event bounds.

**Pros**

- Small schema change or no schema change.
- Keeps the original "corrections survive Pass 2 reruns" behavior.
- Minimal frontend API churn.

**Cons**

- Still ambiguous when multiple segmentation jobs exist for one region job.
- Overlap heuristics can hide true added events or preserve stale ones.
- Sequence Models cannot know whether corrected events belong to the selected
  segmentation job.
- Does not fix duplicate saved `add` rows caused by identifying adds by edited
  time range.

**Verdict:** rejected. It treats a data ownership problem as display logic.

### Approach B: Scope Corrections to Segmentation Jobs

Add `event_segmentation_job_id` to boundary corrections, and resolve
vocalization corrections against that same effective event set when reviewing a
classification job. Corrections still carry `region_detection_job_id` for query
and cleanup convenience, but the segmentation job becomes the owner of event
boundary edits.

**Pros**

- Matches reality: an event boundary correction edits one segmentation output.
- Prevents stale corrections from a previous segmentation run from appearing in
  a newer segmentation job.
- Allows a single canonical effective-event loader keyed by segmentation job.
- Makes Sequence Models provenance explicit.
- Smaller migration than creating a full SQL event registry.

**Cons**

- Corrections no longer automatically carry across Pass 2 reruns.
- Existing region-scoped correction rows need migration/backfill.
- Window Classify needs an explicit segmentation-job context if it wants event
  boundary editing, rather than only a region job.

**Verdict:** recommended.

### Approach C: Materialize Reviewed Event Sets

Create a durable SQL or parquet "reviewed event set" artifact. Review surfaces
write into that set, and downstream jobs choose raw segmentation output or a
reviewed event set ID.

**Pros**

- Very explicit provenance.
- Easy for Sequence Models to depend on a stable reviewed event set.
- Supports future branching review workflows.

**Cons**

- Larger conceptual and migration surface.
- Requires new lifecycle management: stale sets, regenerated sets, deletion,
  naming, and UI selection.
- Duplicates a lot of what read-time overlays already provide.

**Verdict:** defer. This is attractive if reviewed event sets become first-class
research artifacts, but it is more machinery than needed for the current bug.

## Recommended Design

Adopt Approach B: segmentation-scoped boundary corrections plus a shared
effective-event loader.

### Data Model

Extend `event_boundary_corrections`:

| Column | Type | Purpose |
|---|---|---|
| `id` | String UUID | Stable correction row identity |
| `region_detection_job_id` | String | Parent Pass 1 source, retained for query and cleanup |
| `event_segmentation_job_id` | String | Pass 2 event artifact this correction edits |
| `region_id` | String | Region containing the event/correction |
| `source_event_id` | String nullable | Original event ID for `adjust` and `delete`; null for true adds |
| `correction_type` | String | `"adjust"`, `"add"`, or `"delete"` |
| `original_start_sec` | Float nullable | Original event start for `adjust/delete` |
| `original_end_sec` | Float nullable | Original event end for `adjust/delete` |
| `corrected_start_sec` | Float nullable | Corrected event start for `adjust/add` |
| `corrected_end_sec` | Float nullable | Corrected event end for `adjust/add` |
| `created_at`, `updated_at` | DateTime | Existing timestamp mixin |

Add indexes:

- `ix_event_boundary_corrections_region_detection_job`
- `ix_event_boundary_corrections_segmentation_job`
- `ix_event_boundary_corrections_source_event`

Service-layer upsert rules:

- `adjust/delete`: match by `(event_segmentation_job_id, source_event_id)` when
  `source_event_id` is provided; fall back to
  `(event_segmentation_job_id, region_id, original_start_sec, original_end_sec)`
  for older clients/tests.
- `add`: match by `id` when updating an existing saved add. New add requests
  without `id` create one row. The frontend must retain the saved correction ID
  for later drags/deletes.
- Validate every `add` and `adjust` against the resulting effective event set
  before committing the batch. Within one `(event_segmentation_job_id,
  region_id)`, no two effective events may overlap after applying the proposed
  corrections.

Overlap validation details:

- Treat intervals as half-open ranges, `[start_sec, end_sec)`, with a small
  epsilon for float noise.
- Reject zero-length or negative-length corrected ranges.
- For `adjust`, exclude the source event being adjusted from the conflict check.
- For an update to a saved `add`, exclude that same correction row from the
  conflict check.
- Validate the final batch result, not only each row independently, so two
  corrections saved together cannot overlap each other.
- Return a conflict response that includes the conflicting effective event ID,
  region ID, and bounds. The frontend can use this to keep the user in context
  and show the colliding event.

Do not add a database unique constraint at first. SQLite float equality and
legacy rows make a service-level invariant easier to migrate and test. Add a
partial unique index later only if the semantics stabilize cleanly.

No historical-data migration is required for the current cleanup path. Existing
ambiguous correction rows will be fixed manually before this design is
implemented. New columns may therefore be added nullable, and new writes should
populate them. Legacy null-scope rows should be ignored by new
segmentation-scoped reads unless an endpoint is explicitly operating in
compatibility mode.

### Vocalization Corrections

Keep `vocalization_corrections` region-scoped for now, but treat them as
time-range labels applied to an effective event set, not as event identities.

Rationale:

- Window Classify labels are naturally time-range labels over Pass 1 windows.
- Classify Review already resolves saved vocalization corrections by overlap.
- The immediate duplicate-boundary problem is in `event_boundary_corrections`.

Future hardening can add optional `event_segmentation_job_id` and `event_id` to
`vocalization_corrections`, but this design does not require that extra schema
move.

### Effective Event Loader

Create a shared backend utility, for example:

```python
async def load_effective_events(
    session: AsyncSession,
    *,
    event_segmentation_job_id: str,
    storage_root: Path,
    include_boundary_corrections: bool = True,
) -> list[Event]
```

Behavior:

1. Load the `EventSegmentationJob` and its `region_detection_job_id`.
2. Read `events.parquet` for that segmentation job.
3. Query `event_boundary_corrections` by `event_segmentation_job_id`.
4. Overlay corrections:
   - `delete`: remove source event.
   - `adjust`: preserve source `event_id`, replace bounds.
   - `add`: synthesize a stable event ID from the correction row, e.g.
     `added-{correction.id}`.
5. Return events sorted by `(start_sec, end_sec, event_id)`.

`load_corrected_events()` can remain as a compatibility wrapper during the
implementation, but all new callers should use `load_effective_events()`.

For classification review, also expose or reuse a raw event identity map from
the same segmentation artifact. The canonical effective-event loader preserves
adjusted source event IDs going forward, but older `typed_events.parquet` rows
may have been written before later boundary adjustments. Those rows must still
resolve their original raw `event_id` and `region_id`.

### API

Update event-boundary correction endpoints to accept and return segmentation
scope.

`POST /call-parsing/event-boundary-corrections`

Request body keeps `region_detection_job_id` for compatibility but requires
`event_segmentation_job_id`:

| Field | Required | Notes |
|---|---|---|
| `region_detection_job_id` | yes | Must match the segmentation job's parent |
| `event_segmentation_job_id` | yes | Correction owner |
| `corrections` | yes | Batch of correction items |

Correction items add:

- `id` optional, used to update/delete saved add rows.
- `source_event_id` optional, required from modern clients for adjust/delete.

The endpoint rejects an `add` or `adjust` that would create overlapping
effective events in the same region for the same segmentation job. This check is
authoritative even if a frontend preflight check misses the conflict.

`GET /call-parsing/event-boundary-corrections` accepts:

- `event_segmentation_job_id` as the preferred filter.
- `region_detection_job_id` as a compatibility filter only where the UI needs
  cross-job counts.

Add or update:

`GET /call-parsing/segmentation-jobs/{job_id}/events?effective=true`

- Default remains raw `events.parquet` for backward compatibility.
- `effective=true` returns `load_effective_events()` output.

`GET /call-parsing/classification-jobs/{job_id}/typed-events`

- Continue returning persisted typed-event rows from `typed_events.parquet`.
- Use the canonical `load_effective_events()` path for effective event identity
  and reviewed bounds, but resolve `region_id` from raw segmentation events
  first, then effective events. This preserves renderability for classification
  artifacts produced before the latest boundary corrections, while still
  supporting added/effective event IDs produced by newer classification workers.
- Never return an empty `region_id` for a typed event whose raw source event ID
  exists in the upstream segmentation job.

### Frontend

Segment Review:

- Fetch boundary corrections by selected `event_segmentation_job_id`.
- Include `source_event_id` in adjust/delete saves.
- Preserve saved add correction IDs and send `id` when an added event is dragged
  or deleted.
- Run the same overlap check locally before save where practical, so the user
  gets immediate feedback. The backend conflict response remains the source of
  truth.
- Prefer the API's effective events if available, or keep local overlay logic
  matching `load_effective_events()`.

Classify Review:

- The classification job already carries `event_segmentation_job_id`; use that
  for boundary corrections.
- Keep resolving vocalization corrections by overlap against the effective event
  set.
- Continue showing `approved`, `correction`, `negative`, and `inference` type
  sources using the existing UI rules.

Window Classify:

- If boundary editing is enabled, require the user to choose a segmentation job
  for the same region detection job.
- If no segmentation job is selected, allow vocalization/window labeling but not
  event boundary edits.

### Downstream Consumers

Pass 3 classification worker:

- Replace `load_corrected_events(region_detection_job_id, segmentation_job_id)`
  with `load_effective_events(event_segmentation_job_id)`.
- Classification output should represent the selected reviewed event set.

Event classifier feedback worker:

- Use the same loader for audio crops and label resolution.

Segmentation feedback/dataset extraction:

- Query corrections by `event_segmentation_job_id`.
- Dataset samples come only from jobs whose segmentation output was actually
  reviewed.

Sequence Models:

- Add event source semantics to continuous embedding jobs that depend on Pass 2
  events:
  - `event_source_mode = "raw"`: read `events.parquet` exactly.
  - `event_source_mode = "effective"`: read `load_effective_events()`.
- Include `event_source_mode` and a correction revision fingerprint in the
  encoding signature for event-aware sources.
- The correction fingerprint can be a hash of correction row IDs and
  `updated_at` values for the selected segmentation job.
- Existing Sequence Models behavior should default to `raw` unless the user
  explicitly asks for reviewed/effective events.

Motif extraction:

- When resolving event anchors for a continuous embedding job, follow the
  continuous embedding job's `event_source_mode` instead of independently
  reading raw segmentation artifacts.

## Data Cleanup Assumption

No migration/backfill of existing correction intent is part of this feature.
Before implementation, manually remove stale or duplicate correction rows from
the current working database. In particular, for
`49e94a4a-61bc-4a20-98e8-b1675286c985`, stale `add` rows that fully contain a
matching adjusted raw event in the same region should be deleted, preserving the
adjusted raw event as the canonical event.

The code change may still need an Alembic schema migration to add nullable
columns and indexes, but it should not attempt to classify or repair historical
correction rows.

## Test Plan

Backend:

- Schema migration test: adds nullable segmentation/event identity columns and
  indexes without attempting data backfill.
- Service tests:
  - Adjust by `source_event_id`.
  - Delete by `source_event_id`.
  - Add creates a stable row.
  - Dragging a saved add updates by `id` rather than creating a duplicate.
  - Add is rejected when it overlaps an existing effective event in the same
    region.
  - Adjust is rejected when its corrected range overlaps another effective event
    in the same region.
  - Adjust is allowed to replace its own source event bounds.
  - Batch validation rejects two proposed corrections that would overlap each
    other.
- Loader tests:
  - Raw events are unchanged.
  - Effective events exclude deleted events.
  - Effective events preserve source event IDs for adjusted events.
  - Effective events synthesize stable IDs for adds.
  - Corrections for a different segmentation job are ignored.
- Worker tests:
  - Pass 3 classification uses effective events.
  - CRNN continuous embedding can choose raw versus effective event source.
  - Effective mode encoding signature changes when correction revision changes.
- API tests:
  - Classification typed-events endpoint returns raw segmentation `region_id`
    for a typed event whose source event was adjusted after classification.
  - Classification typed-events endpoint still resolves added/effective event
    IDs through the effective event loader.
  - Event-boundary correction endpoint returns a conflict payload for overlapping
    add/adjust requests.

Frontend:

- Segment Review regression test for the motivating case: a saved add can be
  selected, dragged, saved, and still appears once after reload.
- Segment Review blocks or cleanly surfaces backend conflicts when an add or
  adjust would overlap another event.
- Segment Review does not show corrections from another segmentation job for the
  same region job.
- Classify Review labels resolve against effective event bounds.
- Classify Review regression for the observed Event 57 case: navigating to an
  adjusted-after-classification event still renders the spectrogram because
  `region_id` is populated from the raw source event.
- Window Classify boundary editing requires a segmentation context.

Verification:

- `uv run ruff format --check` on modified Python files.
- `uv run ruff check` on modified Python files.
- `uv run pyright`.
- `uv run pytest tests/`.
- `cd frontend && npx tsc --noEmit`.

## Documentation Updates

- Append a new ADR superseding ADR-054's broad region-scoped correction overlay
  contract for event boundaries.
- Update `docs/reference/call-parsing-api.md` with segmentation-scoped
  correction fields and effective events query semantics.
- Update `docs/reference/sequence-models-api.md` with `event_source_mode` and
  correction fingerprint behavior.
- Update `docs/reference/behavioral-constraints.md` to define raw versus
  effective event contracts.
