# Sequence Models Epoch Timestamp Alignment — Design Spec

## Overview

Sequence Models artifacts currently drift from the project timestamp rule. The
design says continuous embedding and HMM window rows use UTC epoch seconds, but
the worker writes job-relative seconds into `start_time_sec` / `end_time_sec`.
PR #151 then reused those relative values as timeline-provider coordinates,
which broke the HMM State Timeline Viewer spectrogram for real hydrophone jobs.

This spec aligns Sequence Models on the project convention:

- `start_timestamp` / `end_timestamp` mean UTC epoch seconds.
- Relative offsets are allowed only as narrow internal calculation variables, or
  as explicitly named `*_offset_sec` fields when there is a documented reason to
  persist them.
- Ambiguous persisted/API fields such as `start_time_sec` / `end_time_sec` are
  removed from Sequence Models contracts.

## Goals

- Make every stored and API-passed Sequence Models window/span time an epoch
  timestamp.
- Fix the HMM State Timeline Viewer so spectrogram tiles, the state bar, and
  audio playback share one epoch coordinate system.
- Provide a one-time migration path for existing relative-time Sequence Models
  artifacts.
- Update docs, API types, UI labels, and tests so nonzero epoch starts are
  covered.
- Fail loudly on ambiguous old Sequence Models artifacts after migration, rather
  than silently normalizing them forever.

## Non-Goals

- Rework all Call Parsing region/event/correction artifacts in this change.
  Those surfaces use relative event geometry in several correction keys and
  training paths. They are listed below and should be handled by a dedicated
  future design if we decide to remove job-relative event geometry globally.
- Change the HMM math, PCA/HMM models, labels, or interpretation algorithms.
- Add a database migration for Sequence Models. The affected timestamp drift is
  in parquet/JSON artifacts and API schemas, not DB columns.

## Current Bug

For HMM job `ccac1cab-d0aa-4953-bd92-f1e9b8e209cd`:

- Source region job: `05ab8b4d-7cae-4a3e-b68b-b21b4d49f9f3`
- Region epoch range: `1311033600.0` to `1311120000.0`
- HMM states artifact range: `84.0` to `85079.0`

The HMM values are offsets into the 24-hour region job. PR #151 passes those
offsets directly to `TimelineProvider`, whose spectrogram tile math expects
epoch seconds. The result is a state bar at relative times but spectrogram tiles
for the wrong part of the job.

The same panel also passes the timeline coordinate directly into
`regionAudioSliceUrl`, while that endpoint currently accepts job-relative
`start_sec`. The fix is to make the region audio API accept epoch timestamps and
make HMM, Region Timeline, and future consumers call it with epoch coordinates.

## Decisions

### Canonical Field Names

Use `start_timestamp` / `end_timestamp` for all persisted and API-exposed
Sequence Models window/span intervals.

Remove `start_time_sec` / `end_time_sec` from:

- Continuous embedding parquet rows
- Continuous embedding manifest span summaries
- HMM states parquet rows
- HMM PCA/UMAP overlay parquet rows
- HMM exemplar JSON records
- Sequence Models Pydantic schemas
- Sequence Models TypeScript API types
- Sequence Models UI components
- Sequence Models reference docs/specs

Do not add `start_offset_sec` / `end_offset_sec` to Sequence Models artifacts in
this pass. Offsets can be derived as `timestamp - RegionDetectionJob.start_timestamp`
when needed, and persisting both coordinate systems increases drift risk.

### Worker Output

The continuous embedding worker may continue using job-relative geometry
internally while merging padded spans and extracting audio, but the row emitted
to parquet must add the source `RegionDetectionJob.start_timestamp` before
writing:

- `record.start_time_sec + region_job.start_timestamp -> start_timestamp`
- `record.end_time_sec + region_job.start_timestamp -> end_timestamp`

The HMM worker should copy these epoch fields from the continuous embedding
parquet into `states.parquet`.

Interpretation artifact generation should use and emit the same epoch fields.
The label-distribution service should no longer add a region-job offset to HMM
state rows because the state rows are already epoch-based.

### API Behavior

The `/sequence-models/hmm-sequences/{id}/states` and overlay/exemplar endpoints
must return `start_timestamp` / `end_timestamp`.

The HMM job detail endpoint must include enough source timing metadata for the
frontend to validate and build URLs:

- `region_detection_job_id`
- `region_start_timestamp`
- `region_end_timestamp`

The Call Parsing region audio endpoint should accept `start_timestamp` as the
canonical query parameter. Existing frontend callers should be updated to pass
epoch timestamps. `start_sec` should not remain part of the internal frontend
client contract.

### Existing Artifacts

Existing relative-time Sequence Models artifacts must be migrated or
regenerated. They are not an ongoing supported schema.

The migration script is dry-run by default and rewrites artifacts only with
`--apply`. It should:

- Locate continuous embedding and HMM jobs via the DB.
- Resolve the source `RegionDetectionJob.start_timestamp`.
- Rewrite legacy `start_time_sec` / `end_time_sec` fields to
  `start_timestamp` / `end_timestamp`.
- Add the region job start timestamp when legacy values are job-relative.
- Leave already-canonical epoch artifacts unchanged.
- Abort on ambiguous values rather than guessing.
- Write parquet/JSON atomically.

## Timestamp Surface Audit

| Surface | Fields | Current meaning | Modify? | Target |
|---|---|---|---|---|
| `detection_jobs` DB | `start_timestamp`, `end_timestamp` | Epoch job bounds | No | Already canonical |
| detection row store parquet | `start_utc`, `end_utc` | Epoch row bounds | No | Already canonical |
| classifier detection media APIs | `start_utc`, `duration_sec` | Epoch media start | No | Already canonical |
| `call_parsing_runs` DB | `start_timestamp`, `end_timestamp` | Epoch source bounds | No | Already canonical |
| `region_detection_jobs` DB | `start_timestamp`, `end_timestamp` | Epoch source bounds | No | Already canonical |
| region `trace.parquet` | `offset_sec`, `end_sec` | Job-relative dense trace geometry | No | Dedicated future pass if global Call Parsing conversion is desired |
| region `regions.parquet` | `start_sec`, `end_sec`, `padded_start_sec`, `padded_end_sec` | Job-relative region geometry | No | Dedicated future pass; correction/training keys depend on these today |
| `/call-parsing/region-jobs/{id}/regions` | same as region parquet | Job-relative region geometry | No | Dedicated future pass |
| `/call-parsing/region-jobs/{id}/tile` | `tile_index`, `zoom_level` | Tile index; backend derives epoch from job start | No | Already epoch internally |
| `/call-parsing/region-jobs/{id}/audio-slice` | `start_sec`, `duration_sec` | Job-relative media start | Yes | Replace frontend/client contract with `start_timestamp`, `duration_sec` |
| `region_boundary_corrections` DB | `start_sec`, `end_sec` | Job-relative corrected region bounds | No | Dedicated future pass; correction identity depends on these |
| `event_segmentation_jobs` DB | no source timestamps | N/A | No | N/A |
| event `events.parquet` | `start_sec`, `end_sec`, `center_sec` | Job-relative event geometry | No | Dedicated future pass |
| `event_boundary_corrections` DB | `original_*_sec`, `corrected_*_sec` | Job-relative event bounds | No | Dedicated future pass; correction identity depends on these |
| event classification `typed_events.parquet` | `start_sec`, `end_sec` | Job-relative event geometry | No | Dedicated future pass |
| `vocalization_corrections` DB | `start_sec`, `end_sec` | Job-relative event/window bounds | No | Dedicated future pass |
| `segmentation_training_samples` DB | `start_timestamp`, `end_timestamp` | Epoch crop source bounds | No | Already canonical |
| `segmentation_training_samples` DB | `crop_start_sec`, `crop_end_sec` | Offset inside the training crop | No | Explicit offset fields; acceptable |
| segmentation sample `events_json` | `start_sec`, `end_sec` | Offset inside the crop | No | Explicit local crop geometry; acceptable |
| `continuous_embedding_jobs` DB | created/updated timestamps only | Row lifecycle timestamps | No | Not source geometry |
| CEJ `embeddings.parquet` | `start_time_sec`, `end_time_sec` | Intended epoch, currently job-relative | Yes | `start_timestamp`, `end_timestamp` epoch |
| CEJ `manifest.json` spans | `start_time_sec`, `end_time_sec` | Intended epoch, currently job-relative | Yes | `start_timestamp`, `end_timestamp` epoch |
| CEJ detail API/TS types | `start_time_sec`, `end_time_sec` | Mirrors manifest | Yes | `start_timestamp`, `end_timestamp` |
| `hmm_sequence_jobs` DB | created/updated timestamps only | Row lifecycle timestamps | No | Not source geometry |
| HMM `states.parquet` | `start_time_sec`, `end_time_sec` | Intended epoch, currently job-relative | Yes | `start_timestamp`, `end_timestamp` epoch |
| HMM `/states` API/TS types | `start_time_sec`, `end_time_sec` | Mirrors states parquet | Yes | `start_timestamp`, `end_timestamp` |
| HMM `pca_overlay.parquet` | `start_time_sec`, `end_time_sec` | Intended epoch, currently job-relative | Yes | `start_timestamp`, `end_timestamp` epoch |
| HMM overlay API/TS types | `start_time_sec`, `end_time_sec` | Mirrors overlay parquet | Yes | `start_timestamp`, `end_timestamp` |
| HMM `exemplars.json` | `start_time_sec`, `end_time_sec` | Intended epoch, currently job-relative | Yes | `start_timestamp`, `end_timestamp` epoch |
| HMM exemplars API/TS types | `start_time_sec`, `end_time_sec` | Mirrors exemplar JSON | Yes | `start_timestamp`, `end_timestamp` |
| HMM `label_distribution.json` | no window timestamps | Derived counts only | No | Regenerate only if state/artifact logic changes counts |
| HMM State Timeline Viewer | `TimelineProvider.jobStart/jobEnd`, state bar item times | Receives relative state artifact times | Yes | Pass epoch span/state times |
| HMM State Timeline Viewer audio URL | passes timeline coordinate as `start_sec` | Mixed coordinate systems | Yes | Pass epoch `start_timestamp` to canonical region audio URL |
| `SpanNavBar` | `startSec`, `endSec` props | Ambiguous UI naming | Yes | `startTimestamp`, `endTimestamp` |
| `HMMStateBar` | `start_time_sec`, `end_time_sec` props | Ambiguous UI naming | Yes | `start_timestamp`, `end_timestamp` |

## Implementation Sketch

### Backend

1. Update continuous embedding parquet schema:
   - Replace `start_time_sec` / `end_time_sec` with
     `start_timestamp` / `end_timestamp`.
   - Add source job epoch before writing rows and span summaries.
2. Update HMM states schema:
   - Read `start_timestamp` / `end_timestamp` from CEJ parquet.
   - Write those fields to `states.parquet`.
3. Update interpretation generators:
   - Overlay metadata uses `start_timestamp` / `end_timestamp`.
   - Exemplar records use `start_timestamp` / `end_timestamp`.
   - Label distribution consumes epoch HMM states directly.
4. Update Sequence Models API schemas and routers:
   - Response models expose `start_timestamp` / `end_timestamp`.
   - HMM detail includes `region_start_timestamp` / `region_end_timestamp`.
   - API rejects missing canonical timestamp fields in Sequence Models artifacts.
5. Update region audio endpoint:
   - Canonical query param: `start_timestamp`.
   - Internally convert to job-relative only where archive resolution requires it.

### Frontend

1. Update `frontend/src/api/sequenceModels.ts` timestamp fields.
2. Update Continuous Embedding detail table headers and values.
3. Update HMM detail page:
   - Derive spans from `start_timestamp` / `end_timestamp`.
   - Pass epoch spans into `TimelineProvider`.
   - Pass epoch state rows into `HMMStateBar`.
   - Build region audio URLs with epoch `start_timestamp`.
4. Update `SpanNavBar` and `HMMStateBar` prop names and tooltips.
5. Update E2E mocks to use a nonzero epoch range and assert audio/tile URL
   coordinates.

### Migration

Add `scripts/migrate_sequence_model_timestamps.py`.

Dry run:

```
uv run python scripts/migrate_sequence_model_timestamps.py
```

Apply:

```
uv run python scripts/migrate_sequence_model_timestamps.py --apply
```

The script should report:

- CEJ parquets scanned, already canonical, migrated, skipped, failed.
- CEJ manifests scanned, already canonical, migrated, skipped, failed.
- HMM state parquets scanned, already canonical, migrated, skipped, failed.
- HMM overlay parquets scanned, already canonical, migrated, skipped, failed.
- HMM exemplar JSONs scanned, already canonical, migrated, skipped, failed.

## Testing

Backend tests:

- Continuous embedding worker with `RegionDetectionJob.start_timestamp != 0`
  writes epoch `start_timestamp` / `end_timestamp` values.
- HMM worker preserves epoch timestamps from CEJ artifacts.
- HMM `/states`, `/overlay`, `/exemplars`, and detail responses expose canonical
  timestamp fields.
- Label distribution no longer double-adds the region job start timestamp.
- Region audio endpoint accepts epoch `start_timestamp` and resolves the same
  audio as the previous job-relative calculation.
- Migration script dry-run detects relative artifacts and apply rewrites temp
  fixtures atomically.

Frontend tests:

- HMM State Timeline Viewer receives nonzero epoch state rows and requests the
  tile index implied by `timestamp - region_start_timestamp`.
- Audio playback URL uses `start_timestamp`, not `start_sec`.
- Span navigation labels format epoch timestamps correctly.
- Continuous Embedding detail no longer labels relative values as UTC seconds.

Manual verification:

- Run the migration script dry-run against the local production-like DB.
- Apply migration to a copy of the DB/storage first.
- Open
  `/app/sequence-models/hmm-sequence/ccac1cab-d0aa-4953-bd92-f1e9b8e209cd`.
- Verify the HMM state bar and PCEN spectrogram align on the first span around
  `2011-07-18T00:01:24Z`.
- Verify playback starts at the same visual playhead location.

## Rollout

1. Land code changes and tests.
2. Run migration dry-run on local storage.
3. Apply migration to local storage after backing up `continuous_embeddings/` and
   `hmm_sequences/`.
4. Re-run verification gates.
5. Treat any remaining `start_time_sec` / `end_time_sec` in Sequence Models
   artifacts as invalid and fix by regeneration or migration.
