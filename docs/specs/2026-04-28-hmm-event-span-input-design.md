# HMM Event-Span Input Refactor

## Problem

The current HMM sequence training pipeline consumes all continuous embedding
windows from merged detection regions. These regions contain sparse
vocalizations separated by stretches of background noise, producing an HMM that
spends most of its states modeling silence rather than vocalization structure.

## Goal

Refactor the continuous embedding job to consume segmentation events (Pass 2
output) instead of detection regions. Each event becomes an independent,
padded span fed to the HMM, producing a training set focused on vocalization
structure. The HMM timeline viewer gains dual region/event navigation matching
the Segment review page pattern.

## Design

### 1. Data Model Changes

#### `continuous_embedding_jobs` table

| Change | Detail |
|--------|--------|
| Remove column | `region_detection_job_id` (FK to `region_detection_jobs`) |
| Add column | `event_segmentation_job_id` (String, FK to `event_segmentation_jobs.id`, NOT NULL) |
| Rename column | `total_regions` → `total_events` |
| Semantics change | `pad_seconds` now means event padding (default 2.0), not region padding |

The detection job is reachable through the chain:
`ContinuousEmbeddingJob.event_segmentation_job_id` →
`EventSegmentationJob.region_detection_job_id` →
`RegionDetectionJob`.

#### `hmm_sequence_jobs` table

No schema changes. The FK `continuous_embedding_job_id` remains the only
input reference.

#### Encoding signature

The idempotency hash changes its input from `region_detection_job_id` to
`event_segmentation_job_id`. All other components stay the same
(`model_version`, `hop_seconds`, `window_size_seconds`, `pad_seconds`,
`target_sample_rate`, `feature_config`). Existing signatures are naturally
invalidated since the field name changes.

#### Embeddings parquet schema

| Column | Change |
|--------|--------|
| `merged_span_id` | Stays — now a sequential event index (0, 1, 2, ...), one per event |
| `event_id` | **New** — string FK back to the segmentation event |
| `source_region_ids` | Stays — populated from the event's `region_id` |
| All other columns | Unchanged |

#### Manifest JSON

- `total_regions` → `total_events`
- Each span entry gains an `event_id` field
- `merged_spans` reflects event count (1:1 with events)

### 2. Continuous Embedding Worker

#### Input resolution

- Load `events.parquet` from the segmentation job (via `read_events()`)
  instead of `regions.parquet` from the detection job
- Still resolve `RegionDetectionJob` row for hydrophone metadata
  (`hydrophone_id`, `start_timestamp`, `end_timestamp`) through
  `event_segmentation_job.region_detection_job_id`
- Validate segmentation job is complete before proceeding

#### Span construction

Replaces `merge_padded_regions()`:

1. Sort events by `start_sec`
2. For each event, build a span: `[event.start_sec - pad_seconds, event.end_sec + pad_seconds]`
3. Clamp to audio envelope (no negative offsets, no past-stream-end)
4. **No merging** — each event is its own independent span, even if pads overlap
   with adjacent events
5. Assign sequential `merged_span_id` (0, 1, 2, ...)

This keeps each span centered on a single vocalization event with a small
context buffer.

#### Embedding production

The core loop is unchanged: for each span, resolve audio, build window batch,
run model, collect rows. The `EmbedderProtocol` still receives the
`RegionDetectionJob` for audio resolution — the worker resolves it indirectly
through the segmentation job.

### 3. Service Layer & API

#### Continuous Embedding Service

- `create_continuous_embedding_job` accepts `event_segmentation_job_id` instead
  of `region_detection_job_id`
- Validates segmentation job exists and is complete
- Encoding signature uses `event_segmentation_job_id`

#### Continuous Embedding API Endpoints

- Create/queue endpoint accepts `event_segmentation_job_id`
- Detail/list endpoints return `event_segmentation_job_id` and `total_events`

#### HMM Sequence API

- Detail endpoint resolves the detection job through
  `cej.event_segmentation_job_id → seg_job.region_detection_job_id`
- Manifest spans include `event_id` for frontend navigation

### 4. Frontend: HMM Timeline Viewer Navigation

#### Dual navigation controls

Two sets of prev/next controls matching the `SegmentReviewWorkspace` /
`ReviewToolbar` pattern:

**Region-level nav:**
- Prev/Next buttons for detection regions
- Selecting a region scrolls to show that region's time range

**Event-level nav:**
- Prev/Next buttons for events (the padded spans)
- Keyboard shortcuts: **A** (prev event), **D** (next event)
- Flat list of all events sorted by `start_sec`
- Crossing region boundaries auto-switches the active region

#### Scroll behavior

"Bring into view without centering" — same logic as `SegmentReviewWorkspace`:
- 15% viewport padding margin
- Scroll direction-aware targeting (forward nav targets event end,
  backward nav targets event start)
- Sequence-numbered scroll requests to avoid races

#### State bar + spectrogram

When an event is selected, the state bar and spectrogram show the padded
event span with overlaid Viterbi states. The existing rendering works
as-is since states are indexed by `merged_span_id` and timestamps.

#### Create job form

The continuous embedding creation form changes from selecting a detection
job to selecting a completed segmentation job.

### 5. Alembic Migration

Single migration to:

1. Add `event_segmentation_job_id` column to `continuous_embedding_jobs`
2. Remove `region_detection_job_id` column
3. Rename `total_regions` → `total_events`
4. Change `pad_seconds` default from current value to `2.0`

Uses `op.batch_alter_table()` for SQLite compatibility. Existing continuous
embedding jobs have been deleted via the UI, so no data migration is needed.

## Non-Goals

- Merging overlapping padded events into combined spans
- Asymmetric event padding (before vs. after)
- Supporting both region-based and event-based continuous embedding modes
- Changes to the HMM training algorithm itself
