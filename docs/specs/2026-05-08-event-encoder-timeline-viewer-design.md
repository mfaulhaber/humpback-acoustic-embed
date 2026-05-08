# Event Encoder Timeline Viewer - Design

**Date:** 2026-05-08
**Status:** Draft
**Primary domain:** Sequence Models
**Neighbor domains:** Signal Timeline, Call Parsing, Frontend Shell

## 1. Goal

Add a read-only timeline viewer to the Event Encoder job detail page so a user
can inspect tokenized events in audio context. The viewer should mirror the
review affordances already used by the Call Parsing Segment and Classifier job
detail surfaces: event navigation, keyboard shortcuts, audio playback, zoom
controls, selected-event highlighting, and compact event badges.

The timeline viewer is the second panel on
`/app/sequence-models/event-encoder/:jobId`, directly after the job summary
card and before the existing report panel.

## 2. Scope

### In scope

- Add an Event Encoder timeline panel to `EventEncoderDetailPage`.
- Reuse the shared `TimelineProvider`, `Spectrogram`, region tile URLs, and
  region audio-slice playback path used by Call Parsing review pages.
- Expose per-event token assignments from a completed Event Encoder job through
  a small read-only Sequence Models API endpoint.
- Show tokenized events as timeline bars positioned by absolute timestamps from
  `event_tokens.parquet`.
- Highlight the selected event and render a small token badge such as `T17` on
  each event, visually similar to the Call Parsing classifier vocal label badge.
- Add previous and next event controls in the panel toolbar.
- Support keyboard shortcuts for event navigation and playback:
  - `A` / `D`: previous / next event.
  - `Space`: play or pause the selected event slice.
  - `+` / `-`: zoom in / out.
  - `ArrowLeft` / `ArrowRight`: pan the viewport.
- Include audio playback and zoom controls in the timeline panel.
- Support selecting which valid `k` tokenization to view when a job produced
  multiple valid k values.
- Add focused backend and frontend tests.

### Non-goals

- No editing of event boundaries, event types, token labels, or correction rows.
- No mutation of `events.parquet`, `typed_events.parquet`, or Event Encoder
  parquet artifacts.
- No new Event Encoder artifact format for v1.
- No global token vocabulary or cross-job token-color semantics.
- No display of skipped source events in v1. Skipped events do not currently
  have per-event skip records in an artifact, so the first viewer should show
  encoded/tokenized events only and leave skipped-event diagnostics in the
  report summary.

## 3. Existing Context

- `EventEncoderDetailPage` currently renders:
  1. job summary card;
  2. report card;
  3. descriptor table;
  4. artifacts card.
- `GET /sequence-models/event-encoders/{id}` returns the job row plus
  `manifest.json` and `report.json`, but not full per-event token assignments.
- `event_tokens.parquet` already contains the timeline-ready event token data:
  `k`, `event_id`, `region_id`, `source_sequence_key`, `sequence_index`,
  `start_timestamp`, `end_timestamp`, `token_id`, `token_label`,
  `distance_to_centroid`, `second_centroid_distance`, and `token_confidence`.
- Event Encoder timestamps are absolute epoch timestamps. They are produced by
  adding the upstream region-detection job start timestamp to Pass 2
  event-relative seconds.
- Call Parsing review pages already use `RegionAudioTimeline`, which wraps
  `TimelineProvider` with:
  - region job start/end bounds;
  - `/call-parsing/region-jobs/{id}/audio-slice`;
  - shared timeline playback state.
- Call Parsing Segment and Classifier review bodies disable provider keyboard
  shortcuts and install local shortcut handlers so event navigation and playback
  work together without duplicate listeners.

## 4. Approaches Considered

### Approach A: Add a Dedicated Event Encoder Timeline Endpoint

Add `GET /sequence-models/event-encoders/{job_id}/timeline?k={k}`. The endpoint
reads the completed job's `event_tokens.parquet`, filters to one valid `k`,
resolves the upstream `region_detection_job_id` from the CRNN Continuous
Embedding job, and returns only the fields needed by the viewer.

Pros:

- Keeps the existing detail endpoint small and report-focused.
- Uses `event_tokens.parquet` as the source of truth for token assignments.
- Preserves Event Encoder raw/effective semantics because the completed token
  artifact freezes the source event set used when the job ran.
- Gives the frontend direct access to selected-k event rows without parsing
  local artifact paths.
- Allows targeted backend tests around k filtering, artifact-missing states,
  and incomplete jobs.

Cons:

- Adds a new API surface and schema.
- The endpoint may return many event rows for large jobs, though the returned
  fields are compact.

Verdict: recommended.

### Approach B: Compose Existing APIs In The Frontend

Fetch Event Encoder detail, Call Parsing segmentation events, region metadata,
and existing report JSON, then infer token labels from `sequence_preview` or
`token_examples`.

Pros:

- Avoids a new backend endpoint.
- Reuses existing Call Parsing event and region APIs.

Cons:

- Existing report JSON intentionally contains compact previews, not complete
  event-to-token assignments.
- `token_examples` is exemplar-focused and cannot reconstruct the full
  sequence.
- Frontend cannot read `event_tokens_path` because it is a backend filesystem
  path.
- Effective-mode timelines could drift if current correction rows differ from
  the correction revision used by the completed encoder job.

Verdict: rejected.

### Approach C: Expand Event Encoder Detail To Include Timeline Rows

Append all per-event token rows to
`GET /sequence-models/event-encoders/{job_id}` whenever a job is complete.

Pros:

- One fetch for the detail page.
- Simple frontend hook shape.

Cons:

- Bloats the default detail response even when the user only needs the report.
- Harder to cache by selected `k`.
- Mixes report summary data with high-cardinality timeline data.

Verdict: not recommended for v1.

## 5. Recommended API Design

Add Pydantic response models in `src/humpback/schemas/sequence_models.py`:

- `EventEncoderTimelineEvent`
- `EventEncoderTimelineResponse`

Add a route in `src/humpback/api/routers/sequence_models.py`:

`GET /sequence-models/event-encoders/{job_id}/timeline?k={optional_int}`

Response fields:

| Field | Type | Notes |
|---|---|---|
| `job_id` | string | Event Encoder job id |
| `event_segmentation_job_id` | string | Pass 2 source id |
| `event_source_mode` | `"raw" | "effective"` | Source semantics used by the completed job |
| `continuous_embedding_job_id` | string | Upstream CRNN Continuous Embedding job |
| `region_detection_job_id` | string | Upstream Pass 1 region job for tiles/audio |
| `selected_k` | int | k used for the returned token rows |
| `valid_k_values` | int array | k values available in `event_tokens.parquet` |
| `job_start_timestamp` | number | Region job absolute start timestamp |
| `job_end_timestamp` | number | Region job absolute end timestamp |
| `events` | event array | Tokenized events for `selected_k` |

Event fields:

| Field | Type | Notes |
|---|---|---|
| `event_id` | string | Event identity from Event Encoder artifact |
| `region_id` | string | Region id from Pass 2 source |
| `source_sequence_key` | string | Sequence grouping key |
| `sequence_index` | int | Event order within the source sequence |
| `start_timestamp` | number | Absolute epoch seconds |
| `end_timestamp` | number | Absolute epoch seconds |
| `token_id` | int | Numeric token id |
| `token_label` | string | Display label, for example `T17` |
| `token_confidence` | number | Confidence derived from centroid distance |
| `distance_to_centroid` | number | Tokenization diagnostic |
| `second_centroid_distance` | number or null | Tokenization diagnostic |

Endpoint behavior:

- `404` when the Event Encoder job does not exist.
- `409` when the job is not `complete`.
- `404` when the completed row has no `event_tokens_path` or the artifact is
  missing.
- `422` when the caller requests a `k` that is not present in the artifact.
- If `k` is omitted, use the lowest valid k from the artifact.
- Sort returned events by `(source_sequence_key, start_timestamp,
  end_timestamp, event_id)`.
- Treat the Event Encoder artifact as authoritative. Do not re-load current
  raw/effective Pass 2 events, because correction rows may have changed after
  the encoder job completed.
- Resolve `region_detection_job_id` from the referenced
  `ContinuousEmbeddingJob`. If the upstream embedding is not a region-CRNN
  source, return `409`; this should only happen with corrupt historical data
  because Event Encoder creation already validates the source.

## 6. Frontend Data Flow

Add typed API support in `frontend/src/api/sequenceModels.ts`:

- `EventEncoderTimelineEvent`
- `EventEncoderTimelineResponse`
- `fetchEventEncoderTimeline(jobId, k?)`
- `useEventEncoderTimeline(jobId, k, enabled)`

Query behavior:

- Enable the timeline query only when the Event Encoder job is complete.
- Keep the selected `k` in local component state.
- Initial query may omit `k`; after the response arrives, use
  `response.selected_k` as the active value.
- When the user changes `k`, refetch the timeline for that k.
- Preserve the selected event id across k changes when possible; otherwise keep
  the nearest previous sequence index.

## 7. Frontend Component Design

Add a dedicated panel component under
`frontend/src/components/sequence-models/`, for example:

- `EventEncoderTimelinePanel.tsx`
- `EventEncoderTokenOverlay.tsx`

`EventEncoderDetailPage` layout becomes:

1. Back link.
2. Job summary card.
3. Event Encoder Timeline card.
4. Report card.
5. Acoustic Descriptors card.
6. Artifacts card.

The timeline card should render a compact toolbar above the spectrogram:

- Previous event button with `ChevronLeft`.
- Next event button with `ChevronRight`.
- Current event counter, for example `Event 3 / 128`.
- Selected token badge and confidence text.
- k selector when more than one valid k exists.
- Playback button using the shared playback handle.

The spectrogram body should:

- Use `RegionAudioTimeline` or an equivalent local wrapper around
  `TimelineProvider` with:
  - `jobStart = job_start_timestamp`;
  - `jobEnd = job_end_timestamp`;
  - `zoomLevels = REVIEW_ZOOM`;
  - default zoom `30s`;
  - `playback = "slice"`;
  - `audioUrlBuilder = regionAudioSliceUrl(region_detection_job_id, ...)`.
- Render `Spectrogram` with `regionTileUrl(region_detection_job_id, ...)` and
  the same frequency range as the Call Parsing review surfaces.
- Render a read-only token overlay, not the editable `EventBarOverlay`.
- Render `ZoomSelector` below the spectrogram.

The token overlay should:

- Use `useOverlayContext()` so it shares the spectrogram coordinate system.
- Draw one event bar per returned timeline event.
- Position bars with `start_timestamp` and `end_timestamp` directly; no
  job-relative conversion is needed.
- Color bars deterministically by `token_id` and `selected_k`, using the
  existing Sequence Models `labelColor(token_id, selected_k)` helper.
- Highlight the selected event with a high-contrast ring.
- Render a small badge at the event start containing the full `token_label`.
  The badge should fit labels such as `T0`, `T17`, and `T199`.
- Select an event on click and update the toolbar counter.
- Avoid edit affordances: no drag handles, no add mode, no correction styling.

## 8. Navigation And Keyboard Behavior

The Event Encoder timeline should follow the Call Parsing review pattern:
disable `TimelineProvider` keyboard shortcuts and register one local shortcut
handler for the whole panel.

Shortcut behavior:

- Ignore shortcuts when focus is inside `input`, `textarea`, `select`, or a
  contenteditable element.
- Ignore modified keypresses using `Meta`, `Ctrl`, or `Alt`.
- `A`: select previous tokenized event and center the viewport on it.
- `D`: select next tokenized event and center the viewport on it.
- `Space`: toggle playback for the selected event slice. If no event is
  selected, play from the current viewport start for the smaller of the
  viewport duration or 30 seconds.
- `+` / `=`: zoom in.
- `-`: zoom out.
- `ArrowLeft` / `ArrowRight`: pan by ten percent of the active viewport span.

Selection rules:

- On initial load, select the first event if any events exist.
- Previous/next navigation is disabled at the ends of the returned event list.
- Changing `k` should keep the same event selected if the event id exists for
  the new k.
- Changing jobs resets selection, playback state, and zoom to defaults.
- Clicking an event bar selects it and centers the viewport on its midpoint.

## 9. Empty And Error States

- Queued/running jobs: show the timeline card in the second position with a
  muted message that the timeline will appear when tokenization is complete.
- Failed/canceled jobs: show the existing job error and a muted unavailable
  state in the timeline card.
- Complete job with no encoded events: show an empty-state message and keep the
  report summary visible.
- Missing timeline artifact: show an error message in the panel, but do not
  hide the report card.
- Missing upstream region metadata: show an unavailable state for audio/tile
  playback. The backend should normally prevent this state.

## 10. Tests

### Backend

Add coverage to `tests/integration/test_sequence_models_api.py` or a focused
Sequence Models API test:

- Complete Event Encoder job with `event_tokens.parquet` returns timeline
  metadata, valid k values, selected k, and sorted event rows.
- `?k=` filters rows to the requested k.
- Invalid requested k returns `422`.
- Incomplete job returns `409`.
- Missing job returns `404`.
- Complete job with missing token artifact returns `404`.
- Response uses the artifact timestamps and does not reload current effective
  events.

### Frontend

Update `frontend/e2e/sequence-models/event-encoder.spec.ts`:

- Mock the new timeline endpoint for a complete job.
- Assert the timeline panel appears between the job summary and report panels.
- Assert token badges render with labels such as `T17`.
- Assert next/previous buttons update the selected event counter and highlight.
- Assert `D` and `A` keyboard navigation update the selection.
- Assert the k selector requests the new k and updates token badges.
- Assert a non-complete job shows a timeline unavailable state without hiding
  the report/error content.

Add a focused component test for `EventEncoderTokenOverlay` if the existing
frontend test setup supports it:

- Bars use overlay context positioning.
- The selected event gets the selected styling.
- Badge labels remain visible for multi-digit token labels.

Verification commands:

- `uv run pytest tests/integration/test_sequence_models_api.py -q`
- `uv run pytest tests/services/test_event_encoder_service.py tests/workers/test_event_encoder_worker.py -q`
- `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
- `cd frontend && npx tsc --noEmit`
- Final full backend gate remains `uv run pytest tests/`.

## 11. Risks And Follow-Ups

- Large Event Encoder jobs may return many timeline rows. The v1 endpoint
  returns compact fields for one k only; if this becomes heavy, add source
  sequence or time-window pagination later.
- Skipped events are not visible in the first timeline because there is no
  per-event skip artifact. If skipped-event review becomes important, add an
  explicit skip-details artifact in the worker and extend the endpoint.
- Token colors are deterministic within a selected k but not semantically
  stable across jobs. This matches current Event Encoder tokenization semantics,
  where tokens are job-local.
- The viewer should not share the editable Call Parsing overlay directly.
  Keeping a read-only token overlay avoids accidental correction affordances on
  a Sequence Models report page.
