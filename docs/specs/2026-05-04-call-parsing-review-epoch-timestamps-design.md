# Call Parsing Review Workspaces — Epoch Timestamp Alignment

## Overview

Commit `5166ed7` ("Align Sequence Models timestamps to epoch") changed the
`/call-parsing/region-jobs/{id}/audio-slice` endpoint to accept
`start_timestamp` as an absolute UTC epoch and to validate strictly against
the source region job's absolute `[start_timestamp, end_timestamp]` window.
The HMM and Masked Transformer detail pages were updated to send absolute
epoch timestamps. The three call-parsing review workspaces —
`SegmentReviewWorkspace`, `ClassifyReviewWorkspace`, and
`WindowClassifyReviewWorkspace` — were not updated. They still mount
`<TimelineProvider jobStart={0}>` and call `playbackRef.current?.play(startSec, …)`
with job-relative seconds. Every audio-slice request from those pages now
sends `start_timestamp=<small relative number>` against absolute job bounds
and returns `400`.

This spec aligns those three workspaces with the same epoch coordinate
convention HMM and MT pages use, fixes the broken audio playback, and adds
two small shared frontend utilities so the wiring is not duplicated three
times.

## Goals

- Fix `400` from audio-slice on Segment, Classify, and Window Classify
  review pages by sending absolute `start_timestamp` values that fall
  inside the region job's `[start_timestamp, end_timestamp]` window.
- Run the timeline coordinate space inside these three workspaces in
  absolute UTC epoch seconds, matching `HMMSequenceDetailPage` and
  `MaskedTransformerDetailPage`.
- Consolidate the conversion seam into one hook + one wrapper component
  used by all three workspaces, instead of repeating the relative→epoch
  arithmetic at every play, recenter, and overlay site.
- Render real UTC times in the workspace headers, region tables, event
  detail panels, and review toolbars (replacing the current
  `00:00:00`-anchored output that came from `jobStartEpoch=0`).
- Keep workspace state, correction payloads, and the artifact storage
  contract on job-relative `start_sec` / `end_sec`.

## Non-Goals

- No changes to `regions.parquet`, `events.parquet`, `typed_events.parquet`,
  or `trace.parquet` schemas.
- No changes to `region_boundary_corrections`, `event_boundary_corrections`,
  or `vocalization_corrections` DB columns or REST payloads. Correction
  identity remains keyed on relative `start_sec` / `end_sec`.
- No changes to `TimelineProvider`, `usePlayback`, `OverlayContext`,
  `TileCanvas`, or any overlay component. The shared timeline primitive
  is not modified by this spec.
- No changes to `regionAudioSliceUrl`, `regionTileUrl`, or any backend
  endpoint, schema, worker, or migration.
- No migration script, no data backfill.
- No changes to `HMMSequenceDetailPage`, `MaskedTransformerDetailPage`, or
  `RegionDetectionTimeline`. They already send absolute epoch and are out
  of scope.
- No refactor of `RegionEditOverlay` / `RegionOverlay` to drop the
  `jobStart` prop in favor of `useTimelineContext()` (pre-existing wart;
  workspaces continue to pass the right value into it).

## Current Bug

For an audio-slice request issued from
`SegmentReviewWorkspace` with `selectedEvent.startSec = 7916.1` and a region
job whose absolute bounds are `[1635638400, 1635724800]`:

- The workspace sets `jobStart={0}` on `TimelineProvider`.
- The provider's playback handle is invoked with `play(7916.1, dur)`.
- `audioUrlBuilder` calls
  `regionAudioSliceUrl(jobId, 7916.1, dur)` →
  `…/audio-slice?start_timestamp=7916.1&duration_sec=…`.
- The backend rejects the request because `7916.1` is outside
  `[1635638400, 1635724800]`.

The same mismatch produces `400` from `ClassifyReviewWorkspace` and
`WindowClassifyReviewWorkspace`.

## Decisions

### Coordinate space inside review workspaces

Run the timeline coordinate space in absolute UTC epoch seconds inside
the three review workspaces. `TimelineProvider.jobStart` becomes
`region.start_timestamp` and `jobEnd` becomes `region.end_timestamp`,
matching HMM/MT pages.

### Workspace state stays relative

Workspace-local state, correction payloads, viewer body data, and the
artifact-storage contract continue to use job-relative
`start_sec` / `end_sec`. Only the timeline-coordinate handoff (provider
props, `playbackRef.play(...)`, props plumbed into overlays/panels that
operate in epoch space) is converted at the seam.

### Conversion seam: one hook, one wrapper

All three workspaces use:

1. A `useRegionEpoch(regionDetectionJobId)` hook that returns
   `{regionStartTimestamp, regionEndTimestamp, toEpoch}` derived from the
   `useRegionDetectionJobs` query, or `null` while loading or when the
   region job has no absolute timestamps.
2. A `RegionAudioTimeline` `forwardRef` wrapper that mounts
   `<TimelineProvider>` with epoch `jobStart`/`jobEnd`, configures
   `audioUrlBuilder` against `regionAudioSliceUrl`, and forwards the
   `TimelinePlaybackHandle` ref. Other `TimelineProvider` props
   (`zoomLevels`, `defaultZoom`, `disableKeyboardShortcuts`,
   `scrollOnPlayback`, `onZoomChange`, `onPlayStateChange`) are
   passthrough.

Both files live under `frontend/src/components/call-parsing/`. Neither is
exported beyond that directory. The shared `TimelineProvider` is not
modified; the wrapper is a call-parsing-specific composition layered on
top.

### Why not also change Sequence Models

HMM and MT detail pages already do the right thing, but via a different
data path: the HMM/MT job detail response carries
`region_start_timestamp` and `region_end_timestamp` fields, and each page
mounts `<TimelineProvider>` inline in a panel-specific component. The
review-workspace job rows (`EventSegmentationJob`,
`EventClassificationJob`, `WindowClassificationJob`) already FK directly
to `region_detection_job_id`, and the frontend already caches
`useRegionDetectionJobs()`. A frontend join is strictly cheaper than
adding redundant fields to three more backend job schemas. Sequence
Models stays as-is — touching it risks regressing two stable feature
areas for no functional gain.

### Recentering math

Each workspace has viewport recentering logic that operates in the
timeline coordinate space (`viewStart`, `viewSpan`). When `jobStart`
switches from `0` to `regionStartTimestamp`, the recentering targets and
comparisons must add `regionStartTimestamp` exactly once:

- Target: `regionEpoch.toEpoch(currentNavEvent.startSec - pad + viewSpan / 2)`.
- Comparison: `regionEpoch.toEpoch(currentNavEvent.startSec) >= viewStart + pad`.

`SegmentReviewWorkspace` has three such sites; `ClassifyReviewWorkspace`
has one.

### Overlay `jobStart` prop

`RegionEditOverlay` and `RegionOverlay` accept `jobStart` as a prop today
(pre-existing wart that conflicts with the §10.10 "children read context"
guidance, but unrelated to this fix). Workspaces pass
`regionEpoch.regionStartTimestamp` into that prop instead of `0`. No
changes to the overlay components themselves.

### `formatRecordingTime` callers

`formatRecordingTime(offsetSec, jobStartEpoch)` keeps its existing
signature. Workspaces pass `regionEpoch.regionStartTimestamp` for the
second argument instead of `0`. The presentational components
(`ReviewToolbar`, `EventDetailPanel`, `ClassifyDetailPanel`,
`RegionTable`, the inline event-row block in `WindowClassify`) keep
their existing `jobStartEpoch: number` prop — the workspace decides what
to pass.

### Loading gate

When `useRegionEpoch` returns `null` because the region job has not
hydrated yet, the workspace renders a loading placeholder consistent
with the existing "Select a region" empty-state pattern. The
`<RegionAudioTimeline>` is not mounted until `regionEpoch != null`.

## Surface Audit

| Surface | Current | Target |
|---|---|---|
| `<TimelineProvider jobStart>` in three review workspaces | `0` | `regionEpoch.regionStartTimestamp` (via `<RegionAudioTimeline>`) |
| `<TimelineProvider jobEnd>` in three review workspaces | `timelineExtent.end` (or job duration) | `regionEpoch.regionEndTimestamp` (via wrapper) |
| `playbackRef.current?.play(startSec, dur)` sites | relative seconds | `regionEpoch.toEpoch(startSec)` |
| Recenter targets / comparisons in `SegmentReviewWorkspace` and `ClassifyReviewWorkspace` | relative seconds | `regionEpoch.toEpoch(...)` |
| Overlay `jobStart` props on `RegionEditOverlay` / `RegionOverlay` | `0` (currently agrees with provider) | `regionEpoch.regionStartTimestamp` |
| `jobStartEpoch` prop passed to `ReviewToolbar`, `EventDetailPanel`, `ClassifyDetailPanel`, `RegionTable`, inline event-row block | `0` | `regionEpoch.regionStartTimestamp` |
| `audioUrlBuilder` definition in each workspace | inline `useCallback` | absorbed into `<RegionAudioTimeline>` |
| `tileUrlBuilder` definition in each workspace | inline `useCallback` | unchanged (left at workspace level; not part of this fix) |
| Workspace state (`startSec`, `endSec`) | relative | unchanged |
| Correction payload fields (`corrected_start_sec`, etc.) | relative | unchanged |
| `regions.parquet`, `events.parquet`, `typed_events.parquet` | relative | unchanged |
| Backend audio-slice / tile / job-detail endpoints | absolute / unchanged | unchanged |
| HMM / MT detail pages | absolute via `region_start_timestamp` field | unchanged |
| `RegionDetectionTimeline` | already correct | unchanged |

## Implementation Sketch

### New files

1. `frontend/src/components/call-parsing/useRegionEpoch.ts` — hook that
   reads from `useRegionDetectionJobs` and returns
   `RegionEpoch | null`. Exposes `regionStartTimestamp`,
   `regionEndTimestamp`, and `toEpoch(relativeSec) =>
   regionStartTimestamp + relativeSec`.

2. `frontend/src/components/call-parsing/RegionAudioTimeline.tsx` —
   `forwardRef` wrapper around `TimelineProvider`. Takes
   `regionDetectionJobId`, `regionEpoch`, `resetKey`, plus passthrough
   props (`zoomLevels`, `defaultZoom`, `disableKeyboardShortcuts`,
   `scrollOnPlayback`, `onZoomChange`, `onPlayStateChange`, `children`).
   Configures `playback="slice"` and
   `audioUrlBuilder = (startEpoch, dur) => regionAudioSliceUrl(jobId, startEpoch, dur)`.
   Forwards the `TimelinePlaybackHandle` ref.

### Modified files

3. `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`
   - Add `useRegionEpoch(regionDetectionJobId)`.
   - Extend the render gate to require `regionEpoch != null`.
   - Replace inline `<TimelineProvider>` with `<RegionAudioTimeline>`.
   - Convert all `playbackRef.current?.play(...)` sites to use
     `regionEpoch.toEpoch(...)`.
   - Convert recentering targets and comparisons (3 sites).
   - Pass `regionEpoch.regionStartTimestamp` into all `jobStartEpoch`
     props on `ReviewToolbar`, `EventDetailPanel`, `RegionTable` (two
     instances).
   - Pass `regionEpoch.regionStartTimestamp` to overlay `jobStart` props
     inside `SegmentViewerBody`.
   - Drop the `jobStart = 0` constant.

4. `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`
   - Same pattern. Replace `<TimelineProvider jobStart={0}>` with
     `<RegionAudioTimeline>`. Convert one `play` site, one recenter
     site, the overlay `jobStart` prop, and the
     `<ClassifyDetailPanel jobStartEpoch={0}>` site.

5. `frontend/src/components/call-parsing/WindowClassifyReviewWorkspace.tsx`
   - Same pattern. Convert two `play` sites
     (`selectedEvent.start_sec`, `selectedAddedEvent.startSec`), the
     `jobStart={0}` overlay prop site, and two `jobStartEpoch={0}`
     prop sites (the inline event-row block uses
     `formatRecordingTime(_, jobStartEpoch)` directly).

### Component prop signatures (no changes)

Presentational subcomponents keep their existing signatures:

- `ReviewToolbar({ ..., jobStartEpoch: number })`
- `EventDetailPanel({ ..., jobStartEpoch: number })`
- `ClassifyDetailPanel({ ..., jobStartEpoch: number })`
- `RegionTable({ ..., jobStartEpoch: number })`

Workspace decides what to pass; the components themselves are unchanged.

## Testing

### Shared fixture

`frontend/src/components/call-parsing/__test-helpers__/regionEpoch.ts`
exports `REGION_EPOCH_BASE = 1635638400` (`2021-10-31T00:00:00Z`) for
all three workspace tests and any new unit tests.

### New unit tests

- `useRegionEpoch.test.tsx`
  - Returns `null` when `regionDetectionJobId` is `null`.
  - Returns `null` when no matching job is in cache.
  - Returns `null` when the matching job has `start_timestamp == null`
    or `end_timestamp == null`.
  - Returns `{regionStartTimestamp, regionEndTimestamp, toEpoch}` when
    fully populated. `toEpoch(0) === REGION_EPOCH_BASE`;
    `toEpoch(7916.1) === REGION_EPOCH_BASE + 7916.1`.

- `RegionAudioTimeline.test.tsx`
  - Renders the inner `TimelineProvider` with epoch `jobStart` and
    `jobEnd`.
  - The `audioUrlBuilder` produces an absolute `start_timestamp` query
    param.
  - The forwarded `TimelinePlaybackHandle` ref is reachable from the
    outer ref.

### Extended workspace tests

- `ClassifyReviewWorkspace.test.ts` (existing) — add a region-job
  fixture with `start_timestamp = REGION_EPOCH_BASE` and
  `end_timestamp = REGION_EPOCH_BASE + 86400`. Add cases:
  - Selecting an event with `startSec = 7916.1` and clicking Play
    captures `playbackRef.play(REGION_EPOCH_BASE + 7916.1, …)`.
  - Existing correction round-trip assertions pass unchanged
    (`corrected_start_sec` in POST body remains relative).
  - `formatRecordingTime` rendered text matches the absolute UTC
    time-of-day for the selected event (not the
    epoch-zero-anchored value).

- `SegmentReviewWorkspace.test.ts` (new) — mirror of the above plus
  recenter math: when an event navigation step recenters the viewport,
  the captured `setScrollToCenter` target equals
  `regionEpoch.toEpoch(...)` of the relative target value.

- `WindowClassifyReviewWorkspace.test.ts` (new) — mirror plus the
  `selectedAddedEvent.startSec` play path and the inline event-row
  block's UTC time-of-day rendering.

### Playwright (conditional)

Audit `frontend/e2e/call-parsing/*.spec.ts`. For specs that already
mock `/call-parsing/region-jobs` and exercise an audio-slice request in
one of the three workspaces:

- Set `start_timestamp` and `end_timestamp` on the region-job fixture
  to a non-zero base.
- Assert each captured audio-slice request URL contains
  `start_timestamp=<value within [REGION_EPOCH_BASE, REGION_EPOCH_BASE + 86400]>`.

If no specs currently exercise the audio-slice path in these
workspaces, do not add new ones in this PR.

### Manual browser verification

1. Open a Segment review page for a job whose region has
   `start_timestamp = 1635638400`. Verify spectrogram tiles render
   for the actual recording window, the timeline header shows real
   UTC times, and clicking Play on an event returns `200` with audio
   that matches the visible spectrogram region.
2. Same for Classify review.
3. Same for Window Classify review.
4. Save a correction in Segment review. Confirm the network payload
   contains relative `start_sec` / `end_sec` (e.g. `7920.0`), not
   absolute epoch.

## Verification Gates

In order, after all tasks:

1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx vitest run src/components/call-parsing/`
3. `cd frontend && npx playwright test call-parsing/` (only if E2E
   specs were updated)
4. `uv run pytest tests/`
5. Manual browser verification per above.

## Risks

- **`RegionDetectionJob.start_timestamp` is nullable.** The hook returns
  `null` and the workspace renders a loading placeholder. No user-visible
  regression vs today's behavior, which silently used `jobStart=0`.
- **Recentering math has multiple sites.** Section "Decisions →
  Recentering math" enumerates them; tests assert the captured target.
- **Overlay `jobStart` prop is easy to leave at `0`** when replacing the
  inline `TimelineProvider` mount with the wrapper. Once `jobStart`
  switches to a real epoch, overlays drawn with `jobStart=0` would
  appear off-screen — visual/unit tests catch this immediately.
- **`WindowClassifyReviewWorkspace` did not previously call
  `useRegionDetectionJobs`.** Adding it via the hook is deduped by
  TanStack Query against the sibling components that already cache it,
  so there is no extra request waterfall.
- **Sequence Models continues to use a different data path** for the
  same problem. Acknowledged divergence; future unification would
  require backend changes for review-workspace job detail responses and
  is not justified here.

## Rollout

1. Land the two new utilities with their unit tests.
2. Migrate the three workspaces, one workspace per commit.
3. Run the verification gates and manual browser verification.
4. Open PR.
