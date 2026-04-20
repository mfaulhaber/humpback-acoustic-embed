# Region Detection Timeline Viewer

Read-only timeline viewer for Call Parsing Detection (Pass 1) jobs, showing generated regions overlaid on a PCEN spectrogram with trace score heatmap and audio playback.

## Context

Region detection jobs produce two artifacts: `trace.parquet` (dense per-window classifier scores) and `regions.parquet` (padded whale-active regions merged via hysteresis). The existing `RegionSpectrogramViewer` shows a single region's spectrogram, but there is no job-level view of all regions in context. The existing classifier detection `TimelineViewer` has the right infrastructure (tiled spectrograms, heatmap strip, overlays, playback) but is tightly coupled to classifier detection jobs.

## Approach

Adapt the existing `TimelineViewer` / `SpectrogramViewport` infrastructure with a read-only region mode. Reuse tile rendering, pan/zoom, playback, and heatmap strip. Add a new `RegionOverlay` for full-height shaded regions. Fix the existing confidence heatmap alignment bug (benefits both viewers).

## Design

### Routing & Entry Point

- New route: `/app/call-parsing/region-timeline/:jobId`
- New component: `RegionTimelineViewer.tsx` in `frontend/src/components/call-parsing/`
- Thin wrapper that fetches the region job, its regions, and trace-based confidence scores, then renders `SpectrogramViewport` in read-only region mode
- Enable the currently disabled "Timeline" button in `RegionJobTable` to navigate here
- Header shows job metadata (hydrophone, time range, region count) plus zoom and playback controls — no overlay toggles or label mode buttons

### RegionOverlay Component

- New `RegionOverlay.tsx` in `frontend/src/components/timeline/`, sibling to `DetectionOverlay.tsx`
- Regions converted from job-relative `start_sec`/`end_sec` to absolute epoch seconds by adding `jobStart`
- Rendered as full-height semi-transparent rectangles on the spectrogram canvas
- Single color with opacity derived from `max_score` (higher confidence = more visible)
- No click handling, no labels, no hover tooltips — purely visual
- `SpectrogramViewport` gets a new `overlayMode` value `"region"` that renders `RegionOverlay` instead of `DetectionOverlay`

### Heatmap Bug Fix

The existing confidence heatmap strip in `SpectrogramViewport` has an alignment bug with two causes:

1. **Backend** (`src/humpback/api/routers/timeline.py`): bucket count uses `int(job_duration / window_sec)` (floor division), dropping detections in the final partial window. Fix: use ceiling division so all data is captured.
2. **Frontend** (`SpectrogramViewport.tsx`): recalculates `windowSec = totalDuration / scores.length` instead of using the backend's returned `window_sec`. Fix: use the backend's `window_sec` value for bar positioning.

This fix applies to the existing classifier detection timeline heatmap as well.

### Trace Score Heatmap

- New endpoint: `GET /call-parsing/region-jobs/{job_id}/confidence` — reads `trace.parquet`, buckets raw `(time_sec, score)` pairs into the same response format the existing heatmap expects (array of scores + `window_sec`)
- New hook: `useRegionJobConfidence(jobId)` in `useCallParsing.ts`
- `SpectrogramViewport` renders the same (fixed) heatmap strip for region jobs using trace-derived scores

### URL Abstraction

`SpectrogramViewport` currently hardcodes classifier detection tile and audio URLs. Refactor to accept `tileUrlBuilder` and `audioSliceUrlBuilder` props (functions that produce URLs given zoom/tile/start/duration parameters). The existing `TimelineViewer` passes classifier detection URL builders; `RegionTimelineViewer` passes region job URL builders (`regionJobTileUrl`, `regionJobAudioSliceUrl`).

### Read-Only Gating

`RegionTimelineViewer` passes: `labelMode=false`, `labelEditMode=null`, no `renderLabelEditor` or `renderVocLabelEditor`, `overlayMode="region"`. No labeling, vocalization, or detection overlay code paths are exercised.

### Playback

Reuses the existing `SpectrogramViewport` audio playback infrastructure (double-buffered audio elements, gapless prefetch). Audio fetched via the region job audio-slice endpoint through the `audioSliceUrlBuilder` prop. Note: the region job audio-slice endpoint uses job-relative seconds (`start_sec` = 0 at job start), so the `audioSliceUrlBuilder` must subtract `jobStart` from absolute epoch timestamps before building the URL.

## Files Changed

| File | Change |
|------|--------|
| `frontend/src/App.tsx` | Add route `/app/call-parsing/region-timeline/:jobId` |
| `frontend/src/components/call-parsing/RegionTimelineViewer.tsx` | **New** — wrapper component |
| `frontend/src/components/call-parsing/RegionJobTable.tsx` | Enable "Timeline" button |
| `frontend/src/components/timeline/RegionOverlay.tsx` | **New** — full-height region shading |
| `frontend/src/components/timeline/SpectrogramViewport.tsx` | Accept `tileUrlBuilder`, `audioSliceUrlBuilder`, `overlayMode="region"`; fix heatmap to use backend `window_sec` |
| `frontend/src/components/timeline/TimelineViewer.tsx` | Pass URL builders to `SpectrogramViewport` (no behavioral change) |
| `frontend/src/hooks/queries/useCallParsing.ts` | Add `useRegionJobConfidence` hook |
| `frontend/src/api/client.ts` | Add `fetchRegionJobConfidence` |
| `frontend/src/api/types.ts` | Add confidence response type if needed |
| `src/humpback/api/routers/call_parsing.py` | Add `GET /region-jobs/{job_id}/confidence` endpoint |
| `src/humpback/api/routers/timeline.py` | Fix bucket count (ceiling division), return effective `window_sec` |

## Not In Scope

- Label editing or vocalization overlays in this viewer
- Click-to-navigate on regions or heatmap
- Region list/table sidebar
- Changes to the existing `RegionSpectrogramViewer` (single-region viewer)
