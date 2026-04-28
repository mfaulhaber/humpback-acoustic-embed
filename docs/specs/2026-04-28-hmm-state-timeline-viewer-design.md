# HMM State Timeline Viewer — Design Spec

## Overview

Add an interactive timeline viewer panel to the HMM Sequence job detail page. The panel displays a PCEN spectrogram with an HMM state bar component directly below, allowing users to visually correlate Viterbi-decoded hidden states with the underlying audio spectrogram. Navigation is by merged span (prev/next), with zoom, pan, and audio playback.

## Goals

- Show HMM state sequences in temporal context alongside the spectrogram
- Reuse the shared timeline infrastructure without modifying it
- Provide audio playback so users can hear what each state transition sounds like
- Keep the existing Plotly-based State Timeline panel (shifted below the new panel)

## Non-Goals

- Click-to-filter (e.g., clicking a state to highlight it in other panels)
- Editing or correcting HMM state assignments
- New spectrogram tile generation (reuses existing region detection tiles)

---

## Page Layout

The new panel is inserted between the Job Config panel and the existing State Timeline panel. All existing panels shift down. Panel order (9 total):

1. Job Config / Details (existing)
2. **HMM State Timeline Viewer (new)**
3. State Timeline — Plotly chart (existing)
4. PCA / UMAP Overlay (existing)
5. Transition Matrix (existing)
6. Label Distribution (existing)
7. Dwell-Time Histograms (existing)
8. State Exemplars (existing)
9. State Summary Table (existing)

---

## Component Architecture

### Composition

```
<Card>
  <CardHeader>  "HMM State Timeline Viewer"
  <CardContent>
    <SpanNavBar />          // page-level, left-justified
    <TimelineProvider>      // shared, unchanged
      <Spectrogram />       // shared, unchanged — PCEN tiles from region detection job
      <HMMStateBar />       // NEW — consumes useTimelineContext()
      <ZoomSelector />      // shared, unchanged — centered
      <PlaybackControls />  // shared, unchanged — centered
    </TimelineProvider>
  </CardContent>
</Card>
```

### Shared components (zero modifications)

- `TimelineProvider` — zoom/pan/playback state management
- `Spectrogram` — PCEN tile rendering, frequency axis, time axis, drag-to-pan, playhead
- `ZoomSelector` — centered zoom preset buttons
- `PlaybackControls` — centered play/pause, skip, speed, +/- zoom

### New components

#### SpanNavBar

Location: `frontend/src/components/sequence-models/SpanNavBar.tsx`

A navigation bar rendered above the `TimelineProvider`, outside the shared timeline boundary.

Props:
- `spans: Array<{ id: number; startSec: number; endSec: number }>` — merged span list
- `activeIndex: number` — currently selected span index
- `onPrev: () => void`
- `onNext: () => void`

Layout: left-justified, matching the `ReviewToolbar` pattern from `SegmentReviewWorkspace`.
- `‹` `›` chevron buttons (disabled at boundaries)
- Label: `Span {idx+1}/{total} · {startTime} – {endTime}`
- Time formatting via `formatRecordingTime`

#### HMMStateBar

Location: `frontend/src/components/sequence-models/HMMStateBar.tsx`

A canvas-based visualization rendered between `Spectrogram` and `ZoomSelector` inside the `TimelineProvider`.

Props:
- `items: Array<{ start_timestamp: number; end_timestamp: number; viterbi_state: number; max_state_probability: number }>` — Viterbi windows for the active span
- `nStates: number` — total number of hidden states

Viewport sync:
- Consumes `useTimelineContext()` for `viewStart`, `viewEnd`, `pxPerSec`, `centerTimestamp`
- Uses `FREQ_AXIS_WIDTH_PX` constant for left offset alignment with the spectrogram canvas
- Uses `ResizeObserver` on container for responsive width

Rendering:
- Canvas-based for performance (spans can have thousands of 1-second windows)
- Fixed height: 60px
- Y-axis: state number (0 at bottom, n_states-1 at top), with a label strip on the left matching the frequency axis width
- Each Viterbi window renders as a colored horizontal bar at its state's Y position
- Bar colors from the `STATE_COLORS` palette (currently defined inline in `HMMSequenceDetailPage.tsx` — extract to a shared constant in `sequence-models/constants.ts` so both `HMMStateBar` and the page can import it)
- Playhead: vertical red line synced to `centerTimestamp`

Hover tooltip:
- Mouse X position converted to time, then binary-searched against the sorted window list
- Tooltip rendered as a positioned `<div>` (not canvas) showing: `State {n} · {startTime}–{endTime} · prob {value}`
- Tooltip follows the mouse horizontally, anchored vertically near the hovered state row

---

## Data Flow

### Existing data (no new fetches)

The HMM states data is already loaded by the page via `useHMMStates(jobId, 0, 5000)`. This returns all Viterbi windows with `merged_span_id`, `start_timestamp`, `end_timestamp`, `viterbi_state`, `max_state_probability`.

The page already computes `spanIds` (unique sorted `merged_span_id` values) and manages `selectedSpan` state. The new panel reuses this same data and state.

### Span-to-timeline mapping

Each merged span has a contiguous time range derived from the min `start_timestamp` and max `end_timestamp` of its windows. These become `jobStart` and `jobEnd` for the `TimelineProvider`.

When the user navigates to a different span:
1. `SpanNavBar` calls `onPrev`/`onNext`, updating the active span index
2. The `TimelineProvider` re-keys on the span ID, resetting viewport state
3. Spectrogram and HMMStateBar render data for the new time range

### Spectrogram tiles

Tiles are sourced from the region detection job via the existing `regionTileUrl(regionDetectionJobId, zoomLevel, tileIndex)` endpoint. Tiles are generated on demand by either consumer (Classify Review or this viewer).

### Audio playback

Slice-based playback via `regionAudioSliceUrl(regionDetectionJobId, startEpoch, durationSec)` — same source as Classify Review.

---

## TimelineProvider Configuration

| Prop | Value |
|------|-------|
| `key` | `hmm-timeline-${spanId}` |
| `jobStart` | Active span's min `start_timestamp` |
| `jobEnd` | Active span's max `end_timestamp` |
| `zoomLevels` | `REVIEW_ZOOM` (5m, 1m, 30s, 10s) |
| `defaultZoom` | Best-fit preset for span duration |
| `playback` | `"slice"` |
| `audioUrlBuilder` | `regionAudioSliceUrl(regionDetectionJobId, ...)` |

---

## Backend Change

### HMM job detail response

The `GET /sequence-models/hmm-sequences/{jobId}` response must include `region_detection_job_id`, `region_start_timestamp`, and `region_end_timestamp` so the frontend can build tile and audio URLs and validate epoch ranges.

Resolution path: `hmm_sequence_job.continuous_embedding_job_id` → `continuous_embedding_job.region_detection_job_id`.

Change:
- Add `region_detection_job_id: str`, `region_start_timestamp: float | None`, and `region_end_timestamp: float | None` to the Pydantic response schema (`HMMSequenceJobResponse` or `HMMSequenceJobDetail`)
- Populate them in the router by loading the parent CEJ and source Region Detection Job

No new endpoints. No database migration.

---

## Interaction Details

### Zoom and pan
- Zoom presets: 5m, 1m, 30s, 10s (shared `REVIEW_ZOOM`)
- Drag-to-pan on the spectrogram (handled by shared `Spectrogram`)
- Keyboard: `+`/`-` for zoom, arrow keys for pan

### Playback
- Play/pause via `PlaybackControls` button or Space key
- Skip forward/back buttons
- Speed cycling: 0.5x, 1x, 2x
- Playhead (red vertical line) visible in both spectrogram and HMMStateBar

### Span navigation
- Prev/Next buttons in `SpanNavBar`
- Switching spans resets the entire viewport (new `TimelineProvider` key)

### State bar hover
- Hovering over the HMMStateBar shows a tooltip with state number, time range, and probability
- No click behavior

---

## File Changes

### New files
- `frontend/src/components/sequence-models/HMMStateBar.tsx`
- `frontend/src/components/sequence-models/SpanNavBar.tsx`

### Modified files
- `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx` — add new panel
- `frontend/src/api/sequenceModels.ts` — add source region metadata to response type
- `src/humpback/schemas/sequence_models.py` — add source region metadata to response schema
- `src/humpback/api/routers/sequence_models.py` — resolve and return source region metadata

### Not modified
- Any shared timeline component (`TimelineProvider`, `Spectrogram`, `ZoomSelector`, `PlaybackControls`, `TimelineFooter`, `TimelineHeader`)
- Any existing consumer of shared timeline components
- Database schema (no migration)

---

## Testing

### Unit tests
- `HMMStateBar`: viewport-to-canvas coordinate mapping, hover hit detection (binary search), edge cases (empty items, single window)
- `SpanNavBar`: prev/next disabled states at boundaries, label formatting, click callbacks

### Playwright
- Load the HMM Sequence detail page with a completed job
- Verify the new panel renders with spectrogram and state bar
- Navigate between spans via prev/next
- Verify zoom presets change the viewport
