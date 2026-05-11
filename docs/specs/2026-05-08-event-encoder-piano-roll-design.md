# Event Encoder Token Frequency Piano Roll — Design

**Date:** 2026-05-08
**Status:** Approved
**Primary domain:** Sequence Models
**Neighbor domains:** Signal Timeline, Frontend Shell

## 1. Goal

Add a dedicated full-viewport Token Frequency Piano Roll viewer as a new route
for Event Encoder jobs. The viewer renders each tokenized event as a rectangle
on a time × frequency canvas, encoding tonal range, dominant frequency, ridge
slope direction, and token identity simultaneously. This gives researchers a
single view that surfaces spacing patterns between events, repeating token
sequences, precise timing intervals within token families, and acoustic
character differences across tokens.

The existing Event Encoder detail page and its timeline panel are unchanged.
The piano roll is a separate, dedicated page optimized for full-screen
exploration of token–frequency patterns.

## 2. Motivation

Anecdotal review of recent Event Encoder job runs (k=50, 1273 events, 88
minutes of audio) shows:

- Token families align well with vocalization types: tonal sweeps (T00, T01,
  T18), steady tones (T09, T11, T15), pulsed units (T15, T19, T20), broadband
  noise (T13, T14, T17).
- Repeating n-gram motifs appear in token sequences (e.g. T00→T00→T00 4×,
  T13→T25→T14 3×, T24→T01→T22→T38 2×).
- Same-token events burst in tight clusters (0.6–1.5 s apart).
- Dense phrase regions (~0.3 events/s) alternate with long silences (15–110 s).

The existing timeline viewer shows tokens as colored bars on a spectrogram. It
is good for verifying individual events against the spectrogram but does not
expose the frequency structure or slope character of events at a glance.

A frequency piano roll, where Y encodes pitch and rectangle height encodes
tonal range, makes all four patterns visible without requiring the user to
inspect events one at a time.

## 3. Scope

### In scope

- New frontend route `/app/sequence-models/event-encoder/:jobId/piano-roll`.
- Full-viewport canvas-based renderer with pan/zoom on the time axis.
- Shift+scroll zoom on the frequency axis.
- Event rectangles positioned by median F0 (Y center) with height proportional
  to F0 range, width proportional to duration, colored by token family.
- Unvoiced events (voicing fraction ≤ 0.3) rendered with solid border and
  light fill at peak frequency position, visually distinct from voiced events.
- Ridge log-frequency slope rendered as a directional line inside each event
  rectangle when zoomed in, with arrowhead showing sweep direction.
- Token label rendered centered inside each rectangle with opaque background
  pill when zoomed in.
- Minimap for event-range overview and click-to-navigate.
- Token legend panel with click-to-filter (dims non-matching events).
- Event selection via click with tooltip showing all descriptor values.
- K-value selector reusing the timeline endpoint's `valid_k_values`.
- Audio playback for selected events using the same `regionAudioSliceUrl`
  builder as the existing timeline panel.
- Playback controls: play/pause button, Space keyboard shortcut, bounded to
  selected event duration.
- Link from the existing Event Encoder detail page to the piano roll route.
- Back-navigation link to the detail page from the piano roll.

### Non-goals

- No backend API changes. The existing timeline endpoint provides all required
  data including `descriptor_values` per event.
- No spectrogram rendering in this view. The piano roll replaces the
  spectrogram with a frequency-axis canvas.
- No changes to Event Encoder artifacts or tokenization output.
- No self-similarity matrix, inter-onset interval histogram, or transition
  diagram (future features).
- No editing of event tokens or labels.

## 4. Data Source

The piano roll reuses the existing timeline endpoint:

```
GET /sequence-models/event-encoders/{job_id}/timeline?k={k}
```

Response fields used per event:

| Field | Visual encoding |
|---|---|
| `start_timestamp`, `end_timestamp` | X position, rectangle width |
| `descriptor_values.median_f0` | Y center (voiced events) |
| `descriptor_values.f0_range` | Rectangle height (tonal range) |
| `descriptor_values.peak_frequency` | Y center (unvoiced events, Y-mode fallback) |
| `descriptor_values.voicing_fraction` | Voiced/unvoiced visual treatment, fill opacity |
| `descriptor_values.ridge_log_frequency_slope` | Slope line direction and arrowhead |
| `token_id`, `token_label` | Color, centered label text |
| `token_confidence` | Tooltip display |

Response fields used for setup:

| Field | Usage |
|---|---|
| `region_detection_job_id` | Audio slice URL builder for playback |
| `job_start_timestamp`, `job_end_timestamp` | Recording bounds for audio clamping and relative time labels |
| `valid_k_values`, `selected_k` | K-value selector |
| `descriptor_feature_names`, `descriptor_feature_units` | Tooltip descriptor listing |

No additional backend endpoint is required.

The piano roll time domain is the first event start through the last event end,
with 30 seconds of buffer on both sides. If the timeline has no events, it
falls back to the recording bounds from `job_start_timestamp` and
`job_end_timestamp`.

## 5. Visual Encoding

### 5.1 Event Rectangles

Each event renders as a rectangle on the time × frequency canvas:

- **X** = `start_timestamp` mapped to viewport. **Width** = event duration.
  Minimum rendered width of 2 px to keep short events visible.
- **Y center** = `median_f0` when voiced (voicing fraction > 0.3), or
  `peak_frequency` otherwise. User can toggle Y mode to always use peak
  frequency.
- **Height** = `f0_range` mapped to the frequency axis. Minimum rendered
  height of 4 px. Unvoiced events with zero F0 range get the 4 px minimum.
- **Fill color** = token family color from `labelColor(token_id, k)`, using
  the same categorical palette and HSL ramp as the existing token overlay.
- **Fill opacity** = scaled by voicing fraction for voiced events
  (0.55–0.90). Unvoiced events use 0.35 fill opacity with full-opacity solid
  border and 0.15 interior fill.
- **Border** = same token color at higher opacity.

### 5.2 Ridge Slope Line

When the rectangle is at least 12 px wide and 6 px tall:

- A white line is drawn inside the rectangle from left edge to right edge.
- The line tilts according to `ridge_log_frequency_slope`: positive slope
  tilts upward left-to-right (rising pitch), negative tilts downward (falling
  pitch), zero is horizontal.
- Visual deflection is clamped: a slope magnitude of ~4 uses the full
  available deflection (40% of rectangle height in each direction).
- When the rectangle is at least 30 px wide and the slope magnitude exceeds
  0.3, an arrowhead renders at the right end of the line.
- The arrowhead is drawn as a connected V-shape with `lineJoin: round`.

### 5.3 Token Label

When the rectangle is at least 20 px wide and 10 px tall:

- The token label (e.g. `T18`) renders centered inside the rectangle.
- A small opaque background pill in the token's color sits behind the label
  text to keep it legible over the slope line.
- White text, 9 px monospace font.

### 5.4 Token Legend Filtering

- A collapsible legend panel lists all token families with color swatch,
  label, occurrence count, and mean F0 (or "noise" for unvoiced families).
- Clicking a token in the legend dims all other events (alpha reduced to
  ~0.12). Clicking the same token again clears the filter.
- Clicking an event on the canvas selects it (white border highlight) but does
  not activate token filtering.
- Escape clears both selection and filter.

### 5.5 Unvoiced Display Modes

A dropdown selector offers three modes for unvoiced events:

- **At peak frequency** (default): render at `peak_frequency` Y position with
  solid border and light fill.
- **Bottom lane**: collapse all unvoiced events into a fixed-height lane at
  the bottom of the canvas, separated by a dashed line.
- **Hide**: do not render unvoiced events.

### 5.6 Frequency Y-Mode

A dropdown selector toggles between:

- **Median F0** (default): Y center from `median_f0` for voiced events.
- **Peak Frequency**: Y center from `peak_frequency` for all events.

## 6. Interaction

### 6.1 Pan and Zoom

- **Scroll** (no modifier): zoom the time axis, centered on cursor.
- **Shift+Scroll**: zoom the frequency axis, centered on cursor.
- **Drag**: pan the time axis only (no frequency panning).
- Cursor affordance is open hand over the canvas, closed hand while dragging,
  and pointer when hovering an event rectangle.
- **F key**: fit all events in view (reset time bounds to the event range plus
  the 30 second buffer).
- Time and frequency zoom have minimum and maximum bounds to prevent
  degenerate views.

### 6.2 Selection and Tooltip

- **Click** an event: select it (white border). Shows a tooltip with all
  descriptor values, token label, confidence, gap from previous, position.
- **Click** outside an event: clear the selected event while preserving the
  visible playhead position.
- **Hover**: light border highlight, tooltip follows cursor.
- Tooltip placement flips horizontally/vertically near viewport edges so the
  detail window remains visible.
- **Double-click**: zoom to fit the clicked event with padding.
- **Escape**: clear selection and token filter.

### 6.3 Minimap

- A small canvas (240 × 40 px) in the bottom-right corner shows all events as
  2 px dots at their time × frequency position, colored by token.
- A white rectangle outline shows the current viewport bounds.
- Clicking the minimap centers the viewport on the clicked time position.

### 6.4 Audio Playback

- A play/pause button in the toolbar and the **Space** keyboard shortcut
  control audio playback.
- A vertical playback head tracks the absolute timestamp currently being
  heard.
- The playback head is always visible as a thin center line when audio is idle.
- Playback uses the shared timeline `usePlayback` hook with the same
  `regionAudioSliceUrl` builder as `EventEncoderTimelinePanel`.
- When an event is selected, playback uses slice mode and is bounded to the
  event's start timestamp and duration.
- When no event is selected, playback starts from the current visible playhead
  position and uses gapless 300-second audio windows so it can continue beyond
  the initially requested slice.
- Playback centers the time viewport on the playback head as audio advances,
  keeping the head fixed in the middle of the piano roll except at hard
  recording boundaries. The centered view may show blank space before the
  first recording timestamp or after the last one, matching the timeline
  provider's center-timestamp model.

### 6.5 Keyboard Shortcuts

| Key | Action |
|---|---|
| Space | Play/pause selected event or viewport |
| A | Select previous event and center it |
| D | Select next event and center it |
| Escape | Clear selection and token filter |
| F | Fit all events in view |
| + / = | Zoom in (time axis) |
| - | Zoom out (time axis) |
| ArrowLeft | Pan left (10% of viewport) |
| ArrowRight | Pan right (10% of viewport) |

Shortcuts are suppressed when focus is inside `input`, `textarea`, `select`,
or a contenteditable element.

## 7. Page Layout

The piano roll is a full-viewport page with three bands:

1. **Top toolbar** (~40 px): back link, job ID, event/duration/token stats,
   Y-mode selector, frequency max selector, unvoiced mode selector,
   play/pause button, k-value selector.
2. **Canvas area** (remaining height minus toolbar and status bar): the piano
   roll rendering surface. Contains the minimap overlay (bottom-right) and
   the collapsible token legend (top-right).
3. **Status bar** (~28 px): cursor time, cursor frequency, current zoom span,
   keyboard shortcut hints.

## 8. Routing and Navigation

- New route: `/app/sequence-models/event-encoder/:jobId/piano-roll`
- The existing Event Encoder detail page gains a link/button to navigate to
  the piano roll route for the current job.
- The piano roll toolbar includes a back link to the detail page.
- The route uses the same `useEventEncoderJob` hook to validate the job
  exists and is complete before rendering.

## 9. Canvas Rendering

Use an HTML5 Canvas element (not SVG) for the main visualization. At k=50
with 1273 events, SVG would be acceptable, but Canvas scales better for
larger jobs and avoids DOM overhead during pan/zoom.

- Use `devicePixelRatio` scaling for crisp rendering on Retina displays.
- Frustum-cull events outside the visible viewport before drawing.
- Draw order: grid lines → event rectangles → slope lines → labels → selection
  highlight → minimap.
- The minimap uses a separate canvas element.

## 10. Reuse from Existing Components

- `labelColor()` from `sequence-models/constants.ts` for token coloring.
- `useEventEncoderTimeline()` hook for data fetching.
- `useEventEncoderJob()` hook for job validation.
- `regionAudioSliceUrl()` from `api/client.ts` for audio playback URLs.
- Audio playback pattern from `EventEncoderTimelinePanel` (HTML audio element,
  play with start offset and bounded duration).
- The piano roll does NOT use `TimelineProvider` or `Spectrogram` components
  since it manages its own canvas, pan/zoom, and coordinate system.

## 11. State Management

All state is local to the piano roll component:

| State | Type | Initial |
|---|---|---|
| `selectedK` | `number \| null` | From response `selected_k` |
| `selectedEventIndex` | `number \| null` | `null` |
| `highlightToken` | `number \| null` | `null` |
| `viewStartTime` | `number` | `first_event.start_timestamp - 30` |
| `viewEndTime` | `number` | `last_event.end_timestamp + 30` |
| `freqMin` | `number` | `0` |
| `freqMax` | `number` | `2000` |
| `yMode` | `"f0" \| "peak"` | `"f0"` |
| `unvoicedMode` | `"peak" \| "bottom" \| "hide"` | `"peak"` |
| `isPlaying` | `boolean` | `false` |

## 12. Tests

### Frontend component tests

- Canvas renders without error when timeline data is loaded.
- K-value selector changes refetch timeline data.
- Keyboard shortcuts are suppressed when focus is in a text input.

### Playwright coverage

Add `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts` with
mocked timeline data:

- Route renders and shows the toolbar with job stats.
- K-value selector is present when multiple k values exist.
- Back link navigates to the detail page.
- Play button exists and is clickable.
- Canvas element is rendered with expected dimensions.

### Verification commands

1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test e2e/sequence-models/event-encoder-piano-roll.spec.ts`
3. `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`

No backend tests are required.

## 13. Prototype Reference

A working HTML/Canvas prototype exists at `/tmp/token-swimlane-prototype.html`
with real job data (`/tmp/event_encoder_k50.json`). It demonstrates the full
visual encoding (frequency positioning, F0-range height, ridge slope lines
with arrowheads, token labels with opaque backgrounds, token legend filtering,
minimap, pan/zoom, tooltips) and can be served from `python3 -m http.server`
in `/tmp/`. The implementing agent should reference this prototype for
rendering logic and visual behavior but should build the production component
using React, the project's existing API hooks, and the established component
patterns.

## 14. Risks and Follow-Ups

- Very large Event Encoder jobs (10k+ events) may need viewport-based data
  windowing. The current timeline endpoint returns all events for the selected
  k. This is acceptable for current job sizes but may require pagination for
  significantly larger recordings.
- Future enhancements under separate specs: self-similarity matrix overlay,
  inter-onset interval histogram panel, token transition diagram, embedding
  color strip.
- The piano roll does not show the spectrogram. Users who want to verify a
  token against the spectrogram should use the existing timeline viewer on
  the detail page.
