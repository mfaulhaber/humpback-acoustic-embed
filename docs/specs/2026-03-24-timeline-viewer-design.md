# Timeline Viewer — Zoomable Spectrogram for Detection Jobs

## Overview

A full-screen, immersive spectrogram viewer for hydrophone detection jobs, inspired by
Google's Pattern Radio. Users can explore up to 24 hours of audio with a zoomable
spectrogram centered on the playback position, a confidence heatmap showing classifier
scores, and lightweight detection review via popovers.

**Scope:** Hydrophone detection jobs only (MVP). Local file detection jobs may be
added in a future iteration.

**Entry point:** "Timeline View" button on completed hydrophone job rows in the
HydrophoneTab, navigating to `/app/classifier/timeline/{jobId}`.

---

## Layout

The page uses a dedicated dark theme (no app sidebar) to maximize spectrogram real
estate. From top to bottom:

1. **Header bar** — back link to HydrophoneTab, hydrophone name, date range (UTC),
   settings toggles (label overlay on/off, frequency range, settings popover).

2. **Minimap** — full 24h confidence heatmap bar with a draggable/clickable viewport
   indicator rectangle. Lets users see activity clusters at a glance and jump to any
   position.

3. **Main spectrogram viewport** — frequency axis on left gutter (configurable, default
   0–3 kHz), tiled spectrogram canvas with Ocean Depth colormap, fixed center playhead
   line with triangle marker, confidence heatmap strip below, UTC time axis at bottom.

4. **Zoom level selector** — row of discrete zoom level buttons (24h, 6h, 1h, 15m, 5m,
   1m) with the active level highlighted.

5. **Playback controls** — current UTC timestamp, skip-to-previous/play-pause/skip-to-next
   transport, playback speed selector (0.5x, 1x, 2x), zoom +/− buttons.

---

## Color Scheme: Ocean Depth

**Spectrogram:** deep navy (#000510) → dark blue (#051530) → ocean blue (#0a3050) →
teal (#108070) → seafoam (#50c8a0) → near-white (#d0fff0).

**Confidence heatmap:** dark green (#0a1a0a) → green (#2a5a20) → lime (#60a020) →
yellow-green (#a0c820) → yellow (#d0e040) → bright yellow (#f0f060).

**UI chrome:** dark background (#060d14), muted teal text (#a0c8c0), accent teal
(#70e0c0) for playhead and active states, border/divider (#1a3040).

The existing spectrogram popup (Inferno colormap) is unchanged — Ocean Depth is used
only in the timeline viewer.

---

## Tile System

### Tile Grid

Each zoom level divides the 24h timeline into fixed-duration tiles rendered as PNGs at
512×256 pixels.

| Zoom Level | Viewport Span | Tile Duration | Tiles for 24h | Rendering |
|-----------|---------------|---------------|---------------|-----------|
| 24h       | 24 hours      | 24h           | 1             | Pre-rendered |
| 6h        | 6 hours       | 6h            | 4             | Pre-rendered |
| 1h        | 1 hour        | 10 min        | 144           | On-demand |
| 15m       | 15 min        | 2.5 min       | 576           | On-demand |
| 5m        | 5 min         | 50 sec        | ~1,728        | On-demand |
| 1m        | 1 min         | 10 sec        | ~8,640        | On-demand |

### Pre-rendering

When a hydrophone detection job completes, the worker pre-renders tiles for the 24h and
6h zoom levels. These are always needed (minimap, initial view) and are few in number.
A `timeline_tiles_ready` boolean on the DetectionJob model signals the frontend.

### Audio Resolution for Tiles

Tile rendering must map a (job_id, zoom_level, tile_index) to raw audio samples from
the hydrophone HLS local cache. This is the core technical challenge.

**Timeline coordinate system:** A hydrophone detection job defines a continuous timeline
from `start_timestamp` to `end_timestamp` (UTC epoch seconds). All tile positions and
confidence scores use this absolute timeline. Tile index 0 at any zoom level starts at
`start_timestamp`; each subsequent tile advances by tile_duration seconds.

**Audio resolution:** A new `resolve_timeline_audio(job_id, start_sec, duration_sec)`
function resolves audio for an arbitrary timeline-absolute time range:

1. Reads the job's hydrophone config (hydrophone_id, local_cache_path). MVP supports
   Orcasound HLS providers only; NOAA GCS jobs are out of scope for the timeline viewer.
2. Builds/reuses the HLS timeline metadata for the job (segment list with absolute
   timestamps — the same metadata the detection worker already builds).
3. Identifies which HLS segments overlap the requested `[start_sec, start_sec + duration_sec]`.
4. Decodes and concatenates the overlapping segments, trimming to the exact requested range.
5. Returns a 1D float32 audio array at the job's sample rate.
6. For gaps in the HLS cache (missing segments), inserts silence to preserve timeline alignment.

This function is used by both `generate_timeline_tile()` (for spectrogram rendering) and
a new timeline-aware audio playback endpoint (see Audio Playback section below).

**HLS timeline reuse:** The detection worker already builds HLS timeline metadata during
detection. The timeline viewer should persist or re-derive this metadata so it does not
depend on re-listing the HLS cache directory on every tile request. A cached timeline
manifest (JSON) per job is sufficient.

### On-demand Rendering

Finer zoom levels are rendered on first request and cached to disk. The tile renderer
calls `resolve_timeline_audio()` to obtain audio for the tile's time range, then
generates a marker-free spectrogram PNG with the Ocean Depth colormap. The renderer
builds its own STFT pipeline (not wrapping the existing `generate_spectrogram_png`,
which hardcodes Inferno and 0–3 kHz) but reuses the same signal processing parameters
(n_fft, hop_length, etc.).

### Tile Storage

```
data/detections/{job_id}/timeline_tiles/
├── 24h/tile_0000.png
├── 6h/tile_0000.png … tile_0003.png
├── 1h/tile_0000.png … (on-demand)
└── …
```

### Zoom Transitions

When changing zoom level, the frontend crossfades (300ms opacity transition) from the
current tile layer to the new one. At most 2 zoom levels of tiles are in memory
simultaneously, and only during the transition.

---

## Frontend Architecture

### Route

`/app/classifier/timeline/:jobId` — new route in the classifier section.

### Component Tree

```
TimelineViewer
├── TimelineHeader
│   ├── Back link → /app/classifier/hydrophone
│   ├── Job metadata (hydrophone, date range, status)
│   └── SettingsBar (labels toggle, freq range, settings popover)
├── Minimap
│   ├── ConfidenceBar (24h heatmap, client-rendered canvas)
│   └── ViewportIndicator (draggable rectangle)
├── SpectrogramViewport
│   ├── FrequencyAxis (left gutter, scales with freq range)
│   ├── TileCanvas (main canvas, composites tile PNGs)
│   ├── ConfidenceStrip (per-window heatmap below spectrogram)
│   ├── Playhead (fixed center line + triangle)
│   ├── TimeAxis (UTC labels, density adapts to zoom)
│   └── DetectionOverlay (optional label markers)
├── ZoomSelector (discrete level buttons)
├── PlaybackControls
│   ├── TimestampDisplay (current UTC position)
│   ├── TransportButtons (skip, play/pause, skip)
│   ├── SpeedSelector (0.5x, 1x, 2x)
│   └── ZoomButtons (+/−)
└── DetectionPopover (on click: row details + link to table)
```

### Core State

Five pieces of state drive the entire view:

| State | Type | Description |
|-------|------|-------------|
| `centerTimestamp` | number (UTC epoch sec) | Center of viewport / playback position |
| `zoomLevel` | enum | Active discrete zoom level |
| `isPlaying` | boolean | When true, centerTimestamp advances in real-time |
| `freqRange` | [number, number] | Frequency range in Hz, default [0, 3000] |
| `showLabels` | boolean | Label overlay toggle |

All other derived values (visible tile indices, time axis labels, minimap viewport
position) are computed from these.

### Tile Loading

TileCanvas calculates visible tile indices for the current center + zoom, plus 1–2
buffer tiles on each side. An in-memory LRU `Map<string, HTMLImageElement>` caches
loaded tile images. TanStack Query handles network-level fetch deduplication and
caching.

### Audio Playback

A new timeline-aware audio endpoint serves audio by absolute timeline position:

**`GET /classifier/detection-jobs/{job_id}/timeline/audio?start_sec={epoch}&duration_sec=30`**

This endpoint calls `resolve_timeline_audio()` (same function used by tile rendering)
and returns a WAV stream. The frontend pre-fetches ~30-second chunks ahead of the
playhead using this endpoint. The `centerTimestamp` advances tied to
`AudioContext.currentTime` for precise spectrogram-audio sync.

The existing `audio-slice` endpoint (which requires a detection-row `filename` and
file-relative offset) is not suitable for arbitrary timeline playback and is left
unchanged.

---

## Interaction Model

### Panning (when paused)

- Click-and-drag horizontally on spectrogram to pan.
- Drag the minimap viewport indicator for fast panning.
- Click anywhere on minimap to jump to that position.
- Left/right arrow keys nudge by 1/10th of viewport width.

### Zooming

- Scroll wheel / trackpad pinch steps through discrete zoom levels.
- Zoom is always centered on the playhead (center of viewport).
- Click zoom level buttons or +/− buttons for direct level jumps.
- Keyboard: + and − keys.
- Transitions: 300ms crossfade between tile layers.

### Playback

- Space bar toggles play/pause.
- During playback: spectrogram scrolls left under fixed center playhead.
- Scroll rate matches real-time audio, adjusted by speed multiplier.
- Skip forward/back jumps to next/previous detection above confidence threshold.
- Panning while playing auto-pauses (hybrid mode). Play re-centers on audio position.

### Detection Popover

- Click on confidence heatmap strip or spectrogram at a detection window's position.
- Popover shows: timestamp range, avg/peak confidence, current labels, small spectrogram
  thumbnail, "View in table" link (navigates to HydrophoneTab with row highlighted).
- Dismissed by clicking elsewhere or pressing Esc.
- Does not appear during active playback.

### Label Overlay (toggled)

- Semi-transparent colored rectangles at labeled detection row positions.
- Colors: humpback = teal, orca = amber, ship = red, background = gray.
- 15% opacity, full frequency-axis height.
- Hidden by default; toggled via header button.

---

## Backend Changes

### New API Endpoints

**`GET /classifier/detection-jobs/{job_id}/timeline/tile`**

Query params: `zoom_level` (enum), `tile_index` (int), `freq_min` (int, default 0),
`freq_max` (int, default 3000).

Returns PNG. Serves from tile cache, renders on miss. 404 for out-of-range tile index.

**`GET /classifier/detection-jobs/{job_id}/timeline/confidence`**

Returns JSON: `{ window_sec: number, scores: number[], start_timestamp: number,
end_timestamp: number }`. The `scores` array is a single flat list ordered by
timeline-absolute position, one entry per detection window from `start_timestamp` to
`end_timestamp`. The backend constructs this by reading the window diagnostics parquet,
converting each row's per-file `(filename, offset_sec)` to a timeline-absolute position
using the HLS timeline manifest, then sorting by absolute time and extracting the
confidence values. Gaps (where no HLS audio existed) are filled with `null`
(TypeScript type: `(number | null)[]`). Cached in memory after first load.

**`GET /classifier/detection-jobs/{job_id}/timeline/audio`**

Query params: `start_sec` (float, UTC epoch), `duration_sec` (float, default 30).

Returns WAV stream. Uses `resolve_timeline_audio()` to map timeline-absolute position
to HLS cache segments. Gaps in cache are silent. Used by the frontend for playback at
arbitrary timeline positions.

**`POST /classifier/detection-jobs/{job_id}/timeline/prepare`**

Triggers pre-rendering of 24h + 6h zoom level tiles. Called automatically on job
completion; can be called manually. Idempotent — skips existing tiles. Returns
immediately; rendering is async.

### Spectrogram Renderer Changes

- New `OceanDepthColormap` — custom `LinearSegmentedColormap` with Ocean Depth stops.
- New `generate_timeline_tile()` function:
  - Own STFT + colormap pipeline (does not wrap `generate_spectrogram_png`, which
    hardcodes Inferno and 0–3 kHz). Reuses same signal processing parameters (n_fft,
    hop_length) from settings.
  - Calls `resolve_timeline_audio()` to obtain audio for the tile's time range.
  - Outputs marker-free images (no axes, labels, or padding).
  - Accepts `freq_min`/`freq_max` for configurable frequency cropping.
  - Fixed output: 512px wide × 256px tall.
  - Uses Ocean Depth colormap.
- Existing spectrogram rendering (Inferno, with axes) is untouched.

### Model Changes

- Add `timeline_tiles_ready: bool = False` column to `DetectionJob`.
- Alembic migration required.

### Tile Cache

- Directory structure per job and zoom level (not flat SHA256).
- No eviction for pre-rendered tiles (job-scoped, small).
- On-demand tiles: global FIFO eviction at 5,000 items across all jobs (bounds total
  disk usage regardless of how many jobs have been viewed).

### Worker Integration

- After hydrophone job status → `complete`, worker calls tile prepare step automatically.
- Sets `timeline_tiles_ready = True` on completion.
- The "Timeline View" button appears only after `timeline_tiles_ready` is true. If the
  user navigates to the viewer URL before pre-rendering completes, the viewer falls back
  to on-demand rendering for all zoom levels.

### Unchanged

- Detection TSV/parquet format.
- Existing `audio-slice` endpoint (still used by detection row popup/thumbnails).
- Existing `spectrogram` endpoint (still used by popup/thumbnails).
- Existing diagnostics endpoints (reused as data source for confidence array).

---

## Testing

### Backend Unit Tests

- `test_ocean_depth_colormap` — verify RGB values at key stops.
- `test_generate_timeline_tile` — render tile from synthetic audio, verify PNG
  dimensions and marker-free output.
- `test_tile_index_bounds` — verify tile count per zoom level, 404 for out-of-range.
- `test_confidence_endpoint` — fixture diagnostics parquet → verify JSON shape and
  score count.
- `test_tile_cache_storage` — verify directory structure, cache hits on re-request.
- `test_prepare_idempotency` — double prepare does not re-render.

### Backend Integration Tests

- `test_timeline_tile_api` — create job with fixture data, request tiles at each zoom
  level, verify PNG responses.
- `test_timeline_prepare_api` — trigger prepare, verify coarse tiles on disk.
- `test_freq_range_parameter` — same tile with different freq ranges → different PNG.

### Playwright E2E Tests

- `test_timeline_navigation` — click "Timeline View" on completed job, verify route
  and spectrogram loads.
- `test_zoom_levels` — click each zoom button, verify active indicator and tile loads.
- `test_minimap_click` — click minimap, verify viewport position change.
- `test_playback_controls` — play/pause, verify audio element state.
- `test_detection_popover` — click confidence strip, verify popover with details and
  "View in table" link.

Tests use fixture data (small hydrophone job with pre-generated diagnostics and audio
segments). No real model required.

---

## Out of Scope (MVP)

- Local file detection job support (future iteration).
- Direct labeling from the timeline viewer (use table view via popover link).
- WebGL rendering (Canvas 2D with pre-colored tiles is sufficient).
- Real-time streaming / live detection visualization.
- Vertical frequency-axis zoom (frequency range is set via settings, not interactive zoom).
