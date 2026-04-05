# Timeline Export Consumer Contract

**Date**: 2026-04-05
**Status**: Approved
**Audience**: Developers building the readonly timeline viewer MVP

This spec defines the data contract between the humpback platform's timeline
export function and the static React 19 viewer app. Everything the viewer needs
to render a detection timeline is described here.

## Hosting Layout

```
s3://bucket/
├── index.html              # React SPA (Vite build)
├── assets/                 # JS, CSS
└── data/
    ├── index.json          # Timeline registry
    ├── {job_id_1}/
    │   ├── manifest.json
    │   ├── tiles/
    │   │   ├── 24h/tile_0000.png
    │   │   ├── 6h/tile_0000.png ...
    │   │   ├── 1h/tile_0000.png ...
    │   │   ├── 15m/tile_0000.png ...
    │   │   ├── 5m/tile_0000.png ...
    │   │   └── 1m/tile_0000.png ...
    │   └── audio/
    │       ├── chunk_0000.mp3
    │       └── ...
    └── {job_id_2}/
        └── ...
```

Everything is same-origin. No CORS configuration needed.

## URL Routing

| Route | View | Data Source |
|-------|------|-------------|
| `/` | Landing page — list of available timelines | `data/index.json` |
| `/{job_id}` | Timeline viewer | `data/{job_id}/manifest.json` |

## Timeline Registry (`index.json`)

Maintained outside the export function (manually or via a scan script in the
viewer project). Provides enough metadata for a landing page.

```typescript
interface TimelineIndex {
  timelines: TimelineEntry[];
}

interface TimelineEntry {
  job_id: string;
  hydrophone_name: string;
  species: string;
  start_timestamp: number;  // epoch seconds UTC
  end_timestamp: number;    // epoch seconds UTC
}
```

## Manifest Schema (Version 1)

Each exported timeline has a `manifest.json` at its root. The viewer loads this
on initial page load — it contains all metadata needed to render.

```typescript
interface TimelineManifest {
  version: 1;
  job: JobMetadata;
  tiles: TileMetadata;
  audio: AudioMetadata;
  confidence: ConfidenceData;
  detections: Detection[];
  vocalization_labels: VocalizationLabel[];
  vocalization_types: VocalizationType[];
}
```

### `job` — Job-Level Metadata

```typescript
interface JobMetadata {
  id: string;                    // UUID
  hydrophone_name: string;       // e.g. "Orcasound Lab"
  hydrophone_id: string;         // e.g. "rpi_orcasound_lab"
  start_timestamp: number;       // job start, epoch seconds UTC
  end_timestamp: number;         // job end, epoch seconds UTC
  species: string;               // e.g. "humpback"
  window_selection: string;      // "nms" | "prominence" | "tiling"
  model_name: string;            // classifier model name
  model_version: string;         // classifier model version
}
```

### `tiles` — Tile Layout

```typescript
interface TileMetadata {
  zoom_levels: ZoomLevel[];
  tile_size: [number, number];   // [width, height] in pixels — always [512, 256]
  tile_durations: Record<ZoomLevel, number>;   // seconds per tile
  tile_counts: Record<ZoomLevel, number>;      // tiles at this zoom level
}

type ZoomLevel = "24h" | "6h" | "1h" | "15m" | "5m" | "1m";
```

**Tile path computation:**
```typescript
function tilePath(jobId: string, zoom: ZoomLevel, index: number): string {
  return `data/${jobId}/tiles/${zoom}/tile_${String(index).padStart(4, '0')}.png`;
}
```

**Tile time range:**
```typescript
function tileTimeRange(manifest: TimelineManifest, zoom: ZoomLevel, index: number) {
  const duration = manifest.tiles.tile_durations[zoom];
  const start = manifest.job.start_timestamp + index * duration;
  const end = Math.min(start + duration, manifest.job.end_timestamp);
  return { start, end };
}
```

### `audio` — Audio Chunk Layout

```typescript
interface AudioMetadata {
  chunk_duration_sec: number;   // 300 (5 minutes)
  chunk_count: number;          // ceil(job_duration / chunk_duration_sec)
  format: "mp3";
  sample_rate: number;          // 32000
}
```

**Audio chunk path computation:**
```typescript
function audioChunkPath(jobId: string, index: number): string {
  return `data/${jobId}/audio/chunk_${String(index).padStart(4, '0')}.mp3`;
}
```

**Chunk time range:**
```typescript
function chunkTimeRange(manifest: TimelineManifest, index: number) {
  const duration = manifest.audio.chunk_duration_sec;
  const start = manifest.job.start_timestamp + index * duration;
  const end = Math.min(start + duration, manifest.job.end_timestamp);
  return { start, end };
}
```

**Last chunk:** May be shorter than `chunk_duration_sec`. The viewer should
handle audio elements that end before the expected chunk boundary.

### `confidence` — Confidence Strip Data

```typescript
interface ConfidenceData {
  window_sec: number;            // 5.0
  scores: (number | null)[];     // one entry per window, null = no data
}
```

Scores are ordered chronologically starting from `job.start_timestamp`.
Entry `i` covers the time range `[start + i * window_sec, start + (i+1) * window_sec)`.

Array length: `ceil((end_timestamp - start_timestamp) / window_sec)`.

### `detections` — Detection Rows

```typescript
interface Detection {
  row_id: string;               // UUID — stable identifier
  start_utc: number;            // epoch seconds UTC
  end_utc: number;              // epoch seconds UTC
  avg_confidence: number;       // 0.0 - 1.0
  peak_confidence: number;      // 0.0 - 1.0
  label: string | null;         // "humpback" | "orca" | "ship" | "background" | null
}
```

Labels are mutually exclusive — at most one label per detection row. Null
means unlabeled. `row_id` is preserved for future label editing.

### `vocalization_labels` — Vocalization Type Labels

```typescript
interface VocalizationLabel {
  start_utc: number;            // epoch seconds UTC
  end_utc: number;              // epoch seconds UTC
  type: string;                 // vocalization type name, e.g. "moan"
  confidence: number;           // 0.0 - 1.0
  source: "manual" | "inference";
}
```

Multiple labels can exist for the same time window (multi-label).

### `vocalization_types` — Vocabulary

```typescript
interface VocalizationType {
  id: number;
  name: string;
}
```

## Rendering Hints

### Zoom Levels and Viewport Spans

Each zoom level shows a specific amount of time on screen:

| Zoom Level | Tile Duration (sec) | Viewport Span (sec) | Viewport Span (human) |
|-----------|--------------------|--------------------|----------------------|
| 24h | 86400 | 86400 | 24 hours |
| 6h | 21600 | 21600 | 6 hours |
| 1h | 600 | 3600 | 1 hour |
| 15m | 150 | 900 | 15 minutes |
| 5m | 50 | 300 | 5 minutes |
| 1m | 10 | 60 | 1 minute |

At the 1h zoom level and below, multiple tiles compose the viewport. For
example, at 1h zoom: viewport span is 3600s, tile duration is 600s, so 6
tiles are visible at once.

### Spectrogram Tiles

- **Colormap**: Ocean Depth — navy (low energy) through teal and seafoam to
  white (high energy). The export bakes this into the PNG; the viewer
  renders tiles as plain images.
- **Frequency range**: 0-3000 Hz (baked into tile). The vertical axis
  represents frequency, bottom = 0 Hz, top = 3000 Hz.
- **Tile positioning**: Tiles are placed left-to-right chronologically.
  A tile at index `i` starts at `job.start_timestamp + i * tile_duration`.
  The canvas scrolls by adjusting which tiles are visible based on
  `centerTimestamp`.
- **Tile caching**: Use an in-memory LRU cache (recommended ~200 entries)
  for loaded `Image` objects to avoid re-fetching during pan/zoom.
- **Zoom transitions**: Cross-fade between zoom levels over ~300ms for
  smooth UX. Preload the target zoom layer before starting the transition.

### Confidence Strip

Render as a horizontal bar above or below the spectrogram:

- Map each score to a color gradient: dark/cold (0.0) to bright/warm (1.0)
- Null entries render as transparent/gray (no data)
- Each entry is `window_sec` wide (5 seconds), positioned by index

Suggested 8-stop gradient (score -> color):
```
0.00 → #1a1a2e  (dark navy)
0.15 → #16213e
0.30 → #0f3460
0.45 → #2d6a4f
0.60 → #52b788
0.75 → #95d5b2
0.85 → #f4d35e
1.00 → #f9e547  (bright yellow)
```

### Detection Overlay

Render detection rows as colored bars overlaid on the spectrogram:

- **Labeled rows**: Color by label type
  - `humpback`: warm orange/amber
  - `orca`: blue
  - `ship`: red
  - `background`: gray
- **Unlabeled rows** (`label: null`): Use confidence-based alpha on a
  neutral color
- **Bar positioning**: Horizontal position from `start_utc` to `end_utc`,
  mapped to pixel coordinates using the viewport time range
- **Tooltip on hover**: Show label, start/end time (UTC formatted), avg
  and peak confidence

### Vocalization Overlay

Render as a toggleable layer on top of detection bars:

- Group labels by time window (`start_utc:end_utc`)
- Show type badges within each window
- **Manual labels**: Filled badges
- **Inference labels**: Outlined badges
- Assign colors to types by index from a categorical palette (e.g.,
  D3 category10). Do not persist colors — derive from type order in
  `vocalization_types`.

### Audio Playback

Gapless double-buffered playback using two `<audio>` elements:

1. Determine which chunk contains the current playback timestamp
2. Load that chunk into audio element A, start playback
3. Prefetch the next chunk into audio element B
4. When A ends, swap to B immediately, prefetch the next chunk into A
5. If B isn't ready when A ends, reload from current position as fallback

**Timestamp to chunk mapping:**
```typescript
function chunkForTimestamp(manifest: TimelineManifest, timestamp: number): number {
  const offset = timestamp - manifest.job.start_timestamp;
  return Math.floor(offset / manifest.audio.chunk_duration_sec);
}
```

**Playhead**: Render as a vertical line at viewport center. Audio playback
advances `centerTimestamp`, which scrolls the spectrogram.

### Time Display

All timestamps are UTC. Display format varies by zoom level:

| Zoom Level | Time Axis Format | Example |
|-----------|-----------------|---------|
| 24h, 6h | `MM-DD HH:MM` | `03-15 14:30` |
| 1h, 15m | `HH:MM:SS` | `14:30:45` |
| 5m, 1m | `HH:MM:SS` | `14:30:45` |

Label all time displays as UTC explicitly (e.g., "14:30:45 UTC" or a
persistent "UTC" indicator in the header).

### Keyboard Shortcuts (Suggested)

| Key | Action |
|-----|--------|
| Space | Play / pause |
| `+` / `-` | Zoom in / out |
| Left / Right arrow | Pan 10% of viewport |

### Responsive Considerations

- Tile pixel width is fixed at 512px. Scale tiles to fill the available
  viewport width via CSS/canvas scaling.
- The confidence strip and detection overlay scale with the same horizontal
  time-to-pixel mapping as the spectrogram.

## Schema Versioning

The manifest includes `"version": 1`. The viewer should check this field
on load and display a clear error if it encounters an unsupported version.

When label editing is added in a future version, the manifest version will
bump to 2 with additional fields. Version 1 manifests will remain readable
by the version 2 viewer.
