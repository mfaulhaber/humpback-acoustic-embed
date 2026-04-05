# Timeline Export Function Design

**Date**: 2026-04-05
**Status**: Approved

## Purpose

Export a completed detection job's timeline data as a self-contained static bundle
that a separate readonly React app can serve from S3 without any API server.

The primary audience is researchers who need to browse detection timelines without
running the full humpback platform. A future version of the viewer will support
label editing; the export format accommodates this but the MVP treats everything
as readonly.

## Bundle Structure

Each export produces a directory for one detection job:

```
{output_dir}/{job_id}/
├── manifest.json
├── tiles/
│   ├── 24h/tile_0000.png
│   ├── 6h/tile_0000.png ... tile_0003.png
│   ├── 1h/tile_0000.png ... tile_0005.png
│   ├── 15m/tile_0000.png ... tile_0023.png
│   ├── 5m/tile_0000.png ... tile_0071.png
│   └── 1m/tile_0000.png ... tile_0359.png
└── audio/
    ├── chunk_0000.mp3
    ├── chunk_0001.mp3
    └── ... (288 chunks for 24h)
```

### Tiles

- Format: PNG, 512x256 pixels, Ocean Depth colormap
- Naming: `tile_{index:04d}.png` (zero-padded 4-digit index)
- All 6 zoom levels exported: 24h, 6h, 1h, 15m, 5m, 1m
- Tile durations: 86400s, 21600s, 600s, 150s, 50s, 10s respectively
- Tile count per zoom level: `ceil(job_duration / tile_duration)`
- Copied directly from the timeline tile cache

### Audio

- Format: MP3, 128 kbps, 32 kHz sample rate
- Chunk size: 300 seconds (5 minutes)
- Naming: `chunk_{index:04d}.mp3` (zero-padded 4-digit index)
- Chunk count: `ceil(job_duration / 300)`
- Last chunk may be shorter than 300 seconds
- Generated from the hydrophone audio source via `resolve_timeline_audio()`

### Manifest

Single `manifest.json` containing all metadata the viewer needs. See the
companion spec `2026-04-05-timeline-export-consumer-contract.md` for the full
schema definition.

## Export Service

### Function Signature

```python
async def export_timeline(
    job_id: str,
    output_dir: Path,
    db: Session,
    settings: Settings,
) -> ExportResult
```

### Pipeline

1. **Validate** — confirm job exists, is complete, and has all tiles cached.
   If tiles are not fully rendered, raise an error (do not implicitly trigger
   tile preparation).
2. **Create output directory** — `{output_dir}/{job_id}/` with `tiles/` and
   `audio/` subdirectories. Create parents if needed.
3. **Copy tiles** — for each zoom level, copy PNGs from the timeline cache.
   Preserve the `tile_{index:04d}.png` naming.
4. **Generate audio chunks** — for each 300s window within the job's time
   range, resolve audio via the existing `resolve_timeline_audio()`, encode
   to MP3, write to `audio/chunk_{index:04d}.mp3`.
5. **Build manifest** — query the database for job metadata, detection rows,
   vocalization labels, and vocalization types. Read confidence scores from
   parquet diagnostics. Assemble and write `manifest.json`.
6. **Return summary** — path, tile count, audio chunk count, total size.

### Preconditions

- Detection job must exist and be complete.
- All tiles at all zoom levels must be rendered in the timeline cache. If not,
  the caller must first run `POST /timeline/prepare` with `scope=full` and
  wait for completion.

### Idempotency

Re-exporting the same job to the same output directory overwrites existing
files. Content is deterministic for a given job state.

### Error Cases

| Condition | Behavior |
|-----------|----------|
| Job not found | 404 |
| Job not complete | 409, message: "job must be complete" |
| Tiles not fully rendered | 409, message: "run timeline prepare with scope=full first" |
| Output directory not writable | 500 with path in message |

## API Endpoint

```
POST /classifier/detection-jobs/{job_id}/export-timeline
```

**Request body:**
```json
{ "output_dir": "/path/to/export" }
```

**Response (200):**
```json
{
  "job_id": "uuid",
  "output_path": "/path/to/export/{job_id}",
  "tile_count": 11494,
  "audio_chunk_count": 288,
  "manifest_size_bytes": 1523400
}
```

Synchronous — blocks until export completes. Acceptable for MVP since this is
an operational tool, not user-facing in the viewer.

## CLI Script

```
uv run python scripts/export_timeline.py \
  --job-id <uuid> \
  --output-dir /path/to/export
```

- Loads settings from `.env`
- Creates DB session
- Calls `export_timeline()` service function
- Prints progress (tile copy, audio encoding) to stderr
- Prints summary to stdout
- Non-zero exit on failure

## Size Estimates (24-hour job)

| Component | Count | Approximate Size |
|-----------|-------|-----------------|
| Tiles (all zoom levels) | ~11,000 | ~500 MB |
| Audio chunks | 288 | ~1.4 GB |
| manifest.json | 1 | ~1-2 MB |
| **Total** | | **~2 GB** |

## Scope

### In scope
- `export_timeline()` service function in `src/humpback/services/`
- API endpoint on the timeline router
- CLI script in `scripts/`
- Manifest assembly logic

### Out of scope
- The React 19 readonly viewer (separate project)
- S3/CloudFront infrastructure
- `index.json` generation (lives in viewer project)
- Label editing in the viewer (future version)
- Authentication on S3
- Background job queue for export (MVP is synchronous)
