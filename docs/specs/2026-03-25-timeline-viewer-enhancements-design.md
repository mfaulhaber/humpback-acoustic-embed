# Timeline Viewer Enhancements — Design Spec

## Overview

Four improvements to the timeline viewer for hydrophone detection jobs:

1. **Background tile pre-caching** — eagerly cache all zoom levels on session open
2. **Positive-only detection label bars** — replace full-rectangle overlays with narrow vertical bars for humpback/orca only, with hover tooltips
3. **Playhead-audio sync fix** — make the audio element's `currentTime` the authoritative clock for playhead position
4. **Gapless audio playback** — double-buffered MP3 chunks at 5-minute segments

---

## 1. Background Tile Pre-caching

### Problem

Only 24h and 6h tiles are pre-rendered on job completion. Finer zoom levels (1h, 15m, 5m, 1m) render on-demand, causing visible load delays when zooming — especially at 1m where a 24h job has ~8,640 tiles. Returning to a previously viewed job re-renders everything.

### Design

**Hybrid backend-driven rendering with on-demand fallback.**

When the frontend opens a timeline session, it POSTs to an enhanced `/timeline/prepare` endpoint that queues background rendering of all 6 zoom levels in priority order: 1h, 15m, 5m, 1m, 6h, 24h. The backend renders tiles in a background thread, tracking progress per zoom level. The frontend polls a new `/timeline/prepare-status` endpoint to show caching progress. Any tile can still be requested on-demand immediately if the user zooms before background caching finishes.

### Per-Job LRU Tile Cache

The tile cache switches from a flat 5,000-tile global FIFO to per-job LRU eviction:

- **`HUMPBACK_TIMELINE_CACHE_JOBS`** env var, default 15.
- Each job's tiles are stored in `{cache_dir}/{job_id}/` (existing structure).
- A sentinel file `{cache_dir}/{job_id}/.last_access` is touched on any tile read or write for that job.
- When a new job's tiles are written and the job count exceeds the limit, the least-recently-accessed job's entire directory is evicted.
- Returning to a previously cached job is instant — all ~11,000 tiles are on disk.

### Disk Budget

At ~50-100 KB per tile, 15 fully-cached 24h jobs = ~8-16 GB. Operators can tune the job count via `HUMPBACK_TIMELINE_CACHE_JOBS` for their disk budget.

### Tile Counts Per Zoom Level (24h Job)

| Zoom | Tile Duration | Tiles |
|------|--------------|-------|
| 24h  | 24h          | 1     |
| 6h   | 6h           | 4     |
| 1h   | 10 min       | 144   |
| 15m  | 2.5 min      | 576   |
| 5m   | 50 sec       | ~1,728|
| 1m   | 10 sec       | ~8,640|
| **Total** |         | **~11,093** |

### Backend Changes

**Enhanced `POST /timeline/prepare`:**
- Accepts optional `zoom_levels` query param (default: all 6 levels).
- Renders tiles in priority order (1h, 15m, 5m, 1m, 6h, 24h) in a background thread.
- Returns immediately with status acknowledgment.
- Skips already-cached tiles (idempotent).

**New `GET /timeline/prepare-status`:**
- Returns per-zoom-level progress: `{ "1h": { "total": 144, "rendered": 80 }, "15m": { "total": 576, "rendered": 0 }, ... }`.
- Counts cached tile files on disk per zoom directory.

**`TimelineTileCache` changes:**
- New `touch_job(job_id)` method — updates `.last_access` sentinel mtime.
- New `evict_lru_jobs(max_jobs)` — called on `put()`, removes oldest job directories when count exceeds limit.
- `get()` calls `touch_job()` on cache hit.
- Constructor accepts `max_jobs` (from settings) instead of flat `max_items`.

### Frontend Changes

- On timeline session open, call `prepareTimeline(jobId)` requesting all zoom levels.
- Poll `prepare-status` to display a subtle progress indicator (e.g., "Caching tiles: 1h 80/144").
- Tile requests work immediately via on-demand fallback — no blocking on background completion.

---

## 2. Positive-Only Detection Label Bars

### Problem

The current `DetectionOverlay` renders semi-transparent rectangles for all label types (humpback, orca, ship, background) at 15% opacity spanning the full spectrogram height. This is visually noisy and treats negative labels (ship, background) with the same prominence as positive detections.

### Design

**Narrow vertical bars for positive labels only (humpback, orca), with hover tooltips.**

**Filtering:**
- Only render rows where `humpback === 1` or `orca === 1`.
- Ship, background, and unlabeled rows are not displayed.

**Visual treatment:**
- Full-height vertical bars spanning the spectrogram canvas and the confidence strip below.
- Bar width = detection window's time span in pixels, with a minimum of 2px so narrow windows remain visible at wide zoom levels.
- Colors: humpback = `rgba(64, 224, 192, 0.25)` (teal), orca = `rgba(224, 176, 64, 0.25)` (amber).
- Z-order: above spectrogram tiles, below playhead line.

**Hover interaction:**
- Pointer events enabled on individual bars.
- On hover, show a lightweight tooltip with:
  - Label type (Humpback / Orca)
  - UTC timestamp range (HH:MM:SS — HH:MM:SS)
  - Confidence (avg / peak)
- On click, open the existing `DetectionPopover` with full details.

### Component Changes

**`DetectionOverlay.tsx`:**
- Filter detection rows to positive labels only.
- Change rendering from filled rectangles to full-height narrow bars.
- Enable pointer events on individual bar elements.
- Add hover tooltip state and rendering.

**`SpectrogramViewport.tsx`:**
- Extend the overlay area to also cover the confidence strip (currently stops above it).

**`TimelineHeader.tsx`:**
- "Labels" toggle unchanged (still toggles the overlay on/off).

---

## 3. Playhead-Audio Sync Fix

### Problem

Two independent clocks drive the timeline with no feedback between them:

1. A `requestAnimationFrame` loop advances `centerTimestamp` by `dt * speed` each frame.
2. A `<audio>` element plays WAV chunks independently.

The RAF loop starts immediately on play, but the audio element needs time to load and decode. This latency (~1-2 seconds) means the playhead is always ahead of the audio.

### Design

**Audio-authoritative playhead.** The audio element's `currentTime` becomes the single source of truth.

**New state in `TimelineViewer.tsx`:**
- `playbackOriginEpoch: number` — the UTC epoch second where the current audio chunk begins. Set when a chunk is loaded.

**RAF loop change:**
```
// Before (RAF-driven):
centerTimestamp += dt * speed

// After (audio-authoritative):
centerTimestamp = playbackOriginEpoch + activeAudioRef.currentTime
```

The `playbackRate` property on the audio element handles speed changes:
```
activeAudioRef.playbackRate = speed
```

Since `playbackRate` makes the audio element advance `currentTime` proportionally faster, the timestamp mapping is direct — `currentTime` already reflects actual audio progress.

**Playback lifecycle:**

1. User presses play → set `playbackOriginEpoch = centerTimestamp`, request audio chunk starting at that epoch, wait for `canplay` event.
2. Audio starts → RAF loop reads `activeAudioRef.currentTime`, computes `centerTimestamp = playbackOriginEpoch + activeAudioRef.currentTime`.
3. Audio stalls (buffering) → `currentTime` stops advancing → spectrogram freezes in sync.
4. Chunk ends → next chunk loads via double-buffer swap (Section 4), `playbackOriginEpoch` updates to next chunk's start time.

**User pans while paused:** `centerTimestamp` is set directly by the pan handler. When play resumes, a new chunk is fetched from the new position.

### Component Changes

**`TimelineViewer.tsx`:**
- Replace `dt * speed` RAF advancement with `playbackOriginEpoch + activeAudioRef.currentTime`.
- Add `playbackOriginEpoch` state.
- Set `audioRef.playbackRate = speed` instead of using speed in RAF math.

---

## 4. Gapless Audio Playback

### Problem

A single `<audio>` element plays 30-second WAV chunks. When a chunk ends, the `ended` event fires, then a new URL is set, the browser fetches and decodes, and only then does playback resume. This creates a perceptible silence gap. Additionally, 30-second WAV at 32kHz 16-bit mono = ~1.9 MB per chunk — frequent fetches with no compression.

### Design

**Two `<audio>` elements with MP3 encoding and 5-minute segments.**

**Double-buffer ping-pong:**

1. Press play → load 5-minute MP3 chunk into element A, start playback.
2. While A plays, compute the next chunk's start time and pre-fetch into element B (silent, paused).
3. When A's `timeupdate` shows it's within ~2 seconds of ending, B should be ready (`readyState >= HAVE_ENOUGH_DATA`). On A's `ended`:
   - Start playback on B immediately.
   - Update `playbackOriginEpoch` to B's start epoch.
   - Begin pre-fetching the next chunk into A (now idle).
4. Repeat, ping-ponging between A and B.

**MP3 encoding:**

- New optional `format` query param on `GET /timeline/audio`: `wav` (default, backward compat) or `mp3`.
- Backend encodes via `ffmpeg` subprocess: float32 audio → temp WAV → MP3 128kbps mono → bytes.
- 5-minute MP3 at 128kbps mono = ~4.7 MB (vs ~18.8 MB WAV).
- All modern browsers decode MP3 natively.

**Segment size increase:**

- `AUDIO_PREFETCH_SEC` changes from 30 to 300 (5 minutes).
- Backend `duration_sec` max increases from 120s to 600s.
- A 1-hour viewport = 12 chunks instead of 120.

**Graceful degradation:**

If the next chunk fails to pre-load (network error, slow backend), fall back to single-element behavior — a brief gap is acceptable. The swap only happens if the standby element's `readyState >= HAVE_ENOUGH_DATA`.

### Backend Changes

**`GET /timeline/audio`:**
- New optional `format` query param (`wav` or `mp3`, default `wav`).
- `duration_sec` max raised from 120 to 600.
- MP3 path: encode float32 audio via `ffmpeg` subprocess, return with `Content-Type: audio/mpeg`.

**New `_encode_mp3()` helper:**
- Takes float32 audio array + sample rate.
- Writes temp WAV, converts to MP3 via `ffmpeg -i input.wav -codec:a libmp3lame -b:a 128k -ac 1 output.mp3`.
- Returns MP3 bytes, cleans up temp files.

### Frontend Changes

**`TimelineViewer.tsx`:**
- Two `<audio>` refs (`audioRefA`, `audioRefB`) and an `activeRef` pointer.
- Swap logic on chunk boundaries: on `ended` event of active element, start the standby element, swap refs.
- Request `format=mp3` in audio URLs.

**`constants.ts`:**
- `AUDIO_PREFETCH_SEC` = 300 (was 30).
- New `AUDIO_FORMAT = "mp3"`.

---

## 5. Configuration

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `timeline_cache_max_jobs` | `HUMPBACK_TIMELINE_CACHE_JOBS` | 15 | Max detection jobs with fully cached timeline tiles. LRU eviction removes the oldest job when exceeded. ~8-16 GB at default. |

No database migration required. All changes are filesystem (tile cache), API parameters, and frontend components.

---

## 6. Testing

### Backend Unit Tests

- `test_tile_cache_lru_eviction` — write tiles for 3 jobs with max_jobs=2, verify oldest job directory removed.
- `test_tile_cache_touch_access` — verify tile access updates job's `.last_access` sentinel.
- `test_prepare_all_zoom_levels` — trigger prepare with all 6 zoom levels, verify tile counts per zoom.
- `test_prepare_status` — verify per-zoom progress reporting matches rendered tile counts.
- `test_audio_mp3_encoding` — verify MP3 output has correct content type, is decodable, and duration matches input.
- `test_audio_duration_600s` — verify 600s max is accepted, 601s is rejected.

### Backend Integration Tests

- `test_prepare_all_and_status_api` — POST prepare, poll status until complete, verify all tile counts.
- `test_audio_mp3_format_param` — request with `format=mp3`, verify `Content-Type: audio/mpeg` header.
- `test_cache_eviction_api` — open tiles for N+1 jobs (N=max), verify first job's tiles are evicted.

### Frontend Playwright Tests

- `test_label_overlay_positive_only` — verify only humpback/orca bars render, no ship/background.
- `test_label_hover_tooltip` — hover a detection bar, verify tooltip shows label type and confidence.
- `test_playback_no_drift` — play for 10 seconds, verify playhead position matches audio `currentTime` within 100ms.
- `test_double_buffer_swap` — mock two audio chunks, verify gapless transition.

---

## Documentation Updates

- **CLAUDE.md §8.6** — add `HUMPBACK_TIMELINE_CACHE_JOBS` to runtime config table.
- **CLAUDE.md §8.5** — note MP3 audio format option for timeline playback.
- **CLAUDE.md §9.1** — update timeline viewer capabilities (background caching, gapless playback, positive-only labels).

---

## Out of Scope

- Audio caching on the frontend (beyond the two double-buffered chunks in memory).
- WebGL or OffscreenCanvas tile rendering.
- Offline/service-worker tile caching.
- Label editing from the timeline viewer (still use table view via popover link).
