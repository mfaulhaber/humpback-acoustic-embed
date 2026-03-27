# Timeline Tile Performance — Design Spec

**Date:** 2026-03-26

## Problem

The timeline viewer currently feels slow to open for long hydrophone jobs and expensive to keep warm in cache.

The current backend path does the following on a tile miss or `/timeline/prepare`:

1. Rebuild the audio slice for the tile from the archive provider
2. Compute STFT power for the tile
3. Render a PNG through Matplotlib
4. Persist the tile to the disk cache

That work compounds badly on long jobs because `/timeline/prepare` currently walks every tile for every zoom level in priority order. The tile grid grows much faster than the viewer’s immediate needs:

- 1 hour job: 464 tiles across all zoom levels
- 24 hour job: 11,093 tiles across all zoom levels
- 7 day job: 77,651 tiles across all zoom levels

By contrast, the frontend only needs a small neighborhood of tiles around the current viewport. At the default zoom, that is roughly a single-digit number of tiles, not tens of thousands.

## Hotspot Findings

### 1. Full-cache preparation is doing far more work than startup needs

`src/humpback/api/routers/timeline.py` renders all tiles for all zoom levels during `/prepare`. Even though this now runs off the detection critical path, it still turns “open the timeline” into “start a very large background rendering job”.

### 2. Audio/timeline resolution is rebuilt per tile

`src/humpback/processing/timeline_audio.py` calls into HLS/provider timeline builders on each tile render. For Orcasound HLS this means rebuilding the segment timeline and decoding overlapping `.ts` segments even when adjacent tiles share most of the same source data.

### 3. The renderer has high fixed Python overhead per tile

`src/humpback/processing/timeline_tiles.py` computes an STFT and then creates a Matplotlib figure, paints the image, and saves it to PNG bytes for every tile. That is a relatively expensive path for something repeated thousands of times.

## Design Options

### Option 1: Viewport-first progressive preparation

Change `/timeline/prepare` so the default behavior prepares only the tiles needed for fast startup:

- prepare the current zoom level first
- prepare only a bounded neighborhood around the requested center timestamp
- optionally prepare one or two coarser zoom levels around the same center
- leave full all-zoom prewarming as an explicit background mode, not the default open-path behavior

Potential API shape:

- `scope="startup" | "full"`
- `zoom_level`
- `center_timestamp`
- `radius_tiles`

Pros:

- Biggest startup win with the smallest architectural change
- Aligns rendering work with what the viewer actually needs
- Keeps the existing tile format and disk cache model

Cons:

- Does not reduce the per-tile cost of a miss
- Long-range panning can still trigger expensive misses unless paired with reuse/prefetch

### Option 2: Reusable timeline manifest and render-session caches

Introduce job-scoped reuse for the expensive non-image work:

- build a timeline manifest once per job and reuse it across tile renders
- during a prepare batch, keep an in-memory render session with:
  - resolved segment timeline
  - cached decoded segment audio keyed by segment path and target sample rate
  - cached per-job normalization data
- on single-tile misses, reuse the persisted manifest and opportunistically prewarm adjacent tiles

Pros:

- Speeds both `/prepare` and ad hoc cache misses
- Directly attacks repeated folder listing, playlist parsing, and ffmpeg decode work
- Preserves current visual output

Cons:

- More state management and invalidation complexity
- Needs explicit memory bounds for decoded-audio reuse

### Option 3: Replace Matplotlib with a faster tile renderer

Keep the current STFT math but remove per-tile figure creation:

- compute the dB array with NumPy/SciPy
- map through the Ocean Depth colormap without creating a Matplotlib figure
- encode directly to PNG with Pillow or a similar lightweight encoder
- optionally parallelize rendering across a bounded worker pool once the audio/decode path is reusable

Pros:

- Lowers CPU overhead for every rendered tile
- Helps regardless of provider type
- Makes full-cache preparation less painful if we still need it

Cons:

- Highest implementation risk
- Needs careful pixel-parity validation so the viewer appearance does not regress
- May not be the dominant bottleneck if timeline/decode reuse is not solved first

## Recommendation

Implement a phased fix:

1. **Adopt Option 1 immediately** by making timeline preparation viewport-first and bounded by default.
2. **Pair it with the bounded parts of Option 2** so adjacent tiles reuse timeline and decoded segment work within a prepare batch and on common miss paths.
3. **Defer Option 3** until after we measure the post-change timings. If the renderer still dominates, we can replace the Matplotlib path in a follow-up without redoing the scheduling/caching work.

This sequence is recommended because the current biggest problem is not just “tile rendering is slow”, but “we are rendering far too many tiles up front”. Once that is corrected, reuse gives the next highest payoff with lower risk than a renderer rewrite.

## Proposed Behavior

### Startup path

- Clicking `Timeline` should start a bounded startup prepare around the default initial viewport instead of launching a full all-zoom warmup.
- The viewer should navigate immediately.
- `prepare-status` should reflect the active target set for the startup prepare, not imply that every tile for every zoom is expected up front.

### Miss path

- A requested uncached tile should still render on demand.
- After rendering a miss, the backend should opportunistically queue a small number of adjacent tiles for the same zoom level.
- The miss path should reuse any available job manifest and decoded-segment cache.

### Full-cache path

- Full all-zoom prewarming should remain available as an explicit mode for offline warming or benchmarking.
- It should reuse the same manifest and render-session helpers as the startup path.

## Files likely changed

| File | Change |
|------|--------|
| `src/humpback/api/routers/timeline.py` | Add scoped prepare behavior, bounded tile targeting, background neighbor prewarm, updated status reporting |
| `src/humpback/processing/timeline_audio.py` | Add reusable job manifest / render-session helpers for timeline audio resolution |
| `src/humpback/processing/timeline_cache.py` | Persist manifest/metadata and bounded cache bookkeeping for render reuse |
| `src/humpback/classifier/s3_stream.py` | Expose helpers for reusable HLS timeline metadata and segment reuse |
| `frontend/src/api/client.ts` | Send scoped prepare requests, handle updated status shape if needed |
| `frontend/src/hooks/queries/useTimeline.ts` | Adjust prepare mutation / status polling |
| `frontend/src/components/classifier/HydrophoneTab.tsx` | Trigger startup-scoped prepare before navigation |
| `frontend/src/components/timeline/TimelineViewer.tsx` | Consume updated prepare status semantics |
| `tests/integration/test_timeline_api.py` | Cover scoped prepare, miss-triggered neighbor warming, and status semantics |
| `tests/unit/test_timeline_audio.py` | Cover manifest reuse and bounded decoded-segment caching |
| `tests/unit/test_timeline_cache.py` | Cover manifest persistence and cache bookkeeping |

## Testing

- **Startup prepare:** verify the default prepare path only targets a bounded tile neighborhood and returns without scheduling a full all-zoom render.
- **Reuse:** verify repeated tile renders for the same job reuse the timeline manifest and avoid rebuilding the HLS timeline unnecessarily.
- **Miss smoothing:** verify a cache miss can trigger adjacent same-zoom tile warming without duplicate work.
- **Status correctness:** verify prepare status reflects startup-scoped progress accurately.
- **Regression:** existing tile PNG generation, cache locking, and timeline API tests continue to pass.
- **Frontend smoke:** opening the timeline from Hydrophone jobs still navigates immediately and continues to show caching progress.
