# Hydrophone S3 Cache Performance — Design Spec

**Date:** 2026-04-12
**Status:** Approved

## Problem

Region detection jobs on hydrophone sources take 5+ minutes per chunk when audio
gaps exist, even on a fully warmed local cache. Root cause analysis identified
four compounding issues in the S3 caching layer.

### Root Causes

1. **Per-chunk timeline builds (primary amplifier):** The region detection worker
   calls `iter_audio_chunks()` per chunk, which calls `provider.build_timeline()`
   internally. For a 24-hour job with 48 chunks, that's 48 independent timeline
   builds — each triggering the full S3 lookback machinery.

2. **`CachingS3Client` always hits S3 (primary cause):** Three operations leak
   S3 calls even on a warmed cache:
   - `list_hls_folders()` unconditionally queries S3 (no caching at all)
   - `list_segments()` queries S3 for folders without `.404.json`, even if S3
     returned a non-empty key list on a previous visit (writes nothing durable)
   - `fetch_playlist()` ignores `live.m3u8.404.json` marker and re-queries S3

3. **Missing `.404.json` folder markers:** Folders created by earlier code paths
   have `live.m3u8.404.json` but no `.404.json`, causing `list_segments()` to
   fall through to S3 on every visit. Self-healing after Fix 2 — no standalone
   work needed.

4. **Lookback loop runs 42 iterations for no-audio chunks:** When folders exist
   but no segments overlap the range, the loop increments 4-hour steps up to 7
   days (42 iterations). The existing jump-to-max optimization only fires when
   `timeline` is non-empty.

### Observed Impact

- Chunks 14-15 (audio present): ~45 seconds each (1.5s/min audio)
- Chunks 17-18 (no audio): ~280-293 seconds each — entirely in "fetching audio"
- Estimated 50-100+ S3 API calls per empty chunk

## Fix 1: Single Job-Level Timeline with Per-Chunk Slicing

**Where:** `region_detection_worker.py` — `_load_hydrophone_trace()`

Build the timeline once for `[start_ts, end_ts]` before the chunk loop. For each
chunk, filter the pre-built timeline to segments overlapping `[chunk_start,
chunk_end]`. If no segments overlap, skip the chunk immediately without entering
`iter_audio_chunks`.

**Interface change:** `_iter_audio_chunks()` gains an optional `timeline:
list[StreamSegment] | None = None` parameter. When provided, skips the internal
`build_timeline()` call. Surfaced through `iter_audio_chunks()` kwargs. All
existing callers pass `timeline=None` (default) and behave exactly as before.

The regular hydrophone detection (`hydrophone_detector.py`) already calls
`iter_audio_chunks` once for the full range, so it naturally avoids this issue
and needs no changes.

## Fix 2: Comprehensive Local-First Caching for `CachingS3Client`

Add a `force_refresh: bool` parameter (default `True`) to control whether S3 is
queried. Batch workloads pass `False`; live streaming callers use the default
`True`.

### `list_hls_folders(force_refresh=True)`

- `force_refresh=False`: scan local directories only (existing `has_ts` filter),
  skip S3 entirely
- `force_refresh=True`: current behavior (S3 + local merge)

### `list_segments(force_refresh=True)`

- **New `.segments.json` manifest:** When S3 returns a non-empty segment key
  list, write `.segments.json` alongside `.404.json` (contains segment key list +
  `cached_at_utc`). On subsequent calls with `force_refresh=False`, if
  `.segments.json` exists, return those keys merged with local `.ts` files — no
  S3 call.
- `.404.json` continues to short-circuit regardless of `force_refresh`.
- `force_refresh=True`: current behavior (always query S3, rewrite manifest).

### `fetch_playlist()` marker check

- **New:** Check for `live.m3u8.404.json` before calling S3. If present, return
  `None` immediately. Applies regardless of `force_refresh` — the marker is
  authoritative.

### Threading through the call chain

`ArchiveProvider` protocol does NOT change — `build_timeline()` signature stays
the same. Instead, `CachingHLSProvider` stores `force_refresh` as instance state
set at construction time. `build_orcasound_detection_provider()` gains an optional
`force_refresh` kwarg that flows to the `CachingHLSProvider` constructor. The
provider passes it to its `CachingS3Client` methods internally.

`OrcasoundHLSProvider` and `LocalHLSCacheProvider` ignore the parameter (they
have no S3 cache to refresh). No protocol change means no downstream breakage.

**Callers:**
- Region detection worker: passes `force_refresh=False` when building the provider
- All other callers (hydrophone detector, timeline viewer, etc.): unchanged default `True`

### Result

After one full run, every visited folder has either `.404.json`, `.segments.json`,
or local `.ts` files. A subsequent run with `force_refresh=False` makes zero S3
calls.

## Fix 3: Collapsed into Fix 2

No standalone work. Self-healing: existing orphaned folders (with only
`live.m3u8.404.json`) are handled by:

1. `list_hls_folders(force_refresh=False)` excludes them (no `.ts` files)
2. If discovered via `force_refresh=True`, `list_segments()` writes `.404.json`
   or `.segments.json` on first visit
3. `fetch_playlist()` now checks `live.m3u8.404.json` marker

## Fix 4: Lookback Early Termination

**Where:** `_build_stream_timeline()` in `s3_stream.py`

Extend the existing jump-to-max-lookback condition from requiring `timeline`
non-empty to requiring `found_any_folders` True:

```
# Before:
if timeline and not jumped_to_max_lookback ...
# After:
if found_any_folders and not jumped_to_max_lookback ...
```

When folders exist in the neighborhood but no segments overlap the range, skip
the 40 intermediate lookback steps and jump directly to max lookback (7 days) for
one final check. Reduces worst case from 42 iterations to 2.

**Safety:** The max-lookback check still runs, so a long HLS session (up to ~6
hours observed) started days before the range would still be found.

## Scope

- Backend only — no frontend changes
- No database migrations
- No changes to the `ArchiveProvider` protocol signature
- No changes to the regular hydrophone detection path (already efficient)
- No changes to NOAA/GCS providers (different caching architecture)

## Testing Strategy

- Unit tests for `list_hls_folders(force_refresh=False)` returning local-only results
- Unit tests for `.segments.json` manifest write/read cycle in `list_segments()`
- Unit tests for `fetch_playlist()` respecting `live.m3u8.404.json` marker
- Unit test for `_build_stream_timeline` early termination with `found_any_folders`
- Unit test for `iter_audio_chunks` accepting a pre-built `timeline` parameter
- Integration-level test: region detection worker builds timeline once and skips
  empty chunks without S3 calls
