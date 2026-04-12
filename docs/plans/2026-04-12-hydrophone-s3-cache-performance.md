# Hydrophone S3 Cache Performance — Implementation Plan

**Goal:** Eliminate redundant S3 API calls during hydrophone region detection by building the timeline once per job, adding local-first caching to `CachingS3Client`, and terminating the lookback loop early.
**Spec:** [docs/specs/2026-04-12-hydrophone-s3-cache-performance-design.md](../specs/2026-04-12-hydrophone-s3-cache-performance-design.md)

---

### Task 1: Lookback early termination (Fix 4)

**Files:**
- Modify: `src/humpback/classifier/s3_stream.py`
- Modify: `tests/unit/test_s3_stream.py`

**Acceptance criteria:**
- [ ] In `_build_stream_timeline`, the jump-to-max-lookback condition uses `found_any_folders` instead of `timeline`
- [ ] When folders exist but no segments overlap the range, lookback terminates in at most 2 iterations (initial + max lookback) instead of 42

**Tests needed:**
- Test `_build_stream_timeline` with a mock client that returns folders with no overlapping segments — verify it calls `list_hls_folders` at most twice (initial + max lookback), not 42 times

---

### Task 2: `fetch_playlist()` 404 marker check (Fix 2a)

**Files:**
- Modify: `src/humpback/classifier/s3_stream.py`
- Modify: `tests/unit/test_s3_stream.py`

**Acceptance criteria:**
- [ ] `CachingS3Client.fetch_playlist()` checks for `live.m3u8.404.json` marker before calling S3
- [ ] If marker exists, returns `None` immediately without network call
- [ ] Existing behavior unchanged when marker does not exist

**Tests needed:**
- Test `fetch_playlist` returns `None` without S3 call when `live.m3u8.404.json` exists
- Test `fetch_playlist` still fetches from S3 when no marker exists (existing behavior preserved)
- Test `fetch_playlist` still returns from local `live.m3u8` cache when present (existing behavior preserved)

---

### Task 3: `list_segments()` manifest + `force_refresh` (Fix 2b)

**Files:**
- Modify: `src/humpback/classifier/s3_stream.py`
- Modify: `tests/unit/test_s3_stream.py`

**Acceptance criteria:**
- [ ] `CachingS3Client.list_segments()` accepts `force_refresh: bool = True` parameter
- [ ] When S3 returns a non-empty segment list, writes `.segments.json` manifest (list of segment keys + `cached_at_utc`) in the folder directory
- [ ] With `force_refresh=False` and `.segments.json` present, returns manifest keys merged with local `.ts` files — no S3 call
- [ ] `.404.json` continues to short-circuit regardless of `force_refresh` value
- [ ] With `force_refresh=True`, queries S3 and rewrites manifest (existing behavior plus manifest write)

**Tests needed:**
- Test `.segments.json` is written when S3 returns non-empty segment list
- Test `force_refresh=False` reads `.segments.json` and returns keys without S3 call
- Test `force_refresh=False` with `.404.json` still short-circuits (no S3 call, no manifest needed)
- Test `force_refresh=True` re-queries S3 even when `.segments.json` exists
- Test `.segments.json` keys are merged with any local `.ts` files

---

### Task 4: `list_hls_folders()` `force_refresh` (Fix 2c)

**Files:**
- Modify: `src/humpback/classifier/s3_stream.py`
- Modify: `tests/unit/test_s3_stream.py`

**Acceptance criteria:**
- [ ] `CachingS3Client.list_hls_folders()` accepts `force_refresh: bool = True` parameter
- [ ] With `force_refresh=False`, scans local directories only (existing `has_ts` filter), skips S3 entirely
- [ ] With `force_refresh=True`, current behavior preserved (S3 + local merge)

**Tests needed:**
- Test `force_refresh=False` returns only local folders with `.ts` files, no S3 call
- Test `force_refresh=True` merges S3 and local results (existing test updated)
- Test `force_refresh=False` excludes folders with only `.404.json` or `live.m3u8.404.json`

---

### Task 5: Thread `force_refresh` through provider chain (Fix 2d)

**Files:**
- Modify: `src/humpback/classifier/providers/orcasound_hls.py`
- Modify: `src/humpback/classifier/providers/__init__.py`
- Modify: `tests/unit/test_s3_stream.py` (if needed for provider-level tests)

**Acceptance criteria:**
- [ ] `CachingHLSProvider.__init__` accepts optional `force_refresh: bool = True` parameter, stores as instance state
- [ ] `CachingHLSProvider.build_timeline()` passes stored `force_refresh` through to `_build_stream_timeline` which passes to `list_hls_folders` and `_ordered_folder_segments` (and onward to `list_segments`/`fetch_playlist`)
- [ ] `_build_stream_timeline` accepts optional `force_refresh: bool = True` parameter and threads it to client method calls
- [ ] `_ordered_folder_segments` accepts optional `force_refresh: bool = True` parameter and threads it to `list_segments` and `fetch_playlist` (via the client)
- [ ] `build_orcasound_detection_provider` accepts optional `force_refresh` kwarg, passes to `CachingHLSProvider`
- [ ] `build_archive_detection_provider` accepts optional `force_refresh` kwarg, passes to `build_orcasound_detection_provider`
- [ ] `OrcasoundHLSProvider` and `LocalHLSCacheProvider` are unaffected (no S3 cache to refresh)
- [ ] `ArchiveProvider` protocol signature unchanged

**Tests needed:**
- Test that `CachingHLSProvider` constructed with `force_refresh=False` passes it through to `CachingS3Client` methods during `build_timeline`
- Test that `build_archive_detection_provider(force_refresh=False)` creates a provider that doesn't call S3

---

### Task 6: `iter_audio_chunks` pre-built timeline parameter (Fix 1a)

**Files:**
- Modify: `src/humpback/classifier/s3_stream.py`
- Modify: `tests/unit/test_s3_stream.py`

**Acceptance criteria:**
- [ ] `_iter_audio_chunks()` accepts optional `timeline: list[StreamSegment] | None = None` parameter
- [ ] When `timeline` is provided, skips the internal `provider.build_timeline()` call
- [ ] When `timeline` is `None` (default), existing behavior preserved
- [ ] Parameter is surfaced through `iter_audio_chunks()` kwargs

**Tests needed:**
- Test that passing a pre-built timeline skips `build_timeline` (mock provider, verify `build_timeline` not called)
- Test that passing `timeline=None` still calls `build_timeline` (existing behavior)
- Test that pre-built timeline produces identical audio iteration to the default path

---

### Task 7: Region detection worker integration (Fix 1b + Fix 2 usage)

**Files:**
- Modify: `src/humpback/workers/region_detection_worker.py`

**Acceptance criteria:**
- [ ] `_load_hydrophone_trace` builds the provider with `force_refresh=False`
- [ ] `_load_hydrophone_trace` builds the full timeline once before the chunk loop via `provider.build_timeline(start_ts, end_ts)`
- [ ] For each chunk, segments are filtered from the pre-built timeline to those overlapping `[chunk_start, chunk_end]`
- [ ] Chunks with no overlapping segments are skipped immediately (log "no audio segments, skipping") without calling `iter_audio_chunks`
- [ ] Chunks with overlapping segments pass the filtered timeline to `iter_audio_chunks(timeline=...)`
- [ ] The `FileNotFoundError` catch block is preserved as a fallback for edge cases where the filtered timeline is non-empty but audio decoding fails

**Tests needed:**
- Test that `_load_hydrophone_trace` calls `build_timeline` exactly once (not per-chunk)
- Test that chunks with no overlapping segments are skipped without entering `iter_audio_chunks`
- Test that chunks with overlapping segments pass the filtered timeline

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/classifier/s3_stream.py src/humpback/classifier/providers/orcasound_hls.py src/humpback/classifier/providers/__init__.py src/humpback/workers/region_detection_worker.py`
2. `uv run ruff check src/humpback/classifier/s3_stream.py src/humpback/classifier/providers/orcasound_hls.py src/humpback/classifier/providers/__init__.py src/humpback/workers/region_detection_worker.py`
3. `uv run pyright src/humpback/classifier/s3_stream.py src/humpback/classifier/providers/orcasound_hls.py src/humpback/classifier/providers/__init__.py src/humpback/workers/region_detection_worker.py`
4. `uv run pytest tests/unit/test_s3_stream.py tests/unit/test_region_detection_worker.py -v`
5. `uv run pytest tests/`
