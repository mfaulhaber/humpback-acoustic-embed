# Timeline Spectrogram Brightness Implementation Plan

**Goal:** Make Lifted Ocean the default shared timeline spectrogram renderer while unifying classifier and region timelines onto one hydrophone-span tile repository with better cache reuse and rendering performance.
**Spec:** [docs/specs/2026-05-02-timeline-spectrogram-brightness-design.md](../specs/2026-05-02-timeline-spectrogram-brightness-design.md)

---

### Task 1: Add renderer abstraction and Lifted Ocean default

**Files:**
- Create: `src/humpback/processing/timeline_renderers.py`
- Modify: `src/humpback/processing/timeline_tiles.py`
- Modify: `tests/unit/test_timeline_tiles.py`
- Create: `tests/unit/test_timeline_renderers.py`

**Acceptance criteria:**
- [x] `TimelineTileRenderer` abstract base class exposes stable `renderer_id`, `version`, and `render(...) -> bytes` behavior for timeline PNG tiles
- [x] `OceanDepthRenderer` preserves the current Ocean Depth palette and value mapping as an unused compatibility renderer
- [x] `LiftedOceanRenderer` is the default renderer and uses the approved lifted palette and value transform
- [x] Lifted Ocean keeps the current PCEN parameters unchanged
- [x] `generate_timeline_tile` remains available as a backward-compatible wrapper around the default renderer, or all call sites are migrated in the same task
- [x] Renderer output is marker-free PNG bytes with no axes, labels, or padding
- [x] Renderer id and version are deterministic and suitable for cache keys

**Tests needed:**
- Ocean Depth low, midpoint, and high color behavior remains covered for compatibility
- Lifted Ocean low values are visibly above pure black, high values clamp, and mapping is monotonic
- Lifted Ocean renders brighter than Ocean Depth for a fixed synthetic PCEN-like matrix
- Existing tile PNG smoke tests still pass for non-empty audio, empty audio, silence, warm-up, and frequency cropping

---

### Task 2: Introduce shared hydrophone-span tile repository

**Files:**
- Modify: `src/humpback/processing/timeline_cache.py`
- Create: `src/humpback/processing/timeline_repository.py`
- Modify: `tests/unit/test_timeline_cache.py`
- Create: `tests/unit/test_timeline_repository.py`
- Modify: `tests/unit/test_timeline_cache_migration.py`

**Acceptance criteria:**
- [x] `TimelineSourceRef` captures hydrophone id, archive/cache source identity, job start timestamp, and job end timestamp for both detection jobs and region detection jobs
- [x] Repository paths are keyed by a deterministic hydrophone-span key rather than consumer job id
- [x] Repository paths include renderer id, renderer version, zoom level, frequency range, and tile pixel dimensions
- [x] Classifier and region source refs for the same hydrophone/source/time span resolve to the same span key
- [x] Different renderer ids, renderer versions, frequency ranges, or tile dimensions resolve to different tile paths
- [x] Existing job-id cache paths remain readable only where needed for migration or export compatibility; new writes go to the shared span repository
- [x] Audio manifest persistence is keyed by the same hydrophone-span identity so compatible consumers do not duplicate manifests
- [x] Cache version handling is renderer-aware and does not require deleting unrelated renderer outputs

**Tests needed:**
- Same hydrophone/source/time-span refs produce identical span keys across classifier and region-style fixtures
- Different hydrophone, source path, start/end time, renderer id, renderer version, frequency range, and tile size each produce distinct cache paths
- Repository `get` after `put` returns exact bytes
- Stale legacy cache migration still removes old job-id tiles and legacy sidecars without affecting shared renderer-versioned outputs
- Audio manifest round-trip works through the shared span key

---

### Task 3: Move tile rendering into a shared service with cache locking

**Files:**
- Create: `src/humpback/services/timeline_tile_service.py`
- Modify: `src/humpback/api/routers/timeline.py`
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `tests/integration/test_timeline_api.py`
- Create: `tests/integration/test_region_timeline_cache.py`
- Create: `tests/unit/test_timeline_tile_service.py`

**Acceptance criteria:**
- [x] Shared service exposes `get_or_render_tile` for any `TimelineSourceRef`
- [x] Classifier detection tile endpoint uses the shared service and shared repository
- [x] Region detection tile endpoint uses the shared service and shared repository
- [x] Region tile endpoint checks disk cache before rendering
- [x] First region tile miss renders and stores the tile; repeated request returns cached bytes
- [x] Tile rendering passes timeline cache/repository context into `resolve_timeline_audio` so HLS manifests and decoded PCM caches can be reused
- [x] Same-tile concurrent cache misses are guarded by an in-process or advisory lock so only one render writes a given tile when practical
- [x] Existing frontend URL contracts remain unchanged
- [x] Existing audio-slice and playback behavior is unchanged

**Tests needed:**
- Classifier tile endpoint still returns PNG bytes for valid tile requests
- Region tile endpoint returns PNG bytes and then hits cache on the second request with a fake renderer or spy
- Invalid zoom and out-of-range tile behavior remains unchanged
- Concurrent same-tile requests do not produce corrupt or partial tile files
- `resolve_timeline_audio` receives cache context for region tile rendering in a targeted unit or integration test

---

### Task 4: Generalize prepare, status, and neighbor prefetch

**Files:**
- Modify: `src/humpback/services/timeline_tile_service.py`
- Modify: `src/humpback/api/routers/timeline.py`
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `src/humpback/services/timeline_export.py`
- Modify: `tests/integration/test_timeline_api.py`
- Modify: `tests/unit/test_timeline_export.py`

**Acceptance criteria:**
- [x] Existing classifier prepare endpoint delegates to the shared service while preserving response shape
- [x] Existing classifier prepare-status endpoint reads shared repository progress for the source span
- [x] Neighbor prefetch still launches after classifier tile misses
- [x] Region tile misses can use the same neighbor-prefetch mechanism, even if no public region prepare endpoint is added yet
- [x] `timeline_prepare_workers` remains the single setting controlling multi-threaded tile preparation
- [x] Export prepares and copies tiles from the shared repository path
- [x] Export remains compatible with classifier detection jobs and does not require frontend changes

**Tests needed:**
- Classifier startup prepare and prepare-status integration tests still pass
- Neighbor prefetch writes adjacent tiles into the shared repository
- Export prepares missing shared tiles and copies the expected tile count
- Region tile miss can schedule adjacent tile preparation when configured

---

### Task 5: Optimize marker-free PNG rendering path

**Files:**
- Modify: `src/humpback/processing/timeline_renderers.py`
- Modify: `src/humpback/processing/timeline_tiles.py`
- Modify: `tests/unit/test_timeline_renderers.py`
- Modify: `tests/unit/test_timeline_tiles.py`
- Create: `tests/performance/test_timeline_rendering_performance.py`

**Acceptance criteria:**
- [x] Renderer avoids per-tile Matplotlib figure creation for marker-free tiles, using direct NumPy/Pillow color mapping and PNG encoding where feasible
- [x] Output dimensions exactly match configured tile width and height
- [x] Frequency axis orientation remains visually compatible with current `origin="lower"` behavior
- [x] Bicubic or equivalent resize/interpolation behavior is preserved closely enough for visual review
- [x] Performance benchmark can compare current-style rendering and optimized rendering on a small fixed sample without requiring network access
- [x] If direct rendering causes unacceptable visual differences, the code keeps Matplotlib behind a renderer implementation and records benchmark results in comments or test naming

**Tests needed:**
- Renderer output has exact requested dimensions
- Synthetic low-frequency and high-frequency bands appear in expected vertical positions
- Direct renderer emits valid PNG bytes for empty audio, silence, and synthetic signal
- Performance test is opt-in or lightweight enough not to destabilize normal CI

---

### Task 6: Documentation and reference updates

**Files:**
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/signal-processing.md`
- Modify: `docs/reference/frontend.md`
- Modify: `docs/specs/2026-05-02-timeline-spectrogram-brightness-design.md`

**Acceptance criteria:**
- [x] Storage layout documents shared span-keyed tile repository paths and renderer-versioned subdirectories
- [x] Signal processing reference documents Lifted Ocean as the default timeline display renderer and Ocean Depth as compatibility renderer
- [x] Frontend reference notes that timeline consumers keep existing URL contracts while backend tile storage is shared by hydrophone span
- [x] Spec status is updated from Draft to Implemented or Accepted as appropriate after implementation
- [x] Documentation mentions that renderer id/version are part of cache identity

**Tests needed:**
- Documentation links resolve locally
- Grep confirms no reference claims region timeline tiles are uncached after implementation

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/processing/timeline_tiles.py src/humpback/processing/timeline_renderers.py src/humpback/processing/timeline_cache.py src/humpback/processing/timeline_repository.py src/humpback/services/timeline_tile_service.py src/humpback/api/routers/timeline.py src/humpback/api/routers/call_parsing.py src/humpback/services/timeline_export.py tests/unit/test_timeline_tiles.py tests/unit/test_timeline_renderers.py tests/unit/test_timeline_cache.py tests/unit/test_timeline_repository.py tests/unit/test_timeline_tile_service.py tests/integration/test_timeline_api.py tests/integration/test_region_timeline_cache.py tests/unit/test_timeline_export.py tests/performance/test_timeline_rendering_performance.py`
2. `uv run ruff check src/humpback/processing/timeline_tiles.py src/humpback/processing/timeline_renderers.py src/humpback/processing/timeline_cache.py src/humpback/processing/timeline_repository.py src/humpback/services/timeline_tile_service.py src/humpback/api/routers/timeline.py src/humpback/api/routers/call_parsing.py src/humpback/services/timeline_export.py tests/unit/test_timeline_tiles.py tests/unit/test_timeline_renderers.py tests/unit/test_timeline_cache.py tests/unit/test_timeline_repository.py tests/unit/test_timeline_tile_service.py tests/integration/test_timeline_api.py tests/integration/test_region_timeline_cache.py tests/unit/test_timeline_export.py tests/performance/test_timeline_rendering_performance.py`
3. `uv run pyright src/humpback/processing/timeline_tiles.py src/humpback/processing/timeline_renderers.py src/humpback/processing/timeline_cache.py src/humpback/processing/timeline_repository.py src/humpback/services/timeline_tile_service.py src/humpback/api/routers/timeline.py src/humpback/api/routers/call_parsing.py src/humpback/services/timeline_export.py`
4. `uv run pytest tests/unit/test_timeline_tiles.py tests/unit/test_timeline_renderers.py tests/unit/test_timeline_cache.py tests/unit/test_timeline_repository.py tests/unit/test_timeline_tile_service.py tests/unit/test_timeline_cache_migration.py tests/integration/test_timeline_api.py tests/integration/test_region_timeline_cache.py tests/unit/test_timeline_export.py tests/performance/test_timeline_rendering_performance.py`
5. `uv run pytest tests/`
6. `cd frontend && npx tsc --noEmit`

Verification results:

- `uv run ruff format --check ...` passed
- `uv run ruff check ...` passed
- `uv run pyright ...` passed
- `uv run pytest ...` focused timeline/export suite passed: 99 passed, 1 skipped
- `uv run pytest tests/` passed: 2342 passed, 2 skipped
- `cd frontend && npx tsc --noEmit` passed

Manual verification:

1. Open `/app/sequence-models/masked-transformer/23722a75-3490-44c1-a430-69404e891eff?k=150`.
2. Center the timeline at `2021-10-31 03:17:00 UTC`.
3. Check `5m`, `1m`, `30s`, and `10s` zooms for Lifted Ocean brightness and overlay readability.
4. Reload the page and confirm region tiles return from shared disk cache.
5. Inspect `data/timeline_cache/spans/` and confirm compatible classifier and region consumers share one span cache per renderer/version/frequency/zoom geometry.
