# Timeline Spectrogram Brightness Alternatives

**Date:** 2026-05-02
**Status:** Implemented

## Problem

The shared timeline viewer now uses PCEN-normalized backend PNG tiles, which fixes the earlier quiet/loud hydrophone sections better than the legacy gain-step and ref-db pipeline. The remaining issue is visual: tiles read too dark overall and the custom Ocean Depth palette feels low-energy, especially in the shared timeline surface used by masked-transformer, HMM, region review, and classifier timelines.

Reference view for evaluation:

- Page: `/app/sequence-models/masked-transformer/23722a75-3490-44c1-a430-69404e891eff?k=150`
- Region detection job: `8aced89b-32e7-4582-a82e-c81d7ec8ef26`
- Center time: `2021-10-31 03:17:00 UTC` (`1635650220`)
- Zooms checked: `5m`, `1m`, `30s`, `10s`

## Baseline Before Implementation

Backend:

- `src/humpback/processing/timeline_tiles.py`
  - `generate_timeline_tile` calls `render_tile_pcen`
  - crops to requested frequency range
  - renders with `get_ocean_depth_colormap()`
  - uses fixed `vmin=settings.pcen_vmin`, `vmax=settings.pcen_vmax`
  - current config defaults are `pcen_vmin=0.0`, `pcen_vmax=1.0`
  - interpolation is `bicubic`
- `src/humpback/processing/pcen_rendering.py`
  - STFT magnitude, not power, is passed into `librosa.pcen`
  - current PCEN defaults are `time_constant=2.0`, `gain=0.98`, `bias=2.0`, `power=0.5`
  - filter state is initialized from the first magnitude frame to avoid dark left-edge cold starts
- `src/humpback/api/routers/timeline.py` and `src/humpback/api/routers/call_parsing.py`
  - both classifier and region-detection timelines feed the same `generate_timeline_tile`
- Cache behavior was not standardized:
  - classifier detection timelines use `TimelineTileCache` rooted at `data/timeline_cache/{detection_job_id}/{zoom}/tile_####.png`
  - classifier tile misses render once, persist to disk, and launch neighbor prepare work
  - region-detection timelines render through `/call-parsing/region-jobs/{job_id}/tile`
  - region-detection tile misses rendered every request because `_render_region_tile_sync` bypassed `TimelineTileCache`
  - HMM, masked-transformer, segment review, classify review, and window-classify review all used `regionTileUrl`, so they shared that previously render-on-demand region path

Frontend:

- `frontend/src/components/timeline/TileCanvas.tsx`
  - draws backend PNGs directly onto the canvas
  - no brightness, contrast, gamma, CSS filter, or per-pixel post-processing is applied
  - missing tiles draw a dark placeholder
- `frontend/src/components/timeline/spectrogram/Spectrogram.tsx`
  - supplies the tile canvas and overlay context
  - consumers should not need new required props

The rendered 03:17 contact sheet showed current tile mean luminance around `0.115-0.129` across the checked zooms. Candidate remaps landed around `0.37-0.43`, which is a substantial readability increase.

## Required Architecture Changes

### One Shared Tile Repository

All timeline consumers should resolve tiles through one repository abstraction keyed by the underlying hydrophone time span, not by the UI consumer or model job that happens to request the image.

Proposed repository identity:

- `hydrophone_id`
- archive/cache source identity (`local_cache_path` or provider kind/cache root)
- `job_start_timestamp`
- `job_end_timestamp`
- renderer id and renderer version
- tile geometry (`zoom_level`, `tile_index`, `freq_min`, `freq_max`, tile pixel size)

The repository should expose a stable cache key such as:

- `data/timeline_cache/spans/{span_key}/{renderer_id}/v{renderer_version}/{zoom}/f{freq_min}-{freq_max}/w{width}_h{height}/tile_####.png`

`span_key` should be deterministic from the hydrophone/source/time-span fields, for example a short SHA-256 digest. This lets a classifier detection job, region detection job, HMM detail page, masked-transformer detail page, and review workspace reuse the same tile bytes when they point at the same hydrophone/time range.

Migration path:

1. Add a `TimelineTileRepository` wrapper around `TimelineTileCache` rather than changing every call site at once.
2. Teach the repository to resolve a `TimelineSourceRef` from either a classifier detection job or a region detection job.
3. Move classifier and region tile endpoints onto a shared `get_or_render_tile(source_ref, tile_request, renderer)` function.
4. Keep the current URL contracts initially (`timelineTileUrl` and `regionTileUrl`) so frontend composition stays stable.
5. Add shared prepare/status helpers for region jobs, or at minimum use the repository on region tile misses before adding full region prepare UI.

Acceptance criteria:

- region-detection tile endpoint checks the shared repository before rendering
- classifier detection endpoint stores and reads through the same repository
- the same hydrophone/time span requested by multiple consumers produces one disk tile set per renderer/version/frequency/zoom geometry
- audio manifests continue to be shared through the same span-oriented key, not duplicated per consumer job
- tile cache invalidation is renderer-version-aware, so introducing Lifted Ocean does not require deleting unrelated renderer outputs

### Renderer Abstraction

Introduce a renderer interface so visual style is explicit and swappable:

- new module: `src/humpback/processing/timeline_renderers.py`
- abstract base class: `TimelineTileRenderer`
- method: `render(audio, sample_rate, freq_min, freq_max, n_fft, hop_length, warmup_samples, width_px, height_px) -> bytes`
- properties:
  - `renderer_id: str`
  - `version: int`
  - `pcen_params(settings) -> PcenParams`
  - optional `cache_metadata(settings) -> dict[str, object]`

Renderer classes:

- `OceanDepthRenderer`
  - preserves today's renderer behavior
  - kept as an unused compatibility class for side-by-side experiments and future rollback
  - `renderer_id = "ocean-depth"`
  - version starts at the current cache-compatible value
- `LiftedOceanRenderer`
  - new default
  - same PCEN parameters as current
  - applies the lifted value transform and lifted palette described below
  - `renderer_id = "lifted-ocean"`
  - version starts at `1`

The public function `generate_timeline_tile` can either become a thin backward-compatible wrapper around `LiftedOceanRenderer().render(...)` or be replaced in callers after tests are migrated. The renderer id/version must be part of the shared repository key, so future per-consumer renderers can coexist without cache collisions.

Per-consumer swapping should be configured at the backend endpoint/service boundary, not in `TileCanvas`. The frontend should continue to ask for a tile; the server chooses a renderer unless a future explicit renderer query parameter is added for experiments.

## Research Notes

PCEN is still the right normalization foundation. Librosa documents PCEN as automatic gain control followed by nonlinear compression, intended to suppress background noise and emphasize foreground signals, and as an alternative to decibel scaling. Its `gain`, `bias`, `power`, and `time_constant` parameters affect the signal transform, so visual brightness should first be handled by value-to-color mapping unless we also want to change detection-relevant visual semantics.

Color-map guidance points toward sequential maps with monotonic lightness for ordered scalar data. Matplotlib’s colormap guidance specifically emphasizes perceptually uniform sequential maps because equal data steps should look like equal perceptual steps, and lightness changes are interpreted better than hue-only changes. Crameri’s scientific color maps make the same point for perceptual uniformity, color-vision deficiency, and grayscale robustness.

## Alternatives

### Option A: Lift Existing Ocean Depth Mapping

Keep the current visual identity but adjust the mapping:

- replace the very near-black lower stops with a dark blue that still has visible luminance
- apply a power/gamma normalization before color lookup, around `gamma=0.75-0.8`
- reduce the effective visual ceiling, roughly equivalent to mapping `pcen / 0.65`
- make this the new default renderer via `LiftedOceanRenderer`
- store it under its own renderer id/version in the shared tile repository

Observed preview:

- Good: preserves the aquatic/sonar feel, reveals background texture and calls without changing color language
- Good: lowest migration risk; existing tests around Ocean Depth can be updated rather than deleted
- Risk: can make broadband noise busier at fine zooms
- Risk: if too much of the range saturates to pale seafoam/white, quiet calls may compete with transient clicks

### Option B: Switch to a Built-In Perceptual Map

Use a Matplotlib sequential perceptual map, probably `magma` or `inferno`, plus a power norm and lower ceiling.

Observed preview:

- Good: more vivid and very readable; call traces pop strongly
- Good: built-in Matplotlib maps are stable, documented, and accessible without adding a dependency
- Risk: warm/pink palette feels less domain-specific and may clash with existing overlays and categorical token colors
- Risk: highlights can become too high-energy for a utilitarian review surface

### Option C: Add a New Domain Palette

Introduce a new custom sequential palette such as Aurora Sonar:

- low: near-black blue/violet
- mid: blue/teal
- high: green/yellow/cream
- pair with `gamma=0.75` and effective ceiling around `0.7`

Observed preview:

- Good: less boring than current Ocean Depth while still feeling hydrophone-native
- Good: yellow/cream peaks make chirps and harmonics easy to scan
- Risk: custom palette requires more accessibility testing than `magma`/`cividis`
- Risk: yellow highlights may compete with overlays unless overlay colors are checked in context

### Option D: Frontend-Only Canvas Filter

Leave backend tiles unchanged and add an optional `TileCanvas` display transform:

- CSS/canvas filter such as brightness/contrast/saturation
- or a post-draw canvas pass

Observed preview was not the primary path because frontend filters cannot recover a better data-to-color mapping; they brighten compressed pixels after color decisions have already been made.

- Good: fastest experiment, no cache invalidation
- Good: could become a temporary user preference toggle
- Risk: can wash out bright detail and alter overlay contrast
- Risk: every consumer sees a filtered image whose underlying tile bytes still encode the old visual contract

### Option E: Per-Consumer Renderer Selection

Use the renderer abstraction immediately to choose different styles per page or source kind.

- Good: supports experiments such as `magma` for masked-transformer and `lifted-ocean` for review
- Good: renderer id/version cleanly separates cached outputs
- Risk: undermines visual consistency if used too early
- Risk: increases cache footprint because each renderer/style creates its own tile set

For this session, keep `LiftedOceanRenderer` as the global default and leave per-consumer selection as an extension point.

## Recommendation

Implement Option A as the visual default, but do it inside the broader shared-repository and renderer-abstraction cleanup:

1. Create the renderer abstraction.
2. Preserve the current Ocean Depth behavior as `OceanDepthRenderer`, unused by default.
3. Add `LiftedOceanRenderer` as the new default.
4. Route classifier and region-detection tile endpoints through a shared tile repository keyed by hydrophone/source/time span plus renderer id/version.
5. Keep frontend tile composition unchanged.

Suggested first pass:

- add a small rendering helper that normalizes PCEN values before colormap lookup
- use `display = clip(pcen / 0.65, 0, 1) ** 0.78`
- replace `_OCEAN_DEPTH_COLORS` with lifted lower/mid stops:
  - `#07101d`
  - `#0d2441`
  - `#145579`
  - `#1fa08d`
  - `#6ee4b7`
  - `#effff7`
- keep PCEN parameters unchanged
- keep the frontend compositor unchanged
- do not rely only on the global `TIMELINE_CACHE_VERSION`; include renderer id/version in the cache path and increment any legacy cache version needed for old job-id paths

If Option A still feels too conservative after manual review, Option C is the best second candidate. Option B is a useful fallback if we want a documented built-in map and can accept the warmer visual identity.

## Rendering Performance Opportunities

Baseline performance observations before implementation:

- classifier timeline preparation already uses `ThreadPoolExecutor` controlled by `settings.timeline_prepare_workers`
- classifier tile misses persist to disk and launch neighbor prefetch
- region-detection tile requests did neither; repeated requests recomputed audio resolution, STFT, PCEN, colormap, and PNG encoding
- `generate_timeline_tile` uses Matplotlib figure creation for every tile, which is convenient but expensive for marker-free image tiles
- `timeline_audio` already has reusable HLS manifests and decoded PCM memory caching, but region rendering does not pass `timeline_cache`, `job_id`, or PCM cache limits into `resolve_timeline_audio`, so it misses some reuse available to classifier tiles

Performance changes to include in implementation:

1. **Cache region tiles.** Biggest immediate win. Moving region endpoints to the shared repository avoids repeated rendering across HMM/masked-transformer/review views.
2. **Generalize prepare/prefetch.** Reuse classifier's background prepare and neighbor-prefetch pipeline for any `TimelineSourceRef`, including region jobs.
3. **Keep multi-threaded prepare.** Preserve `timeline_prepare_workers`, but move the worker loop out of `api/routers/timeline.py` into a shared service so region jobs can use it.
4. **Avoid duplicate same-tile renders.** Add a per-tile advisory lock or in-process in-flight set keyed by repository tile path. Today simultaneous cache misses can compute the same tile before one write wins.
5. **Use audio manifest and PCM caches for region jobs.** Pass the shared repository/cache plus source key into `resolve_timeline_audio` so HLS manifests and decoded PCM are reused.
6. **Replace Matplotlib tile encoding with direct NumPy/Pillow rendering.** For marker-free tiles, the pipeline can map normalized scalar values through a `ListedColormap` lookup table and encode with PIL. This avoids per-tile `plt.subplots`, `imshow`, and `savefig` overhead. Keep output pixel-equivalent enough for tests but allow small interpolation differences if documented.
7. **Benchmark before changing PCEN/STFT.** STFT and PCEN are likely real CPU costs, but Matplotlib overhead and repeated region renders are lower-risk wins. Do not tune PCEN parameters for speed unless profiling shows they dominate.
8. **Keep process-level parallelism conservative.** STFT/librosa/scipy may use native libraries that release the GIL in places, but over-threading can contend with audio decode and backend request handling. Default worker count should remain small and configurable.

Suggested performance benchmark:

- render a fixed set of 03:17-centered tiles for `5m`, `1m`, `30s`, and `10s`
- measure cold cache and warm cache for classifier and region endpoints
- compare current Matplotlib renderer vs direct NumPy/Pillow renderer
- record wall time per tile, total prepare time, and cache hit latency

## Testing And Verification

Unit tests:

- update `test_ocean_depth_colormap_endpoints` and `test_ocean_depth_colormap_midpoint_is_teal`
- add tests for the display normalization helper:
  - monotonic mapping
  - low values lift above pure black
  - high values clamp at 1.0
  - finite output for zeros and silence
- existing tile tests continue to verify valid PNG output
- add renderer tests:
  - `OceanDepthRenderer` preserves current low/mid/high palette behavior
  - `LiftedOceanRenderer` is brighter than `OceanDepthRenderer` on a fixed synthetic PCEN matrix
  - renderer id/version are stable and included in repository keys
- add shared repository tests:
  - classifier and region source refs for the same hydrophone/time span resolve to the same span key
  - different renderer ids or versions resolve to different tile paths
  - different frequency ranges resolve to different tile paths
  - repeated region tile request hits cache after first render
- add concurrency test where two requests for the same missing tile render only once if practical with a fake renderer

Manual/browser verification:

- open the masked-transformer reference page and check `5m`, `1m`, `30s`, and `10s` centered on `2021-10-31 03:17:00 UTC`
- check at least one mostly quiet section and one louder/broadband section
- verify motif overlays, playhead, axes, and token strip still read clearly
- compare HMM detail and region-review timelines because they share the same tile renderer
- inspect the cache directory after loading classifier and region timelines for the same hydrophone/time span; confirm there is one shared span cache, not consumer-specific duplicates
- measure a warm reload of the masked-transformer page; region tiles should return from disk cache rather than re-rendering

Verification gates:

- `uv run ruff format --check src/humpback/processing/timeline_tiles.py src/humpback/processing/timeline_renderers.py src/humpback/processing/timeline_cache.py tests/unit/test_timeline_tiles.py tests/unit/test_timeline_cache.py`
- `uv run ruff check src/humpback/processing/timeline_tiles.py src/humpback/processing/timeline_renderers.py src/humpback/processing/timeline_cache.py tests/unit/test_timeline_tiles.py tests/unit/test_timeline_cache.py`
- `uv run pyright src/humpback/processing/timeline_tiles.py src/humpback/processing/timeline_renderers.py src/humpback/processing/timeline_cache.py`
- `uv run pytest tests/unit/test_timeline_tiles.py tests/unit/test_timeline_cache.py`
- broader `uv run pytest tests/` before session review

## Preview Artifact

Local preview contact sheet generated from actual app-served 03:17 tiles:

- `/tmp/humpback-timeline-previews/spectrogram_tile_remap_alternatives_0317.png`
