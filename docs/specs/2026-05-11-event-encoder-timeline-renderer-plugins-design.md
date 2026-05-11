# Event Encoder Timeline Renderer Plugins - Design

**Date:** 2026-05-11
**Status:** Draft
**Primary domain:** Signal Timeline
**Neighbor domains:** Sequence Models, Frontend Shell, Call Parsing

## 1. Goal

Add a selectable timeline spectrogram renderer path so the Event Encoder job
detail timeline can experiment with DSP-heavy hydrophone visualization without
changing playback, machine-learning feature extraction, or the global default
timeline look.

The first experiment should target noisy Orcasound hydrophone recordings where
boat or propeller background energy makes whale vocalizations difficult to scan.
It should keep the current PCEN tile renderer available, then add an
experimental tonal/ridge-enhanced renderer that suppresses stationary and
broadband vertical noise while preserving whale call ridges.

## 2. Triggering Example

User reference:

- Event Encoder job:
  `dcb6da9c-3ca3-4c04-b961-73444307915d`
- Upstream region detection job:
  `8aced89b-32e7-4582-a82e-c81d7ec8ef26`
- Hydrophone:
  `rpi_orcasound_lab`
- Source span:
  `2021-10-31 00:00:00 UTC` to `2021-11-01 00:00:00 UTC`
- Selected Event Encoder event:
  event 16 of 1273 at `2021-10-31 02:12:43.640 UTC`
- Selected token:
  `T03`, confidence `0.702`
- Selected-event descriptors:
  duration `1.6 s`, peak frequency about `468.8 Hz`, median F0 about
  `468.2 Hz`, log energy about `-8.909`

Local audio inspection of the 30 s viewport around `02:12:30 UTC` found the
strongest Welch power bands clustered around `296.9 Hz` to `351.6 Hz`, with a
secondary selected-call band around `460.9 Hz`. This matches the screenshot:
whale ridges are present, but recurring vertical broadband/propeller texture is
boosted alongside them by the current per-frequency whitening renderer.

## 3. Current Architecture

The timeline renderer architecture is partially present but not yet exposed as
a user-selectable plugin system.

Backend:

- `src/humpback/processing/timeline_renderers.py` defines
  `TimelineTileRenderer` with `renderer_id`, `version`, `pcen_params()`,
  `cache_metadata()`, `render()`, `display_values()`, and `encode_png()`.
- Existing renderers:
  - `OceanDepthRenderer`, compatibility renderer, `renderer_id="ocean-depth"`.
  - `LiftedOceanRenderer`, brighter PCEN display baseline.
  - `PerFrequencyWhitenedOceanRenderer`, current default,
    `renderer_id="per-frequency-whitened-ocean"`, version `3`.
- `TimelineTileRepository` already includes `renderer_id` and
  `renderer_version` in the disk cache key:
  `spans/{span_key}/{renderer_id}/v{version}/{zoom}/f{min}-{max}/w{width}_h{height}/tile_NNNN.png`.
- `get_or_render_tile()` already accepts a `renderer` argument.
- Public tile endpoints do not accept a renderer id, so the API always uses
  `DEFAULT_TIMELINE_RENDERER`.

Frontend:

- `Spectrogram` and `TileCanvas` are already renderer-agnostic: they draw URLs
  from a caller-provided `tileUrlBuilder`.
- `EventEncoderTimelinePanel` uses region-backed tiles via
  `regionTileUrl(timeline.region_detection_job_id, zoomLevel, tileIndex)`.
- The Event Encoder timeline hardcodes `freqRange={[0, 3000]}`.
- `regionTileUrl()` currently ignores `freq_min` and `freq_max`, unlike
  `timelineTileUrl()`, so Event Encoder cannot currently experiment with a
  0-5 kHz view like the user-provided reference image.

## 4. Research Notes

These notes only inform display rendering. They do not justify changing audio
playback or ML feature extraction.

- PCEN remains the right base normalization. Librosa documents
  `librosa.pcen` as automatic gain control followed by nonlinear compression,
  and as an alternative to log-amplitude scaling for emphasizing foreground
  signals over background. The existing project PCEN path already follows this
  pattern.
  Source: https://librosa.org/doc/latest/generated/librosa.pcen.html
- Lostanlen et al. frame PCEN as a frontend for far-field audio and
  bioacoustic sound recognition, with per-channel adaptive gain control useful
  for background-noise variation.
  Source: https://www.lostanlen.com/pubs/lostanlen2019spl/
- Passive acoustic tooling commonly uses FFT-domain noise removal such as
  median filtering, average subtraction, Gaussian smoothing, and thresholding
  to clean spectrogram displays before detection or review. These are natural
  display-only experiments for hydrophone timelines.
  Source: https://www.pamguard.org/olhelp/sound_processing/fftManagerHelp/docs/noise_removal.html
- HPSS-style spectrogram decomposition uses median filtering along time and
  frequency axes to separate horizontal tonal/harmonic structure from vertical
  percussive/transient structure. That maps well to this sample: whale ridges
  should be retained while propeller-like vertical texture should be dimmed.
  Source: https://librosa.org/doc/latest/generated/librosa.decompose.hpss.html
- Vessel noise can mask whale signals, especially where low-frequency vessel
  energy overlaps vocalizations. This supports a renderer whose goal is
  human-review contrast, not source-faithful amplitude display.
  Source: https://sanctuaries.noaa.gov/science/condition/sbnms/2020-report-content-summaries.html

## 5. Scope

### In scope

- Add a renderer registry around the existing `TimelineTileRenderer` classes.
- Add an optional renderer id query parameter to classifier and region tile
  endpoints.
- Keep renderer id and version in cache identity, preserving current cache
  safety.
- Add an experimental Event Encoder timeline renderer selector.
- Add an experimental tonal/ridge-enhanced renderer for hydrophone review.
- Teach Event Encoder region tile URLs to pass frequency range and renderer id.
- Allow Event Encoder timeline to switch between 0-3 kHz and 0-5 kHz review
  bands.
- Add focused backend and frontend tests.
- Update Signal Timeline and Sequence Models documentation/capsules if the
  renderer plugin contract becomes a domain-local rule.

### Non-goals

- No changes to playback audio normalization.
- No changes to classifier, Perch, CRNN, Continuous Embedding, or Event Encoder
  feature extraction.
- No mutation of `event_tokens.parquet`, `event_vectors.parquet`,
  `events.parquet`, or `typed_events.parquet`.
- No global default renderer change in this first experiment.
- No persistent user preference storage in v1. Local component state or URL
  search params are enough for experimentation.
- No database migration.

## 6. Approaches Considered

### Approach A: Backend Renderer Registry plus Event Encoder Selector

Add a registry of renderer plugins on the backend, expose renderer selection as
an optional tile query parameter, and add an Event Encoder-only selector that
switches the region tile URL.

Pros:

- Uses the architecture already in place.
- Keeps cache keys renderer-version-aware.
- Allows safe A/B review on one surface before promoting globally.
- Keeps frontend tile composition simple: a tile URL changes, then
  `TileCanvas` reloads the image.
- Lets classifier detection timelines and call-parsing region timelines share
  the same renderer registry later.

Cons:

- Requires plumbing renderer id through tile endpoints and prepare/prefetch
  helpers.
- Each renderer creates its own cache tree for the same hydrophone span.
- The frontend needs a small control for renderer and frequency range.

Verdict: recommended.

### Approach B: Frontend-Only Canvas Filters

Apply CSS/canvas brightness, contrast, gamma, or color filters after drawing
the existing PNG tiles.

Pros:

- Fastest prototype.
- No backend cache changes.

Cons:

- Cannot recover call ridges that were already compressed or color-mapped into
  the same pixel range as background noise.
- Cannot do frequency-wise background subtraction or vertical-burst
  suppression because the frontend receives RGB pixels, not the spectrogram
  matrix.
- Makes visual semantics implicit in the canvas rather than the renderer cache
  identity.

Verdict: reject for the main experiment.

### Approach C: Event Encoder-Only Tile Proxy

Create a new Sequence Models endpoint like
`/sequence-models/event-encoders/{job_id}/tile` that proxies to the upstream
region job but hardcodes the experimental renderer.

Pros:

- Very narrow UI blast radius.
- Avoids touching classifier detection tile URLs.

Cons:

- Duplicates tile endpoint validation and prefetch logic.
- Hides renderer selection inside Event Encoder rather than making it a
  reusable timeline capability.
- Undermines the existing span-oriented shared tile repository.

Verdict: not recommended unless Approach A proves too invasive.

### Approach D: Promote the New Renderer Globally

Replace `DEFAULT_TIMELINE_RENDERER` with the tonal/ridge-enhanced renderer.

Pros:

- Simple UI: no selector.
- All timeline surfaces benefit if the renderer is clearly better.

Cons:

- Too risky for the first pass. Different review tasks may prefer source-like
  amplitude context over aggressive noise suppression.
- Global cache footprint and visual behavior change immediately.

Verdict: defer. Promotion should happen only after Event Encoder review proves
the renderer useful.

## 7. Recommended Design

### 7.1 Renderer Registry

Extend `src/humpback/processing/timeline_renderers.py` or add a nearby module
with a registry:

- `TimelineRendererOption`
  - `renderer_id`
  - `version`
  - `display_name`
  - `is_default`
  - `is_experimental`
  - `recommended_freq_max`
- `DEFAULT_TIMELINE_RENDERER_ID = "per-frequency-whitened-ocean"`
- `TIMELINE_RENDERERS: dict[str, TimelineTileRenderer]`
- `get_timeline_renderer(renderer_id: str | None) -> TimelineTileRenderer`
- `list_timeline_renderer_options() -> list[TimelineRendererOption]`

Validation rules:

- Missing renderer id resolves to the current default renderer.
- Unknown renderer id returns HTTP `422` from public API routes.
- Renderer classes must have stable `renderer_id` and `version`.
- Renderer version increments whenever cached pixels for that renderer would
  change.

### 7.2 Tile Endpoint Selection

Add an optional query parameter:

- `renderer: str | None = Query(None)`

Apply it to:

- `GET /classifier/detection-jobs/{job_id}/timeline/tile`
- `GET /call-parsing/region-jobs/{job_id}/tile`

Both endpoints should resolve the renderer id once, pass the renderer object to
`get_or_render_tile()`, and pass the same renderer through neighbor prefetch.

Prepare behavior:

- Extend the internal prepare helpers so startup/full prepare can accept a
  renderer.
- Preserve default prepare behavior for existing classifier timeline flows.
- For renderer-selected tile misses, neighbor prefetch must render neighbors
  with the selected renderer, not the default renderer.
- If adding renderer selection to full prepare is too much for v1, document
  that non-default renderers prepare only through on-demand and neighbor
  prefetch. The tile request path itself must still be correct.

### 7.3 Renderer Options API

Add a lightweight shared endpoint:

- `GET /timeline/renderers`

Response:

- ordered renderer options from the backend registry
- default renderer first
- experimental renderers included, because the Event Encoder selector is an
  experiment surface

This avoids hardcoding backend renderer ids only in frontend code. If a new
top-level timeline router is undesirable during implementation, a local
frontend constant is acceptable for the first iteration, but the registry
should still exist backend-side.

### 7.4 Event Encoder UI

In `EventEncoderTimelinePanel`:

- Add local state for `rendererId`, defaulting to backend default or
  `"per-frequency-whitened-ocean"`.
- Add local state for frequency max, initially `3000`, with `5000` as an
  experiment option.
- Update `tileUrlBuilder` to pass `freqMin`, `freqMax`, and `rendererId` into
  `regionTileUrl()`.
- Update `regionTileUrl()` to accept optional `freqMin`, `freqMax`, and
  `rendererId` query params.
- Keep playback and token navigation unchanged.
- Keep the selector scoped to the Event Encoder timeline toolbar. Do not add
  this control to all timeline surfaces yet.

Control labels:

- Default renderer: `PCEN`
- Experimental renderer: `Tonal`
- Frequency control: `3k` / `5k`

The UI should remain compact and tool-like. It should not add explanatory
paragraphs inside the app.

### 7.5 Experimental Tonal/Ridge Renderer

Add a renderer such as:

- class: `TonalRidgeOceanRenderer`
- id: `tonal-ridge-ocean`
- version: `1`
- experimental: `true`

Processing pipeline:

1. Resolve audio exactly as current timeline tiles do.
2. Compute STFT magnitude and PCEN using the existing `render_tile_pcen()`
   logic and existing PCEN settings.
3. Crop to the requested frequency range.
4. Estimate and subtract slow/stationary background per frequency:
   - rolling median over time, around `2-4 s`, or
   - low percentile per frequency when the tile is too short.
5. Estimate and dim broadband vertical transients:
   - subtract a per-frame low/mid percentile across frequency, or
   - use an HPSS-style soft mask comparing horizontal median energy with
     vertical median energy.
6. Normalize local detail per frequency using robust percentiles, for example
   background `20th` and foreground `97th`.
7. Blend with a floor from the current PCEN display so the viewer still shows
   context and does not become a binary edge detector.
8. Encode with a sequential blue/green/yellow palette where high ridges can
   reach pale yellow/white, closer to the provided reference spectrogram.

Important tuning guardrails:

- Preserve diagonal upsweeps and short chirps. HPSS masks must be soft and
  blended, not a hard harmonic-only gate.
- Keep enough background context to spot false-positive detections.
- Do not suppress the 250-700 Hz band where the inspected sample carries both
  vessel energy and the selected whale call.
- Do not change the underlying audio path or Event Encoder descriptors.

Prototype observations from the sample:

- Pure local per-frequency percentile remap improves contrast but keeps many
  vertical stripes.
- Rolling median residual produces a darker background and more visible call
  arcs, but needs blending so it does not over-edge the display.
- Broadband frame subtraction helps target vertical propeller texture but can
  damage true broadband call components if applied too aggressively.

The first implementation should favor conservative blending over maximum
denoising.

## 8. Data and Cache Contracts

- Renderer id and version remain part of the disk cache identity.
- Frequency range remains part of the disk cache identity.
- Tile URLs include renderer id so `TileCanvas`'s in-memory image cache
  naturally separates renderer outputs.
- Existing cached default tiles remain valid.
- Experimental renderer caches should coexist under the same span key.
- The renderer should be display-only; it must not write sidecar artifacts or
  mutate source audio.

## 9. Testing Strategy

Backend unit tests:

- Registry lists default and experimental renderers with stable ids and
  versions.
- Unknown renderer id raises the route-level validation error.
- `TonalRidgeOceanRenderer` returns a valid PNG with exact requested
  dimensions.
- Renderer output differs from the default for a synthetic mixture containing
  tonal chirps plus broadband vertical pulses.
- Synthetic tonal chirp contrast is higher after tonal/ridge enhancement while
  frame-wide broadband pulse contrast is lower than in the default display.
- Cache paths differ by renderer id/version and frequency range.

Backend integration tests:

- Classifier timeline tile endpoint accepts `renderer=...` and returns PNG.
- Region timeline tile endpoint accepts `renderer=...&freq_min=0&freq_max=5000`
  and returns PNG.
- Region tile miss launches neighbor prefetch with the selected renderer.
- Existing default tile requests remain backward compatible.

Frontend tests:

- `regionTileUrl()` includes optional `freq_min`, `freq_max`, and `renderer`
  params when supplied.
- Event Encoder timeline tile builder sends selected renderer and frequency
  range.
- Event Encoder renderer/frequency selector changes tile URLs without
  disturbing selected event, selected k, playback controls, or token-scoped
  navigation.
- Existing `EventEncoderTokenOverlay` tests remain unchanged.

Manual verification:

- Open Event Encoder job `dcb6da9c-3ca3-4c04-b961-73444307915d`.
- Center around event 16 at `2021-10-31 02:12:43.640 UTC`.
- Compare `PCEN` and `Tonal` renderers at `30s` and `10s` zoom.
- Confirm the `T03` event near 469 Hz and nearby `T16` events near 300 Hz are
  easier to scan in `Tonal`.
- Confirm propeller-like vertical texture is dimmer but still visible enough
  to understand recording quality.
- Compare `3k` and `5k` frequency ranges.

Suggested targeted commands for implementation:

1. `uv run pytest tests/unit/test_timeline_renderers.py tests/unit/test_timeline_repository.py tests/unit/test_timeline_tile_service.py -q`
2. `uv run pytest tests/integration/test_timeline_api.py tests/integration/test_region_timeline_cache.py -q`
3. `uv run pytest tests/integration/test_sequence_models_api.py -q`
4. `cd frontend && npx vitest run src/components/sequence-models/EventEncoderTokenOverlay.test.tsx src/components/sequence-models/eventEncoderTimelineNavigation.test.ts`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`

## 10. Open Questions

- Should the renderer selector state be shareable through URL search params
  immediately, or remain local component state for v1?
- Should 0-5 kHz be limited to Event Encoder review zooms (`5m`, `1m`, `30s`,
  `10s`) to avoid forcing high sample rates for coarse full-day timelines?
- Should the first tonal renderer use a custom HPSS-style implementation over
  the existing PCEN matrix, or call `librosa.decompose.hpss()` directly?
- After Event Encoder review, should `TonalRidgeOceanRenderer` become available
  on Call Parsing Segment/Classify review surfaces?

## 11. Recommendation

Implement Approach A. The codebase already has most of the backend plugin
foundation: renderer classes, renderer ids, renderer versions, and cache
identity. The missing pieces are registry lookup, public renderer selection,
and an Event Encoder UI control.

Keep the current `PerFrequencyWhitenedOceanRenderer` as the default. Add
`TonalRidgeOceanRenderer` as an experimental display-only plugin and expose it
only in the Event Encoder timeline first. This gives us a low-risk place to
test whether rolling background subtraction plus soft tonal/ridge enhancement
actually makes whale calls easier to review in boat-noisy hydrophone audio.
