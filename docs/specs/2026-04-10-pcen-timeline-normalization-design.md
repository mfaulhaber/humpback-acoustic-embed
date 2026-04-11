# PCEN Timeline Normalization

**Date:** 2026-04-10
**Status:** Approved

## Problem

The timeline viewer currently uses a heuristic gain-step detector (`gain_normalization.py`, ~350 lines of DSP heuristics) to correct abrupt mic gain changes in hydrophone recordings. It walks the entire job in a coarse pre-pass, detects sustained high-RMS segments via a median-plus-threshold rule, and attenuates those segments before STFT and before MP3 encoding. A second per-job pre-pass computes `ref_db` as the max of sampled tiles' power stats, which `generate_timeline_tile` then uses to normalize the colormap.

This approach has accumulated pain:

1. **Pre-pass latency and cache artifacts.** Two per-job pre-passes (`.gain_profile.json`, `.ref_db.json`) run before anything renders. Both require invalidation strategies and clutter the timeline cache.
2. **Step detection is fragile.** Thresholds, min-duration filtering, internal-drop splitting, and boundary refinement all need to be tuned; edge cases misfire.
3. **Only handles discrete gain jumps.** When whale calls are buried in a broadband noise floor with no sharp step, the viewer provides no help.
4. **Maintenance burden.** ~350 lines of tuned DSP heuristics that no one wants to touch.
5. **Too-dark spectrograms.** Even after gain correction, residual bright content pulls `ref_db` up via `max()`, and the 80 dB dynamic range paints most of the colormap near black. Normal regions are visibly under-bright.
6. **Visual/audio coupling is wrong.** Spectrogram visualization and audio playback have different ideal normalizations, but the current code forces them through one pass of the same gain correction.

Reference job: `8224c4a6-bc36-43db-ad59-e8933ef09115` (21:24–22:02 gain-jump region).

## Approach

Replace the entire gain-normalization and ref_db pipeline with two independent, well-understood normalizations — one per output path.

### Spectrogram path: PCEN

Use **PCEN (Per-Channel Energy Normalization)** via `librosa.pcen`. PCEN is the field-standard bioacoustic AGC used by Google Perch, BirdNET, and most whale/bird detection pipelines. It is a running per-frequency-bin divider that auto-adapts to the local noise floor, followed by root compression. Output is bounded, so the colormap range is a fixed constant — no `ref_db` computation.

Each tile renders independently:

1. Resolve `[tile_start − warmup_sec, tile_end]` worth of audio from the archive providers (the existing `resolve_timeline_audio` path, just with an extended time range).
2. Compute STFT on the full extended chunk.
3. Apply `librosa.pcen(power, sr, hop_length, time_constant, gain, bias, power, eps)` to the power spectrogram.
4. Trim the warm-up frames off the front of the PCEN output.
5. Render with fixed `vmin` / `vmax`.

**No state chaining between tiles.** Each tile computes its own PCEN state from scratch, using the warm-up region to let the low-pass smoother settle. Simpler, parallelizable, and avoids the cross-tile coordination that would defeat the simplicity goal.

For tiles at the very start of a job (before `warmup_sec` of audio is available), PCEN runs with a shorter warm-up. The first tile is slightly under-normalized; this is acceptable and undetectable at any zoom level above the finest.

### Audio playback path: RMS target + soft limiter

A new helper in `audio_encoding.py`:

```
normalize_for_playback(audio, target_rms_dbfs=-20.0, ceiling=0.95)
```

1. Compute the RMS of the chunk.
2. Scale to the target RMS — within a single chunk this equalizes gain jumps so the listener does not need to ride the volume knob.
3. Apply `tanh` soft-clip at the ceiling to prevent harsh clipping on transients.

Stateless, ~15 lines, zero user-visible knobs. Applied in both:

- `timeline.router.get_audio` (interactive playback)
- `timeline_export` audio chunk generation (static bundle)

The export currently bypasses `_apply_gain_correction` entirely — a latent inconsistency where exported MP3s are raw while exported tiles are gain-corrected. The refactor fixes this in passing; post-refactor, both viewer and export share the same audio normalization path.

### PCEN parameter defaults

Based on bioacoustic literature (Lostanlen et al., *Per-Channel Energy Normalization: Why and How*):

| Parameter | Default | Purpose |
|---|---|---|
| `pcen_time_constant_sec` | 0.8 | Longer than librosa's speech default (0.4) so a 1–3 s whale call does not AGC itself out mid-call |
| `pcen_gain` (α) | 0.8 | Lower than speech default 0.98; less aggressive gain tracking preserves dynamics |
| `pcen_bias` (δ) | 10.0 | DC offset in the compression stage |
| `pcen_power` (r) | 0.25 | Root compression exponent |
| `pcen_eps` | 1e-6 | Numerical floor |
| `pcen_warmup_sec` | 2.0 | ≈ 2.5 × τ; pre-tile audio fetched to settle PCEN state |
| `pcen_vmin` | 0.0 | Fixed colormap floor |
| `pcen_vmax` | 2.5 | Fixed colormap ceiling (tunable after eyeballing reference job) |
| `playback_target_rms_dbfs` | −20.0 | Target loudness for playback chunks |
| `playback_ceiling` | 0.95 | Soft-clip ceiling for tanh limiter |

All `HUMPBACK_`-prefixed env-var overridable, matching the existing settings pattern.

### Pipeline integration

**Before:**
```
resolve audio → _apply_gain_correction → STFT → 10·log10 → colormap(vmin=ref_db-80, vmax=ref_db)
resolve audio → _apply_gain_correction → encode_mp3
```

**After:**
```
resolve [start − warmup, end] → STFT → librosa.pcen → colormap(vmin=0, vmax=2.5) → trim warmup frames
resolve audio → normalize_for_playback → encode_mp3
```

`_prepare_tiles_sync` loses its Pass 0 (gain profile) and ref_db computation. Its only remaining job is to ensure tiles exist at all zoom levels.

### Cache invalidation

Existing timeline caches were rendered under the old normalization. They must be invalidated on upgrade or users will see a mix of old (dark, gain-corrected) and new (PCEN) tiles inside a single job.

**Cache version marker.** Add a `.cache_version` file to each job cache directory, containing an integer (new value: `2`). On first access to a job's cache (via a new `ensure_job_cache_current(job_id)` helper, called from `get_tile` and `_prepare_tiles_sync`):

- If `.cache_version` is missing or `< 2`: delete all `tile_*.png`, `.ref_db.json`, `.gain_profile.json`, and `.prepare_plan.json` under the job dir, then write `.cache_version = 2`.
- Otherwise: no-op.

`.audio_manifest.json` and `.last_access` are preserved. Tiles regenerate on demand under the new pipeline. One-time per job; subsequent accesses short-circuit.

### Scope boundaries

**In scope:**
- Timeline viewer spectrogram rendering (`generate_timeline_tile`, `_render_tile_sync`)
- Timeline viewer audio playback (`get_audio`)
- Timeline static export (tiles are copied from the cache; audio chunk generation gets the new playback normalization)
- Deletion of the entire gain-normalization module, ref_db machinery, and associated tests
- Timeline cache version marker and one-time migration
- New configuration settings; removal of old ones

**Not in scope:**
- Detection / classifier feature extraction (`features.py`) — intentionally untouched. Extending PCEN to classifier training is a separate session.
- Database schema changes (none required)
- UI controls for PCEN tuning (config is env-var only)
- Per-job PCEN parameter overrides

### Design decisions

- **Independent pipelines for visual and audio.** PCEN is not a good listening normalization, and audio compression is not a good visualization. Decoupling them produces better results on both sides and simplifies each.
- **No per-job pre-passes.** Eliminates both `.gain_profile.json` and `.ref_db.json`. Tile rendering is stateless per tile (modulo the warm-up region, which lives in the tile's own audio fetch).
- **Fixed colormap range.** PCEN output is bounded by design. Using fixed `vmin`/`vmax` removes an entire class of "sample the job, take the max, cache it" logic.
- **No state chaining between tiles.** Tiles already exist at different zoom levels and render in parallel; sharing PCEN filter state across tiles would force serialization or complex coordination for little benefit. The 2 s warm-up handles settlement cheaply.
- **RMS-target + tanh for audio**, not a full attack/release compressor. Playback chunks are ≤ 600 s and the user scrubs to specific regions; per-chunk RMS scaling is more than sufficient and has zero tunable knobs.
- **Cache version marker over "delete everything".** Automatic invalidation on first access is strictly better UX than asking users to `rm -rf data/timeline_cache/*` on upgrade. One file per job, one integer, trivial to evolve.
- **Keep `_prepare_tiles_sync` signature stable.** The export service imports it from the router module; breaking the signature would ripple into the export with no benefit.

### File changes

**New:**
- `src/humpback/processing/pcen_rendering.py` — thin wrapper over `librosa.pcen` with project defaults and a `render_tile_pcen(audio, sr, warmup_sec, …)` helper that handles warm-up trimming
- `tests/unit/test_pcen_rendering.py`
- `tests/unit/test_audio_playback_normalization.py`
- `tests/unit/test_timeline_cache_migration.py`

**Modified:**
- `src/humpback/processing/timeline_tiles.py` — `generate_timeline_tile` uses PCEN with fixed `vmin/vmax` instead of `10·log10` + `ref_db`
- `src/humpback/processing/audio_encoding.py` — add `normalize_for_playback` helper
- `src/humpback/processing/timeline_cache.py` — add `ensure_job_cache_current`; remove `get_ref_db`, `put_ref_db`, `get_gain_profile`, `put_gain_profile`
- `src/humpback/api/routers/timeline.py` — rewire `_render_tile_sync` (warm-up fetch, PCEN path), rewire `get_audio` (playback normalization), remove `_compute_job_ref_db`, `_apply_gain_correction`, `_compute_job_gain_profile`, Pass 0 of `_prepare_tiles_sync`, and ref_db loading in `get_tile`
- `src/humpback/services/timeline_export.py` — call `normalize_for_playback` before `encode_mp3` in the audio chunk loop
- `src/humpback/config.py` — remove `gain_norm_threshold_db`, `gain_norm_min_duration_sec`; add `pcen_*` and `playback_*` settings
- `CLAUDE.md` §8.6, §8.7, §9.1 — update runtime configuration, behavioral constraints, and implemented capabilities
- `DECISIONS.md` — append new ADR for the replacement
- `docs/reference/storage-layout.md` — remove `.gain_profile.json`, `.ref_db.json`; add `.cache_version`
- `docs/reference/signal-processing.md` — short PCEN subsection if the file mentions gain normalization

**Deleted:**
- `src/humpback/processing/gain_normalization.py` (~350 lines)
- `tests/unit/test_gain_normalization.py`

### Testing strategy

**Unit tests:**
- PCEN rendering: synthetic whale-call chirp + broadband noise → chirp visibly above noise floor in the output; step-gain synthetic input → output uniform across the step; warm-up comparison → first frames differ without warm-up, bulk matches; tile at job start (no warm-up audio) renders without error.
- Playback normalization: varied input levels produce output at target RMS; clipping-level input stays below ceiling; silent input produces silent output (no NaN/divide-by-zero).
- Cache migration: stale cache with old artifacts and no version file → `ensure_job_cache_current` deletes old tiles and sidecars, writes `.cache_version = 2`; already-current cache → no-op.

**Integration tests:**
- Timeline export end-to-end on a small synthetic job: exported tile PNGs exist at expected counts per zoom, exported MP3 chunks have RMS within tolerance of the target, source cache has `.cache_version = 2` written.

**Existing tests to update:**
- Any test in `tests/unit/test_timeline_rendering.py`, `tests/unit/test_timeline_tiles.py`, or integration tests touching `_render_tile_sync`, `_compute_job_ref_db`, or `get_audio` that asserts ref_db / gain-profile behavior.

**Manual verification:**
- Open reference job `8224c4a6-bc36-43db-ad59-e8933ef09115` in the browser. Confirm: (a) 21:24–22:02 gain-jump region no longer bright-yellow, (b) normal regions visibly brighter than before, (c) audio playback comfortable across the gain-jump boundary without volume fiddling, (d) no seam artifacts at tile boundaries at any zoom level.
- Run `scripts/export_timeline.py` against the same job and spot-check the exported bundle in isolation.
