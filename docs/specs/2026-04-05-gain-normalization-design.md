# Gain Normalization for Timeline Rendering

**Date:** 2026-04-05
**Status:** Approved

## Problem

Hydrophone recordings frequently contain abrupt mic gain changes — instant step
increases that sustain for seconds to minutes before dropping back. These high-gain
regions:

1. Pull the per-job `ref_db` upward, darkening spectrograms in normal regions and
   making whale vocalizations hard to see
2. Create uncomfortable volume jumps during playback when the listener has volume
   set for normal regions

Example: detection job `8224c4a6-bc36-43db-ad59-e8933ef09115` has a gain spike at
21:24:10-21:24:15 and a sustained high-gain region from there until approximately
22:01:50-22:02:00.

## Approach

**Sliding RMS with step detection.** Compute a per-job gain profile that identifies
high-gain segments and their attenuation amounts. Apply the correction to raw audio
before it feeds into both spectrogram rendering and MP3/WAV playback encoding.

### Gain Profile Computation

New module: `src/humpback/processing/gain_normalization.py`

1. Walk the job's audio in 1-second RMS windows at a low sample rate (4000 Hz)
2. Compute RMS (dB) for each 1-second window
3. Compute the **median RMS** across all non-silent windows — the "normal" level
4. Flag windows where RMS exceeds `median + threshold_db` (default: 6 dB, ~4x power)
5. Group consecutive flagged windows into gain segments
6. Discard segments shorter than `min_duration_sec` (default: 5.0 seconds) to filter
   whale calls and transient noise
7. For each retained segment, compute `attenuation_db = segment_median_rms - global_median_rms`

**Output:** A gain profile — list of `{"start_sec": float, "end_sec": float,
"attenuation_db": float}` — plus `global_median_rms` for reference.

**Cached artifact:** Persisted as `.gain_profile.json` in the job's timeline cache
directory alongside `.ref_db.json`. Computed once per job, reused for all subsequent
renders.

### Audio Correction Function

`apply_gain_profile(audio, sample_rate, start_epoch, gain_profile) -> np.ndarray`

- For each gain segment overlapping the audio's time range, attenuate corresponding
  samples by `attenuation_db`
- Apply 50ms cosine crossfade at segment boundaries to prevent click artifacts from
  the correction itself

### Pipeline Integration

The gain profile is applied after `resolve_timeline_audio()` and before spectrogram
generation or audio encoding. Single integration point for both paths.

**Tile rendering** (`_render_tile_sync`):
resolve audio -> apply gain correction -> compute STFT -> render tile

**Audio playback** (`get_audio`):
resolve audio -> apply gain correction -> encode WAV/MP3

**Computation order during prepare:**
1. Compute gain profile (one pass over coarse audio)
2. Compute ref_db (samples tiles with gain correction applied)
3. Render tiles (with gain correction applied)

The ref_db computation applies the gain profile when sampling, so ref_db reflects
corrected audio and gives better dynamic range for normal regions.

### Configuration

Two new settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `gain_norm_threshold_db` | 6.0 | RMS jump above median to flag as gain change |
| `gain_norm_min_duration_sec` | 5.0 | Minimum sustained duration to distinguish from whale calls |

### Existing Job Behavior

- New jobs and re-rendered jobs (cache cleared) get gain normalization automatically
- Existing cached jobs are unaffected until their cache is cleared
- Manual reset: delete `data/timeline_cache/{job_id}/` and re-open the timeline

### Scope Boundaries

**In scope:**
- Gain profile computation and caching
- Audio correction for tile rendering and playback
- Integration with existing ref_db computation
- Unit tests for detection and correction

**Not in scope:**
- UI controls for gain normalization
- Per-job enable/disable toggle
- Manual gain profile editing
- Changes to detection/classifier pipeline
- Database schema changes

### File Changes

- **New:** `src/humpback/processing/gain_normalization.py` — profile computation + apply function
- **Modified:** `src/humpback/api/routers/timeline.py` — compute gain profile before ref_db, apply in render and audio paths
- **Modified:** `src/humpback/config.py` — two new settings
- **New:** `tests/test_gain_normalization.py` — unit tests

### Export Compatibility

Static timeline export renders through the same `_render_tile_sync` / `encode_mp3`
paths. Exported bundles automatically include corrected tiles and audio with no
additional work.

### Design Decisions

- **Correct audio, not spectrogram scaling:** A single correction point feeds both
  visual and auditory output, keeping them consistent
- **Bake into cache, not on-the-fly:** Exported timeline artifacts are static and
  complete; on-the-fly would break the S3 readonly viewer
- **Median as target level:** The median is robust to outliers from both gain spikes
  and silence, giving a stable "normal" reference
- **5-second minimum duration:** Whale calls are typically 1-3 seconds; gain changes
  sustain for tens of seconds to minutes. The threshold cleanly separates them.
- **50ms crossfade:** Prevents click artifacts at correction boundaries without
  audibly smoothing the transition
