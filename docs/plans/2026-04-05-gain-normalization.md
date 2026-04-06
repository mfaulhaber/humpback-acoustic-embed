# Gain Normalization Implementation Plan

**Goal:** Detect abrupt mic gain changes in hydrophone recordings and attenuate high-gain regions so timeline spectrograms and audio playback reflect the normal recording level.
**Spec:** [docs/specs/2026-04-05-gain-normalization-design.md](../specs/2026-04-05-gain-normalization-design.md)

---

### Task 1: Gain Profile Computation Module

**Files:**
- Create: `src/humpback/processing/gain_normalization.py`

**Acceptance criteria:**
- [ ] `compute_gain_profile(audio_resolver, job_start, job_end, target_sr, threshold_db, min_duration_sec)` walks the job audio in 1-second RMS windows
- [ ] Computes per-window RMS in dB, skipping silent windows (below a floor like -80 dB)
- [ ] Derives `global_median_rms` from all non-silent windows
- [ ] Flags windows exceeding `median + threshold_db`
- [ ] Groups consecutive flagged windows into segments, discards segments shorter than `min_duration_sec`
- [ ] Computes `attenuation_db = segment_median_rms - global_median_rms` per segment
- [ ] Returns a dataclass or dict with `segments` list and `global_median_rms`

**Tests needed:**
- Synthetic audio with a known step gain change — verify correct segment detection and attenuation value
- Audio with no gain changes — verify empty segments list
- Short transient spike (<5s) — verify it is filtered out
- Multiple gain regions — verify all are detected independently

---

### Task 2: Audio Correction Function

**Files:**
- Modify: `src/humpback/processing/gain_normalization.py`

**Acceptance criteria:**
- [ ] `apply_gain_profile(audio, sample_rate, start_epoch, gain_profile)` returns corrected audio array
- [ ] For each gain segment overlapping the audio time range, attenuates samples by `attenuation_db`
- [ ] Applies 50ms cosine crossfade at segment boundaries to prevent click artifacts
- [ ] No-op when gain profile has no segments (returns audio unchanged)
- [ ] Handles partial overlap (segment starts before or ends after the audio chunk)
- [ ] Does not modify the input array (returns a copy when corrections are applied)

**Tests needed:**
- Apply a known attenuation to a synthetic signal — verify amplitude reduction matches expected dB
- Verify crossfade region has smooth transition (no discontinuity)
- Partial overlap — segment extends beyond audio boundaries
- Empty gain profile — verify audio is returned unchanged (same object identity)

---

### Task 3: Gain Profile Caching

**Files:**
- Modify: `src/humpback/processing/timeline_cache.py`

**Acceptance criteria:**
- [ ] `put_gain_profile(job_id, profile)` writes `.gain_profile.json` to the job's cache directory
- [ ] `get_gain_profile(job_id)` reads and returns the cached profile, or `None` if missing
- [ ] JSON format matches the output of `compute_gain_profile` — serializable segments list plus `global_median_rms`
- [ ] Follows the same pattern as existing `put_ref_db` / `get_ref_db`

**Tests needed:**
- Round-trip write and read of a gain profile
- Missing profile returns None

---

### Task 4: Configuration Settings

**Files:**
- Modify: `src/humpback/config.py`

**Acceptance criteria:**
- [ ] `gain_norm_threshold_db: float = 6.0` setting added
- [ ] `gain_norm_min_duration_sec: float = 5.0` setting added
- [ ] Both configurable via `HUMPBACK_GAIN_NORM_THRESHOLD_DB` and `HUMPBACK_GAIN_NORM_MIN_DURATION_SEC` environment variables

**Tests needed:**
- Verify defaults load correctly (covered by existing settings tests pattern)

---

### Task 5: Pipeline Integration — Tile Rendering

**Files:**
- Modify: `src/humpback/api/routers/timeline.py`

**Acceptance criteria:**
- [ ] New helper `_compute_job_gain_profile()` resolves audio at coarse SR and calls `compute_gain_profile`, caching the result
- [ ] `_prepare_tiles_sync` calls gain profile computation before ref_db computation
- [ ] `_compute_job_ref_db` applies gain correction to sampled tile audio before computing power stats
- [ ] `_render_tile_sync` applies gain correction to resolved audio before passing to `generate_timeline_tile`
- [ ] The gain profile is resolved once per prepare pass and threaded through, not re-computed per tile

**Tests needed:**
- Integration test: a synthetic job with a gain jump produces a non-empty gain profile and corrected ref_db

---

### Task 6: Pipeline Integration — Audio Playback

**Files:**
- Modify: `src/humpback/api/routers/timeline.py`

**Acceptance criteria:**
- [ ] `get_audio` endpoint loads the cached gain profile for the job
- [ ] Applies `apply_gain_profile` to resolved audio before `encode_wav` / `encode_mp3`
- [ ] When no gain profile is cached (old jobs, profile not yet computed), audio is served unchanged

**Tests needed:**
- Verify the audio endpoint applies gain correction when a profile exists
- Verify graceful fallback when no profile is cached

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/processing/gain_normalization.py src/humpback/processing/timeline_cache.py src/humpback/api/routers/timeline.py src/humpback/config.py tests/test_gain_normalization.py`
2. `uv run ruff check src/humpback/processing/gain_normalization.py src/humpback/processing/timeline_cache.py src/humpback/api/routers/timeline.py src/humpback/config.py tests/test_gain_normalization.py`
3. `uv run pyright src/humpback/processing/gain_normalization.py src/humpback/processing/timeline_cache.py src/humpback/api/routers/timeline.py src/humpback/config.py tests/test_gain_normalization.py`
4. `uv run pytest tests/`
