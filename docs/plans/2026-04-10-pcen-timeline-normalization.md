# PCEN Timeline Normalization Implementation Plan

**Goal:** Replace heuristic gain-step detection and per-job `ref_db` pre-passes with PCEN spectrogram normalization and RMS-targeted playback normalization, scoped to the timeline viewer and export.
**Spec:** [docs/specs/2026-04-10-pcen-timeline-normalization-design.md](../specs/2026-04-10-pcen-timeline-normalization-design.md)

---

### Task 1: PCEN rendering helper

**Files:**
- Create: `src/humpback/processing/pcen_rendering.py`
- Create: `tests/unit/test_pcen_rendering.py`

**Acceptance criteria:**
- [ ] Exposes `render_tile_pcen(audio, sample_rate, hop_length, n_fft, warmup_samples, params) -> np.ndarray` returning a 2-D `(n_freqs, n_frames)` PCEN magnitude array with warm-up frames already trimmed
- [ ] Wraps `librosa.pcen` with project defaults: `time_constant`, `gain`, `bias`, `power`, `eps` from a `PcenParams` dataclass
- [ ] Accepts `warmup_samples=0` and returns full output unchanged
- [ ] When the caller supplies fewer than `warmup_samples` of pre-roll audio (tile at job start), the function trims only what was actually prepended and returns the remaining frames without error
- [ ] Input audio of length zero returns an empty `(n_freqs, 0)` array without raising
- [ ] No silent suppression: NaN or Inf in PCEN output surfaces as a ValueError from the helper

**Tests needed:**
- Synthetic narrowband chirp embedded in broadband noise → output at chirp frequency is at least `delta_db` above the adjacent noise floor frequency bins (verify chirp is enhanced, not lost)
- Step-gain synthetic input (same signal at two gain levels spliced together) → PCEN output has comparable magnitude across the step (verify AGC effect)
- Warm-up comparison: render same tile with `warmup_samples=0` and `warmup_samples=2*sr`; verify bulk frames differ in early region and converge in later region
- Tile-at-start edge case: supply less pre-roll than `warmup_samples`; verify no error, output length matches expected trimmed length
- Empty audio: returns `(n_freqs, 0)` shape
- NaN input audio: raises ValueError

---

### Task 2: Playback audio normalization helper

**Files:**
- Modify: `src/humpback/processing/audio_encoding.py`
- Create: `tests/unit/test_audio_playback_normalization.py`

**Acceptance criteria:**
- [ ] `normalize_for_playback(audio, target_rms_dbfs, ceiling) -> np.ndarray` added to `audio_encoding.py`
- [ ] Computes RMS of the input chunk, scales to the target RMS level in dBFS
- [ ] Applies `tanh`-based soft-clip at `ceiling` to prevent harsh clipping
- [ ] Silent input (RMS of 0 or below a small floor) returns silence unchanged (no divide-by-zero, no NaN)
- [ ] Does not mutate the input array; returns a new float32 array
- [ ] Order of operations documented in the docstring: RMS scale first, then soft-clip

**Tests needed:**
- Varied-level inputs (−6, −20, −40 dBFS sine waves): output RMS within 0.5 dB of target
- Clipping-level input (amplitude > 1.0): output absolute max does not exceed `ceiling`
- Silent input (all zeros): output is all zeros, no exceptions
- Near-silent input (amplitude ~1e-10): output is silent, no divide-by-zero artifacts
- Verify input array identity is not returned when a correction applies (input is not mutated)

---

### Task 3: Timeline cache version marker and migration

**Files:**
- Modify: `src/humpback/processing/timeline_cache.py`
- Create: `tests/unit/test_timeline_cache_migration.py`

**Acceptance criteria:**
- [ ] Constant `TIMELINE_CACHE_VERSION = 2` added to `timeline_cache.py`
- [ ] `TimelineTileCache.ensure_job_cache_current(job_id)` added
- [ ] On call, reads `.cache_version` from the job's cache directory; if missing or `< TIMELINE_CACHE_VERSION`, deletes all `tile_*.png` under every zoom subdirectory and sidecar files (`.ref_db.json`, `.gain_profile.json`, `.prepare_plan.json`), then writes the current version atomically
- [ ] Preserves `.audio_manifest.json` and `.last_access`
- [ ] Already-current cache (version file present and equal): no filesystem writes, no deletions
- [ ] Handles absent job cache directory: no-op, no error
- [ ] Uses the same atomic temp-file-rename pattern as existing sidecar writers in `timeline_cache.py`
- [ ] Old sidecar accessors `get_ref_db`, `put_ref_db`, `get_gain_profile`, `put_gain_profile` removed from the class

**Tests needed:**
- Stale cache fixture: job dir with old tiles, `.ref_db.json`, `.gain_profile.json`, `.prepare_plan.json`, no `.cache_version` → after `ensure_job_cache_current`, all stale artifacts gone, `.cache_version = 2` present, `.audio_manifest.json` and `.last_access` preserved
- Current cache: job dir with `.cache_version = 2` and tiles → `ensure_job_cache_current` leaves mtimes and file list unchanged
- Missing job dir: `ensure_job_cache_current` does not create anything and does not raise
- Version mismatch in the future: hand-write `.cache_version = 1`, verify migration fires

---

### Task 4: Configuration settings

**Files:**
- Modify: `src/humpback/config.py`

**Acceptance criteria:**
- [ ] Removed: `gain_norm_threshold_db`, `gain_norm_min_duration_sec`
- [ ] Added: `pcen_time_constant_sec` (default 0.8), `pcen_gain` (0.8), `pcen_bias` (10.0), `pcen_power` (0.25), `pcen_eps` (1e-6), `pcen_warmup_sec` (2.0), `pcen_vmin` (0.0), `pcen_vmax` (2.5)
- [ ] Added: `playback_target_rms_dbfs` (−20.0), `playback_ceiling` (0.95)
- [ ] All new settings overridable via `HUMPBACK_`-prefixed environment variables
- [ ] Any existing `ref_db_dynamic_range_db` setting is removed if present (since fixed `vmin`/`vmax` replaces it)

**Tests needed:**
- Existing config test pattern covers defaults and env-var overrides; extend it to cover the new settings
- Verify old settings are not importable (would indicate an accidental residual reference)

---

### Task 5: Wire PCEN into spectrogram rendering pipeline

**Files:**
- Modify: `src/humpback/processing/timeline_tiles.py`
- Modify: `src/humpback/api/routers/timeline.py`

**Acceptance criteria:**
- [ ] `generate_timeline_tile` in `timeline_tiles.py` no longer references `ref_db`; instead takes PCEN parameters plus fixed `vmin`/`vmax` and routes its STFT through `render_tile_pcen` from Task 1
- [ ] `generate_timeline_tile` accepts a `warmup_frames` parameter (number of STFT frames corresponding to the warm-up region) and trims them before rendering
- [ ] `_render_tile_sync` in `timeline.py` computes the warm-up duration from `settings.pcen_warmup_sec`, extends the audio fetch to `[tile_start − warmup, tile_end]`, and passes the warm-up frame count through to `generate_timeline_tile`
- [ ] For tiles at the start of a job (less than `warmup_sec` of pre-roll available), `_render_tile_sync` clamps the fetch start to `job_start` and passes a smaller `warmup_frames` value
- [ ] `_render_tile_sync` no longer calls `_apply_gain_correction` or loads `.gain_profile.json`
- [ ] `_prepare_tiles_sync` no longer runs the gain-profile pre-pass (Pass 0) and no longer runs `_compute_job_ref_db`; its sole remaining responsibility is filling missing tiles
- [ ] `_prepare_tiles_sync` calls `cache.ensure_job_cache_current(job_id)` at the start
- [ ] `get_tile` handler calls `cache.ensure_job_cache_current(job_id)` on entry and no longer loads `.ref_db.json`
- [ ] Signature of `_prepare_tiles_sync` is preserved so `timeline_export` imports continue to work
- [ ] The `_compute_job_ref_db` and `_apply_gain_correction` functions are removed from `timeline.py` along with any now-unused imports and helpers (`_compute_job_gain_profile`, related constants)

**Tests needed:**
- Existing `tests/unit/test_timeline_tiles.py` updated: assertions on `ref_db` parameters replaced with PCEN-based assertions (render produces output in `[vmin, vmax]` range, not NaN)
- Existing `tests/unit/test_timeline_rendering.py` (or equivalent) updated: remove gain-profile-dependent assertions; add at least one test that calls `generate_timeline_tile` with warm-up frames and verifies the first columns are absent from the returned PNG
- Integration behavior: `_prepare_tiles_sync` against a small synthetic job writes `.cache_version = 2` and tile PNGs, but does not write `.ref_db.json` or `.gain_profile.json`
- Tile at job start: synthetic tile where `tile_start == job_start` renders without raising and produces non-empty output

---

### Task 6: Wire playback normalization into audio paths

**Files:**
- Modify: `src/humpback/api/routers/timeline.py`
- Modify: `src/humpback/services/timeline_export.py`

**Acceptance criteria:**
- [ ] `get_audio` handler in `timeline.py` calls `normalize_for_playback` on resolved audio before passing it to `encode_wav` / `encode_mp3`, using `settings.playback_target_rms_dbfs` and `settings.playback_ceiling`
- [ ] `get_audio` no longer loads `.gain_profile.json` or calls `_apply_gain_correction`
- [ ] `timeline_export.export_timeline` audio chunk loop calls `normalize_for_playback` on each resolved chunk before `encode_mp3`, using the same settings
- [ ] Exported chunks of silent source audio remain silent (sanity check — normalization does not introduce noise)
- [ ] The export service still calls `_prepare_tiles_sync` with the same arguments as today (signature preservation from Task 5 honored)

**Tests needed:**
- Existing tests for `get_audio` (if any) updated to reflect the new normalization path
- Integration test in `tests/integration/test_timeline_export.py` (extended or new) checks that exported MP3 chunks decode to audio with RMS within tolerance of the target; see Task 8

---

### Task 7: Remove legacy gain normalization code

**Files:**
- Delete: `src/humpback/processing/gain_normalization.py`
- Delete: `tests/unit/test_gain_normalization.py`
- Modify: `src/humpback/processing/timeline_cache.py` (if Task 3 left any residual gain/ref_db accessors)
- Modify: `src/humpback/api/routers/timeline.py` (if Task 5 left any residual imports)

**Acceptance criteria:**
- [ ] `gain_normalization.py` no longer present in the repo
- [ ] No file imports from `humpback.processing.gain_normalization` — verified via grep
- [ ] `test_gain_normalization.py` no longer present
- [ ] `timeline_cache.py` has no `.gain_profile.json` or `.ref_db.json` accessors remaining (they were removed in Tasks 3 and 5; this task is a final grep to confirm)
- [ ] `timeline.py` has no residual references to gain-profile or ref_db machinery
- [ ] Test suite has no broken imports after deletions

**Tests needed:**
- Grep-based verification: no matches for `gain_normalization`, `gain_profile`, `_apply_gain_correction`, `_compute_job_ref_db`, `ref_db.json`, `gain_profile.json` under `src/humpback/` and `tests/`
- Full pytest run passes (covered in verification section)

---

### Task 8: Timeline export end-to-end integration test

**Files:**
- Create or Modify: `tests/integration/test_timeline_export.py`

**Acceptance criteria:**
- [ ] Integration test exports a small synthetic detection job (≤ 2 minutes duration, single zoom level is acceptable, seeded audio)
- [ ] Asserts: exported tile PNGs exist at the expected counts for each zoom level; each tile file is non-empty
- [ ] Asserts: exported MP3 chunks exist; decoding one chunk and computing its RMS shows the value within 1.5 dB of `settings.playback_target_rms_dbfs`
- [ ] Asserts: source cache directory has `.cache_version = 2` after export runs
- [ ] Asserts: source cache directory does not contain `.ref_db.json` or `.gain_profile.json` after export
- [ ] Test uses a temporary cache directory and output directory (pytest `tmp_path`)
- [ ] Seeded audio includes a synthetic gain step so the test also verifies that export works on the exact case the old gain normalization was designed to handle

**Tests needed:**
- The test itself is the deliverable

---

### Task 9: Documentation updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `DECISIONS.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/signal-processing.md` (if it references gain normalization)
- Modify: `README.md` (only if it mentions gain normalization)

**Acceptance criteria:**
- [ ] `CLAUDE.md` §8.6 Runtime Configuration: `gain_norm_*` bullets removed; `pcen_*` and `playback_*` bullets added with one-line descriptions each
- [ ] `CLAUDE.md` §8.7 Behavioral Constraints: the "Timeline gain normalization" bullet is rewritten. New text covers: PCEN rendering with per-tile warm-up, fixed colormap range, RMS-targeted audio playback, and the deliberate decoupling of visual and audio normalization. Explicitly notes detection/features pipeline is untouched.
- [ ] `CLAUDE.md` §9.1 Implemented Capabilities: Timeline viewer bullet no longer mentions "automatic gain normalization"; now mentions "PCEN spectrogram normalization and RMS-targeted playback".
- [ ] `DECISIONS.md` new ADR appended: "Replace heuristic gain-step detection with PCEN". Covers context (clunkiness, too-dark spectrograms), decision (PCEN + RMS limiter split), and consequences (pre-passes removed, cache simplified, path open to extend to detection later).
- [ ] `docs/reference/storage-layout.md`: remove `.gain_profile.json` and `.ref_db.json` from the timeline cache section; add `.cache_version`.
- [ ] `docs/reference/signal-processing.md`: if it mentions gain normalization, update that section; otherwise leave unchanged.
- [ ] `README.md`: grep for "gain normalization" and update or remove as needed.

**Tests needed:**
- None (documentation-only task)

---

### Task 10: Manual verification

**Files:** none

**Acceptance criteria:**
- [ ] Reference job `8224c4a6-bc36-43db-ad59-e8933ef09115` opens in the timeline viewer at `http://localhost:5173/app/classifier/timeline/8224c4a6-bc36-43db-ad59-e8933ef09115`
- [ ] The 21:24–22:02 gain-jump region no longer appears as bright yellow; it is visually consistent with surrounding regions
- [ ] Normal (non-gain-jump) regions are visibly brighter than in the current production build — whale calls are more readable
- [ ] Audio playback across the 21:24 boundary is comfortable without the listener adjusting volume
- [ ] At each zoom level, no seam artifacts or brightness discontinuities appear at tile boundaries
- [ ] `uv run scripts/export_timeline.py --job-id 8224c4a6-bc36-43db-ad59-e8933ef09115 --output-dir /tmp/pcen-export-test` runs to completion
- [ ] A spot-check of the exported bundle shows tiles and audio consistent with the live viewer

**Tests needed:**
- None (manual verification)

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/processing/pcen_rendering.py src/humpback/processing/audio_encoding.py src/humpback/processing/timeline_cache.py src/humpback/processing/timeline_tiles.py src/humpback/api/routers/timeline.py src/humpback/services/timeline_export.py src/humpback/config.py tests/unit/test_pcen_rendering.py tests/unit/test_audio_playback_normalization.py tests/unit/test_timeline_cache_migration.py tests/integration/test_timeline_export.py`
2. `uv run ruff check` on the same file list
3. `uv run pyright` on the same file list
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit` (only if frontend files changed; expected: no frontend changes)
6. Manual verification per Task 10
