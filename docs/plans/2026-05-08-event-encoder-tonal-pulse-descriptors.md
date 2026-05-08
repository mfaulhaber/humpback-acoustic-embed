# Event Encoder Tonal & Pulse Descriptors Implementation Plan

**Goal:** Add six new acoustic descriptors (median F0, F0 range, voicing
fraction, inflection count, pulse rate, pulse rate slope) to the Event Encoder
pipeline with clustering mitigations.

**Spec:** `docs/specs/2026-05-08-event-encoder-tonal-pulse-descriptors-design.md`

**Primary domain:** sequence-models

**Neighbor domains:** none

---

### Task 1: Refactor ridge path extraction

Extract the Viterbi ridge path computation into a shared private function so
both the existing slope and the new inflection count can consume it without
duplicating the dynamic programming pass.

**Files:**
- Modify: `src/humpback/sequence_models/event_encoder.py`

**Acceptance criteria:**
- [ ] New `_compute_ridge_path(...)` private function returns the log₂
      frequency path array (same result the current inline code produces)
- [ ] `compute_ridge_log_frequency_slope` delegates to `_compute_ridge_path`
      and `_theil_sen_slope`
- [ ] Existing ridge slope tests pass with identical values

**Tests needed:**
- Existing `test_ridge_log_frequency_slope_tracks_log_chirp` and
  `test_ridge_log_frequency_slope_is_stable_across_harmonics` must pass
  unchanged — values must be bitwise identical

---

### Task 2: Add inflection count descriptor

Compute `inflection_count` from the ridge path extracted in Task 1.

**Files:**
- Modify: `src/humpback/sequence_models/event_encoder.py`
- Modify: `tests/sequence_models/test_event_encoder.py`

**Acceptance criteria:**
- [ ] `_ridge_inflection_count(path)` counts sign changes in `np.diff(path)`
      and returns `log(1 + N)` as a float
- [ ] Returns 0.0 for paths with fewer than 3 points
- [ ] `inflection_count` appended to `DESCRIPTOR_ORDER` and `DESCRIPTOR_UNITS`
- [ ] `compute_acoustic_descriptors` returns `inflection_count` in its dict

**Tests needed:**
- Ascending log chirp → inflection_count == 0.0
- Concatenated ascending-then-descending chirp → inflection_count ≈ log(2)
- Empty/silent/short audio → inflection_count == 0.0

---

### Task 3: Add F0 descriptors (median_f0, f0_range, voicing_fraction)

Add the three pYIN-based descriptors via a private helper.

**Files:**
- Modify: `src/humpback/sequence_models/event_encoder.py`
- Modify: `tests/sequence_models/test_event_encoder.py`

**Acceptance criteria:**
- [ ] `_compute_f0_descriptors(audio, sample_rate, fmin, fmax)` returns a dict
      with `median_f0`, `f0_range`, `voicing_fraction`
- [ ] Uses `librosa.pyin` with configurable `fmin` (default 70.0) and `fmax`
      (default 1200.0)
- [ ] Returns 0.0 for all three when audio is empty, silent, or pYIN finds no
      voiced frames
- [ ] `median_f0`, `f0_range`, `voicing_fraction` appended to
      `DESCRIPTOR_ORDER` and `DESCRIPTOR_UNITS`
- [ ] `compute_acoustic_descriptors` accepts `f0_fmin` and `f0_fmax` kwargs
      and merges F0 results into its return dict

**Tests needed:**
- 440 Hz sine → median_f0 ≈ 440, f0_range ≈ 0, voicing_fraction ≈ 1.0
- Log chirp 300→1200 Hz → median_f0 between 300 and 1200, f0_range > 0
- Empty audio → all three == 0.0
- Silent audio → all three == 0.0

---

### Task 4: Add pulse rate descriptors (pulse_rate, pulse_rate_slope)

Add the two envelope-autocorrelation-based descriptors via a private helper.

**Files:**
- Modify: `src/humpback/sequence_models/event_encoder.py`
- Modify: `tests/sequence_models/test_event_encoder.py`

**Acceptance criteria:**
- [ ] `_compute_pulse_descriptors(audio, sample_rate, min_rate_hz, max_rate_hz,
      confidence_threshold, envelope_smooth_ms)` returns a dict with
      `pulse_rate`, `pulse_rate_slope`
- [ ] Envelope extracted via `scipy.signal.hilbert`, smoothed with
      moving-average window
- [ ] Dominant pulse rate from normalized autocorrelation peak in the lag range
      corresponding to `min_rate_hz`–`max_rate_hz`
- [ ] Confidence gate: autocorrelation peak height below threshold → both
      values 0.0
- [ ] Pulse rate slope via Theil-Sen on inter-peak instantaneous rates;
      requires ≥ 3 envelope peaks, else 0.0
- [ ] `pulse_rate`, `pulse_rate_slope` appended to `DESCRIPTOR_ORDER` and
      `DESCRIPTOR_UNITS`
- [ ] `compute_acoustic_descriptors` accepts pulse config kwargs and merges
      results

**Tests needed:**
- Amplitude-modulated tone (20 Hz AM on 500 Hz carrier) → pulse_rate ≈ 20
- Smooth 440 Hz sine → pulse_rate == 0.0 (below confidence)
- Decaying-rate AM signal → pulse_rate_slope < 0
- Empty/silent audio → both == 0.0
- Very short audio (< 1 envelope period) → both == 0.0

---

### Task 5: Update worker config forwarding and default descriptor weight

Forward the new config keys from `descriptor_config` to
`compute_acoustic_descriptors` and change the default `descriptor_weight`.

**Files:**
- Modify: `src/humpback/workers/event_encoder_worker.py`

**Acceptance criteria:**
- [ ] `_build_encoded_events` forwards `f0_fmin`, `f0_fmax`,
      `pulse_min_rate_hz`, `pulse_max_rate_hz`, `pulse_confidence_threshold`,
      `pulse_envelope_smooth_ms` from `descriptor_config` to
      `compute_acoustic_descriptors` with correct defaults
- [ ] Default `descriptor_weight` fallback in `_run_event_encoder_job` changes
      from 1.0 to 0.571
- [ ] `descriptor_feature_names` in manifest reflects the 14-entry
      `DESCRIPTOR_ORDER`

**Tests needed:**
- Existing worker tests pass (they exercise the full pipeline with default
  configs and will now produce 14-d descriptor vectors)

---

### Task 6: Update descriptor vector shape assertions in tests

Existing tests assert `descriptor_vector().shape == (8,)`. These need to
reflect the new 14-descriptor order.

**Files:**
- Modify: `tests/sequence_models/test_event_encoder.py`

**Acceptance criteria:**
- [ ] All shape assertions updated from (8,) to (14,)
- [ ] All existing descriptor value assertions unchanged (ridge slope, spectral
      features, gap, duration, etc.)

**Tests needed:**
- Full test suite passes with no regressions

---

### Task 7: Update domain documentation

Update the sequence-models domain capsule to reflect the expanded descriptor
set.

**Files:**
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/invariants.md`

**Acceptance criteria:**
- [ ] README mentions the six new descriptors in the artifact manifest
      description
- [ ] Invariants document records the new 14-entry descriptor order including
      F0 and pulse features

**Tests needed:**
- None (documentation only)

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/event_encoder.py src/humpback/workers/event_encoder_worker.py`
2. `uv run ruff check src/humpback/sequence_models/event_encoder.py src/humpback/workers/event_encoder_worker.py`
3. `uv run pyright src/humpback/sequence_models/event_encoder.py src/humpback/workers/event_encoder_worker.py`
4. `uv run pytest tests/sequence_models/test_event_encoder.py -q`
5. `uv run pytest tests/workers/test_event_encoder_worker.py -q`
6. `uv run pytest tests/services/test_event_encoder_service.py -q`
7. `uv run pytest tests/`
