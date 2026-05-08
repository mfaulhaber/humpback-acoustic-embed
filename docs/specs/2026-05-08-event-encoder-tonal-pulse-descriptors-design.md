# Event Encoder Tonal & Pulse Descriptors

**Date:** 2026-05-08
**Status:** Draft
**Domain:** sequence-models

## Summary

Add six new acoustic descriptors to the Event Encoder that capture tonal
progression (F0 contour) and temporal texture (pulse/buzz/creak modulation).
These complement the existing eight descriptors by encoding fundamental
frequency, contour complexity, and amplitude modulation — features that the
current spectral descriptors and CRNN embeddings do not explicitly represent.

## Motivation

Humpback whale vocalizations exhibit tonal progressions (ascending/descending
pitch), varying contour complexity (pure sweeps vs. warbles), and temporal
modulation textures (smooth tones vs. buzzes, creaks, and vibrato with evolving
pulse rates). The existing descriptors capture spectral shape
(centroid, bandwidth, entropy), overall pitch direction (ridge slope), and
timing (duration, gap). They do not capture the true fundamental frequency
(distinct from harmonic-weighted centroid), contour shape complexity, or
rhythmic pulsing.

Literature precedent:

- Cazau, Adam & Aubin (2016, Scientific Reports) divided humpback song units
  into three temporal segments and found chaos concentrated at initiation (49%
  in first third), frequency jumps in middle thirds (38%).
- Ou et al. (2013, JASA) used start frequency, end frequency, and number of
  up/down sweeps as classification features.
- Schall, Roca & Van Opzeeland (2021, JASA) used 44 acoustic metrics in a
  random forest to discriminate 12 humpback unit types at 84% accuracy.
- Humpback unit taxonomy classifies by tonal direction: "ascending moan",
  "descending shriek", "short ascending moan" (sam), etc.

Per-segment (start/middle/end) F0 features were considered but rejected: for
short events (<200ms), pYIN cannot produce stable per-segment estimates, and
noisy values would scatter events in the embedding space and fragment clusters
that should group by spectral character.

## New Descriptors

Six entries appended to `DESCRIPTOR_ORDER`:

| Descriptor | Units | Computation | Degenerate fallback |
|---|---|---|---|
| `median_f0` | Hz | Median of `librosa.pyin()` voiced frames | 0.0 (no voiced frames) |
| `f0_range` | Hz | max(F0) − min(F0) across voiced frames | 0.0 (< 2 voiced frames) |
| `voicing_fraction` | normalized | Fraction of pYIN frames with non-NaN F0 | 0.0 (no frames) |
| `inflection_count` | log count | `log(1 + N)` where N = sign changes in ridge path slope | 0.0 (path < 3 points) |
| `pulse_rate` | Hz | Dominant peak of amplitude-envelope autocorrelation | 0.0 (confidence below threshold) |
| `pulse_rate_slope` | Hz/s | Theil-Sen slope of inter-peak instantaneous rates | 0.0 (< 3 peaks or low confidence) |

Total descriptors: 8 → 14.

## F0 Extraction: median_f0, f0_range, voicing_fraction

Private helper `_compute_f0_descriptors(audio, sample_rate, fmin, fmax)` returns
a dict of three floats.

Algorithm:

1. Call `librosa.pyin(audio, fmin=fmin, fmax=fmax, sr=sample_rate)` which
   returns `(f0_array, voiced_flags, voiced_probs)`.
2. Extract voiced frames: `voiced_f0 = f0_array[~np.isnan(f0_array)]`.
3. `voicing_fraction = len(voiced_f0) / len(f0_array)` (0.0 if no frames).
4. `median_f0 = np.median(voiced_f0)` if any voiced frames, else 0.0.
5. `f0_range = max(voiced_f0) - min(voiced_f0)` if ≥ 2 voiced frames, else 0.0.

Config parameters (via `descriptor_config`):

- `f0_fmin`: 70.0 Hz — humpback F0 floor (Au et al. 2006).
- `f0_fmax`: 1200.0 Hz — humpback F0 ceiling.

## Inflection Count

Derived from the existing ridge path — no new signal processing.

The ridge tracker already computes a path of log₂(frequency) values per STFT
frame via Viterbi decoding. Currently only the Theil-Sen slope of this path is
exposed. The inflection count reuses the same path:

1. Extract `_compute_ridge_path(...)` as a private function returning the path
   array. Both `compute_ridge_log_frequency_slope` and `_ridge_inflection_count`
   consume this shared path.
2. Compute first differences: `d = np.diff(path)`.
3. Count sign changes: `N = np.sum(np.diff(np.sign(d)) != 0)`.
4. Store as `log(1 + N)` to compress the right-skewed integer distribution and
   reduce outlier pull from high-inflection events.
5. Return 0.0 if the path has fewer than 3 points.

Interpretation: ascending chirp → 0, up-then-down arch → log(2) ≈ 0.69,
warble → higher values.

## Pulse Rate Extraction: pulse_rate, pulse_rate_slope

Private helper `_compute_pulse_descriptors(audio, sample_rate, ...)` returns a
dict of two floats.

### Envelope extraction

1. Analytic signal via `scipy.signal.hilbert(audio)`.
2. Amplitude envelope: `np.abs(analytic_signal)`.
3. Smooth with a moving-average window (default 5 ms) to remove carrier
   frequency ripple.

### Dominant pulse rate

1. Compute normalized autocorrelation of the smoothed envelope.
2. Search for the first prominent peak in the lag range corresponding to
   `pulse_min_rate_hz` (2 Hz) through `pulse_max_rate_hz` (200 Hz).
3. `pulse_rate = sample_rate / peak_lag` in Hz.
4. **Confidence gate:** the autocorrelation peak height must exceed
   `pulse_confidence_threshold` (default 0.3). Below threshold →
   `pulse_rate = 0.0`.

### Pulse rate slope

1. Find individual peaks in the smoothed envelope using
   `scipy.signal.find_peaks` with minimum distance = half the dominant period.
2. Compute inter-peak intervals, convert to instantaneous rate:
   `rate_i = 1.0 / interval_i`.
3. Fit Theil-Sen slope of rate vs. time (reuses existing `_theil_sen_slope`).
4. Requires ≥ 3 envelope peaks; otherwise 0.0.
5. Same confidence gate: if no dominant pulse detected, slope is also 0.0.

Config parameters (via `descriptor_config`):

- `pulse_min_rate_hz`: 2.0
- `pulse_max_rate_hz`: 200.0
- `pulse_confidence_threshold`: 0.3
- `pulse_envelope_smooth_ms`: 5.0

## Clustering Mitigations

### Descriptor weight scaling

Default `descriptor_weight` changes from 1.0 → 8/14 ≈ 0.571 to maintain
constant total descriptor influence relative to the 128-d PCA embedding block.
This is a default-only change; the worker reads it from
`preprocessing_config_json`, so existing jobs are unaffected.

### Descriptor outlier clipping

After robust z-score normalization, descriptor vectors are clipped to
`[-descriptor_clip_value, descriptor_clip_value]` before weighting and
concatenation. The default is 3.0; `null` disables clipping. This prevents a
descriptor with a very small MAD, such as near-zero pulse-rate slope across most
events, from stretching UMAP/PCA projections into long outlier arms.

### Confidence-gated imputation

`pulse_rate` and `pulse_rate_slope` return 0.0 when below the confidence
threshold. After robust z-score normalization (median/MAD), these values sit
near the population median, making non-pulsed events neutral on these axes
rather than forming an artificial cluster. Same logic applies to `median_f0`
and `f0_range` when no voiced frames are detected.

### Log-transformed inflection count

`log(1 + count)` compresses the integer distribution, reducing outlier pull
from high-inflection events and making the feature more continuous for k-means.

### Correlation with existing features

`median_f0` correlates with `spectral_centroid` and `peak_frequency` but
measures a distinct quantity (fundamental vs. harmonic-weighted). With 14
descriptors in a 142-d vector (128 PCA + 14 descriptors), per-feature
influence is modest enough that correlation does not dominate clustering axes.

## Code Changes

### event_encoder.py

- Append 6 entries to `DESCRIPTOR_ORDER` and `DESCRIPTOR_UNITS`.
- Add private helpers: `_compute_f0_descriptors()`, `_compute_pulse_descriptors()`.
- Refactor ridge computation: extract `_compute_ridge_path()` returning the
  path array. `compute_ridge_log_frequency_slope` and `_ridge_inflection_count`
  both consume the shared path.
- `compute_acoustic_descriptors()` calls the new helpers and merges results.
- New config parameters forwarded: `f0_fmin`, `f0_fmax`, `pulse_min_rate_hz`,
  `pulse_max_rate_hz`, `pulse_confidence_threshold`, `pulse_envelope_smooth_ms`.

### event_encoder_worker.py

- Forward new config keys from `descriptor_config` to
  `compute_acoustic_descriptors`.
- Default `descriptor_weight` in new jobs changes to 0.571.
- Forward `descriptor_clip_value` from `preprocessing_config_json` to
  preprocessing.

### event_tokenization.py

- Add optional `descriptor_clip_value` preprocessing after robust z-score
  normalization and before descriptor weighting.

### Dependencies

- `librosa` — already in the dependency tree.
- `scipy.signal.hilbert` and `scipy.signal.find_peaks` — scipy already a
  dependency.

No new dependencies required.

## Testing

### F0 descriptors

- 440 Hz sine → `median_f0 ≈ 440`, `f0_range ≈ 0`, `voicing_fraction ≈ 1.0`.
- Log chirp 300→1200 Hz → `f0_range > 0`, `median_f0` between 300 and 1200.

### Inflection count

- Ascending chirp → 0.
- Concatenated up-then-down chirp → `log(1 + 1) ≈ 0.69`.

### Pulse rate

- Amplitude-modulated tone (20 Hz AM on 500 Hz carrier) → `pulse_rate ≈ 20`.
- Smooth tone → `pulse_rate = 0.0` (below confidence threshold).
- Decaying-rate AM → negative `pulse_rate_slope`.

### Degenerate inputs

- Empty audio, silence, ultra-short events → 0.0 for all new descriptors.

### Descriptor vector shape

- Grows from (8,) to (14,).

## Non-Goals

- Per-segment (start/middle/end) F0 features.
- Formant extraction, harmonics-to-noise ratio, or other Praat-style features.
- Changes to CRNN pooling strategies or PCA pipeline.
- Frontend display of new descriptors (separate follow-up).
- CREPE or other deep F0 estimators (pYIN is sufficient and dependency-light).

## Acceptance Criteria

1. `DESCRIPTOR_ORDER` has 14 entries; `descriptor_vector()` returns shape (14,).
2. All six new descriptors computed correctly for synthetic test signals.
3. All degenerate inputs return finite 0.0 for new descriptors.
4. Existing descriptor values unchanged (ridge slope, spectral features, etc.).
5. Default `descriptor_weight` is 0.571 for new encoder jobs.
6. Descriptor robust-z outliers are clipped to 3.0 by default, with clipping
   disable-able via `descriptor_clip_value=null`.
7. `uv run pytest tests/` passes.
8. `uv run pyright` passes.
9. `uv run ruff check` passes.
