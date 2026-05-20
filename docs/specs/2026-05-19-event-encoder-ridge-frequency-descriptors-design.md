# Event Encoder Ridge Frequency Descriptors

**Date:** 2026-05-19
**Status:** Approved
**Primary domain:** Sequence Models
**Neighbor domains:** Signal Timeline, Frontend Shell

## 1. Goal

Improve Event Encoder piano roll rendering for high-frequency whistles,
shrieks, and noisy tonal calls by persisting robust spectral ridge frequency
summaries and using those summaries to place a single token rectangle on the
frequency axis.

The immediate goal is not to draw detailed contours or multiple ridges inside
each piano roll token. The immediate goal is to make each token rectangle
truthfully occupy the frequency band where the event's dominant ridge lives.

## 2. Motivation

The current piano roll defaults to `median_f0` placement with fallback to
`peak_frequency`. This works for clean low and mid-frequency voiced tones, but
it fails on high whistles and shrieks when:

- `librosa.pyin()` returns no F0, or locks onto a low subharmonic.
- `peak_frequency` is dominated by low-frequency rumble or broadband floor.
- `f0_range` is zero or tiny, so the rendered rectangle has no meaningful
  vertical extent.

In Event Encoder job `66ec763d-f7ca-4edb-8921-e14fe0a216a7`, sequence
positions 1049-1057 show this failure clearly. Several events render near the
bottom of the piano roll even though the spectrogram strip shows strong
high-frequency energy.

Artifact inspection showed representative rows:

| sequence | current `peak_frequency` | current `median_f0` | `spectral_centroid` | reconstructed ridge median |
|---:|---:|---:|---:|---:|
| 1051 | 62.5 Hz | 0 Hz | 3169 Hz | about 2320 Hz |
| 1052 | 62.5 Hz | 0 Hz | 3289 Hz | about 2438 Hz |
| 1053 | 62.5 Hz | 71 Hz | 3267 Hz | about 2790-2930 Hz |
| 1057 | 62.5 Hz | 0 Hz | 3122 Hz | about 2219 Hz |

The existing Viterbi ridge tracker can already recover the high-frequency path
for these events. It currently persists only `ridge_log_frequency_slope` and
`inflection_count`, so the UI cannot use the ridge's actual frequency position.

## 3. Scope

### In Scope

- Add ridge-frequency summary descriptors to Event Encoder output artifacts.
- Keep one piano roll token rectangle per event.
- Use trimmed ridge low/high bounds to set token height.
- Add a ridge-first piano roll Y mode and make it the default for new artifact
  versions that contain ridge summaries.
- Preserve existing F0 and peak-frequency display modes as alternates.
- Keep completed timelines artifact-authoritative.
- Version the Event Encoder tokenizer/default descriptor contract so existing
  completed jobs remain readable.

### Non-goals

- No multi-ridge rendering inside piano roll tokens.
- No contour sidecar artifact in this implementation.
- No MIDI/MPE export in this implementation.
- No backend changes to Call Parsing tiles or spectrogram rendering.
- No reinterpretation of existing completed Event Encoder artifact values.

## 4. Proposed Approach

Add a small ridge-summary descriptor family derived from the existing STFT
ridge path:

| Descriptor | Units | Computation | Degenerate fallback |
|---|---|---|---|
| `ridge_median_frequency` | Hz | Median tracked ridge frequency | 0.0 |
| `ridge_low_frequency` | Hz | Lower trimmed ridge bound, default 10th percentile | 0.0 |
| `ridge_high_frequency` | Hz | Upper trimmed ridge bound, default 90th percentile | 0.0 |
| `ridge_frequency_span` | Hz | `ridge_high_frequency - ridge_low_frequency` | 0.0 |
| `ridge_coverage` | normalized | Tracked ridge frame count / total STFT frame count | 0.0 |
| `ridge_energy_ratio` | normalized | Median local ridge-bin energy ratio across tracked frames | 0.0 |
| `band_limited_peak_frequency` | Hz | Mean-spectrum peak after excluding low-frequency rumble | 0.0 |
| `high_band_energy_ratio` | normalized | Energy above a configured high-band floor divided by total energy | 0.0 |

These descriptors are appended to `DESCRIPTOR_ORDER` after the existing 14
entries. Existing descriptor names and meanings do not change.

## 5. Ridge Bounds

The piano roll should not use literal ridge min/max values. One bad frame can
turn a short token into an implausibly tall vertical bar. Instead, store and
render trimmed bounds:

- `ridge_low_frequency = percentile(ridge_hz, ridge_low_percentile)`
- `ridge_high_frequency = percentile(ridge_hz, ridge_high_percentile)`
- Defaults: 10th and 90th percentiles.

Percentile defaults should be configurable through `descriptor_config`:

- `ridge_summary_low_percentile`: `10.0`
- `ridge_summary_high_percentile`: `90.0`

The implementation must validate:

- `0 <= ridge_summary_low_percentile < ridge_summary_high_percentile <= 100`

This keeps token height robust while still expressing whistles, sweeps, and
shrieks as visibly high-frequency events.

For broad harmonic moans, the single tracked ridge can follow one dominant
partial while other visible ridges extend much higher. In Ridge mode, the UI may
conservatively expand the rendered upper bound to the spectral centroid when
`high_band_energy_ratio`, `bandwidth`, and, for moderate high-band ratios, low
`spectral_entropy` show a broad tonal high-band envelope and the centroid is
well above `ridge_high_frequency`. This preserves the ridge-derived low/center
placement while avoiding a misleading thin rectangle for multi-ridge moan
events.

## 6. Ridge Trust and Fallbacks

The UI should only use ridge bounds when the ridge summary is trustworthy.

Recommended defaults:

- `ridge_coverage >= 0.35`
- `ridge_energy_ratio >= 0.01`
- `ridge_median_frequency > 0`
- `ridge_high_frequency >= ridge_low_frequency`

If those conditions fail, the piano roll should fall back in this order:

1. `median_f0` when `voicing_fraction > VOICED_THRESHOLD`.
2. `band_limited_peak_frequency`.
3. Existing `peak_frequency`.
4. Spectral centroid as a last-resort display coordinate.

The fallback should be centralized in a small frontend helper so tests can
exercise the decision without canvas assertions.

## 7. Band-Limited Peak

Keep the existing `peak_frequency` descriptor for compatibility, but add
`band_limited_peak_frequency` for display and analysis.

The current `peak_frequency` is computed from the full mean spectrum, so it can
be hijacked by low-frequency rumble. The band-limited peak should compute the
mean-spectrum peak only inside:

- lower bound: `band_peak_min_frequency_hz`, default `100.0`
- upper bound: `band_peak_max_frequency_hz`, default equal to the effective
  ridge maximum

The high-band energy ratio should use:

- lower bound: `high_band_min_frequency_hz`, default `1000.0`

For high-frequency whistle/shriek work, new v3 jobs should increase the
effective ridge upper bound from the current 3000 Hz default. Recommended
default:

- `ridge_max_frequency_hz`: `6000.0`

This stays below the 16 kHz target sample rate Nyquist frequency while allowing
events above the current 3 kHz ceiling to be represented. If the sample rate is
lower or a user supplies a higher value, clamp to a safe fraction of Nyquist.

## 8. Descriptor Versioning

Introduce a new default tokenizer/descriptor version:

- `crnn-event-encoder-v3`

Why version:

- Existing artifacts have a 14-entry descriptor vector.
- The tokenization signature already includes `tokenizer_version`,
  `descriptor_config_json`, and preprocessing config.
- Completed timeline views are artifact-authoritative and must remain readable.

For v3, append the eight ridge display descriptors to the descriptor block.
Total descriptor count becomes 22.

The default `descriptor_weight` should be adjusted so the overall descriptor
block contribution does not jump simply because eight display descriptors were
added. A reasonable default is:

- current v2: `8 / 14 = 0.571`
- proposed v3: `8 / 22 = 0.364`

This mirrors the existing scaling intent: keep aggregate descriptor influence
roughly comparable to the original eight-descriptor block. Existing explicit
preprocessing configs remain honored.

## 9. Piano Roll Rendering

Add a new Y mode:

- `ridge`: label "Ridge"

For jobs with ridge descriptors, `ridge` should be the default Y mode. For
older jobs without the descriptors, keep the existing default behavior.

Rendering logic for `ridge` mode:

1. Resolve a display band using the ridge trust rules.
2. If trusted ridge bounds exist:
   - `centerFrequency = ridge_median_frequency`
   - `lowFrequency = ridge_low_frequency`
   - `highFrequency = ridge_high_frequency`, optionally expanded to the
     spectral centroid for broad high-band harmonic envelopes
3. Else use the fallback center frequency and a compact minimum height.
4. Convert low/high to y coordinates.
5. Render one rectangle per event.
6. Apply a minimum visual height so very steady tones remain selectable.

The existing slope line remains useful and should continue to use
`ridge_log_frequency_slope`.

Tooltips should include:

- `ridge_median`
- `ridge_band` or `ridge_low` / `ridge_high`
- `ridge_coverage`
- `ridge_energy`
- `band_peak`

No explanatory in-app text is required beyond compact labels in existing
tooltip rows.

## 10. UI Frequency Range

The piano roll already allows 1500, 2000, 3000, 4000, and 5000 Hz views.

For v3 ridge descriptors, consider two small UI adjustments:

- Increase the maximum frequency option to 6000 Hz and use it as the default
  vertical range for v3 artifacts with ridge-frequency descriptors.
- When switching to Ridge mode, avoid silently changing the user's current
  frequency range, but allow the token rectangles to clamp visibly at the top
  if the selected range is too low.

Automatic range expansion is intentionally out of scope for the first pass. It
can make review feel slippery when users are zooming manually.

## 11. API and Artifact Impact

No new endpoint is required. The existing Event Encoder timeline endpoint
already returns descriptor maps from the artifact rows.

Artifact changes:

- `event_vectors.parquet`: add appended ridge summary descriptor columns.
- `event_tokens.parquet`: add appended ridge summary descriptor columns.
- `manifest.json`: `descriptor_feature_names` grows from 14 to 22 for v3 jobs.
- `report.json`: descriptor summaries include the appended descriptors.

Old artifacts continue to work because the frontend reads descriptor maps and
must tolerate missing ridge fields.

## 12. Implementation Sketch

### `src/humpback/sequence_models/event_encoder.py`

- Add ridge summary descriptors to `DESCRIPTOR_ORDER` and `DESCRIPTOR_UNITS`.
- Refactor `_compute_ridge_path()` into a helper that can also report:
  - total STFT frames
  - selected ridge frequencies in Hz
  - selected ridge strengths
  - per-frame band energy or local energy denominator
- Add `_compute_ridge_summary_descriptors(...)`.
- Add `_band_limited_peak_frequency(...)`.
- Merge new descriptors into `compute_acoustic_descriptors()`.

### `src/humpback/schemas/sequence_models.py`

- Add descriptor config fields:
  - `ridge_summary_low_percentile`
  - `ridge_summary_high_percentile`
  - `band_peak_min_frequency_hz`
  - `band_peak_max_frequency_hz`
  - `high_band_min_frequency_hz`
- Update default `tokenizer_version` to `crnn-event-encoder-v3`.
- Update default `ridge_max_frequency_hz` to `6000.0`.
- Update default `descriptor_weight` to `0.364`.

### `src/humpback/workers/event_encoder_worker.py`

- Forward new descriptor config fields to `compute_acoustic_descriptors()`.
- Preserve artifact-authoritative read behavior.

### Frontend

- Add a small helper that resolves event display frequency bands.
- Add `ridge` Y mode to `EventEncoderPianoRollPage`.
- Use ridge low/high for rectangle height when trusted.
- Fall back cleanly for v2 artifacts.
- Add tooltip rows for ridge summaries.

## 13. Testing

### Backend Unit Tests

- Clean high-frequency sine or chirp:
  - `ridge_median_frequency` near expected frequency.
  - `ridge_low_frequency` and `ridge_high_frequency` bracket the path.
- Low-rumble plus high whistle:
  - legacy `peak_frequency` may be low.
  - `band_limited_peak_frequency` and ridge median remain high.
- One-frame outlier in a ridge path:
  - trimmed low/high bounds resist the outlier.
- Silence and degenerate audio:
  - all new descriptors are finite 0.0.
- Descriptor vector shape:
  - v3 descriptor vector has 22 entries.

### Frontend Unit Tests

- Trusted ridge descriptors choose ridge median and low/high bounds.
- Weak ridge coverage falls back to F0.
- Unvoiced event with no F0 falls back to band-limited peak.
- Missing v3 fields keep v2 artifacts renderable.

### E2E Coverage

- Piano roll route still renders v2 mock data.
- Piano roll route renders v3 mock data with high-frequency tokens above
  low-frequency tokens in Ridge mode.
- Tooltip displays compact ridge summary rows when fields are present.

## 14. Risks

- Raising `ridge_max_frequency_hz` can make the ridge tracker chase high-band
  noise in weak events. `ridge_energy_ratio`, `ridge_coverage`, trimmed bounds,
  and fallback rules mitigate this.
- More descriptor columns can shift tokenization. Versioning and descriptor
  weight scaling reduce surprise and keep existing jobs stable.
- Ridge summaries are still event-level summaries, not frame-level pitch
  curves. They are enough for piano roll placement, but not enough for high
  quality MIDI/MPE extraction.

## 15. Future Work

If MIDI/MPE extraction becomes a committed feature, add a separate contour
artifact rather than overloading event descriptors:

- `event_contours.parquet`
- one or more rows per event frame
- primary ridge frequency, amplitude, confidence, optional secondary ridges

That future artifact can drive pitch-bend curves and expression data. It is not
needed for the current piano roll readability fix.
