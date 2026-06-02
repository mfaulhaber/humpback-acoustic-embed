# Piano Roll Notes v7 — Residual Discontinuity Splits and Ridge-Guided Bend Rescue

**Status:** Draft design (2026-06-02)
**Supersedes default:** v6 (`DEFAULT_EXTRACTOR_VERSION` would bump `"v6"` -> `"v7"`)
**Primary domain:** `sequence-models`
**Neighbor domains:** none for the extractor-only change; load `frontend-shell` only if the Notes view rendering contract changes.

## 1. Background

Recent Piano Roll Notes commits established this progression:

- v3: ridge-aligned notes and MPE contours.
- v4: HPS-style F0 selection with lower-band support.
- v5: direct harmonic-sum CQT emission plus log-frequency Viterbi smoothing.
- v6: v5 decode plus an out-and-back slope de-spike pass.

The v6 pass is intentionally surgical: it bridges short excursions that return
to the local slope envelope, while preserving non-returning level changes so it
does not destroy real register shifts. That solved the known `669849...` style
spikes and preserved the corrected event `e3e8e9c4b5c0403a91ef3f50e965fdf0`,
where v6 reduces v5's jagged >40 octaves/s contour to a smooth <=5.7
octaves/s pitch change matching the spectrogram energy.

New review of Event Encoder job `690580c5-7804-43c9-bd8d-690691b5d6d4`
shows two remaining failure modes.

## 2. Findings

### 2.1 False-positive bends: residual discontinuities

Persisted v6 contours still contain large steps after de-spiking:

| Event | v6 residual F0 step |
|---|---|
| `5a50b01f7a014ceaaf5ca52e81e6ad42` | about MIDI 34.7 -> 71.0 in ~23 ms; later about 71.3 -> 34.3 over ~58 ms |
| `dad6357550ba4a9cadcf05e103f923ee` | about MIDI 70.7 -> 34.7 in ~23 ms; other branch jumps remain |
| `0d698411766142ee908fa6a4c2ac81ec` | about MIDI 34.3 -> 51.7 in ~23 ms |

The debug renders show these are not the same as v6's original spike target.
They are mostly non-returning jumps between branches, often across short
unvoiced gaps that v5/v6 intentionally merge with `min_break_frames = 6`.
Because `_build_f0_note` emits one note for the whole merged segment, the
renderer and MPE synth connect the contour rows across the gap and encode the
branch change as a violent pitch bend.

This means v6 is doing the right thing by not bridging those jumps, but the
next stage needs to avoid representing the jump as a continuous bend.

### 2.2 False-negative bends: flat low-F0 decode under a moving ridge

Event `f470e42f95974358885c9392dffd83ee` has a clear rising dominant ridge in
the debug render, roughly from the 800 Hz band toward the 1500 Hz band. The
Event Encoder descriptor sidecar agrees: `ridge_log_frequency_slope` is about
0.58 octaves/s and the ridge frequency span is about 514 Hz.

Persisted v5/v6 notes, however, render a mostly flat harmonic stack. The v6 F0
span is only about 0.3 semitones, so all harmonic siblings inherit nearly zero
cents movement. This is upstream of de-spiking: the harmonic-Viterbi path has
parked on a low subfundamental while the dominant acoustic ridge moves.

## 3. Goals

1. Remove high-slope MPE pitch bends that come from residual branch
   discontinuities.
2. Preserve correct continuous pitch changes like `e3e8e9c4...`.
3. Recover obvious upsweep bends when the decoded F0 is flat but the persisted
   STFT ridge is confidently moving.
4. Keep the v3-v6 parquet schema, MPE synth, and frontend Notes view contracts.

Non-goals:

- Multi-F0 event extraction.
- Retuning the whole v5 Viterbi objective globally.
- Changing MIDI synthesis to post-filter bend events. The contour sidecar
  should be correct before MIDI export.

## 4. Approaches

### Approach A — Split Residual Steep Discontinuities

After v6 de-spiking, scan each decoded F0 segment for adjacent retained frames
whose absolute log-frequency slope still exceeds `max_continuous_slope_oct_per_s`
over the actual time delta. Cut the segment at that boundary before
`_build_f0_note`.

This converts a non-returning branch jump into separate notes rather than one
MPE note with a huge pitch bend. Harmonic siblings are already built per F0
note, so the fix naturally propagates without MIDI or frontend changes.

Pros:

- Directly addresses `5a50...`, `dad635...`, and `0d698...`.
- Leaves v6's out-and-back bridging semantics intact.
- Uses the existing note-building path.
- Protects `e3e8...` if the threshold stays at v6's 6.0 octaves/s, since its
  corrected contour is below that threshold.

Cons:

- Does not recover missing upsweeps such as `f470...`.
- Can create additional short F0 notes. Fragments below `min_note_frames`
  should be dropped to avoid clutter.

### Approach B — Ridge-Guided Bend Rescue for Flat Decodes

Use the persisted Event Encoder ridge sidecar that v6 already loads but ignores.
For each post-v6 F0 segment, apply a targeted rescue only when all of these are
true:

- The decoded F0 span is nearly flat, for example <= 2 semitones.
- The overlapping STFT ridge span is clearly moving, for example >= 5
  semitones or a minimum octaves/s slope.
- Ridge coverage over the segment is sufficient.
- The ridge-to-F0 ratio is plausibly harmonic-stable over the segment.

When the gates pass, choose a carrier harmonic index from the median
`ridge_frequency / decoded_f0_frequency` ratio, then replace the segment's F0
log-frequency with `ridge_log_frequency - log2(carrier_harmonic)`.

This keeps the single-F0-plus-harmonics architecture while borrowing the
dominant acoustic ridge as the bend carrier. The visible harmonic stack then
inherits the ridge's upsweep via cents conservation.

Pros:

- Targets the `f470...` false negative without loosening global Viterbi
  smoothness.
- Reuses the high-quality ridge sidecar already written by the Event Encoder.
- Keeps output schemas and MIDI/frontend behavior unchanged.

Cons:

- Needs careful gating so broad high-band energy does not hijack flat true-F0
  events.
- Carrier harmonic choice can saturate at `HarmonicSearchParams.max_harmonic`
  when the dominant ridge sits above the represented harmonic range.
- Requires visual test-bed review on more examples than split-only v7.

### Approach C — Global Viterbi Retune

Increase `transition_lambda`, adjust `min_harmonics_present`, relax H1
prominence, or widen the harmonic stack so the decoder avoids low flat paths
and high branch jumps.

Pros:

- One-stage algorithm; no post-processing.

Cons:

- High risk: the current v5/v6 decode is already a hard-won compromise.
- Tightening transitions can flatten real glides; loosening gates can bring
  back harmonic flapping.
- Does not directly solve contour rows being connected across short gaps.

Rejected for v7 as the primary strategy.

## 5. Proposed v7

v7 should be v6 plus two post-decode steps before note building:

1. `split_residual_discontinuities(...)`
   - Input: de-spiked F0 segments from v6.
   - Default threshold: `max_continuous_slope_oct_per_s = 6.0`.
   - Cut between adjacent frames when the actual slope exceeds the threshold.
   - Keep only resulting fragments with at least `segmentation.min_note_frames`.
   - This is the baseline fix for false-positive bends.

2. `rescue_flat_segments_from_ridge(...)`
   - Input: split segments plus optional `ridge_sidecar_rows`.
   - Defaults to conservative gates: decoded span <= 2 semitones, ridge span >=
     5 semitones, minimum overlap frames, and stable harmonic ratio.
   - Rewrite only the F0 log-frequency values for gated flat segments.
   - Keep timing, frame indexes, strength, and `subharmonic_octave = 0`.
   - This is the targeted fix for false-negative upsweeps.

Then reuse `_build_f0_note` and `_build_harmonic_notes` unchanged.

Parameter shape:

| Section | Field | Default | Purpose |
|---|---|---|---|
| `despike` | existing v6 fields | unchanged | Preserve v6 out-and-back bridging and trailing trim |
| `discontinuity` | `enabled` | `True` | Enable residual split pass |
| `discontinuity` | `max_continuous_slope_oct_per_s` | `6.0` | Maximum continuous bend slope before splitting |
| `ridge_rescue` | `enabled` | `True` | Enable flat-segment ridge rescue |
| `ridge_rescue` | `max_decoded_span_semitones` | `2.0` | Only rescue near-flat decoded F0 |
| `ridge_rescue` | `min_ridge_span_semitones` | `5.0` | Require a clearly moving ridge |
| `ridge_rescue` | `min_overlap_frames` | `8` | Avoid short/noisy decisions |
| `ridge_rescue` | `max_ratio_mad_semitones` | TBD, start 2.0 | Require stable harmonic relationship |

## 6. Packaging

- Add `src/humpback/processing/note_extractor_v7.py`.
- Add `_V7_EXTRACTOR_VERSION = "v7"` and `_extract_notes_v7` in
  `src/humpback/workers/piano_roll_notes_worker.py`.
- Include v7 in ridge-aware extractor dispatch because it consumes the Event
  Encoder ridge sidecar for rescue.
- Bump `DEFAULT_EXTRACTOR_VERSION` to `"v7"` in
  `src/humpback/models/piano_roll_notes.py`.
- Register `"v7"` in `tools/piano_roll_notes_registry.py`.
- No Alembic migration. Output sidecars remain
  `event_notes_v7.parquet` and `event_note_contours_v7.parquet` with the
  v3-v6 schema.
- MIDI export resolver and frontend Notes view require no changes.

## 7. Test Plan

Pure processing tests:

- Residual non-returning branch jump splits into two notes instead of one
  contour with a high-slope bend.
- Correct `e3e8...`-style smooth pitch change stays one note under the default
  threshold.
- Short post-split fragments below `min_note_frames` are dropped.
- `ridge_rescue` leaves non-flat decodes unchanged.
- `ridge_rescue` rewrites a synthetic flat F0 under a moving harmonic ridge and
  preserves cents conservation for harmonics.
- `ridge_rescue` refuses unstable ridge/F0 ratios.

Worker and tooling tests:

- v7 worker writes `event_notes_v7.parquet` and
  `event_note_contours_v7.parquet`.
- `_resolve_params(..., "v7")` parses `discontinuity` and `ridge_rescue`
  sections and inherits v6's v5-style defaults (`pad_seconds = 0.25`,
  `min_break_frames = 6`, 30 Hz STFT floor).
- Debug registry accepts `"v7"`.

Manual visual checks with `tools/piano_roll_notes_debug.py`:

- `5a50b01f7a014ceaaf5ca52e81e6ad42`: no vertical bend between low and high
  branches.
- `dad6357550ba4a9cadcf05e103f923ee`: branch changes become note boundaries.
- `0d698411766142ee908fa6a4c2ac81ec`: offscreen low branch no longer bends
  into the visible harmonic stack.
- `e3e8e9c4b5c0403a91ef3f50e965fdf0`: smooth correct pitch change remains
  continuous.
- `f470e42f95974358885c9392dffd83ee`: harmonic stack inherits the visible
  rising ridge instead of staying flat.

## 8. Open Questions

- Should ridge rescue run before or after residual splitting? Proposed order is
  split first, then rescue each stable flat fragment; this avoids using a ridge
  to smooth over a discontinuity that should become a note boundary.
- Should the carrier harmonic be capped at `max_harmonic` or allowed above it
  only for bend estimation? Capping keeps harmonic note construction unchanged,
  but allowing a higher virtual carrier may align the F0 contour more exactly
  when the dominant ridge is above H16.
- Should short unvoiced gaps produce renderer contour breaks even when kept in
  one note? That would require frontend/MIDI contour semantics beyond v7's
  extractor-only scope, so it is deferred unless split-only proves insufficient.
