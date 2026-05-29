# Piano Roll Notes v6 — F0 Contour De-spike

**Status:** Approved (brainstorming, 2026-05-29)
**Supersedes default:** v5 (`DEFAULT_EXTRACTOR_VERSION` bumps `"v5"` → `"v6"`)
**ADR:** ADR-072 (to be written in implementation)

## 1. Problem

The v5 harmonic-Viterbi F0 extractor produces contours that are usually
smooth (the Viterbi `transition_lambda` penalises frame-to-frame jumps),
but occasionally a strong-enough wrong-octave / wrong-harmonic emission
over a short run of frames produces a surviving **spike**: the F0 jumps
far off the local trajectory and immediately returns. Because harmonic
siblings inherit the F0 bend by cents conservation, every harmonic
ribbon mirrors the same excursion, and the exported MPE pitch bends carry
the artifact.

Reference case — Event Encoder job `690580c5-7804-43c9-bd8d-690691b5d6d4`:

- Event `669849340bff411390e5eaaf1ec9b9e9`: F0 glides gently around
  MIDI 73 for ~1.1 s, then plunges ~15 semitones to ~MIDI 58 and recovers
  over ~0.1 s near t≈1.2 s. The harmonic ribbons (MIDI 82–100) inherit
  the identical dip.
- Event `0be3d3520789414cb1f494eec50ba641`: same family of short steep
  excursions.

The spike's slope (~25 oct/s) is roughly 60× the legitimate glide's slope
(~0.4 oct/s), so a slope threshold separates the two with a large margin.

## 2. Goal

Add a post-decode **de-spike pass** over the F0 contour so spikes are
erased while the note stays a single continuous contour, and ship it as a
new canonical extractor version **v6** that becomes the default.

Non-goals: changing the Viterbi emission/decode model, the CQT, the STFT
ridge, the parquet schema, the MPE synth, or any frontend behaviour.

## 3. Decisions (locked during brainstorming)

1. **Removal semantics — excise + bridge.** Spike frames are *retained*
   in the contour (timing, `frame_index`, `contour_frame_count`, and
   per-frame `strength` unchanged); only their log-frequency is
   overwritten by linear interpolation across the excised span. The note
   remains one continuous, smoothed contour with the excursion gone. No
   note splitting, no time gaps.
2. **Detection — steep is always an error.** A pure slope threshold; no
   return-to-baseline guard. Any transition steeper than the threshold is
   treated as an artifact. (The data has no genuine pitch motion faster
   than the threshold.)
3. **Algorithm — slew-rate anchor walk (Approach A).** A single-pass
   anchor walk, the robust generalisation of "find the steep jump out,
   find the steep jump back, remove between."

## 4. Algorithm

Operates on each decoded F0 segment — the `list[_RefinedFrame]` runs that
v5's `_decode_f0` returns — *before* note building. Per-frame log
frequencies are `lf[0..n-1]`; the per-frame time step is
`dt = cqt.hop_length / target_sample_rate` (≈ 11.6 ms at 22050 Hz /
hop 256). The per-frame log-frequency budget is
`max_step_log = max_slope_oct_per_s * dt` (log2 units = octaves).

Walk left → right holding a trusted `anchor` index (starts at 0):

1. For the next frame `i`, the legal envelope from the anchor is
   `|lf[i] - lf[anchor]| <= (i - anchor) * max_step_log`.
2. If frame `i` is inside the envelope, accept it: any frames strictly
   between `anchor` and `i` that were skipped get bridged (see step 4),
   then `anchor = i`.
3. If frame `i` is outside the envelope, it is a spike frame; advance
   without moving the anchor and keep scanning for the next in-envelope
   frame.
4. **Bridge.** When a later frame `j` re-enters the envelope, every
   excised frame `k` with `anchor < k < j` has its log frequency replaced
   by linear interpolation between `lf[anchor]` and `lf[j]`:
   `lf[k] = lf[anchor] + (lf[j] - lf[anchor]) * (k - anchor) / (j - anchor)`.
5. **Max-spike-frames guard.** If an excursion runs longer than
   `max_spike_frames` without re-entering the envelope, accept the frame
   at `anchor + max_spike_frames` as a new anchor (treat it as a genuine
   level change rather than excising the remainder of the segment). This
   bounds worst-case behaviour; in practice spikes are far shorter.
6. **Edges.** A spike that begins at frame 0 (no left anchor) or runs to
   the final frame (no right anchor to bridge to) is filled by holding
   the nearest legal value — constant extrapolation, not interpolation.

The pass returns a new `list[_RefinedFrame]` (the dataclass is frozen);
`strength` and `subharmonic_octave` (always 0 in v5/v6) are carried
through unchanged.

### 4.1 Harmonic inheritance

De-spiking runs before `_build_f0_note` and `_build_harmonic_notes`
(reused unchanged from v3). Harmonic presence is searched at
`n · (cleaned f0)` and harmonic bends reuse the cleaned F0 cents (cents
conservation), so the harmonic ribbons are corrected with no separate
pass.

## 5. Parameters

New `DespikeParams` dataclass on the v6 extractor params:

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `True` | Master switch; `False` makes v6 byte-identical to v5. |
| `max_slope_oct_per_s` | `6.0` | Slope threshold (≈ 72 semitones/s). Sits ~4× above the reference glide and ~4× below the reference spike. First knob to tune on the test-bed. |
| `max_spike_frames` | `12` | Excursion-width guard (~140 ms; the reference spike is ~9 frames). |

## 6. Packaging

`extract_notes_v6` in `src/humpback/processing/note_extractor_v6.py` is
structurally `extract_notes_v5` with one inserted step:

```
v5._decode_f0(...)  ->  despike_f0_segments(...)  ->  v3._build_f0_note / _build_harmonic_notes
```

It returns the same `NotesV3Result`. Output artifacts are
`event_notes_v6.parquet` / `event_note_contours_v6.parquet` with the
identical column set used by v3–v5; `subharmonic_octave` is written as 0.

### 6.1 Worker, model, and resolver wiring (single-phase promotion)

- `DEFAULT_EXTRACTOR_VERSION` bumps `"v5"` → `"v6"` in
  `src/humpback/models/piano_roll_notes.py`. New Event Encoder
  completions auto-enqueue v6; new MIDI export rows default to v6.
- `piano_roll_notes_worker.py`: add `_V6_EXTRACTOR_VERSION = "v6"`, add it
  to `_RIDGE_AWARE_VERSIONS` and `_RIDGE_AWARE_EXTRACTORS`, add an
  `_extract_notes_v6` async wrapper mirroring `_extract_notes_v5`, add a
  `despike` section + `DespikeParams` to `_ResolvedParams` /
  `_resolve_params` / `to_json_dict`, and extend the version-conditional
  defaults (30 Hz STFT floor, `min_break_frames = 6`, `pad_seconds = 0.25`)
  to include v6.
- MIDI export resolver requires **no change**: it orders by
  `desc(extractor_version)` and `"v6" > "v5"` lexicographically, so a
  complete v6 row wins automatically. MPE routing keys on the presence of
  the `note_uid` column (present in v6), so v6 routes to the MPE Lower
  Zone path identically to v3–v5.
- Frontend requires **no change**: the version string flows through the
  API verbatim and the only special-cased label is v3 ("MPE v3"); v6
  displays as `"v6"`.
- Legacy v1–v5 rows on disk remain queryable via explicit
  `extractor_version` pinning; no auto-backfill of v6.

### 6.2 Test-bed

Register a `"v6"` variant in `tools/piano_roll_notes_registry.py` (a
`_run_v6` wrapper mirroring `_run_v5`) so the permanent debug CLI can
render v5-vs-v6 side by side on any event.

## 7. Testing

- **Pure de-spike unit tests** (`tests/processing/test_note_extractor_v6.py`):
  synthetic contour with an out-and-back spike → bridged to the straight
  line; a sustained legal glide → unchanged; a leading spike and a
  trailing spike → held by extrapolation; an excursion longer than
  `max_spike_frames` → far level accepted; a multi-spike segment → all
  bridged. Plus an `extract_notes_v6` smoke test on synthetic audio
  asserting the same `NotesV3Result` contract as v5 and that
  `enabled=False` reproduces v5 output.
- **Worker dispatch test**: v6 routes through the ridge-aware path, writes
  `event_notes_v6.parquet` + `event_note_contours_v6.parquet`, and
  `_resolve_params(..., "v6")` applies the v6 defaults and parses a
  `despike` override section.
- **Registry/test-bed test** (`tests/tools/test_piano_roll_notes_debug.py`):
  `"v6"` is a known variant.
- **Visual confirmation**: render both reference events v5-vs-v6 with the
  test-bed and confirm the spikes are gone and the surrounding contour is
  unchanged.

## 8. Acceptance

- The reference event `669849340bff411390e5eaaf1ec9b9e9` renders under v6
  with the t≈1.2 s plunge erased and the rest of the contour visually
  unchanged from v5.
- `enabled=False` makes v6 output byte-identical to v5 for the same event.
- All standard gates pass (`ruff format --check`, `ruff check`,
  `pyright`, `pytest tests/`).
