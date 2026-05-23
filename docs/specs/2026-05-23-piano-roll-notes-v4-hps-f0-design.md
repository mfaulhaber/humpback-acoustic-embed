# Piano Roll Notes v4 — HPS-based F0 with extended low band

**Date**: 2026-05-23
**Status**: Draft
**Builds on**: ADR-063 (v3 ridge descriptors), ADR-067 (per-frame harmonic labeling), ADR-069 (ridge-aligned MPE export)
**Triggered by**: User report on Event Encoder job `690580c5-7804-43c9-bd8d-690691b5d6d4` — low-frequency spectrum visible in the spectrogram is not reflected in the extracted Piano Roll Notes; F0 notes routinely lock to H2/H3 above the true fundamental.

---

## 1. Problem

ADR-069 introduced the v3 ridge-aligned F0 + harmonics extractor. The pipeline is: STFT ridge (Viterbi) → per-frame subharmonic refinement (`_refine_subharmonic`) → coherent-contour F0 segmentation → CQT harmonic-sibling search. In production the refinement step under-corrects on the textbook humpback song case: a low F0 whose 2nd or 3rd harmonic carries comparable or greater spectral energy. The Viterbi tracker locks onto the harmonic; `_refine_subharmonic` only shifts down when both the noise-floor gate (`k_sub · MAD`) **and** the relative-magnitude gate (`min_relative_log_magnitude = -2.5`, ≈22 dB) pass — frames where the true F0 is more than ~22 dB weaker than the ridge get rejected.

### Evidence from job 690580c5 (1635 events, 56,352 notes, 463,963 contour rows)

Pitch distribution is severely top-heavy:

| Band | MIDI 45–69 (≤ A4) | MIDI 70–89 | MIDI 90–109 |
|---|---|---|---|
| Note count | 5,022 (9%) | 11,481 (20%) | 39,847 (71%) |

Per-partial breakdown shows only **10% of notes are F0 (`partial_index = 0`)**; 90% are harmonics H2..H16. F0 notes' median pitch is **MIDI 69 (440 Hz)** — too high for typical humpback song fundamentals.

Ridge-tracker output range:

| Quantile | Frequency | MIDI |
|---|---|---|
| min | 109 Hz | 44.9 |
| p10 | 297 Hz | 62.2 |
| median | 687 Hz | 76.7 |
| p90 | 1281 Hz | 87.5 |
| max | 5266 Hz | 112.0 |

- **0 ridge frames below 100 Hz** (the `STFTParams.min_frequency_hz = 100.0` floor silently caps every candidate).
- Only **6% of events** have any ridge frame below 200 Hz.
- Subharmonic refinement fires on **39% of F0 contour frames** (shift > 0), confirming the raw ridge often locks on a harmonic — but post-refinement the median is still 440 Hz, so the existing gates over-reject true F0s.

### Sub-100 Hz noise characterization (wide-CQT analysis of this job's source audio)

| Band | log-mag p50 | log-mag p99 | log-mag max |
|---|---|---|---|
| 20–100 Hz | -6.61 | -3.02 | -0.83 |
| 200–1500 Hz (mid) | -5.89 | -3.15 | +0.99 |

- Sub-100Hz median is ~0.7 log units (~6 dB) **weaker** than the mid band (broadband hydrophone self-noise + infrasonic rumble dominate).
- **3.8% of sub-100Hz frame-bins exceed the mid band's 90th percentile** — these are real moans poking out of the noise floor.
- Sub-100Hz p99 actually *exceeds* mid p99: at the strong end, the bands overlap.

Implication: a hard single-bin energy threshold cannot separate sub-100 Hz signal from noise. A multi-bin harmonic-support test can, because broadband noise lacks coherent harmonic structure.

## 2. Goal

Recover low-frequency F0s that v3 missed, without re-introducing the false-positive subharmonic behavior that motivated v3's `min_relative_log_magnitude` gate. Concretely:

1. F0 selection considers candidates as far down as `ridge / 6`, not just `ridge / {1, 2, 4, 8}`.
2. Selection is driven by total harmonic-stack support, not single-bin magnitude at `f₀/2`.
3. The ridge tracker can descend to ~30 Hz so HPS isn't capped by a band floor that pre-dates this design.
4. A sub-100 Hz candidate only wins when it has genuine harmonic support, so broadband noise doesn't pull the F0 down.
5. Existing v3 sidecars, MIDI exports, MPE channel layout, and the renderer are untouched.

## 3. Non-goals

- Multi-F0 per event (still out of scope; one F0 contour per event).
- Changes to harmonic-sibling search (`_build_harmonic_notes`), MPE Lower Zone layout, MIDI export, or `event_notes_*.parquet` / `event_note_contours_*.parquet` schemas.
- Auto-backfill of v4 for completed v3 jobs (manual via job-admin UI, mirroring the v3 launch pattern).
- Frontend rendering changes beyond what falls out of the wider pitch range becoming populated.

## 4. Decision

Ship as **note extractor v4**, a single-stage replacement of `_refine_subharmonic` with HPS-style harmonic-stack F0 scoring, plus the prerequisite band-floor change.

### 4.1 Pipeline shape

```
STFT ridge (Viterbi, single path)          ← unchanged
        ↓
[v3] _refine_subharmonic (octave halving)  ← REMOVED in v4 code path
[v4] _score_f0_candidates (HPS)            ← NEW
        ↓
_segment_f0_runs                            ← unchanged
        ↓
_build_f0_note + _build_harmonic_notes     ← unchanged
```

### 4.2 HPS F0 scoring (per frame)

For each ridge frame at log₂-frequency `L` and CQT column `c`:

1. Build candidate F0 log₂-frequencies `L_c = L − log₂(d)` for `d ∈ candidate_divisors = (1, 2, 3, 4, 5, 6)`.
2. For each candidate `L_c`, evaluate harmonic-stack support across `n = 1 .. n_harmonics` (default 8):
   - Target log₂-frequency for harmonic `n`: `H_n = L_c + log₂(n)`.
   - If `H_n` exceeds `log₂(cqt_bin_freqs[-1])`, skip (harmonic out of CQT range).
   - In the CQT column, take the max log-magnitude within `±cents_tolerance` (default 50¢) of `H_n`: call this `m_n`.
   - Per-frame noise floor `floor = median(bottom_half(c))`, `mad = MAD(bottom_half(c))` (already used by `_frame_noise_floor`). Define `threshold_n = floor + k_noise · mad`.
   - `is_present[n] = (m_n ≥ threshold_n)`.
   - `contribution[n] = max(0.0, m_n − floor)` (clipped to 0 when sub-floor so absent harmonics don't subtract).
3. Score:
   ```
   raw_score = Σ_n contribution[n]
   count_present = Σ_n is_present[n]
   k_min = low_band_min_harmonics if (2^L_c < low_band_threshold_hz) else high_band_min_harmonics
   if count_present < k_min:
       score(L_c) = -inf
   else:
       score(L_c) = raw_score − (low_band_penalty if 2^L_c < low_band_threshold_hz else 0.0)
   ```
4. Choose `d* = argmax_d score(L_c)`. On ties, prefer the largest `d` (i.e., the lowest-frequency candidate), preserving the "prefer F0 over an alias" intent that motivated v3's subharmonic refinement.
5. Record `(frame_index, time_offset_s, log_frequency = L_c*, strength = ridge_strength, divisor = d*)`.

After per-frame scoring, majority-smooth the integer `divisor` stream over `smoothing_frames = 5` (reuse `_majority_smooth`). Substitute each frame's `log_frequency` with `L_original − log₂(smoothed_divisor)`.

### 4.3 Parameter table

```python
@dataclass(frozen=True, slots=True)
class HPSParams:
    n_harmonics: int = 8
    cents_tolerance: float = 50.0
    k_noise: float = 2.0
    candidate_divisors: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    smoothing_frames: int = 5
    low_band_penalty: float = 0.5
    low_band_threshold_hz: float = 100.0
    low_band_min_harmonics: int = 3
    high_band_min_harmonics: int = 2
```

`STFTParams` modified:

```python
@dataclass(frozen=True, slots=True)
class STFTParams:
    n_fft: int = 1024
    hop_length: int = 512
    min_frequency_hz: float = 30.0   # was 100.0 in v3
    max_frequency_hz: float = 6000.0  # unchanged
    candidate_count: int = 5
    smoothness_penalty: float = 8.0
    peak_prominence_ratio: float = 0.0
```

`ExtractNotesV4Params` is a sibling of `ExtractNotesV3Params` differing only in `hps: HPSParams` replacing `subharmonic: SubharmonicParams`. All other fields (CQT, segmentation, harmonic, MIDI, identity) carry over.

### 4.4 Why these defaults

- **`candidate_divisors = (1, 2, 3, 4, 5, 6)`** — covers ridge-locked-on-H1..H6 cases. H7+ is rare in humpback song context and increases compute. Worker logs warn on `d == 6` selections in case real-world traces show that bound being hit often (then a v5 would extend to 8).
- **`n_harmonics = 8`** — sums enough partials to score 30 Hz fundamentals (8th harmonic at 240 Hz, well within the CQT max ~4.4 kHz at default `target_sample_rate = 22050`, `fmin = 27.5`, `bins_per_octave = 36`, `n_bins = 264`). Higher N adds diminishing returns and noise.
- **`cents_tolerance = 50.0`** — ≈1.5 CQT bins at 36 bins/octave; tolerates moderate vibrato and CQT quantization. Tighter than the 75¢ harmonic-sibling search (which has to accommodate sweeps across the F0 segment, not just a single frame).
- **`low_band_penalty = 0.5`** — 0.5 log units ≈ 4 dB. Activates only when sub-100 Hz and ≥ 100 Hz candidates score within ~4 dB of each other (genuinely ambiguous). Strong moans outscore noise by far more than 0.5.
- **`low_band_min_harmonics = 3` / `high_band_min_harmonics = 2`** — based on the 3.8% sub-100 Hz "above mid-band p90" frequency: requiring 3 coincident above-floor bins at expected harmonic ratios drops the joint probability of an infrasonic noise false-positive to a negligible level. Above 100 Hz, signal is cleaner, so 2 harmonics suffice to keep short whistles addressable.

### 4.5 Persisted contracts

- **`extractor_version = "v4"`**.
- **`DEFAULT_EXTRACTOR_VERSION = "v4"`** in `piano_roll_notes_worker.py`. Newly-completing encoder jobs auto-enqueue v4 notes jobs.
- **Sidecar paths**:
  - `event_encoders/{job_id}/event_notes_v4.parquet` — identical schema to `event_notes_v3.parquet`.
  - `event_encoders/{job_id}/event_note_contours_v4.parquet` — identical schema to `event_note_contours_v3.parquet`.
- **`subharmonic_octave` column repurposed in v4**: stores `candidate_divisor − 1` (0 = ridge is F0, 1 = ridge is H2, …, 5 = ridge is H6). v3 semantics (octaves halved) and v4 semantics (divisor − 1) are both "how much was the ridge shifted down" so renderers using it for diagnostic display work unchanged. The interpretation is documented in `docs/reference/storage-layout.md` and in the parquet column docstring.
- **`params_json`** in the `piano_roll_notes_jobs` row records every `HPSParams` field plus the `STFTParams` floor, so a re-run with the same row reproduces identical bytes.
- **Export resolver** in `piano_roll_midi_export_worker.py`: change the version-resolution call from `max("v1", "v2", "v3")` to `max("v1", "v2", "v3", "v4")` so completed v4 rows win automatically over v3.

### 4.6 Module structure

- Keep `src/humpback/processing/note_extractor_v3.py` as-is; it is now a frozen legacy module that the worker only loads for `extractor_version == "v3"`.
- Add `src/humpback/processing/note_extractor_v4.py` with:
  - `HPSParams`, `STFTParams` (re-exported with new default for clarity), `SegmentationParams` (unchanged), `HarmonicSearchParams` (unchanged), `MidiRangeParams` (unchanged), `ExtractNotesV4Params`.
  - `extract_notes_v4()` entry point with the same `(audio, sample_rate, *, params, ridge_sidecar_rows=None) -> NotesV3Result` signature as v3 (reuses the `NotesV3Result`, `NoteV3`, `ContourFrame` row types — no schema change).
- Update `piano_roll_notes_worker.py` to dispatch `extract_notes_v3` vs `extract_notes_v4` based on `job.extractor_version`. Add `"v4"` to the recognized version constants near `_V3_EXTRACTOR_VERSION` and rename to `_KNOWN_EXTRACTOR_VERSIONS`.

`HarmonicParams` (already retired in v3) stays retired. `SubharmonicParams` stays in `note_extractor_v3.py` for v3-history reproducibility but is never instantiated for new work.

### 4.7 Migration / rollout

- **No DB migration.** The `piano_roll_notes_jobs` table already supports arbitrary `extractor_version` strings.
- **No auto-backfill** of v4 for completed v3 jobs (mirrors the v3 launch pattern). The frontend job-admin UI already supports deleting per-version rows; user re-enqueues v4 when they want to free space and regenerate.
- **Export worker** automatically prefers v4 once any encoder job has a complete v4 row (see §4.5).
- **Existing v1/v2/v3 sidecars** stay readable on disk; nothing is deleted.

### 4.8 Test plan

Unit tests in `tests/processing/test_note_extractor_v4.py`:
- **Pure tone at 200 Hz**: HPS picks `d=1` (ridge is F0); no shift applied; emitted F0 ≈ 200 Hz.
- **Pure tone at 200 Hz with strong H2 at 400 Hz (12 dB louder)**: Ridge Viterbi locks at 400 Hz; HPS picks `d=2`; emitted F0 ≈ 200 Hz.
- **Pure tone at 200 Hz with strong H3 at 600 Hz (12 dB louder)**: Ridge locks at 600 Hz; HPS picks `d=3`; emitted F0 ≈ 200 Hz.
- **40 Hz fundamental with H2..H6 above broadband sub-100 Hz noise**: HPS picks `d=k_ridge_locked_on / 1`; emitted F0 ≈ 40 Hz; `low_band_min_harmonics = 3` clears.
- **Pure sub-100 Hz noise burst (no harmonics)**: All sub-100 Hz candidates fail `low_band_min_harmonics`; F0 stays at the ridge frequency.
- **Pure tone at 1 kHz, no harmonics**: All `d>1` candidates score 0; `d=1` wins; emitted F0 ≈ 1 kHz.
- **Synthetic FM sweep 50 Hz → 80 Hz over 1 s with H2..H4**: Per-frame HPS picks `d=ridge/F0`; smoothed divisor stays at the right value; emitted F0 segment tracks the sweep.

Integration test in `tests/workers/test_piano_roll_notes_worker.py`:
- Capture three real event audio fixtures from job `690580c5` (events with known low-F0 calls visible in the spectrogram).
- Run both `extract_notes_v3` and `extract_notes_v4`; assert F0 median MIDI drops by ≥ 12 semitones for v4 on each fixture.
- Assert v4 emits ≥ 1 F0 note below MIDI 50 on at least one fixture (v3 emits 0).

### 4.9 Alternatives considered

- **Patch `_refine_subharmonic` (Approach B from brainstorming)**: windowed magnitude check, looser relative gate, more halvings. Cheaper but still assumes the true F0 sits at an octave subdivision of the ridge; misses H3 lock. Rejected.
- **Window + thirds/fifths (Approach C)**: adds explicit `f/3`, `f×2/3` candidates without full HPS. Catches H3 but is ad-hoc and doesn't generalize to H5 or H6 lock. Rejected.
- **Full HPS scan over a coarse F0 grid (no ridge dependency)**: cleaner algorithm but loses the ridge tracker's smoothness prior and adds compute per frame. Rejected for v4 — the ridge is a good seed and the divisor set covers realistic lock cases.
- **Multi-F0 per event**: out of scope (see §3).

## 5. Risks

- **HPS picks `d=6` too often**: would mean ridge tracker routinely locks on H6 (a 6× frequency offset is ~31 semitones, which is a lot of harmonic energy concentration). Mitigated by worker-side logging that emits a warning when `d == max(candidate_divisors)` exceeds a per-job threshold; user can then bump divisors to `(1..8)` in a v5.
- **Visual harmonic flood gets worse**: pushing F0 down means harmonic siblings push down too. Total note count is unchanged, but the harmonic stack starts at lower MIDI. User explicitly chose not to tighten the harmonic gate in this design — acknowledged trade-off.
- **HPS picks a non-existent low F0 from rich-overtone band-limited content**: a harmonic-rich call with weak fundamental and a transmission band that genuinely cuts the F0 (water-air channel) could still be valid as "ridge at H2 with no recoverable F0." The `low_band_min_harmonics = 3` guard limits this; if it becomes a real issue in production, raise the requirement to 4 in a v5.
- **`subharmonic_octave` column semantics change**: documented (§4.5) but downstream code that hard-codes "octave count up to 3" needs a quick audit. The renderer and MPE encoder don't use this field as more than a diagnostic.

## 6. Open questions

- Do we want a per-job aggregate report (e.g., divisor-selection histogram, F0 pitch-range histogram) in `report.json` for tuning visibility? Recommend yes; cheap to compute alongside velocity calibration.
- Should the worker emit a structured log line per event with `(n_frames, divisor_histogram, low_band_hits)` for offline analysis? Recommend yes, at DEBUG level.
