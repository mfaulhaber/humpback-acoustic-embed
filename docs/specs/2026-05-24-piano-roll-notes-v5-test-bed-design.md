# Piano Roll Notes v5 — debug test-bed + harmonic-Viterbi F0 (iteration plan)

**Date**: 2026-05-24
**Status**: Draft
**Builds on**: ADR-064 (notes sidecar worker), ADR-067 (channelized MIDI), ADR-068 (windowed export), ADR-069 (ridge-aligned MPE), ADR-070 (v4 HPS F0)
**Triggered by**: User report on Event Encoder job `690580c5-7804-43c9-bd8d-690691b5d6d4` token #47 — the spectrogram shows smooth, gently-sloping spectral ridges, but the v4 piano roll and exported MIDI for the same event show major frame-to-frame pitch spikes. The same pattern is present on most events in the job (4 of the 5 longest events have F0 spans wider than their ridge spans).

---

## 1. Problem

The v4 HPS extractor (ADR-070) selects, per frame independently, an integer divisor `d ∈ {1..6}` of the STFT ridge frequency that maximizes total harmonic-stack support in the CQT column. Divisors are then majority-smoothed and 3-point-median smoothed, and `F0(t) = ridge(t) / d(t)` is used unchanged.

Two assumptions in this design break for tonal humpback song events:

1. **The ridge isn't a stable harmonic** across frames. The STFT Viterbi tracker picks the strongest CQT bin per frame; for events whose harmonic envelope shifts as the call evolves, the strongest bin jumps between H2/H3/H4/H5/H6 frame-to-frame. The Viterbi smoothness prior on the ridge keeps step sizes locally small but does not prevent the ridge from migrating across the harmonic series over the event's duration.

2. **Divisor selection is per-frame independent**. Even when the ridge actually sits on a different harmonic from one frame to the next, HPS's divisor change should produce a near-identical F0 — but only if the ridge step exactly equals an integer-harmonic ratio. In practice the ridge jumps by non-integer amounts (e.g., ridge 562 → 750 → 828 Hz is `1.33×` then `1.10×`), so `ridge/d` after divisor adjustment is several semitones off the true F0.

### Evidence from token #47 in job `690580c5…`

Event ID `4351c893a3514642b9ea094d09b825df`, 11 ridge frames (~352 ms padded). Raw STFT ridge frequencies, HPS-chosen divisors, and resulting v4 F0:

| Frame | Ridge (Hz) | Ridge MIDI | HPS d | F0 (Hz) | F0 MIDI |
|---|---|---|---|---|---|
| 0 | 515.6 | 71.75 | 4 | 128.9 | 47.75 |
| 1 | 515.6 | 71.75 | 4 | 128.9 | 47.75 |
| 2 | 546.9 | 72.76 | 4 | 136.7 | 48.76 |
| 3 | 562.5 | 73.25 | 5 | 112.5 | 45.39 |
| 4 | 562.5 | 73.25 | 5 | 112.5 | 45.39 |
| 5 | 562.5 | 73.25 | 5 | 112.5 | 45.39 |
| 6 | 750.0 | 78.23 | 5 | 150.0 | 50.37 |
| 7 | 828.1 | 79.95 | 5 | 165.6 | 52.09 |
| 8 | 593.8 | 74.19 | 5 | 118.8 | 46.33 |
| 9 | 578.1 | 73.73 | 6 |  96.4 | 42.71 |
| 10 | 578.1 | 73.73 | 6 |  96.4 | 42.71 |

F0 span: ~9 semitones over 320 ms, with ±5 semitone frame-to-frame jumps. The spectrogram for this event shows smooth, slowly-varying harmonic lines (the true F0 sits near ~125 Hz with gentle motion). The v4 contour does not match.

### Severity across the job

Per-event diagnostic on the 5 longest events in job `690580c5…` (≥160 ridge frames each):

| Event | Ridge frames | Ridge MIDI span | F0 MIDI span | Divisors used |
|---|---|---|---|---|
| `21d13e3c…` | 208 | 28.2 st | **56.3 st** | 1..6 |
| `ae21e747…` | 188 | 20.0 st | **47.7 st** | 1..6 |
| `db7e1b9e…` | 171 | 28.4 st | 28.4 st | 3..6 |
| `2fe09647…` | 164 | 16.7 st | 14.4 st | 4..6 |
| `f8825fe3…` | 161 | 17.5 st | **40.2 st** | 1..6 |

In 4 of 5 cases, F0 span exceeds ridge span — divisor selection per frame amplifies the ridge's frame-to-frame jumps rather than canceling them. The mid-event divisor flips visible in token #47's table (4→5 at frame 3, 5→6 at frame 9) explain most of the resulting F0 discontinuities.

## 2. Goal

Recover F0 trajectories that visually track the smooth spectral lines in the spectrogram, even when (a) the ridge wanders across harmonics within an event, or (b) the strongest CQT bin per frame does not sit at a clean integer multiple of the true F0. Concretely:

1. F0 estimation enforces temporal smoothness as part of the cost function, not as a post-filter on per-frame independent decisions.
2. The algorithm is iterated on against real problem cases (starting with token #47 of job `690580c5…`) using a repeatable rendered-PNG feedback loop.
3. The iteration tool stays in the repository as a permanent debug surface for future Piano Roll Notes investigations.
4. The shipped algorithm becomes the new default (`v5`) following the established versioning pattern (v3 → v4 → v5), leaving v4 sidecars intact and the worker dispatch table consistent.
5. Higher harmonics are an acceptable trade — fewer summed partials in exchange for a more reliable F0 is preferred per user direction.

## 3. Non-goals

- Multi-F0 per event (one F0 contour per event remains, per ADR-069).
- Changes to harmonic sibling search (`_build_harmonic_notes`), MPE Lower Zone synthesis, MIDI export resolver semantics, parquet schemas, or contour API shape.
- Auto-backfill of v5 for completed v4 jobs (manual via job-admin UI, mirroring v3 → v4 launch).
- Frontend rendering changes beyond what falls out of `latestExtractorVersion = "v5"` flowing through the existing status pill.
- Replacing the STFT ridge sidecar consumed by the encoder for descriptors — the ridge tracker stays as-is for descriptor computation and as a voicing seed for v5.
- Production performance regressions are tolerated only up to ~2× v4 per-event wall-clock; the test-bed iteration must measure this.

## 4. Decision

Ship in a single feature branch as three sequential phases:

1. **Phase 1 — Test-bed and candidate scaffold.** A CLI tool that loads any event in any encoder job and renders a side-by-side spectrogram + piano-roll PNG using one or more registered algorithm variants. A `v5-candidate` extractor module is registered alongside v3/v4 with a starting harmonic-Viterbi implementation.
2. **Phase 2 — Iteration.** Run the test-bed against token #47 plus a handful of known-problematic events, inspect the rendered PNGs in chat, adjust `note_extractor_v5_candidate.py` (parameters, scoring, voicing, transition cost), repeat until the F0 traces visually align with the spectral ridges. Commits land on the same feature branch. The starting candidate algorithm (§4.3) may be wholly replaced if iteration shows it is unsuitable.
3. **Phase 3 — Promote v5.** Rename `note_extractor_v5_candidate.py` → `note_extractor_v5.py`, wire worker dispatch on `"v5"`, bump `DEFAULT_EXTRACTOR_VERSION`, update reference docs and the sequence-models capsule, append ADR-071. The test-bed stays in the repo and continues to render v5 alongside v3/v4.

### 4.1 Test-bed (Phase 1 deliverable)

`tools/piano_roll_notes_debug.py` — a `uv run python tools/piano_roll_notes_debug.py …` async CLI.

**Inputs**

```
--job <encoder_job_id>                 required
--token <sequence_index>               required (mutually exclusive with --event-id)
--event-id <event_uuid>                required (mutually exclusive with --token)
--variants <comma-separated names>     default "v4"
--out <path>                           required, .png
--pad-seconds <float>                  default 0.05
--width <int> --height <int>           default 1600 / 900
```

`--token N` resolves to the row in `event_tokens.parquet` whose `sequence_index == N`. Currently every job has a single `source_sequence_key`, so `--token` is unambiguous; if a future job has multiple sequences, the script errors with an instruction to pass `--event-id` explicitly.

**Algorithm registry** lives at `tools/piano_roll_notes_registry.py`:

```python
EXTRACTORS: dict[str, Callable[..., NotesV3Result]] = {
    "v3":           ...,
    "v4":           ...,
    "v5-candidate": ...,
}
```

Production code never imports this module. The worker continues to dispatch on `extractor_version` strings as today; the registry exists only for the test-bed CLI. Phase 3 adds `"v5"` to both the registry and the worker dispatch table.

**Resolution chain**

1. Open the project DB session.
2. Resolve `EventEncoderJob` → `RegionDetectionJob` → audio source (`AudioFile` or `hydrophone_id`), reusing `humpback.call_parsing.audio_loader.build_event_audio_loader`.
3. Resolve the target `Event` from the encoder job's `events.parquet` or the region's `events.parquet` (whichever path the worker uses today).
4. Slice padded audio for the event (reuse `_slice_event_audio` from the worker if not already a separable helper; otherwise inline the same logic).
5. For each variant, call `EXTRACTORS[variant](audio, sr, params=…, ridge_sidecar_rows=…)` and capture the returned `NotesV3Result`.

**Rendering** uses matplotlib with the `Agg` backend (no display required).

The output PNG has two panels sharing the same x-axis (event time, seconds since padded start):

- **Top panel** — CQT log-magnitude spectrogram (reuse `compute_event_cqt`). Y-axis log Hz, ticks at 50/100/200/400/800/1600/3200/6400 Hz with secondary MIDI labels. Raw STFT ridge (reuse `compute_ridge_path` or the persisted `event_ridges_*.parquet` sidecar) overlaid as a 1 px line.
- **Bottom panel(s)** — Piano-roll. One sub-panel per `--variants` entry, vertically stacked, all sharing the top panel's x-axis. F0 notes drawn as thick pitch-bend ribbons (cents_from_pitch mapped to a per-frame MIDI offset on top of the note's `midi_pitch`); harmonic notes as thinner desaturated ribbons. Y-axis MIDI 12..120 with octave labels and black-key shading to match the production frontend.

Each panel carries a small variant-name label. Total figure height scales with the number of variants (so two variants render at ~1200 px height; one at ~900 px).

**Out of scope for the test-bed**

- Per-frame internal diagnostics (Viterbi back-pointers, HPS divisor heatmaps). The PNG shows inputs and outputs only; deferred to a `--debug` flag if a later iteration needs it.
- Multi-token batch rendering. One CLI invocation = one event = one PNG.
- Interactive scrubbing or zooming. The user views the static PNG.
- Image diffing or golden tests. Phase 1 ships a small smoke pytest that confirms the resolution chain returns a non-empty audio slice and the renderer writes a non-empty PNG against a synthetic in-memory job; pixel correctness is left to visual review.

### 4.2 Iteration (Phase 2 deliverable, in-chat)

Phase 2 lands one or more commits on the same feature branch, each editing `src/humpback/processing/note_extractor_v5_candidate.py` and rerunning the test-bed against:

- Token #47 of job `690580c5…` (the user-reported case).
- The 5 longest events from §1's diagnostic (`21d13e3c…`, `ae21e747…`, `db7e1b9e…`, `2fe09647…`, `f8825fe3…`).
- One or two short events to confirm we don't break the short-event case.

Each iteration's commit message includes "test-bed iteration N" plus a brief note on the parameter change tried. The chat conversation records the rendered PNGs as visual evidence. No frontend, worker, or backend wiring changes during Phase 2 — only the candidate module and its parameter defaults.

Phase 2 ends when the user signs off on the visual fidelity of the candidate against the iteration set. The signed-off algorithm is what Phase 3 ships.

### 4.3 Starting candidate algorithm (Phase 2 starting point, replaceable)

Direct harmonic-sum F0 over the CQT with log-frequency Viterbi smoothing. Same `(audio, sample_rate, *, params, ridge_sidecar_rows=None) -> NotesV3Result` signature as v3/v4.

**Per-frame F0 score**

For each candidate F0 `f₀` on a dense log-frequency grid (every CQT bin between `f0_min_hz` and `f0_max_hz` — at the production CQT settings of 36 bins/octave, ~156 candidates between 30 Hz and 600 Hz), compute

```
H_t(f₀) = Σ_{k=1..K} w_k · max(0, CQT_log[bin(f₀·k), t] − floor_t)
```

with:

- `K = n_harmonics` (default `4`; deliberately small per user direction "sacrifice higher harmonics for better F0 fidelity").
- `w_k = 1 / √k` by default (`harmonic_weight = "inv_sqrt_k"`). Selectable: `"uniform"`, `"inv_k"`, `"inv_sqrt_k"`.
- `floor_t` and the gating threshold reuse v4's `_frame_noise_floor` (median + `k_noise · MAD`, with the same `_MIN_MAD = 0.3` clamp).
- Each harmonic bin `bin(f₀·k)` is searched within `±cents_tolerance` (default 50¢) of the target log-Hz position and must clear (i) the noise-floor threshold and (ii) the strict 3-bin local-peak test from v4, otherwise it contributes 0 (the `max(0, …)` term covers below-floor cases).

**Voicing per frame**

`voiced_t = (H_t.max() − H_t.median()) > tau_voicing` (default `tau_voicing = 1.5` log units). Unvoiced frames feed into the Viterbi as a single "rest" state.

**Viterbi smoothing**

State space: F0 grid bins + 1 rest state. Per-frame emission cost:

```
cost_voiced(t, f₀) = − H_t(f₀)
cost_rest(t)      = − H_t.median()
```

Transition cost between voiced states `(t, f_i) → (t+1, f_j)`:

```
C(f_i → f_j) = transition_lambda · (log₂ f_j − log₂ f_i)²
```

with `transition_lambda = 2.0` cost-per-squared-octave default. Voiced↔rest transitions carry a fixed entry/exit cost (`voicing_transition_cost = 1.0`) so brief voiced/unvoiced flips don't fragment the contour. Viterbi back-trace yields the smoothed F0 sequence and the voiced-frame mask.

**Segmentation and harmonics**

Continuous voiced runs become F0 segments. Each segment becomes one F0 `NoteV3` via the existing `_build_f0_note` (reused unchanged via the v4 `_adapt_to_v3_params` pattern). Harmonic siblings are derived structurally at `n · f₀(t)` for `n ∈ {2..16}` with the same ±75¢ CQT-peak tolerance as v3/v4 (reuse `_build_harmonic_notes`).

**`subharmonic_octave` semantics in v5**

The field is unused by the harmonic-Viterbi algorithm (no divisor concept exists). Store `0` for every contour row in v5. The column's meaning under v3 (octave halving count) and v4 (divisor − 1) is preserved on disk for those versions; v5 documents the value as "reserved / unused" in `docs/reference/storage-layout.md` and in the parquet column docstring. The frontend renderer only uses the field for hover diagnostics, so the constant zero is benign.

**Parameter dataclass** (`HarmonicViterbiParams`)

```python
@dataclass(frozen=True, slots=True)
class HarmonicViterbiParams:
    n_harmonics: int = 4
    harmonic_weight: Literal["uniform", "inv_k", "inv_sqrt_k"] = "inv_sqrt_k"
    f0_min_hz: float = 30.0
    f0_max_hz: float = 600.0
    cents_tolerance: float = 50.0
    k_noise: float = 2.0
    tau_voicing: float = 1.5
    transition_lambda: float = 2.0
    voicing_transition_cost: float = 1.0
```

`ExtractNotesV5Params` is a sibling of `ExtractNotesV4Params` with `harmonic_viterbi: HarmonicViterbiParams` replacing `hps: HPSParams`. STFT, CQT, segmentation, harmonic, and midi sub-params carry the v4 defaults unchanged (including the 30 Hz STFT band floor from v4).

**Compute estimate**

Per event: emission ≈ `frames × candidates × K × small_window_bins`, Viterbi ≈ `frames × candidates²`. At production scale (~150 frames × 156 candidates × 4 harmonics × ~3 bins) the emission stage is ~280 kFLOPs; the Viterbi at 150 × 156² is ~3.6 MFLOPs. Per-event well under v4 (which runs an N=8 HPS scan for every ridge frame plus majority + median smoothing). Acceptable per §3.

### 4.4 Iteration replacement clause

If Phase 2 iteration shows the harmonic-Viterbi starting point is structurally unsuitable (e.g., genuinely fast pitch sweeps get over-smoothed at any tuning, or noise dominates the emission scores across the iteration set), the candidate may be replaced wholesale with one of:

- `librosa.pyin` wrapped to match the `extract_notes_v*` signature, with `fmin=30, fmax=600, frame_length=4096`.
- A two-stage pipeline keeping the v4 HPS divisor selection but replacing divisor smoothing with Viterbi smoothing on F0 frequency directly.

A replacement does not require respecting the spec — Phase 2's deliverable is "a candidate the user signs off on visually," not "this specific algorithm." Phase 3 then ships whatever Phase 2 produced.

### 4.5 Phase 3 shipping details

- **Module promotion**: `note_extractor_v5_candidate.py` → `note_extractor_v5.py`. Drop the `_candidate` suffix from the registry name. `extract_notes_v5_candidate` → `extract_notes_v5`.
- **Worker dispatch**: `piano_roll_notes_worker.py` gains a `_extract_notes_v5` branch mirroring `_extract_notes_v4`. `_KNOWN_EXTRACTOR_VERSIONS` adds `"v5"`. `DEFAULT_EXTRACTOR_VERSION = "v5"`.
- **Sidecars**: `event_notes_v5.parquet`, `event_note_contours_v5.parquet`. Schemas identical to v3/v4 (no migrations).
- **Export resolver**: `max("v1", …, "v5")` already works via lex ordering.
- **Frontend pill**: `PianoRollNotesStatusPill` already tracks `latestExtractorVersion` per the v4 ship (commit 73f5588), so the v4 → v5 upgrade pill appears automatically. No frontend code change.
- **Auto-enqueue + auto-backfill**: new encoder jobs auto-enqueue v5; existing jobs are user-driven re-enqueue via the v5 pill (matches v4 launch).
- **ADR-071** in `DECISIONS.md` records (a) the v4 → v5 algorithmic change, (b) the test-bed-driven iteration approach, and (c) the `subharmonic_octave` semantics change to "unused" in v5+.
- **Reference docs**: `docs/reference/storage-layout.md`, `docs/reference/signal-processing.md`, the `sequence-models` capsule README + invariants, and `docs/agent-context/current-state.md` get updated to mention v5 as the new default and the test-bed as a permanent debug surface.

### 4.6 Test plan

**Phase 1 — test-bed**

- `tests/tools/test_piano_roll_notes_debug.py`:
  - Synthetic in-memory encoder job (one AudioFile-backed event, fabricated `event_tokens.parquet` row, minimal manifest) resolves through the CLI's lookup chain and returns the expected event audio shape.
  - CLI invocation produces a non-empty PNG at the requested path.
  - `--token` and `--event-id` are mutually exclusive (CLI exits non-zero with a clear message when both or neither are given).
  - Registry contains exactly `{"v3", "v4", "v5-candidate"}` after Phase 1; Phase 3 amends this assertion to include `"v5"`.

**Phase 2 — candidate iteration**

- Unit tests for `note_extractor_v5_candidate.py`:
  - Pure 200 Hz tone with H2–H4: F0 ≈ 200 Hz across all frames; voicing fully active.
  - Linear sweep 50 Hz → 80 Hz over 1 s with H2..H4: Viterbi tracks the sweep within `cents_tolerance`.
  - Pure broadband noise: voicing inactive across all frames; no notes emitted.
  - 200 Hz tone with strong H2 at 12 dB above F0 (the classic ridge-on-H2 case): F0 ≈ 200 Hz, not 400 Hz, because the harmonic-sum at f₀=200 includes H1 + H2 + H3 + H4 while at f₀=400 it includes only H2-of-400 + H4-of-400 (the rest fall above nyquist or onto noise).
  - Two adjacent pure tones at 200 Hz and 400 Hz (no harmonic relationship): higher-energy tone wins; Viterbi does not flap.

These tests stay valid even if the iteration replaces the algorithm under §4.4 — they're fixture-level behavior assertions, not algorithm-internal assertions.

**Phase 3 — v5 shipping**

- `tests/processing/test_note_extractor_v5.py` mirrors the Phase 2 unit tests against the promoted module.
- `tests/workers/test_piano_roll_notes_worker.py` gains a v5 dispatch case asserting the worker writes `event_notes_v5.parquet` + `event_note_contours_v5.parquet` and that `DEFAULT_EXTRACTOR_VERSION == "v5"`.
- Existing MIDI export tests run unchanged; the export resolver picks v5 over v4 automatically via the lex sort.
- The existing `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts` and `event-encoder-piano-roll-perf.spec.ts` re-run unchanged. If a `latestExtractorVersion` constant exists in the frontend, bump it; otherwise zero frontend code change.

### 4.7 Alternatives considered

- **Post-filter v4's F0 in frequency space** (median + Kalman on the resulting F0 stream): cheaper but can't recover from a wrong divisor selection that already produced a wildly off-pitch F0 in a given frame; the bad frames simply become smoothed bad frames.
- **Replace ridge tracker entirely with a Viterbi over CQT** (no STFT ridge): removes the encoder/notes coupling, but the encoder still needs the ridge sidecar for descriptors. Keeping the ridge for descriptors + voicing seed and adding a separate Viterbi for F0 is cleaner.
- **`librosa.pyin` as the v5 algorithm**: tested, simple integration, comes with Viterbi for free. Held in reserve under §4.4 as a fallback. Not the starting candidate because (a) YIN is time-domain autocorrelation-based and our CQT/ridge pipeline already produces the spectral evidence we need, and (b) wrapping it would discard the local-peak + noise-floor gating we already have tuned.
- **Single PR shipping test-bed + v5 together**: faster if the algorithm is right on the first try, but loses the option to iterate before committing to a specific v5 algorithm. The user explicitly chose three-phase, so this is rejected.

## 5. Risks

- **Over-smoothing**: large `transition_lambda` would lock F0 to a near-constant value across a sweep. Mitigated by parameter tuning against the sweep test cases and by Phase 2 visual review.
- **Voicing oscillation in noisy frames**: `tau_voicing` set too low causes brief voicing flips that fragment notes; set too high causes missed quiet onsets. The fixed `voicing_transition_cost` on the rest state limits flapping; if still observed, the cost is tunable in Phase 2.
- **Compute regression for very long events**: Viterbi at `O(frames × candidates²)` is `~3.6 MFLOPs` per event at production scale; events with ≥ 1000 frames (rare but possible) reach `~24 MFLOPs`. Still under v4. If a real regression appears, Phase 2 explores pruning to top-K candidates per frame before Viterbi.
- **Test-bed bit-rot**: tools that aren't run regularly stop working. Mitigated by the Phase 1 pytest exercising the CLI's resolution chain. Phase 3 documentation mentions the tool so future debugging sessions find it.
- **Phase 2 commits churn the feature branch history**: acceptable — the PR description summarizes the final algorithm; reviewers focus on the converged code, not the iteration trail.

## 6. Open questions

- Should the test-bed support rendering v3 too, or only v4 + v5-candidate? Recommendation: include v3 in the registry (cheap) but default `--variants` to `v4` so a no-argument invocation shows the current production baseline.
- Should the rendered PNG carry a header line with job/token/event metadata (job uuid, token index, event uuid, duration)? Recommendation: yes — saves a round-trip when sharing PNGs in chat.
- Should Phase 3 deprecate v4 by removing it from the worker dispatch? Recommendation: no — keep v4 reachable for params reproducibility (matches v3 treatment post-ADR-070).
