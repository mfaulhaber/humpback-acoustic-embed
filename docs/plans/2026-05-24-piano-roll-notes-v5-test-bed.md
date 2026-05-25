# Piano Roll Notes v5 — Test-Bed + Harmonic-Viterbi F0 Implementation Plan

**Goal:** Build a permanent CLI test-bed that renders spectrogram + piano-roll PNGs for any encoder-job event, iterate a harmonic-Viterbi F0 candidate against token #47 of job `690580c5…` until F0 traces visually match the spectral ridges, then promote the converged algorithm to Piano Roll Notes v5.
**Spec:** [docs/specs/2026-05-24-piano-roll-notes-v5-test-bed-design.md](../specs/2026-05-24-piano-roll-notes-v5-test-bed-design.md)
**Primary domain:** sequence-models
**Neighbor domains:** none (no boundary contracts change in `signal-timeline` or `call-parsing` — we only consume existing audio-loader and CQT/ridge helpers)

---

## Phase 1 — Test-bed and v5-candidate scaffold

### Task 1: Add the algorithm registry module for the test-bed

**Files:**
- Create: `tools/piano_roll_notes_registry.py`

**Acceptance criteria:**
- [ ] Module exposes a single dict `EXTRACTORS: dict[str, Callable[..., NotesV3Result]]` keyed by short variant name.
- [ ] Keys at end of Phase 1: `"v3"`, `"v4"`, `"v5-candidate"`. Each value is a thin lambda that constructs the appropriate `ExtractNotesV*Params` from `job_id`, `event_id`, `event_start_utc` and calls the underlying extractor.
- [ ] Module docstring states this registry is for the test-bed only and is not imported by `humpback.workers.piano_roll_notes_worker`.
- [ ] Module imports only from `humpback.processing.note_extractor_v3`, `humpback.processing.note_extractor_v4`, and (once Task 2 lands) `humpback.processing.note_extractor_v5_candidate`.

**Tests needed:**
- Smoke import test asserting the dict's exact key set and that each value is callable.

---

### Task 2: Scaffold `note_extractor_v5_candidate.py` with the starting harmonic-Viterbi algorithm

**Files:**
- Create: `src/humpback/processing/note_extractor_v5_candidate.py`

**Acceptance criteria:**
- [ ] Module exposes `HarmonicViterbiParams`, `ExtractNotesV5Params`, and `extract_notes_v5_candidate(audio, sample_rate, *, params, ridge_sidecar_rows=None) -> NotesV3Result`.
- [ ] Signature is byte-compatible with `extract_notes_v3` and `extract_notes_v4` so the worker (in Phase 3) and the registry (Task 1) can swap dispatch by name only.
- [ ] `HarmonicViterbiParams` fields match spec §4.3 with the listed defaults (`n_harmonics=4`, `harmonic_weight="inv_sqrt_k"`, `f0_min_hz=30.0`, `f0_max_hz=600.0`, `cents_tolerance=50.0`, `k_noise=2.0`, `tau_voicing=1.5`, `transition_lambda=2.0`, `voicing_transition_cost=1.0`).
- [ ] Implementation computes the harmonic-sum emission `H_t(f₀)` per spec §4.3 using `compute_event_cqt`, the v4 `_frame_noise_floor`, and the v4 strict 3-bin local-peak gate, all imported from `note_extractor_v3` / `note_extractor_v4`.
- [ ] Per-frame voicing flag derived from `(H_t.max() − H_t.median()) > tau_voicing`.
- [ ] Log-frequency Viterbi over candidate F0 bins + one rest state, with squared log-frequency transition cost and a fixed voiced↔rest entry/exit cost; back-trace yields the smoothed F0 sequence and voicing mask.
- [ ] Continuous voiced runs feed `_build_f0_note` and `_build_harmonic_notes` (reused via the v4 `_adapt_to_v3_params` namespace pattern); contour rows write `subharmonic_octave = 0`.
- [ ] Module docstring labels it as a Phase-2 candidate that may be revised wholesale before promotion to v5 in Phase 3.
- [ ] Module is NOT imported by `piano_roll_notes_worker.py`. Worker dispatch is unchanged in Phase 1.

**Tests needed:**
- Pure 200 Hz tone with H2–H4 → F0 ≈ 200 Hz across all voiced frames.
- Linear 50 → 80 Hz sweep over 1 s with H2..H4 → Viterbi tracks the sweep within `cents_tolerance`.
- Pure broadband noise → voicing inactive across all frames; zero notes emitted.
- 200 Hz tone with H2 at +12 dB above F0 → F0 ≈ 200 Hz, not 400 Hz.
- Two adjacent pure tones at 200 Hz and 400 Hz with no harmonic relationship → higher-energy tone wins and Viterbi does not flap.

These tests live at `tests/processing/test_note_extractor_v5_candidate.py` and are forward-portable to Phase 3 against `note_extractor_v5.py` with only an import-line rename.

---

### Task 3: Build the test-bed CLI audio-resolution chain

**Files:**
- Create: `tools/piano_roll_notes_debug.py`

**Acceptance criteria:**
- [ ] CLI accepts `--job <encoder_job_id>`, exactly one of `--token <int>` or `--event-id <uuid>`, `--variants <comma-list>` (default `"v4"`), `--out <png-path>` (required), `--pad-seconds <float>` (default `0.05`), `--width <int>` (default `1600`), `--height <int>` (default `900`).
- [ ] Mutually-exclusive validation: if both or neither of `--token`/`--event-id` are present, CLI exits non-zero with a clear message and does not touch the filesystem.
- [ ] `--token N` resolves by filtering the encoder's `event_tokens.parquet` on `sequence_index == N`; multi-sequence jobs cause a non-zero exit pointing at `--event-id`.
- [ ] Resolves `EventEncoderJob` → `RegionDetectionJob` → audio source via the same factory chain (`build_event_audio_loader`) used by `piano_roll_notes_worker._build_audio_provider`.
- [ ] Slices padded audio for the resolved event by reusing `humpback.workers.piano_roll_notes_worker._slice_event_audio` (private import is acceptable for a `tools/` script).
- [ ] Loads the encoder's `event_ridges_*.parquet` sidecar when present and passes the matching event's rows to each variant as `ridge_sidecar_rows`; absence is non-fatal (variants fall back to in-process recompute).
- [ ] Logs job/token/event resolution to stderr (one line) before invoking variants so failures during resolution are easy to triage.
- [ ] Each variant call captures the returned `NotesV3Result` plus wall-clock time per variant.

**Tests needed:**
- Synthetic in-memory job (AudioFile-backed, one event, one-row token parquet, minimal manifest) resolves through the chain and returns the expected event audio shape; covered in Task 5's smoke test.

---

### Task 4: Render spectrogram + piano-roll panels to PNG

**Files:**
- Modify: `tools/piano_roll_notes_debug.py`

**Acceptance criteria:**
- [ ] Renderer uses matplotlib with the `Agg` backend (no display server required) and writes a single PNG to `--out`.
- [ ] Top panel shows the CQT log-magnitude spectrogram for the padded event audio (reusing `compute_event_cqt`) with Y-axis log-Hz, frequency ticks at 50/100/200/400/800/1600/3200/6400 Hz, and a secondary MIDI label axis.
- [ ] Raw STFT ridge (from the sidecar when present, else `compute_ridge_path`) overlays the top panel as a 1 px line in a high-contrast color.
- [ ] One bottom sub-panel per `--variants` entry, vertically stacked, all sharing the top panel's x-axis (event time in seconds since padded start).
- [ ] Each bottom sub-panel renders F0 notes as thick pitch-bend ribbons (derived from each note's `cents_from_pitch` contour) and harmonic notes as thinner desaturated ribbons; Y-axis is MIDI 12..120 with octave labels and black-key shading.
- [ ] Each bottom sub-panel carries the variant name as an in-panel label plus its wall-clock time from Task 3.
- [ ] PNG header text (figure suptitle) includes encoder-job UUID, token index (or event UUID if `--token` was not used), and event duration in milliseconds.
- [ ] Figure dimensions honor `--width` and `--height` at 100 DPI; height scales when multiple variants are rendered so each panel stays legible.

**Tests needed:**
- The Task 5 smoke test asserts the renderer writes a non-empty PNG. Pixel comparison is intentionally out of scope.

---

### Task 5: Smoke test for the test-bed lookup chain and renderer

**Files:**
- Create: `tests/tools/__init__.py`
- Create: `tests/tools/test_piano_roll_notes_debug.py`

**Acceptance criteria:**
- [ ] Test fixture builds an in-memory encoder job: a tiny AudioFile fixture (synthetic sine), one `Event` row, a one-row `event_tokens.parquet`, and a stub manifest, all anchored at a fresh temp `storage_root`.
- [ ] Test exercises `tools/piano_roll_notes_debug.py` end-to-end against the fixture (importing the module's `main` and invoking it; subprocess not required).
- [ ] Test asserts a non-empty PNG is written to the supplied `--out` path and that the resolution log line was emitted to stderr.
- [ ] Negative test covers mutually-exclusive `--token`/`--event-id` (both present and neither present).
- [ ] Registry-key assertion: `EXTRACTORS` keys are exactly `{"v3", "v4", "v5-candidate"}` after Phase 1.

**Tests needed:**
- (This task is the tests.)

---

## Phase 2 — In-chat iteration on the candidate algorithm

### Task 6: Iterate `note_extractor_v5_candidate.py` against the iteration set until user sign-off

**Files:**
- Modify: `src/humpback/processing/note_extractor_v5_candidate.py`
- Possibly modify: `tools/piano_roll_notes_debug.py` (only if iteration surfaces a missing affordance such as a `--debug` flag)

**Acceptance criteria:**
- [ ] Test-bed runs cleanly against the iteration set: token #47 of job `690580c5-7804-43c9-bd8d-690691b5d6d4` plus the 5 longest events from spec §1 (`21d13e3c…`, `ae21e747…`, `db7e1b9e…`, `2fe09647…`, `f8825fe3…`) plus 1–2 short events.
- [ ] Each iteration's commit message starts with `Test-bed iteration N` and names the parameter or sub-routine changed.
- [ ] User signs off in chat that the rendered F0 traces visually track the spectral ridges on the iteration set.
- [ ] The Phase-1 unit tests in `tests/processing/test_note_extractor_v5_candidate.py` continue to pass at the end of iteration; tests are updated only if the iteration replaces the algorithm wholesale per spec §4.4.
- [ ] No worker, frontend, schema, parquet, or API change during Phase 2.

**Tests needed:**
- Existing Phase-1 candidate tests stay green. New tests at most extend the iteration-set coverage.

---

## Phase 3 — Promote candidate to v5 and ship as the new default

### Task 7: Promote candidate module to `note_extractor_v5.py`

**Files:**
- Create: `src/humpback/processing/note_extractor_v5.py`
- Delete: `src/humpback/processing/note_extractor_v5_candidate.py`
- Create: `tests/processing/test_note_extractor_v5.py`
- Delete: `tests/processing/test_note_extractor_v5_candidate.py`
- Modify: `tools/piano_roll_notes_registry.py`

**Acceptance criteria:**
- [ ] `note_extractor_v5.py` is the byte-equivalent of the converged candidate with two renames: `extract_notes_v5_candidate` → `extract_notes_v5` and `ExtractNotesV5Params` references remain unchanged (already the v5 name).
- [ ] Test module promoted under the same renames; all candidate tests pass against the promoted module.
- [ ] Registry replaces the `"v5-candidate"` entry with `"v5"` pointing at the promoted extractor.
- [ ] Phase 1's registry-key assertion in `tests/tools/test_piano_roll_notes_debug.py` is updated to assert `{"v3", "v4", "v5"}`.

**Tests needed:**
- All v5-candidate tests pass against `note_extractor_v5.py`.

---

### Task 8: Wire worker dispatch on `extractor_version = "v5"`

**Files:**
- Modify: `src/humpback/workers/piano_roll_notes_worker.py`

**Acceptance criteria:**
- [ ] `_KNOWN_EXTRACTOR_VERSIONS` (or its equivalent constant) gains `"v5"`.
- [ ] A `_extract_notes_v5` function mirrors `_extract_notes_v4` in shape, calling `extract_notes_v5` from the new module and writing `event_notes_v5.parquet` + `event_note_contours_v5.parquet`.
- [ ] `DEFAULT_EXTRACTOR_VERSION = "v5"` in the worker module and any other location it is mirrored (verify via grep).
- [ ] Auto-enqueue (encoder-complete hook) creates v5 jobs by default; existing v4 jobs remain reachable via explicit `extractor_version="v4"`.
- [ ] `params_json` round-trip for v5 parses and validates against `HarmonicViterbiParams`.
- [ ] Worker still produces byte-identical v3 and v4 outputs when the corresponding `extractor_version` is requested explicitly.

**Tests needed:**
- `tests/workers/test_piano_roll_notes_worker.py` gains a v5 dispatch case asserting (a) the worker writes both v5 parquet sidecars, (b) auto-enqueue picks `"v5"`, and (c) explicit `"v4"` still routes to the v4 path.

---

### Task 9: Update MIDI export resolver and frontend status pill for v5

**Files:**
- Modify: `src/humpback/workers/piano_roll_midi_export_worker.py` (if it hard-codes the version set)
- Modify: `frontend/src/components/sequence-models/PianoRollNotesStatusPill.tsx` (only if a hard-coded `latestExtractorVersion` constant exists)

**Acceptance criteria:**
- [ ] Export resolver picks v5 over v4 automatically when both are complete for an encoder job (verify the lex-ordering picker already does this; otherwise extend the known-version tuple to include `"v5"`).
- [ ] `PianoRollNotesStatusPill` surfaces a "v5 available" pill when an encoder has a v4 (or older) sidecar but no v5 yet; clicking enqueues a v5 notes job. If the pill already reads `latestExtractorVersion` from the backend (per commit 73f5588), no frontend code change is required and this acceptance criterion is satisfied by inspection.
- [ ] Backend `latestExtractorVersion` value exposed to the frontend (wherever it currently surfaces v4) bumps to `"v5"`.

**Tests needed:**
- Existing export-resolver tests cover v3-vs-v4 selection; extend the parameterization to include a (v4-complete + v5-complete) case asserting v5 wins.
- `PianoRollNotesStatusPill.test.tsx` parameterization gains a v4 → v5 upgrade case; or, if the pill is already version-agnostic, add an inspection test asserting the latest version string flows through from a mocked backend response.

---

### Task 10: Append ADR-071 and update reference docs / capsule / current-state

**Files:**
- Modify: `DECISIONS.md`
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/invariants.md`
- Modify: `docs/agent-context/current-state.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/signal-processing.md`

**Acceptance criteria:**
- [ ] ADR-071 records (a) the v4 → v5 algorithm change (harmonic-Viterbi F0 over CQT with log-frequency smoothness baked into the cost function), (b) the test-bed-driven iteration approach as a reusable pattern, and (c) the `subharmonic_octave` column's "reserved / unused" semantics in v5+.
- [ ] Sequence-models capsule lists `note_extractor_v5.py` and `tools/piano_roll_notes_debug.py` under Primary Paths; mentions v5 as the current default; mentions the test-bed under "Before Editing" as the first-stop debug tool for any Piano Roll Notes investigation.
- [ ] Sequence-models invariants gains a v5 entry mirroring the existing v3/v4 entries: pipeline shape, parameter dataclass, `subharmonic_octave` semantics, dispatch behavior.
- [ ] `current-state.md` notes Piano Roll Notes v5 is the new default and the test-bed exists.
- [ ] `storage-layout.md` adds `event_notes_v5.parquet` and `event_note_contours_v5.parquet` paths and the v5 row of the `subharmonic_octave` semantics table.
- [ ] `signal-processing.md` describes the harmonic-Viterbi algorithm at the reference level (inputs, outputs, parameter ranges, why it replaces v4).

**Tests needed:**
- No code tests. Doc changes verified by review.

---

## Verification

Run in order after **each phase**:

1. `uv run ruff format --check` on all Python files touched this phase.
2. `uv run ruff check` on all Python files touched this phase.
3. `uv run pyright` on all Python files touched this phase.
4. `uv run pytest tests/` (full suite).
5. `cd frontend && npx tsc --noEmit` (only if frontend files changed — i.e., Phase 3 Task 9).
6. `cd frontend && npx playwright test` (only if Phase 3 Task 9 touched UI behavior; the existing `event-encoder-piano-roll.spec.ts` and `event-encoder-piano-roll-perf.spec.ts` cover the affected surfaces).
