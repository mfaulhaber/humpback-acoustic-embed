# Piano Roll Notes v4 — HPS F0 Implementation Plan

**Goal:** Replace v3's per-frame octave-halving subharmonic refinement with HPS-style harmonic-stack F0 scoring and lower the ridge tracker's band floor to 30 Hz, so low-frequency humpback fundamentals stop being lost to H2/H3 lock.

**Spec:** [docs/specs/2026-05-23-piano-roll-notes-v4-hps-f0-design.md](../specs/2026-05-23-piano-roll-notes-v4-hps-f0-design.md)

**Primary domain:** sequence-models

**Neighbor domains:** none — no API surface change, no schema change, no frontend behavior change beyond extended-band notes becoming populated.

---

### Task 1: Add `note_extractor_v4` module with HPS F0 selection

**Files:**
- Create: `src/humpback/processing/note_extractor_v4.py`

**Acceptance criteria:**
- [ ] Module exposes `HPSParams`, `STFTParams`, `SegmentationParams`, `HarmonicSearchParams`, `MidiRangeParams`, `ExtractNotesV4Params`, `extract_notes_v4`, `NotesV3Result` (re-exported).
- [ ] `STFTParams.min_frequency_hz` default is `30.0` (was `100.0` in v3); other STFT defaults match v3.
- [ ] `HPSParams` defaults match spec §4.3 exactly: `n_harmonics=8`, `cents_tolerance=50.0`, `k_noise=2.0`, `candidate_divisors=(1,2,3,4,5,6)`, `smoothing_frames=5`, `low_band_penalty=0.5`, `low_band_threshold_hz=100.0`, `low_band_min_harmonics=3`, `high_band_min_harmonics=2`.
- [ ] `ExtractNotesV4Params` mirrors `ExtractNotesV3Params` field-for-field with `hps: HPSParams` replacing `subharmonic: SubharmonicParams`.
- [ ] `extract_notes_v4(audio, sample_rate, *, params, ridge_sidecar_rows=None) -> NotesV3Result` returns the same `NotesV3Result` shape as v3 (reused `NoteV3` / `ContourFrame` row types — no schema change).
- [ ] Pipeline order: ridge resolution (reused from v3) → `_score_f0_candidates` (new, spec §4.2) → majority-smoothing of the chosen `divisor` (reuse `_majority_smooth`) → F0 segmentation (reuse `_segment_f0_runs`) → F0 note build (reuse `_build_f0_note`) → harmonic siblings (reuse `_build_harmonic_notes`).
- [ ] `_score_f0_candidates` implements the per-frame scoring exactly per spec §4.2: max-magnitude in ±`cents_tolerance` window per harmonic, `floor + k_noise·MAD` per-frame threshold, `contribution = max(0, m_n − floor)`, `count_present` gate against `low_band_min_harmonics` / `high_band_min_harmonics`, sub-`low_band_threshold_hz` penalty, tie-break toward the largest divisor (lowest candidate frequency).
- [ ] The `subharmonic_octave` field in emitted `ContourFrame` rows stores `chosen_divisor − 1` (0 = ridge is F0, 5 = ridge is H6).
- [ ] Frames where no candidate clears `k_min` harmonics-above-floor fall back to `divisor = 1` (ridge is F0) before smoothing so the smoothed stream is well-defined.
- [ ] Module docstring references ADR-070 (added by Task 7) and the spec path.
- [ ] No I/O, no imports of `librosa.cqt` outside the ridge fallback path — same pure-function contract as v3.
- [ ] `extract_notes_v3` and v3 helpers are NOT imported or touched; v4 reuses v3 helpers by extracting shared helpers to module-private functions inside v4 or by importing the ones that are already exported from v3 (`_segment_f0_runs`, `_build_f0_note`, `_build_harmonic_notes`, `_majority_smooth`, `_align_cqt_frame`, `_nearest_bin`, `_frame_noise_floor`, `_cqt_bin_frequencies`, `_resolve_ridge_contour`, `_RidgeFrame`, `_RefinedFrame`, `_F0Note`). If those helpers are still module-private in v3, lift them into a new `note_extractor_shared.py` and import from both — keep v3's public surface unchanged.

**Tests needed:**
- New `tests/processing/test_note_extractor_v4.py` covering:
  - Pure tone at 200 Hz → emitted F0 ≈ 200 Hz, no halving.
  - Pure tone at 200 Hz with H2 at 400 Hz +12 dB → ridge locks at 400 Hz, HPS picks `d=2`, emitted F0 ≈ 200 Hz.
  - Pure tone at 200 Hz with H3 at 600 Hz +12 dB → HPS picks `d=3`, emitted F0 ≈ 200 Hz.
  - 40 Hz fundamental with H2..H6 above broadband sub-100 Hz noise → emitted F0 ≈ 40 Hz, `subharmonic_octave > 0` for relevant frames.
  - Pure sub-100 Hz noise burst with no coherent harmonics → no sub-100 Hz F0 chosen (sub-100 candidates fail `low_band_min_harmonics`).
  - Pure tone at 1 kHz, no harmonics → `d=1` wins, emitted F0 ≈ 1 kHz.
  - FM sweep 50 Hz → 80 Hz over 1 s with H2..H4 → F0 segment tracks the sweep, divisor smoothing stable.
  - Empty / too-short ridge contour → returns empty `NotesV3Result`, mirroring v3 behavior.
  - Sidecar ridge rows path: pass synthetic `ridge_sidecar_rows`, assert HPS uses them without recomputing.

---

### Task 2: Worker dispatch for v3 vs v4

**Files:**
- Modify: `src/humpback/workers/piano_roll_notes_worker.py`

**Acceptance criteria:**
- [ ] Add `_V4_EXTRACTOR_VERSION = "v4"` alongside existing `_V3_EXTRACTOR_VERSION`, and a `_KNOWN_EXTRACTOR_VERSIONS = ("v1", "v2", "v3", "v4")` tuple used wherever the worker validates a version string.
- [ ] Import `extract_notes_v4`, `ExtractNotesV4Params`, `HPSParams`, and the v4 `STFTParams` alias (or the v4-defaulted `STFTParams` if shared).
- [ ] Add `_extract_notes_v4` async helper paralleling `_extract_notes_v3` (same signature, same emit pipeline, same parquet write) — constructs `ExtractNotesV4Params` with the resolved per-job params and calls `extract_notes_v4` per event.
- [ ] Dispatch in `run_piano_roll_notes_job`: branch on `job.extractor_version` — `"v3"` → `_extract_notes_v3`, `"v4"` → `_extract_notes_v4`, anything else → fall back to existing v1/v2 behavior unchanged.
- [ ] `_resolve_params` recognizes `"v4"` and constructs `HPSParams` from `params_json["hps"]` with the new defaults. `params_json` schema for v4 includes a top-level `hps` block plus the existing `cqt`, `stft`, `segmentation`, `harmonic`, `midi`, `audio` blocks.
- [ ] When persisting `params_json` for a fresh v4 job, every `HPSParams` field is recorded (no defaults silently embedded; round-trips identically through `_resolve_params`).
- [ ] When `job.extractor_version == "v4"` the worker writes the v4 contours sidecar to `event_encoders/{job_id}/event_note_contours_v4.parquet` and the notes sidecar to `event_encoders/{job_id}/event_notes_v4.parquet` using the existing `event_encoder_note_contours_path` / `event_encoder_notes_path` storage helpers (which already template on `extractor_version`).
- [ ] No path or parquet-schema constants change; `NOTES_V3_SCHEMA` and `NOTE_CONTOURS_V3_SCHEMA` are reused for v4 writes (column shape is identical; only the `subharmonic_octave` semantics differ). Document this in a one-line comment near the v4 write path.
- [ ] Existing v3 tests in `tests/unit/test_piano_roll_notes_worker_v3.py` continue to pass unchanged.
- [ ] Existing v1/v2 dispatch (legacy `piano_roll_tracker` path) is untouched.

**Tests needed:**
- New `tests/unit/test_piano_roll_notes_worker_v4.py` covering:
  - Worker correctly dispatches `extractor_version="v4"` to `_extract_notes_v4` and writes both sidecars to the v4 paths.
  - `params_json` round-trips: write v4 → re-read → identical `HPSParams`.
  - End-to-end on a minimal synthetic encoder fixture (1–2 events, deterministic audio): notes parquet has expected `extractor_version="v4"` column-less semantics (column not present, version implicit from path) and contour parquet rows.
  - Mixed-version coexistence: a job dir with both `event_notes_v3.parquet` and `event_notes_v4.parquet` after running both jobs in sequence; both files exist and neither is overwritten.
- Extend `tests/unit/test_piano_roll_notes_worker_extraction.py` if its helpers benefit from the v4 dispatch path (not required — judgment call during implementation).

---

### Task 3: Bump `DEFAULT_EXTRACTOR_VERSION` to `"v4"`

**Files:**
- Modify: `src/humpback/models/piano_roll_notes.py`

**Acceptance criteria:**
- [ ] `DEFAULT_EXTRACTOR_VERSION = "v4"` (was `"v3"`).
- [ ] No other code change in this file; ORM column default cascades through SQLAlchemy.
- [ ] Auto-enqueue hook in `src/humpback/workers/event_encoder_worker.py` (or wherever it currently invokes the notes service with no explicit version) picks up the new default without modification — verify by inspection.
- [ ] `src/humpback/services/piano_roll_notes_service.py:60` (`version = extractor_version or DEFAULT_EXTRACTOR_VERSION`) picks `"v4"` when callers omit the version — verify by inspection; no code change.

**Tests needed:**
- Existing default-version assertions in `tests/unit/test_piano_roll_notes_service.py` (if any) update from `"v3"` to `"v4"`.
- Tests asserting auto-enqueue creates a `"v3"` row update to `"v4"`. Search: `grep -rn '"v3"' tests/`.

---

### Task 4: Verify MIDI export resolves v4 over v3

**Files:**
- Modify if needed: `src/humpback/services/piano_roll_midi_export_service.py`

**Acceptance criteria:**
- [ ] Confirm `_resolve_latest_notes_version` (around line 265–280) uses `desc(PianoRollNotesJob.extractor_version)` string ordering — `"v4" > "v3"` lexicographically, so v4 wins automatically once present. No code change expected; add a one-line comment confirming the v4 path is covered.
- [ ] If any code path constructs the export by hard-coding `"v3"` (e.g., MPE Lower Zone format detection), confirm it falls through to the version-agnostic `note_uid`-presence test in `humpback.processing.midi_synthesis` and treats v4 sidecars as MPE-equivalent to v3. The MPE channel layout, program changes, and bend stream all derive from the contour shape, which is identical between v3 and v4.

**Tests needed:**
- New test in `tests/unit/test_piano_roll_midi_export_service.py` (or extend existing): given a job with both a complete v3 row and a complete v4 row, the resolver returns v4. Given only v3, it returns v3 (unchanged behavior).
- New test in `tests/unit/test_piano_roll_midi_export_worker.py`: a v4 sidecar pair (notes + contours parquet, valid `note_uid`) synthesizes a valid MPE SMF with the same track count and channel layout as a v3 export, byte-comparing against a v3-equivalent fixture is NOT required (different audio) — instead assert SMF metadata: 17 tracks, MPE master RPN events, per-voice program/CC events.

---

### Task 5: Add ADR-070 to DECISIONS.md

**Files:**
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] New section `## ADR-070: Piano Roll Notes v4 — HPS F0 selection with extended low band` appended after ADR-069.
- [ ] Includes `**Date**: 2026-05-23`, `**Status**: Accepted`, link to the spec file.
- [ ] Context section summarizes the v3 failure mode (H2/H3 lock, sub-100 Hz floor) with the diagnostic numbers from spec §1 (10% F0 / 90% harmonics, median MIDI 69, 0 ridge frames below 100 Hz).
- [ ] Decision section lists the 6 normative bullets from spec §4: HPS scoring, candidate divisors `(1..6)`, `STFTParams.min_frequency_hz = 30.0`, `low_band_min_harmonics = 3` / `high_band_min_harmonics = 2`, `subharmonic_octave` column repurposed, `DEFAULT_EXTRACTOR_VERSION = "v4"`.
- [ ] Alternatives section quotes the rejected approaches (B, C, full-grid HPS, multi-F0) from spec §4.9.
- [ ] Consequences: no DB migration, no auto-backfill of v4 for completed v3 jobs (manual via job-admin UI), MIDI export auto-resolves v4 by string-ordering.

**Tests needed:**
- None (documentation only). Verify the file parses cleanly by reading it back; `grep '^## ADR-' DECISIONS.md` shows ADR-070 in the correct position.

---

### Task 6: Update Sequence Models domain capsule

**Files:**
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/invariants.md`

**Acceptance criteria:**
- [ ] `README.md` "Primary Paths" gains `src/humpback/processing/note_extractor_v4.py` line (mark v3 as "legacy v3 only; superseded by ADR-070").
- [ ] `README.md` "Artifact Roots" gains lines for `event_notes_v4.parquet` and `event_note_contours_v4.parquet`; updates the existing v3 line to note that v3 is legacy and v4 is default.
- [ ] `README.md` "Relevant ADRs" appends ADR-070.
- [ ] `invariants.md` adds invariants matching spec §4.5:
  - `DEFAULT_EXTRACTOR_VERSION = "v4"` (ADR-070).
  - Piano Roll Notes v4 uses HPS-based F0 selection over the shared STFT ridge with `STFTParams.min_frequency_hz = 30.0`.
  - The `subharmonic_octave` column in v4 rows stores `candidate_divisor − 1` (0..5), not octave halvings; v3 semantics remain (0..3).
  - MIDI export resolves the latest `complete` notes-job version by string-ordering (`max("v1","v2","v3","v4")`), so v4 wins over v3 when both exist.
  - Existing v3 invariants about ridge-as-canonical-F0 and harmonic sibling derivation carry over unchanged.
- [ ] No invariant claims v3-specific behavior as universal (audit existing v3 invariants and qualify with "v3" where the rewrite changes meaning).

**Tests needed:**
- None (documentation only).

---

### Task 7: Update reference docs

**Files:**
- Modify: `docs/reference/storage-layout.md`

**Acceptance criteria:**
- [ ] `storage-layout.md` documents the new `event_notes_v4.parquet` and `event_note_contours_v4.parquet` paths under the Event Encoder section, with a one-line note that the schemas match v3 but the `subharmonic_octave` semantics differ.
- [ ] Existing v3 storage entries are left in place (still readable on disk).

**Tests needed:**
- None (documentation only).

---

### Task 8: Update `current-state.md`

**Files:**
- Modify: `docs/agent-context/current-state.md`

**Acceptance criteria:**
- [ ] Add a one-line bullet under Sequence Models noting v4 is the default Piano Roll Notes extractor and references ADR-070.

**Tests needed:**
- None (documentation only).

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/processing/note_extractor_v4.py src/humpback/workers/piano_roll_notes_worker.py src/humpback/models/piano_roll_notes.py src/humpback/services/piano_roll_midi_export_service.py tests/processing/test_note_extractor_v4.py tests/unit/test_piano_roll_notes_worker_v4.py`
2. `uv run ruff check src/humpback/processing/note_extractor_v4.py src/humpback/workers/piano_roll_notes_worker.py src/humpback/models/piano_roll_notes.py src/humpback/services/piano_roll_midi_export_service.py tests/processing/test_note_extractor_v4.py tests/unit/test_piano_roll_notes_worker_v4.py`
3. `uv run pyright src/humpback/processing/note_extractor_v4.py src/humpback/workers/piano_roll_notes_worker.py src/humpback/models/piano_roll_notes.py src/humpback/services/piano_roll_midi_export_service.py tests/processing/test_note_extractor_v4.py tests/unit/test_piano_roll_notes_worker_v4.py`
4. `uv run pytest tests/processing/test_note_extractor_v4.py tests/unit/test_piano_roll_notes_worker_v4.py tests/unit/test_piano_roll_notes_worker_v3.py tests/unit/test_piano_roll_notes_worker_extraction.py tests/unit/test_piano_roll_midi_export_service.py tests/unit/test_piano_roll_midi_export_worker.py tests/unit/test_piano_roll_notes_service.py -q` (v3 and worker-extraction tests must continue to pass).
5. `uv run pytest tests/ -q` (full backend gate).
6. **Manual smoke**: enqueue a v4 notes job against encoder `690580c5-7804-43c9-bd8d-690691b5d6d4` via the existing job-admin UI; load the resulting sidecar in Python and assert F0 median MIDI drops by ≥ 12 semitones vs the existing v3 sidecar (spec §4.8 integration target).
7. **Frontend smoke** (only if any frontend code is touched — none expected): `cd frontend && npx tsc --noEmit` and the relevant Playwright spec from `docs/agent-context/domains/sequence-models/tests.md`. Notes-mode rendering uses `note_uid` and the existing contour endpoint with no version awareness in the UI, so the v4 sidecar should render identically.
