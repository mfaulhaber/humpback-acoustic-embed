# Piano Roll Notes v7 Residual Discontinuities Implementation Plan

**Goal:** Ship Piano Roll Notes v7 so residual high-slope branch jumps split into separate notes and clear flat-decoded upsweeps can borrow the Event Encoder ridge as their bend carrier.
**Spec:** [docs/specs/2026-06-02-piano-roll-notes-v7-residual-discontinuities-design.md](../specs/2026-06-02-piano-roll-notes-v7-residual-discontinuities-design.md)
**Primary domain:** `sequence-models`
**Neighbor domains:** none

---

### Task 1: Add the v7 Extractor Core

**Files:**
- Create: `src/humpback/processing/note_extractor_v7.py`
- Create: `tests/processing/test_note_extractor_v7.py`
- Modify: `src/humpback/processing/note_extractor_v6.py` only if a small shared helper export is needed

**Acceptance criteria:**
- [x] `extract_notes_v7` mirrors v6's decode and de-spike path before adding v7-only post-decode passes.
- [x] Residual discontinuity splitting cuts adjacent retained F0 frames whose actual slope exceeds `max_continuous_slope_oct_per_s`, and drops post-split fragments shorter than `segmentation.min_note_frames`.
- [x] Ridge rescue only rewrites flat decoded F0 fragments when the overlapping ridge has sufficient movement, sufficient frame coverage, and a stable harmonic ratio.
- [x] Rewritten ridge-rescue frames preserve original timing, frame indexes, strength, and `subharmonic_octave = 0`.
- [x] v7 returns the existing `NotesV3Result` shape and writes data compatible with the v3-v6 note and contour parquet schemas.
- [x] v7 reuses `_build_f0_note` and `_build_harmonic_notes` so harmonic bends continue to inherit F0 cents by conservation.
- [x] Disabled v7 sub-passes reduce to v6 behavior for the same input and params.

**Tests needed:**
- Pure helper coverage for residual branch-jump splitting, smooth legal glide preservation, short-fragment dropping, ridge-rescue acceptance, ridge-rescue rejection for non-flat decoded contours, ridge-rescue rejection for unstable ridge/F0 ratios, and disabled-pass parity against v6.
- Synthetic extraction smoke coverage that confirms v7 emits notes and contours with the same schema-facing fields as v6.

---

### Task 2: Wire v7 Into the Worker and Defaults

**Files:**
- Modify: `src/humpback/models/piano_roll_notes.py`
- Modify: `src/humpback/workers/piano_roll_notes_worker.py`
- Create: `tests/unit/test_piano_roll_notes_worker_v7.py`
- Modify: `tests/unit/test_piano_roll_notes_service.py`
- Modify: `tests/unit/test_piano_roll_midi_export_model.py`
- Modify: `tests/integration/test_sequence_models_api.py`

**Acceptance criteria:**
- [x] `DEFAULT_EXTRACTOR_VERSION` is bumped from `"v6"` to `"v7"`.
- [x] Worker dispatch recognizes `"v7"` and writes `event_notes_v7.parquet` plus `event_note_contours_v7.parquet`.
- [x] v7 remains in the ridge-aware worker path and receives the Event Encoder ridge sidecar when present.
- [x] `_resolve_params(..., "v7")` parses `discontinuity` and `ridge_rescue` sections while inheriting v6's v5-style defaults: 30 Hz STFT floor, `min_break_frames = 6`, and `pad_seconds = 0.25`.
- [x] `to_json_dict` persists v7 parameter sections in `params_json`.
- [x] Existing service/API expectations that name the latest default version now expect v7.
- [x] MIDI export version resolution continues to rely on existing lexicographic latest-complete behavior and needs no synthesizer contract change.

**Tests needed:**
- Worker unit coverage for v7 dispatch, parameter parsing, default inheritance, sidecar path status, and parquet path/version naming.
- Existing service, MIDI-export model, and API tests updated where they assert the default extractor version.

---

### Task 3: Register v7 in Debug Tooling

**Files:**
- Modify: `tools/piano_roll_notes_registry.py`
- Modify: `tests/tools/test_piano_roll_notes_debug.py`

**Acceptance criteria:**
- [x] The debug registry exposes a `"v7"` variant that calls `extract_notes_v7`.
- [x] The debug CLI can render v5/v6/v7 comparisons for an Event Encoder event without special-case changes.
- [x] Registry tests assert `"v7"` is available.

**Tests needed:**
- Focused registry/debug test coverage matching the existing v5/v6 test style.

---

### Task 4: Update Project Context, References, and ADRs

**Files:**
- Modify: `DECISIONS.md`
- Modify: `docs/agent-context/current-state.md`
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/invariants.md`
- Modify: `docs/reference/signal-processing.md`
- Modify: `docs/reference/storage-layout.md`

**Acceptance criteria:**
- [x] Add an ADR for Piano Roll Notes v7 that records residual discontinuity splitting, ridge-guided flat-segment rescue, defaults, sidecar compatibility, and no frontend/MIDI schema change.
- [x] Current-state and sequence-models domain context name v7 as the default extractor.
- [x] Sequence Models invariants document v7's split and rescue behavior and preserve v3-v7 schema compatibility.
- [x] Signal-processing reference documents v7's pipeline and parameter defaults.
- [x] Storage-layout reference lists `event_notes_v7.parquet` and `event_note_contours_v7.parquet`.

**Tests needed:**
- No dedicated doc tests unless an existing reference assertion covers version strings; otherwise rely on review plus standard verification.

---

### Task 5: Run Targeted Visual and Regression Verification

**Files:**
- No source files expected; generated visual outputs should stay under `.tmp/` and must not be committed.

**Acceptance criteria:**
- [x] `5a50b01f7a014ceaaf5ca52e81e6ad42` no longer has one continuous bend between low and high branches.
- [x] `dad6357550ba4a9cadcf05e103f923ee` branch changes become note boundaries rather than violent pitch bends.
- [x] `0d698411766142ee908fa6a4c2ac81ec` no longer bends from the offscreen low branch into the visible harmonic stack.
- [x] `e3e8e9c4b5c0403a91ef3f50e965fdf0` remains a smooth continuous pitch change.
- [x] `f470e42f95974358885c9392dffd83ee` gains an upsweep contour from ridge rescue instead of staying flat.
- [x] Generated debug renders are reviewed manually and left untracked.

**Tests needed:**
- Run the debug CLI against the listed events with v6 and v7 variants and inspect the PNGs.
- Run targeted pytest files for the v7 extractor, worker wiring, and debug registry before broader gates.

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/processing/note_extractor_v7.py src/humpback/processing/note_extractor_v6.py src/humpback/workers/piano_roll_notes_worker.py src/humpback/models/piano_roll_notes.py tools/piano_roll_notes_registry.py tests/processing/test_note_extractor_v7.py tests/unit/test_piano_roll_notes_worker_v7.py tests/tools/test_piano_roll_notes_debug.py`
2. `uv run ruff check src/humpback/processing/note_extractor_v7.py src/humpback/processing/note_extractor_v6.py src/humpback/workers/piano_roll_notes_worker.py src/humpback/models/piano_roll_notes.py tools/piano_roll_notes_registry.py tests/processing/test_note_extractor_v7.py tests/unit/test_piano_roll_notes_worker_v7.py tests/tools/test_piano_roll_notes_debug.py`
3. `uv run pyright src/humpback/processing/note_extractor_v7.py src/humpback/processing/note_extractor_v6.py src/humpback/workers/piano_roll_notes_worker.py src/humpback/models/piano_roll_notes.py tools/piano_roll_notes_registry.py`
4. `uv run pytest tests/processing/test_note_extractor_v7.py tests/unit/test_piano_roll_notes_worker_v7.py tests/tools/test_piano_roll_notes_debug.py`
5. `uv run pytest tests/`
