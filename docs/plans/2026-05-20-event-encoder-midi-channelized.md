# Event Encoder MIDI Channelization & Per-Frame Harmonic Labeling — Implementation Plan

**Goal:** Rewrite the Piano Roll Notes harmonic labeler with per-frame ratio matching and channelize the MIDI export onto seven named tracks with per-channel General MIDI programs.

**Spec:** [docs/specs/2026-05-20-event-encoder-midi-channelized-design.md](../specs/2026-05-20-event-encoder-midi-channelized-design.md)

**Primary domain:** sequence-models

**Neighbor domains:** none — all changes live inside `sequence-models` (`src/humpback/processing/piano_roll_tracker.py`, `src/humpback/processing/midi_synthesis.py`, `src/humpback/models/piano_roll_notes.py`, and the corresponding test modules). No boundary contracts with `core-platform`, `signal-timeline`, `call-parsing`, or `frontend-shell` are altered: parquet schema unchanged, DB schema unchanged, API/router unchanged, frontend renderer consumes existing `partial_index` field transparently.

**No backward compatibility:** Per the user, v1 parquet/MIDI artifacts on disk and corresponding job rows will be deleted manually via the UI after this lands. No migration. No batch rebuilder.

---

### Task 1: Rewrite `label_harmonics()` with per-frame ratio matching

**Files:**
- Modify: `src/humpback/processing/piano_roll_tracker.py`

**Acceptance criteria:**
- [ ] `HarmonicParams` gets a new field `min_overlap_frames: int = 3`.
- [ ] `HarmonicParams.max_harmonic` default changes from `8` to `16`.
- [ ] `HarmonicParams.cents_tolerance` default changes from `50.0` to `75.0`.
- [ ] `label_harmonics()` sorts tracks by `median_bin` ascending only (no `start_frame` tiebreaker).
- [ ] F0 anchor selection: each unprocessed track in that order becomes an F0 candidate and is marked `partial_index = 0`.
- [ ] Harmonic check uses per-frame ratios computed from the `bins` lists of the candidate and the prospective harmonic at each frame where both tracks have data; ratios are converted to nearest-integer harmonic and absolute cents deviation; the summary is `median nearest-integer harmonic` + `median absolute cents deviation`.
- [ ] Candidates with fewer than `params.min_overlap_frames` overlapping frames are rejected without consuming the prospective track.
- [ ] Tracks that fail the harmonic check are **not** marked processed — they remain eligible to anchor their own cluster on later iterations.
- [ ] Tracks that pass the check are marked processed and receive `partial_index = median_harmonic - 1`.
- [ ] When `params.enabled is False`, the function is a no-op (all `partial_index` stay at -1) — existing behavior preserved.
- [ ] When `tracks` is empty, the function returns the empty list — existing behavior preserved.
- [ ] Frequency conversion goes through `bin_frequency_hz(...)` from `piano_roll_cqt.py`; do not duplicate the formula in-module.
- [ ] No change to `Track`, `MidiNote`, `TrackerParams`, `MIDIQuantizeParams`, `build_tracks`, or `quantize_to_midi` in this task.

**Tests needed:**
- Two-track exact 2× pair across all frames → `partial_index` becomes `0` and `1`.
- F0 anchor sort fix: higher-bin earlier-starting track + lower-bin later-starting track holding a 0.5× ratio → lower-bin track gets `partial_index = 0`; higher-bin gets `partial_index = 1`.
- Consume-on-overlap fix: non-harmonic overlapping track is left available and anchors its own (singleton) cluster on a later iteration with `partial_index = 0`.
- Per-frame ratio handles sweeping pitches whose medians do not align cleanly (constructed bin trajectories such that `median(other) / median(F0) ≠ 2.0` but per-frame ratios are all ≈ 2.0).
- `min_overlap_frames = 3` rejects a clean-2× candidate that overlaps F0 for only two frames.
- `max_harmonic = 16` lets a track at 10× F0 receive `partial_index = 9`.
- `cents_tolerance = 75` boundary tests: ratio 2.04 (~34¢ off) is accepted, ratio 2.10 (~84¢ off) is rejected. Lock the new policy with explicit boundary assertions.
- Determinism: identical input tracks produce identical `partial_index` across repeated calls.
- `enabled = False` short-circuit: every track keeps `partial_index = -1`.

---

### Task 2: Bump `DEFAULT_EXTRACTOR_VERSION` to `"v2"`

**Files:**
- Modify: `src/humpback/models/piano_roll_notes.py`
- Modify (only if a literal `"v1"` is present): `tests/workers/test_piano_roll_notes_worker.py`
- Modify (only if a literal `"v1"` is present): `tests/workers/test_piano_roll_midi_export_worker.py`
- Modify (only if a literal `"v1"` is present): `tests/services/test_piano_roll_notes_service.py`
- Modify (only if a literal `"v1"` is present): `tests/services/test_piano_roll_midi_export_service.py`
- Modify (only if a literal `"v1"` is present): `tests/integration/test_sequence_models_api.py`

**Acceptance criteria:**
- [ ] `DEFAULT_EXTRACTOR_VERSION` constant is `"v2"`.
- [ ] All re-exports of the constant (e.g., in `piano_roll_midi_export_service.py`) pick up the new value via import; do not duplicate the literal.
- [ ] Grep for `"v1"` across the touched test files; replace literals only when they specify the active default extractor version. Leave alone any literal that intentionally pins a historical version for a regression test.
- [ ] No code path defaults to `"v1"` after this task — the only `"v1"` references that should remain are explicit historical pins.

**Tests needed:**
- Existing piano-roll-notes worker and service tests pass against the new default.
- A new assertion in `tests/unit/test_piano_roll_notes_model.py` (create if absent, or add to an existing piano-roll-notes test module) confirms `DEFAULT_EXTRACTOR_VERSION == "v2"`.

---

### Task 3: Channelize `notes_table_to_midi_bytes()`

**Files:**
- Modify: `src/humpback/processing/midi_synthesis.py`

**Acceptance criteria:**
- [ ] Module-level constant `MIDI_CHANNEL = 0` is removed.
- [ ] New named channel constants `CHANNEL_F0`, `CHANNEL_HARMONIC_2`, `CHANNEL_HARMONIC_3`, `CHANNEL_HARMONIC_4`, `CHANNEL_HARMONIC_5`, `CHANNEL_HARMONIC_HIGH`, `CHANNEL_UNMATCHED` (values 0, 1, 2, 3, 4, 5, 6).
- [ ] New frozen dataclass `ChannelSpec(channel: int, program: int, name: str)`.
- [ ] New module-level tuple `CHANNEL_LAYOUT` of seven `ChannelSpec`s in channel order, with programs `(0, 11, 12, 10, 8, 88, 90)` and names `("F0", "2nd harmonic", "3rd harmonic", "4th harmonic", "5th harmonic", "higher harmonics", "unmatched")`.
- [ ] Internal `_channel_for_partial(partial_index)` returns `CHANNEL_UNMATCHED` for `-1`, `CHANNEL_F0` for `0`, `CHANNEL_HARMONIC_2 + (p - 1)` for `1..4`, and `CHANNEL_HARMONIC_HIGH` for `partial_index >= 5`.
- [ ] `_build_notes_track()` is replaced by `_build_channel_tracks()` which returns `list[mido.MidiTrack]` of length 7 (one per channel in layout order).
- [ ] Each channel track begins at tick 0 with a `track_name` meta-event then a `program_change` for its assigned program on its channel.
- [ ] Note events within a channel track are sorted by `(absolute_tick, off_before_on, original_row_order)` as today.
- [ ] All seven channel tracks are emitted even when one or more carries zero notes; empty channels still write `track_name` + `program_change` + `end_of_track`.
- [ ] No message in any output track has `channel = 9` (GM drum channel left empty).
- [ ] Empty parquet → tempo track only, no channel tracks. Existing defensive behavior preserved.
- [ ] `notes_table_to_midi_bytes(notes_table)` external signature unchanged.
- [ ] Time origin (`t=0` = earliest `start_utc`), 120 BPM tempo, 480 PPQ, pitch clamp `[0, 127]`, zero-duration skipping, deterministic output — all preserved.
- [ ] Module `__all__` is updated to expose the new public surface (constants, `ChannelSpec`, `CHANNEL_LAYOUT`) and remove `MIDI_CHANNEL`.

**Tests needed:**
- Channel routing fixture: parquet rows for `partial_index ∈ {-1, 0, 1, 2, 3, 4, 5, 7, 12}` → eight output tracks (tempo + 7); each note lands on the channel returned by `_channel_for_partial`; `partial_index ∈ {5, 7, 12}` all land on `CHANNEL_HARMONIC_HIGH`.
- Track headers: each channel track's first non-meta-of-zero event sequence is `track_name(name=...)` then `program_change(program=..., channel=...)`, matching `CHANNEL_LAYOUT`.
- Empty-channel handling: parquet containing only `partial_index = 0` notes still emits seven channel tracks; the six empty channels carry only `track_name`, `program_change`, and `end_of_track`.
- GM drum channel exclusion: scan every message in every output track and assert `getattr(msg, "channel", None) != 9`.
- Determinism: bytes produced by two successive calls on the same parquet are byte-identical.
- Existing invariants migrated to the new structure: pitch clamping (clamped to 0/127), zero-duration row skipping, empty parquet → tempo track only (no channel tracks), time-origin at earliest `start_utc`, deterministic note ordering inside one channel.

---

### Task 4: Update worker and integration tests for v2 + channelized export

**Files:**
- Modify: `tests/workers/test_piano_roll_notes_worker.py`
- Modify: `tests/workers/test_piano_roll_midi_export_worker.py`
- Modify: `tests/services/test_piano_roll_midi_export_service.py`
- Modify: `tests/integration/test_sequence_models_api.py`

**Acceptance criteria:**
- [ ] Worker round-trip tests that previously asserted on `event_notes_v1.parquet` now assert on `event_notes_v2.parquet`.
- [ ] Worker round-trip tests that previously asserted on `notes_v1.mid` now assert on `notes_v2.mid`.
- [ ] The `Content-Disposition` filename assertion in the API integration test expects `event_encoder_{job_id}_notes_v2.mid`.
- [ ] Any in-test parquet fixtures that hand-set `partial_index = 0` for every row are reviewed: if the test is asserting on MIDI channel placement, ensure the partials are explicit and cover at least F0 and one harmonic so the assertions remain meaningful under channelization.
- [ ] Tests that synthesize a small in-memory parquet and check the resulting MIDI byte structure are updated to expect the multi-track layout (>= 3 tracks including tempo for any non-empty parquet).

**Tests needed:**
- (Same files; no new test modules.)

---

### Task 5: Update Playwright E2E for `notes_v2.mid` and structural MIDI assertions

**Files:**
- Modify: `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts`

**Acceptance criteria:**
- [ ] The expected download filename suffix is updated from `notes_v1.mid` to `notes_v2.mid`.
- [ ] After download, the spec parses the response body bytes (or just checks header + magic bytes "MThd") and asserts the file claims SMF Type 1 with at least three tracks. Deeper MIDI assertions remain in the backend unit tests.
- [ ] If the spec previously created Piano Roll Notes via API, ensure the request omits `extractor_version` (taking the default) or explicitly passes `"v2"`. No literal `"v1"` left.

**Tests needed:**
- (Existing spec; updated assertions only.)

---

### Task 6: Manual end-to-end smoke against the reference job

**Files:**
- None (manual verification).

**Acceptance criteria:**
- [ ] Via the UI, delete the v1 Piano Roll Notes job and v1 MIDI Export job for `event_encoder_job_id = b759d8bf-0ecf-469a-b169-333b36c60906`, including their on-disk artifacts.
- [ ] Re-enqueue the Piano Roll Notes job; verify it writes `event_notes_v2.parquet`.
- [ ] Inspect the new parquet: assert that the share of `partial_index = -1` is materially below the v1 baseline of 82.8% (target: ≤ 30%). Record the actual distribution in the implementation-session notes.
- [ ] Re-enqueue the MIDI Export job; verify it writes `notes_v2.mid`.
- [ ] Open `notes_v2.mid` in a DAW (e.g., Logic, GarageBand, or Reaper). Confirm: seven named channel tracks visible ("F0" … "unmatched"); muting the F0 track silences the lowest line; harmonic stack audible on the harmonic tracks; the unmatched track is materially less populated than under v1.

**Tests needed:**
- (Manual; no automated test.)

---

### Task 7: Documentation and ADR

**Files:**
- Modify: `DECISIONS.md` (append ADR-067)
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/invariants.md`
- Modify: `docs/agent-context/current-state.md`
- Modify: `docs/specs/2026-05-20-piano-roll-midi-notes-design.md`
- Modify: `docs/specs/2026-05-20-piano-roll-midi-export-design.md`

**Acceptance criteria:**
- [ ] **ADR-067: Per-frame harmonic labeling and channelized MIDI export** is appended to `DECISIONS.md`. Captures: per-frame median-cents algorithm, `max_harmonic` 8 → 16, `cents_tolerance` 50 → 75, new `min_overlap_frames = 3`, F0 anchor sort fix, conditional consume-on-overlap, slim 7-channel MIDI layout (F0 + 2nd–5th + higher + unmatched, channel 10 skipped), multi-track SMF Type 1 with per-channel `program_change` and `track_name`, extractor version `v1 → v2`, and the explicit no-backward-compatibility decision.
- [ ] `sequence-models/README.md` MIDI-export subsection no longer claims "all partials stacked on MIDI channel 1." It instead describes the 7-channel slim layout, GM `program_change` per channel, and named tracks. Artifact-path examples reference `v2`.
- [ ] `sequence-models/invariants.md` adds: harmonic prior labels tracks via per-frame ratios with median-nearest-integer aggregation; F0 anchor is the lowest-bin track per cluster; tracks that fail the harmonic check remain eligible to anchor their own cluster.
- [ ] `docs/agent-context/current-state.md` carries a one-line update mentioning the channelized export and the per-frame labeling.
- [ ] `docs/specs/2026-05-20-piano-roll-midi-notes-design.md` gets a "Superseded in part" banner at the top citing this spec for §6.5 (labeling).
- [ ] `docs/specs/2026-05-20-piano-roll-midi-export-design.md` gets a "Superseded in part" banner at the top citing this spec for §10 (synthesis / channelization).

**Tests needed:**
- None.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/processing/piano_roll_tracker.py src/humpback/processing/midi_synthesis.py src/humpback/models/piano_roll_notes.py tests/processing/test_piano_roll_tracker.py tests/processing/test_midi_synthesis.py tests/workers/test_piano_roll_notes_worker.py tests/workers/test_piano_roll_midi_export_worker.py tests/services/test_piano_roll_midi_export_service.py tests/integration/test_sequence_models_api.py`
2. `uv run ruff check src/humpback/processing/piano_roll_tracker.py src/humpback/processing/midi_synthesis.py src/humpback/models/piano_roll_notes.py tests/processing/test_piano_roll_tracker.py tests/processing/test_midi_synthesis.py tests/workers/test_piano_roll_notes_worker.py tests/workers/test_piano_roll_midi_export_worker.py tests/services/test_piano_roll_midi_export_service.py tests/integration/test_sequence_models_api.py`
3. `uv run pyright src/humpback/processing/piano_roll_tracker.py src/humpback/processing/midi_synthesis.py src/humpback/models/piano_roll_notes.py tests/processing/test_piano_roll_tracker.py tests/processing/test_midi_synthesis.py tests/workers/test_piano_roll_notes_worker.py tests/workers/test_piano_roll_midi_export_worker.py tests/services/test_piano_roll_midi_export_service.py tests/integration/test_sequence_models_api.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test e2e/sequence-models/event-encoder-piano-roll.spec.ts`
7. Manual round-trip per Task 6, including the partial_index distribution inspection.
