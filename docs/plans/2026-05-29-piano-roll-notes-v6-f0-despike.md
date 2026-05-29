# Piano Roll Notes v6 — F0 De-spike Implementation Plan

**Goal:** Add a slope-based F0 contour de-spike pass and ship it as the new default Piano Roll Notes extractor version v6.
**Spec:** [docs/specs/2026-05-29-piano-roll-notes-v6-f0-despike-design.md](../specs/2026-05-29-piano-roll-notes-v6-f0-despike-design.md)
**Primary domain:** sequence-models
**Neighbor domains:** core-platform (`DEFAULT_EXTRACTOR_VERSION` model constant + worker param plumbing)

No Alembic migration: v6 reuses the existing parquet schemas and the
`piano_roll_notes_jobs` / `piano_roll_midi_exports` tables unchanged.
No database backup step is required for this plan.

---

### Task 1: v6 de-spike extractor (pure functions)

**Files:**
- Create: `src/humpback/processing/note_extractor_v6.py`

**Acceptance criteria:**
- [ ] `DespikeParams` dataclass (frozen, slots): `enabled: bool = True`, `max_slope_oct_per_s: float = 6.0`, `max_spike_frames: int = 12`.
- [ ] `ExtractNotesV6Params` mirrors `ExtractNotesV5Params` field-for-field plus a `despike: DespikeParams` field; `pad_seconds` default matches v5 (0.25 is applied by the worker, dataclass default may stay 0.05 for parity with v5's dataclass).
- [ ] A pure `despike_f0_segments(...)` (or per-segment helper) implements the slew-rate anchor walk from spec §4: per-frame budget `max_step_log = max_slope_oct_per_s * dt`; accept frames inside the anchor envelope; excise + linearly bridge log-frequency across skipped frames; `max_spike_frames` guard accepts a new anchor; leading/trailing spikes held by constant extrapolation. Returns new frozen `_RefinedFrame` instances with `strength` and `subharmonic_octave` (0) carried through.
- [ ] `extract_notes_v6` reuses v5's `_decode_f0` and v3's `_build_f0_note` / `_build_harmonic_notes`, inserting the de-spike pass between decode and build. Returns `NotesV3Result`; `subharmonic_octave` is 0 on every contour row.
- [ ] With `despike.enabled = False`, `extract_notes_v6` produces output equal to `extract_notes_v5` for the same audio/params.
- [ ] `dt` is derived from `cqt.hop_length / cqt.target_sample_rate` (no new hard-coded rate).

**Tests needed:**
- `tests/processing/test_note_extractor_v6.py`: synthetic contour with a single out-and-back spike → bridged to the interpolated line; sustained legal glide → unchanged; leading spike and trailing spike → held by extrapolation; excursion longer than `max_spike_frames` → far level accepted (not bridged indefinitely); multi-spike segment → all bridged.
- `extract_notes_v6` smoke test on synthetic audio: same `NotesV3Result` contract as v5; `enabled=False` reproduces v5 output.

---

### Task 2: Worker dispatch + params plumbing

**Files:**
- Modify: `src/humpback/workers/piano_roll_notes_worker.py`

**Acceptance criteria:**
- [ ] Add `_V6_EXTRACTOR_VERSION = "v6"`; include it in `_RIDGE_AWARE_VERSIONS` and `_RIDGE_AWARE_EXTRACTORS`.
- [ ] Add `_extract_notes_v6` async wrapper mirroring `_extract_notes_v5`, constructing `ExtractNotesV6Params` (including `despike`) and calling `extract_notes_v6`.
- [ ] `_ResolvedParams` gains a `despike: DespikeParams` field; `_resolve_params` parses a `"despike"` section (with the spec defaults) and `to_json_dict` emits it.
- [ ] The version-conditional defaults are extended to treat v6 like v5: 30 Hz STFT `min_frequency_hz`, `segmentation.min_break_frames = 6`, and audio `pad_seconds = 0.25`.
- [ ] v6 reuses the v3 parquet schemas and the existing `_note_v3_row` / `_contour_v3_row` writers; output files are `event_notes_v6.parquet` / `event_note_contours_v6.parquet`.

**Tests needed:**
- Worker-level test (extend `tests/unit/test_piano_roll_notes_worker_extraction.py` or add `tests/unit/test_piano_roll_notes_worker_v6.py`): v6 dispatches through the ridge-aware path and writes both parquet artifacts; `_resolve_params(json, "v6")` applies the v6 defaults and round-trips a `despike` override.

---

### Task 3: Default bump + test-bed variant

**Files:**
- Modify: `src/humpback/models/piano_roll_notes.py`
- Modify: `tools/piano_roll_notes_registry.py`

**Acceptance criteria:**
- [ ] `DEFAULT_EXTRACTOR_VERSION` is `"v6"`.
- [ ] `tools/piano_roll_notes_registry.py` adds a `_run_v6` wrapper and registers `"v6"` in `EXTRACTORS`, forwarding `pad_seconds` like the other variants.
- [ ] MIDI export resolver and frontend are confirmed unchanged (resolver orders by `desc(extractor_version)`, so `"v6" > "v5"`; frontend displays the version string verbatim).

**Tests needed:**
- `tests/tools/test_piano_roll_notes_debug.py`: `"v6"` is a known variant.
- Update any test that asserts the default version is `"v5"` to expect `"v6"`.

---

### Task 4: ADR-072 + documentation

**Files:**
- Modify: `DECISIONS.md`
- Modify: `docs/agent-context/current-state.md`
- Modify: `docs/agent-context/domains/sequence-models/invariants.md`
- Modify: `docs/reference/signal-processing.md`

**Acceptance criteria:**
- [ ] `DECISIONS.md` gains `## ADR-072: Piano Roll Notes v6 — slope-based F0 de-spike` summarizing the excise+bridge / steep-is-error / anchor-walk decisions and the default bump.
- [ ] `current-state.md` Sequence Models section notes `DEFAULT_EXTRACTOR_VERSION = "v6"` and the de-spike behaviour; reuses v5 schemas.
- [ ] sequence-models `invariants.md` gains a v6 bullet (de-spike pass, parameters, `subharmonic_octave = 0`, schema parity, no auto-backfill).
- [ ] `signal-processing.md` documents the de-spike algorithm and `DespikeParams`.

**Tests needed:**
- None (documentation only).

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/processing/note_extractor_v6.py src/humpback/workers/piano_roll_notes_worker.py src/humpback/models/piano_roll_notes.py tools/piano_roll_notes_registry.py tests/processing/test_note_extractor_v6.py`
2. `uv run ruff check` on the same files
3. `uv run pyright` on the same files
4. `uv run pytest tests/`
5. Visual confirmation (manual): `uv run python tools/piano_roll_notes_debug.py --job 690580c5-7804-43c9-bd8d-690691b5d6d4 --event-id 669849340bff411390e5eaaf1ec9b9e9 --variants v5,v6 --out /tmp/v6_event2.png` and the same for event `0be3d3520789414cb1f494eec50ba641` — confirm the t≈1.2 s spike is gone in v6 and the surrounding contour matches v5.

No frontend changes → `npx tsc --noEmit` / Playwright not required.
