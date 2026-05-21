# Event Encoder MIDI Channelization & Per-Frame Harmonic Labeling — Design Spec

**Date:** 2026-05-20
**Status:** Draft
**Owner:** Sequence Models / Event Encoder Piano Roll
**Supersedes (in part):**
[2026-05-20-piano-roll-midi-notes-design.md](2026-05-20-piano-roll-midi-notes-design.md) §6.5 (harmonic labeling) and
[2026-05-20-piano-roll-midi-export-design.md](2026-05-20-piano-roll-midi-export-design.md) §10 (single-channel MIDI synthesis).

## 1. Summary

Two coupled changes to the Event Encoder Piano Roll Notes pipeline:

1. **Upstream**: replace `label_harmonics()` with a per-frame ratio matcher,
   fix two F0-selection bugs, and widen the harmonic search range. The current
   labeler leaves ~83% of tracker output as `partial_index = -1` because of a
   `max_harmonic = 8` cap, an over-eager consume-on-overlap step, an F0
   anchor-selection that's sorted on `(start_frame, median_bin)` instead of
   `median_bin`, and a median-bin ratio metric that fails for the sweeping
   pitches that humpback songs are made of. Expected `-1` rate after this
   change: ~17–25%.

2. **Downstream**: channelize the MIDI export. Today every note lands on MIDI
   channel 1. The new export routes notes onto seven channels — F0, 2nd–5th
   harmonic, "higher harmonics" (6+), and "unmatched" — using a multi-track
   SMF Type 1 layout with one track per channel, distinct General MIDI
   program assignments, and named track meta-events so DAWs render each
   partial as its own mute/solo lane out of the box.

The notes parquet schema is unchanged. The extractor version bumps from
`v1` to `v2` to mark the changed labeling semantics; v1 artifacts on disk are
not migrated and are expected to be manually deleted via the existing UI.

## 2. Motivation

The Piano Roll Notes parquet (ADR-064) carries a `partial_index` column that
labels each detected note as F0, an integer harmonic of F0, or unmatched.
Inspecting a representative export of 1,672 events / 80,561 notes
(`event_encoder_job_id = b759d8bf-0ecf-469a-b169-333b36c60906`) reveals:

- **82.8% of notes carry `partial_index = -1`** — far more than the harmonic
  prior should produce on real song data.
- **55% of those `-1` tracks sit above 8× F0**, excluded by the hard cap.
- **6.1% of `-1` tracks fall within ±50¢ of an integer multiple of F0** —
  they *should* have matched but didn't, because the F0 anchor selection
  picked the wrong track first and consumed the real F0 unconditionally.
- **17% of `-1` tracks** are genuinely non-harmonic (sidebands, FM
  modulation, secondary callers) and cannot be labeled by any harmonic
  prior — these are residual after any algorithmic improvement.

Independently, users want to audition harmonics separately in a DAW — mute
the F0 lane to hear just the harmonic stack, or solo the 2nd to study its
modulation. The current single-channel export forces every partial onto one
track. The two changes together close both gaps in one pass.

## 3. Scope

**In scope:**

- Rewrite `label_harmonics()` in `src/humpback/processing/piano_roll_tracker.py`.
- New `HarmonicParams` defaults: `max_harmonic=16`, `cents_tolerance=75`,
  new `min_overlap_frames=3`.
- Channelize `notes_table_to_midi_bytes()` in
  `src/humpback/processing/midi_synthesis.py` using a fixed slim 7-channel
  layout, one SMF track per channel, GM `program_change` and `track_name`
  per channel.
- Bump `DEFAULT_EXTRACTOR_VERSION` from `"v1"` to `"v2"` in
  `src/humpback/models/piano_roll_notes.py`.
- Update sequence-models domain capsule and relevant ADRs.
- Backend unit tests for both the new labeler and the new synthesizer.

**Out of scope (deferred):**

- Backward compatibility with v1 artifacts. v1 parquet files and v1 MIDI
  exports on disk will be deleted manually via the UI; this spec does not
  ship migrations or rebuilders.
- Concurrent / multi-F0 cluster handling in one event. The new algorithm
  remains single-cluster-per-event; tracks not absorbed into the dominant
  cluster stay `partial_index = -1`.
- MPE / pitch-bend support. Still requires a future `v3` parquet with
  sub-semitone pitch contours.
- User-configurable channel layout via `params_json`. Layout is hard-coded.
- Per-channel velocity rebalancing. Velocities continue to derive from
  `peak_magnitude` percentile normalization — higher harmonics being
  quieter is information, not an artifact.
- Frontend changes to the Notes-mode piano roll renderer. The renderer
  already reads `partial_index` for color-coding; the changed distribution
  is consumed transparently.

## 4. Decisions

| Decision | Choice |
|---|---|
| Upstream scope | Bug fixes + per-frame ratio matching, single-cluster-per-event |
| F0 anchor sort key | `median_bin` ascending (no `start_frame` tiebreaker) |
| Consume-on-overlap semantics | Only consume tracks that pass the harmonic check |
| Ratio metric | Per-frame ratios → median nearest-integer harmonic, median abs-cents deviation |
| `max_harmonic` | 16 (was 8) |
| `cents_tolerance` | 75¢ (was 50¢) |
| `min_overlap_frames` | 3 (new) |
| Parquet schema | Unchanged. `partial_index` keeps existing semantics; only the distribution of values changes |
| Extractor version | Bump `v1` → `v2` to mark changed labeling semantics |
| v1 artifact migration | None. User deletes via UI; new jobs write v2 |
| MIDI channel layout | Slim 7-channel: F0, 2nd–5th, higher (≥6), unmatched. GM drum channel 10 left empty |
| SMF track layout | One track per channel + leading tempo track (SMF Type 1) |
| Per-channel GM program | Distinct GM patches; sent at tick 0 of each channel track |
| Per-channel `track_name` | Named meta-event ("F0", "2nd harmonic", …, "unmatched") |
| Empty-channel handling | Track still emitted with program_change + track_name + end_of_track |
| Determinism | Preserved — identical parquet → identical bytes |
| Time origin, tempo, PPQ | Unchanged — t=0 at earliest start_utc, 120 BPM, 480 PPQ |

## 5. Labeling Algorithm

### 5.1 New `HarmonicParams`

```
@dataclass(frozen=True, slots=True)
class HarmonicParams:
    enabled: bool = True
    max_harmonic: int = 16              # was 8
    cents_tolerance: float = 75.0       # was 50.0
    min_overlap_frames: int = 3         # new
```

Existing `HarmonicParams.enabled = False` short-circuit is preserved (no-op,
all tracks keep `partial_index = -1`).

### 5.2 Algorithm

`label_harmonics(tracks, *, cqt_params, params)` is rewritten to:

1. Sort `tracks` by `median_bin` ascending. Iterate over `tracks` in that
   order; each unprocessed track becomes the next F0 anchor candidate.
2. Mark the candidate with `partial_index = 0` and mark it processed.
3. For every other unprocessed track that time-overlaps the candidate for
   at least `min_overlap_frames` frames:
   - Compute per-frame ratios at each overlapping frame using
     `bin_frequency_hz(other_bin, …) / bin_frequency_hz(candidate_bin, …)`.
   - Reduce to two summary numbers: the median nearest-integer harmonic
     across overlapping frames (`median_harmonic`), and the median absolute
     deviation in cents to that integer multiple
     (`median_abs_cents_deviation`).
   - Accept iff `2 ≤ median_harmonic ≤ params.max_harmonic` AND
     `median_abs_cents_deviation ≤ params.cents_tolerance`. On accept, set
     `other.partial_index = median_harmonic - 1` and mark processed.
   - On reject, **leave the track unprocessed** so it remains eligible to
     anchor its own cluster on a later iteration.
4. After the sweep, tracks that were never matched keep
   `partial_index = -1`.

### 5.3 Key behavioral differences vs. v1

| Behavior | v1 | v2 |
|---|---|---|
| F0 anchor priority | first by `(start_frame, median_bin)` | lowest `median_bin` |
| Consume-on-overlap | unconditional `used.add` before check | only on accepted matches |
| Ratio metric | one ratio of medians per track pair | median nearest-integer harmonic over per-frame ratios |
| Temporal-support floor | none | `min_overlap_frames = 3` |
| Harmonic ceiling | 8 | 16 |
| Tolerance | 50¢ | 75¢ |

### 5.4 Predicted impact

From the deep-dive numbers on the reference job:

- `max_harmonic` 8 → 16 reclaims ~55% of current `-1` (the above-8× content)
- F0 sort fix + consume-on-overlap fix + 50→75¢ widening reclaims a
  further ~5–10% (the "would-be" + "near 50–100¢" buckets)
- Residual `-1`: ~17% (genuinely non-harmonic content)
- Expected new `-1` rate: **17–25%**, down from 83%

### 5.5 Out of scope for v2 (callouts)

- Multi-cluster (concurrent F0s) within one event. Tracks that don't fit
  the dominant cluster are left unprocessed and eventually anchor their
  own cluster *only if* iteration reaches them in `median_bin` order. The
  algorithm still cannot represent two simultaneous whales with disjoint
  F0s and overlapping harmonic series — that would require a graph-cluster
  model and is a future project.
- Frame-by-frame harmonic re-labeling. A track gets one `partial_index` for
  its entire lifetime even if its actual harmonic identity shifts
  mid-track.

## 6. MIDI Synthesis

### 6.1 Channel layout

The export uses seven of the sixteen MIDI 1.0 channels. Channel 10 (GM
drum kit) is intentionally left empty so no pitched humpback content gets
re-mapped to drum sounds in any GM-compliant playback engine.

| Channel (0-indexed) | Channel (1-indexed) | Purpose | partial_index | GM program | Track name |
|---|---|---|---|---|---|
| 0 | 1 | F0 | 0 | 0 (Acoustic Grand Piano) | "F0" |
| 1 | 2 | 2nd harmonic | 1 | 11 (Vibraphone) | "2nd harmonic" |
| 2 | 3 | 3rd harmonic | 2 | 12 (Marimba) | "3rd harmonic" |
| 3 | 4 | 4th harmonic | 3 | 10 (Music Box) | "4th harmonic" |
| 4 | 5 | 5th harmonic | 4 | 8 (Celesta) | "5th harmonic" |
| 5 | 6 | Higher harmonics | 5..15 | 88 (New Age Pad) | "higher harmonics" |
| 6 | 7 | Unmatched | -1 | 90 (Polysynth Pad) | "unmatched" |
| 9 | 10 | (skipped — GM drums) | (none) | (none) | (none) |

The mapping function:

```
def _channel_for_partial(partial_index: int) -> int:
    if partial_index == -1: return 6        # unmatched
    if partial_index == 0:  return 0        # F0
    if 1 <= partial_index <= 4: return partial_index   # 2nd..5th
    return 5                                # 6th and above
```

### 6.2 SMF structure

SMF Type 1, 480 PPQ, constant 120 BPM written once at tick 0 of a dedicated
tempo track. All channel tracks are siblings of the tempo track.

```
Track 0: tempo            (set_tempo @ t=0, end_of_track)
Track 1: F0               (program_change ch=0 prog=0,  track_name "F0",              notes…)
Track 2: 2nd harmonic     (program_change ch=1 prog=11, track_name "2nd harmonic",    notes…)
Track 3: 3rd harmonic     (program_change ch=2 prog=12, track_name "3rd harmonic",    notes…)
Track 4: 4th harmonic     (program_change ch=3 prog=10, track_name "4th harmonic",    notes…)
Track 5: 5th harmonic     (program_change ch=4 prog=8,  track_name "5th harmonic",    notes…)
Track 6: higher harmonics (program_change ch=5 prog=88, track_name "higher harmonics",notes…)
Track 7: unmatched        (program_change ch=6 prog=90, track_name "unmatched",       notes…)
```

The leading `track_name` and `program_change` events are written at tick 0
of their track. Note-on / note-off events follow with the same
absolute-tick → delta encoding currently used by `_build_notes_track`.

### 6.3 Per-track event ordering

Within each channel track, events are sorted by `(absolute_tick,
off_before_on, original_row_order)` exactly as today. `mido` interleaves
tracks during save, so the global byte order is determined automatically
once each track is correctly sorted internally.

### 6.4 Empty channels

If a parquet contains zero notes for a given channel, the corresponding
track is still emitted with `program_change`, `track_name`, and
`end_of_track`. This:

- Keeps the file's track layout structurally identical across jobs so DAW
  project templates and routing configs are portable.
- Makes "this channel is genuinely empty" visible rather than ambiguous —
  a missing track could otherwise be mistaken for a parsing failure.

### 6.5 Empty parquet

Same as today: emit the tempo track only. No channel tracks. Defensive
case only — Piano Roll Notes jobs that produce zero notes are themselves
treated as anomalous.

### 6.6 Public API

`notes_table_to_midi_bytes(notes_table: pa.Table) -> bytes` signature
unchanged. The internal `_build_notes_track()` is replaced by
`_build_channel_tracks()` returning `list[mido.MidiTrack]`.

The module-level `MIDI_CHANNEL = 0` constant is removed (no external
callers). New module-level exports:

```
__all__ = [
    "TICKS_PER_QUARTER",
    "TEMPO_BPM",
    "ChannelSpec",
    "CHANNEL_LAYOUT",
    "CHANNEL_F0",
    "CHANNEL_HARMONIC_2", "CHANNEL_HARMONIC_3",
    "CHANNEL_HARMONIC_4", "CHANNEL_HARMONIC_5",
    "CHANNEL_HARMONIC_HIGH",
    "CHANNEL_UNMATCHED",
    "notes_table_to_midi_bytes",
]
```

`ChannelSpec` is a frozen dataclass `(channel: int, program: int, name: str)`.
`CHANNEL_LAYOUT` is a tuple of seven `ChannelSpec`s in channel order.

### 6.7 Determinism

Preserved. Identical parquet input produces byte-identical output —
required for `force=False` no-op semantics on re-export and for
byte-comparison in unit tests.

## 7. Extractor Version And On-Disk Artifacts

### 7.1 Version bump

`DEFAULT_EXTRACTOR_VERSION` in `src/humpback/models/piano_roll_notes.py`
changes from `"v1"` to `"v2"`. The constant is re-exported from
`piano_roll_midi_export_service.py` and used by both notes-job creation and
midi-export-job resolution.

### 7.2 Artifact paths

Unchanged path scheme:

- `event_encoders/{job_id}/event_notes_v2.parquet`
- `exports/event_encoders/{job_id}/notes_v2.mid`

The path helpers `event_encoder_notes_path()` and
`event_encoder_midi_export_path()` already take `extractor_version` as a
parameter and need no code change.

### 7.3 Coexistence with v1 artifacts

v1 parquet and MIDI files on disk are left in place. The user deletes them
via the existing Piano Roll Notes job admin UI after this change ships. No
Alembic migration. No batch cleanup script.

Database rows in `piano_roll_notes_jobs` and `piano_roll_midi_exports` with
`extractor_version = "v1"` are similarly left in place until the user
deletes their parent Event Encoder job (or until the user runs whatever
cleanup affordance the UI offers).

## 8. Schema

### 8.1 Parquet

`NOTES_SCHEMA` in `piano_roll_notes_worker.py` is **unchanged**:

```
event_id        string  not null
event_token     int32   not null
partial_index   int32   not null   # range still {-1, 0..max_harmonic-1}
midi_pitch      uint8   not null
start_utc       float64 not null
start_offset_s  float64 not null
duration_s      float64 not null
velocity        uint8   not null
peak_magnitude  float32 not null
track_id        uint32  not null
```

The valid range of `partial_index` widens from `{-1, 0..7}` to
`{-1, 0..15}` but the column type already accommodates it.

### 8.2 Database

No schema changes. No Alembic migration. The
`piano_roll_notes_jobs.extractor_version` and
`piano_roll_midi_exports.extractor_version` columns hold the new `"v2"`
default for new rows.

## 9. Frontend

No code changes required. The Notes-mode piano roll renderer reads
`partial_index` for note coloring. With the new labeling distribution the
visible color mix shifts (more notes in the "harmonic 6+" and explicit
harmonic colors, fewer in the "unmatched" color); the renderer needs no
logic changes for this.

The MIDI download endpoint streams the new file unchanged — the
`Content-Disposition` filename derives from `extractor_version`, so a v2
notes job produces a download named `event_encoder_{job_id}_notes_v2.mid`.

A Playwright smoke test confirms the renderer survives the new partial
distribution.

## 10. Testing

### 10.1 Labeler unit tests (`tests/processing/test_piano_roll_tracker.py`)

- **Two-track 2× pair**: F0 at bin X, second track at bin X + bins_per_octave
  (exact 2:1 ratio), overlapping every frame → labels are `0` and `1`.
- **F0 sort fix**: a higher-bin track starts at frame 0; a lower-bin track
  enters at frame 5 holding a 0.5× ratio. Verify the lower-bin track is
  selected as F0 (`partial_index = 0`) and the higher-bin track gets
  `partial_index = 1` (2nd harmonic of the lower track).
- **Consume-on-overlap fix**: an F0 candidate at bin X overlaps a track at
  bin X + 1.7 * bins_per_octave (non-harmonic, ratio ≈ 3.25). The
  non-matching track is left unprocessed and on a later iteration becomes
  the F0 of its own (single-member) cluster.
- **Per-frame ratio (sweep)**: two parallel sweeping tracks whose
  bin-trajectories are e.g. `[20, 25, 30]` and `[56, 61, 66]` — medians
  alone do not produce a clean 2:1, but per-frame ratios do. Verify the
  upper track gets `partial_index = 1`.
- **`min_overlap_frames`**: a candidate that overlaps F0 for only two
  frames is rejected even with a clean 2:1 ratio.
- **`max_harmonic` reach**: a track at 10× F0 receives `partial_index = 9`.
- **`cents_tolerance` boundary**: a track at ratio 2.04 (~34¢ off 2x) is
  labeled; a track at ratio 2.10 (~84¢ off) is not.
- **Tolerance widening regression**: a track at ratio 2.04 *is* labeled
  under the new `cents_tolerance = 75` default but *would not* be under
  the old `= 50` default — explicit assertion against both values to lock
  the policy.
- **Determinism**: same inputs produce identical `partial_index`
  assignments across repeated calls.
- **`enabled=False`**: all tracks keep `partial_index = -1`.

### 10.2 MIDI synthesis unit tests (`tests/processing/test_midi_synthesis.py`)

- **Channel routing**: a fixture parquet with one note for each
  `partial_index ∈ {-1, 0, 1, 2, 3, 4, 5, 7, 12}` produces an SMF with
  eight tracks (tempo + 7 channel tracks). The note at `partial_index = 5`,
  `7`, and `12` all land on `CHANNEL_HARMONIC_HIGH`. The note at
  `partial_index = -1` lands on `CHANNEL_UNMATCHED`. Every other note
  lands on its dedicated channel.
- **Track headers**: each channel track begins with a `track_name`
  meta-event matching `CHANNEL_LAYOUT[i].name` and a `program_change`
  with the corresponding `program`.
- **Empty channels emit headers**: a parquet containing only
  `partial_index = 0` notes still emits all seven channel tracks; the
  empty ones have `track_name` + `program_change` + `end_of_track` and
  zero note events.
- **Channel 10 never written**: assert no message in any output track has
  `channel = 9`.
- **Determinism**: byte-identical output for repeated calls.
- **Existing invariants**: pitch clamp to `[0, 127]`, zero-duration note
  skipping, empty parquet → tempo track only, time origin at earliest
  `start_utc`, deterministic note ordering inside each channel — updated
  to assert on the per-channel track indices.

### 10.3 Worker / E2E

- `tests/workers/test_piano_roll_notes_worker.py`: existing fixtures use
  `extractor_version`-aware path helpers; update any literal `"v1"`
  references to the new default. Verify the worker writes
  `event_notes_v2.parquet`.
- `tests/workers/test_piano_roll_midi_export_worker.py`: verify the
  exported file path is `notes_v2.mid` under the v2 default.
- `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts`: the
  existing spec exercises the export round-trip. Update the expected
  filename suffix to `notes_v2.mid` and assert that the downloaded file
  parses as a Type-1 SMF with at least three tracks (tempo + multiple
  channel tracks). No deeper MIDI assertions in the Playwright layer.

### 10.4 Final gates (CLAUDE.md §6)

1. `uv run ruff format --check` on modified Python files
2. `uv run ruff check` on modified Python files
3. `uv run pyright` on modified Python files
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test e2e/sequence-models/event-encoder-piano-roll.spec.ts`

## 11. Documentation

- This spec at `docs/specs/2026-05-20-event-encoder-midi-channelized-design.md`.
- `DECISIONS.md` — append **ADR-067: Per-frame harmonic labeling and
  channelized MIDI export**. Captures: per-frame median-cents algorithm,
  `max_harmonic` 8 → 16, `cents_tolerance` 50 → 75, `min_overlap_frames=3`,
  F0 anchor sort fix, conditional consume-on-overlap, slim 7-channel MIDI
  layout, multi-track SMF Type 1 structure with per-channel GM
  `program_change` and `track_name`, extractor version `v1 → v2`, and the
  explicit no-backward-compatibility decision (user manually deletes v1
  artifacts via UI).
- `docs/agent-context/domains/sequence-models/README.md` — update the
  "Piano Roll Notes" and "MIDI export" subsections:
  - Strike "all partials stacked on MIDI channel 1" and describe the
    7-channel slim layout in its place.
  - Update artifact-path examples to use `v2`.
- `docs/agent-context/domains/sequence-models/invariants.md` — note that
  the harmonic prior labels tracks via per-frame ratios with median
  nearest-integer aggregation, that the F0 anchor is the lowest-bin track
  in each cluster, and that unmatched tracks remain eligible to anchor
  their own clusters.
- `docs/agent-context/current-state.md` — one-line update mentioning the
  channelized export and improved labeling.
- Add a "Superseded in part" header to both
  `docs/specs/2026-05-20-piano-roll-midi-notes-design.md` (§6.5) and
  `docs/specs/2026-05-20-piano-roll-midi-export-design.md` (§10) linking
  this spec.

## 12. Implementation Order (Informational)

1. Labeler rewrite in `piano_roll_tracker.py` plus unit tests.
2. Bump `DEFAULT_EXTRACTOR_VERSION` to `"v2"` and adjust any literal
   `"v1"` references in tests / fixtures.
3. MIDI synthesis channelization in `midi_synthesis.py` plus unit tests.
4. Manual round-trip smoke test against the
   `b759d8bf-0ecf-469a-b169-333b36c60906` event encoder job: delete the
   v1 notes/MIDI artifacts via UI → re-enqueue the notes job (now writes
   v2) → re-enqueue the MIDI export job → open the resulting
   `notes_v2.mid` in a DAW, confirm seven channel tracks, confirm the
   partial separation is musically plausible.
5. Documentation and ADR-067 updates.
6. Run final gates.

## 13. Open Questions

None at spec time. All design decisions resolved during brainstorming.

## 14. Future Work

- Multi-cluster (concurrent F0s) labeling for events with overlapping
  vocalizations.
- Frame-by-frame harmonic re-labeling so a track that changes identity
  mid-life can carry per-frame `partial_index` rather than a single
  label.
- MPE / pitch-bend support — requires a parquet `v3` with sub-semitone
  pitch contours.
- User-configurable channel layout via `params_json` on the export job
  row, if the slim layout proves limiting in practice.
- A "harmonic labeling quality" diagnostic column in the parquet
  (`median_abs_cents_deviation` per track) to make labeling confidence
  inspectable.
