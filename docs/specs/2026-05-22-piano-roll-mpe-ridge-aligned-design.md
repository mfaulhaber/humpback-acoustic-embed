# Piano Roll MPE & Ridge-Aligned Note Extractor — Design Spec

**Date:** 2026-05-22
**Status:** Approved
**Primary domain:** sequence-models
**Neighbor domains:** core-platform, frontend-shell
**Supersedes (in part):**
- [2026-05-20-piano-roll-midi-notes-design.md](2026-05-20-piano-roll-midi-notes-design.md) §6 (entire extraction algorithm) and §8.1 (Notes view rectangle rendering).
- [2026-05-20-event-encoder-midi-channelized-design.md](2026-05-20-event-encoder-midi-channelized-design.md) §5 (harmonic labeler) and §6 (slim 7-channel MIDI synthesis).
- [2026-05-20-piano-roll-midi-export-design.md](2026-05-20-piano-roll-midi-export-design.md) §10 (single-channel MIDI synthesis).

## 1. Goal

Replace the independent CQT peak-tracker in the Piano Roll Notes pipeline with
a **ridge-aware F0 + harmonics extractor** that produces:

1. **One MIDI note per coherent F0 contour** (no more semitone staircase
   artifact when a humpback vocalization glisses across pitch boundaries).
2. **Per-frame sub-semitone pitch trajectories** persisted in a parquet sidecar
   so downstream consumers (renderer, MIDI export, future analysis) read the
   same contour data.
3. An **MPE Standard MIDI File export** whose per-voice pitch bend reproduces
   those trajectories in any MPE-aware DAW.
4. **Curved-ribbon rendering** on the Piano Roll Notes view so the visual
   track matches the spectrogram ridges by construction.

The extractor and the existing Event Encoder STFT ridge tracker become a single
code path: the encoder persists per-frame ridge data, the notes worker reads
it, and the same trajectory feeds both the v3 Ridge mode display and the new
Notes ribbon. This closes the long-standing architectural seam where two DSP
paths computed pitch independently from the same audio.

## 2. Non-goals

- **Multi-F0 in a single event** (concurrent overlapping callers). The
  single-cluster-per-event simplification from ADR-067 is preserved. Tracks
  that don't fit the dominant F0 cluster fall through to their own
  singleton clusters as F0 anchors, same as v2.
- **Backward compatibility with v1/v2 artifacts on disk.** Same policy as
  ADR-067: bump `extractor_version`, leave legacy artifacts in place; the
  user manually deletes them via the existing job-admin UI.
- **Keeping the slim 7-channel partial-routed MIDI layout** as a parallel
  export option in this version. MPE replaces it. A non-expressive
  fallback for non-MPE DAWs is captured in §13 as future work.
- **DAW-side per-partial mute/solo workflow.** ADR-067's "mute the third
  harmonic in one click" affordance is structurally lost under MPE, where
  channels rotate by voice rather than by partial role. Partial identity is
  preserved through per-note program changes, CC 74, and text meta-events,
  but partial mute becomes a manual filter operation rather than a routing
  decision. Documented regression.
- **Vocalization-type-aware coloring.** Token color remains the source.
- **Frame-by-frame per-partial relabeling.** A note carries one
  `partial_index` for its lifetime.
- **Replacing the Sequence Models embedding pipeline or v3 descriptor block.**
  The descriptor block stays untouched; we add a per-frame ridge sidecar to
  the encoder, but `event_vectors.parquet` semantics are unchanged.
- **Real-time playback synthesis from contour data.** Browser audition still
  uses the co-exported FLAC clip from ADR-068.

## 3. Constraints inherited

- Idempotent worker keying on `(event_encoder_job_id, extractor_version)`
  (ADR-064 / ADR-066 / ADR-068).
- UTC end-to-end (CLAUDE.md §5).
- SQLite-safe worker claims via atomic status updates (core-platform
  invariant).
- Token ids and `Txx` labels remain job-local and k-local (sequence-models
  invariant).
- Atomic writes (`.tmp` then rename) for all new parquet artifacts;
  partial-failure cleanup of both notes and contour sidecars on exception.
- Standard MIDI File export is deterministic — identical inputs yield
  byte-identical bytes — required by the `force=false` no-op semantics in
  the existing export worker.

## 4. Architecture overview

```
Event Encoder job
    │
    ├──▶ event_vectors.parquet, event_tokens.parquet  (unchanged)
    │
    └──▶ event_ridges_{tokenizer_version}.parquet     (NEW per-event ridge sidecar)
              │
              │   read by notes worker (with audio-recompute fallback)
              ▼
        Piano Roll Notes worker (v3 branch)
          STFT ridge → subharmonic refine → coherent-contour notes
                     → CQT sibling harmonics → MPE-friendly contours
              │
              ├──▶ event_notes_v3.parquet              (one row per note + note_uid)
              └──▶ event_note_contours_v3.parquet      (NEW one row per frame per note)
              │
              ▼
        Piano Roll MIDI export worker (existing, ADR-068)
          MPE Lower Zone synthesis from contour sidecar
              │
              ├──▶ exports/event_encoders/{job_id}/notes_v3.mid    (MPE SMF)
              └──▶ exports/event_encoders/{job_id}/audio_v3.flac   (unchanged)
              │
              ▼
        Frontend Piano Roll page
          Curved ribbon renderer; Y-axis MIDI 12–120; MPE tooltip
```

### 4.1 Trigger paths

Unchanged from ADR-064:

1. Auto-after-encoding: completing an Event Encoder job auto-enqueues a notes
   job at the current default `extractor_version` (which becomes `"v3"`).
2. Backfill / re-run: `POST .../notes-jobs` with explicit `extractor_version`.
3. MIDI export: user-initiated `POST .../midi-exports` (ADR-068), worker reads
   the latest `complete` notes job's `extractor_version`.

### 4.2 Independence properties

- The notes worker never modifies Event Encoder outputs.
- The ridge sidecar is a pure-output additive artifact; encoder semantics are
  unchanged for any downstream that doesn't consume it.
- Re-running the notes worker at v3 against an encoder job that hasn't been
  re-encoded under the new sidecar-writing code falls back to recomputing
  ridges in-process. Same output, slower.

## 5. DSP pipeline

Per event, the v3 extractor runs five stages.

### 5.1 STFT ridge tracking (extracted module)

`_compute_ridge_path_result()` and its helpers move from
`src/humpback/sequence_models/event_encoder.py` into a new
`src/humpback/processing/ridge_path.py` with a stable public API:

```
compute_ridge_path(
    spectra, freqs, *, sample_rate, hop_length,
    min_frequency_hz=100.0, max_frequency_hz=6000.0,
    candidate_count=5, smoothness_penalty=8.0,
    peak_prominence_ratio=0.0,
) -> RidgePathResult
```

`RidgePathResult` keeps the existing fields (`log_frequencies`, `frame_times`,
`strengths`, `energy_ratios`, `total_frames`). Encoder code imports from the
new module; behavior is byte-identical.

`max_frequency_hz` stays at 6000 Hz in both the encoder (ADR-063 contract) and
the notes worker's in-process fallback. Matching the encoder's persisted
sidecar makes the fallback path produce ridges indistinguishable from the
sidecar (honors ADR-069 §10 "Output is identical"). Humpback F0s sit well
below this ceiling; harmonic siblings are caught by the CQT (its 264 bins at
36 bins/octave from `fmin=27.5` Hz top out near ~4.4 kHz) rather than by the
STFT ridge.

### 5.2 Subharmonic refinement

For each frame `t` with ridge frequency `f₀(t)`, check whether energy at
`f₀(t) / 2` is plausibly present in the CQT log-magnitude (the same CQT
already computed for harmonic search in §5.4). Two gates must pass:

1. The bin nearest `f₀(t)/2` must exceed the per-frame noise floor by
   `≥ k_sub · MAD` where `k_sub = 2.0`.
2. The bin's log magnitude must sit within
   `min_relative_log_magnitude = -2.5` natural-log units (~22 dB) of the
   current ridge bin. The constant-Q filter bank rejects an octave away
   by ~4 log units even for clean sines, so this gate keeps spectral
   leakage at `f₀/2` (which can otherwise pass the bare noise-floor
   test for pure tones with no real sub-fundamental energy) from
   demoting the F0. A real weak fundamental whose CQT magnitude lies
   within ~15 dB of the ridge still passes.

When evidence is present, divide `f₀(t)` by 2 and re-test. Iterate up to 3
times so a dominant ridge that's actually H4 of a very weak fundamental can be
demoted. The frame's "octave offset" (number of halvings applied) is recorded
in a transient diagnostic field.

Frame-level decisions are noisy, so smooth with a 5-frame majority vote of the
octave offset. The refined F0 then equals `f₀(t) / 2^smoothed_offset(t)`.
Smoothed offsets are persisted to the contour sidecar as a `subharmonic_octave`
column so the QA UI (future work) can render where the refinement engaged.

### 5.3 Coherent-contour note segmentation

The refined F0 contour is one continuous curve. It splits into one or more
notes only at:

- **Energy gaps.** `strengths[t]` below an event-relative floor (the
  amplitude_floor_percentile default 5 stays, but now applied per-frame
  rather than per-track-median) for at least `min_break_frames = 3` frames
  (~35 ms at hop=256, sr=22050).
- **Surviving octave jumps.** A `subharmonic_octave` step that persists past
  the 5-frame smoothing — represents a deliberate register shift within the
  event.

Otherwise the entire contour becomes a single F0 note spanning the event.
This is the structural fix for the staircase artifact: a sweep from C5 to E5
over 200 ms emits one note, not three short fragments at C5/D5/E5.

Each F0 note carries:

- `midi_pitch` = `round(median(midi_continuous(f₀(t))))` across the note's
  frames. Median is robust to sweep endpoints; outputs are clamped to the
  new MIDI 12–120 range.
- `cents_from_pitch` per frame in the contour sidecar, equal to
  `1200 · log₂(f₀(t) / f_nominal(midi_pitch))`. Clamped to ±9600 cents for
  MIDI bend safety even though the bend range is set to ±24 semitones
  (=±2400 cents); clamping headroom guards against outlier frames before
  the bend quantizer rounds.
- `start_utc`, `duration_s` from the note's frame bounds plus the event's
  `start_utc` and the audio loader's pad offset.

### 5.4 Harmonic sibling extraction

For each F0 note, at every frame, search the CQT for peaks within
`harmonic_cents_tolerance = 75¢` of `n · f₀(t)` for
`n ∈ {2, ..., max_harmonic}` with `max_harmonic = 16`. These are the values
that ADR-067 specified but that the worker's stale resolver defaults never
delivered (see §10.4).

A harmonic at multiplier `n` is "present" at frame `t` when:

- The nearest CQT peak to `n · f₀(t)` has magnitude above the noise floor.
- The peak's frequency deviates from `n · f₀(t)` by ≤ `harmonic_cents_tolerance`
  in cents.

Group consecutive frames where harmonic `n` is present into harmonic notes,
applying the same `min_break_frames = 3` rule as F0 segmentation. Each
harmonic note carries `partial_index = n - 1` (so H2 → `partial_index = 1`,
matching v2 semantics).

**Harmonic contour is derived, not measured.** The bend stream of a harmonic
in cents equals its parent F0's bend stream in cents, because
`1200 · log₂(n · f / n · f_nominal) = 1200 · log₂(f / f_nominal)`. The
measured CQT peak is used only to validate presence — never to drive the
bend. This enforces strict integer-harmonic relationships and avoids
per-frame drift between H1 and Hn.

Tracks that fail the harmonic search at every frame are not emitted; they
contributed to no note. (In v2, such tracks became `partial_index = -1`
notes; in v3, the architecture has no separate concept of "unmatched
tracks" — every detected ridge segment is either an F0 anchor or a derived
harmonic.)

### 5.5 Velocity calibration

Same job-level percentile mapping as v2, but applied to **per-note peak STFT
magnitude** (taken from `ridge_path.strengths` for F0 notes, and from the
matched CQT peak magnitude for harmonics). Velocity floor / ceiling and
percentiles unchanged from v2 (`floor_percentile = 5.0`,
`ceiling_percentile = 99.0`, `floor = 1`, `ceiling = 127`).

### 5.6 Compute budget

- STFT ridge: ~30 ms per event — amortized when reading the cached sidecar,
  paid once when falling back.
- Subharmonic refinement: ~5 ms per event.
- CQT (still required for harmonic search): ~30 ms per event.
- Harmonic siblings and velocity: ~10 ms per event.

End-to-end: ~45 ms per event with cached ridges, ~75 ms without. A 1672-event
job runs in ~75 s (cached) or ~125 s (uncached) single-threaded — comparable
to v2's 200 s on the same fixture.

## 6. Data model

### 6.1 New encoder artifact — per-event ridge sidecar

**Path:** `event_encoders/{job_id}/event_ridges_{tokenizer_version}.parquet`

**Schema** (one row per frame per event, sorted by `(event_id, frame_index)`):

| column | type | notes |
|---|---|---|
| `event_id` | string | FK to events |
| `frame_index` | uint32 | 0-based within event |
| `frame_time_offset_s` | float32 | seconds from event start |
| `log_frequency` | float32 | `log2(Hz)`; matches `RidgePathResult.log_frequencies` |
| `strength` | float32 | per-frame ridge magnitude |
| `energy_ratio` | float32 | ridge fraction of total frame energy |

Atomic write (`.tmp` then rename) by the encoder worker. Existing manifest
machinery covers the new artifact path — no new manifest schema field is
required; the path follows the established `{name}_{version}.parquet`
convention.

### 6.2 Notes parquet v3 — `event_notes_v3.parquet`

Same path scheme as v2:
`event_encoders/{job_id}/event_notes_v3.parquet`. One row per note (grain
unchanged), sorted by `(start_utc, midi_pitch)`.

| column | type | new in v3 | notes |
|---|---|---|---|
| `event_id` | string | | |
| `event_token` | int32 | | |
| `partial_index` | int32 | range `0..15` | `-1` no longer reachable |
| `midi_pitch` | uint8 | range widens to `12..120` | |
| `start_utc` | float64 | | |
| `start_offset_s` | float64 | | |
| `duration_s` | float64 | | |
| `velocity` | uint8 | | |
| `peak_magnitude` | float32 | | |
| `track_id` | uint32 | | unique within event |
| `note_uid` | string | **new** | deterministic UUID v5; primary key for joining to the contour sidecar |
| `f0_track_id` | uint32 | **new** | the `track_id` of the F0 note this row belongs to (self for F0 rows). Groups a voice and its harmonic siblings for downstream tooling |
| `contour_frame_count` | uint32 | **new** | denormalized count of contour rows in the sidecar; callers can decide whether the contour fetch is worth it |

The `partial_index` column's nullable range stays open to `-1` for forward
compatibility, but v3 never emits it.

`note_uid` is computed deterministically from
`(job_id, event_id, partial_index, track_id, start_utc_rounded_to_ms)` so
re-running the worker against unchanged inputs produces stable identifiers.

### 6.3 New contour sidecar — `event_note_contours_v3.parquet`

**Path:** `event_encoders/{job_id}/event_note_contours_v3.parquet`

**Schema** (one row per frame per note, sorted by `(note_uid, frame_index)`):

| column | type | notes |
|---|---|---|
| `note_uid` | string | FK to `event_notes_v3.note_uid` |
| `frame_index` | uint32 | 0-based within the note |
| `time_offset_s` | float32 | seconds from the note's `start_utc` |
| `cents_from_pitch` | float32 | cents from `midi_pitch`; clamped to ±9600 |
| `harmonic_strength` | float32 | per-frame magnitude evidence (ridge strength for F0, CQT peak for harmonics) |
| `subharmonic_octave` | uint8 | smoothed octave offset applied by §5.2 (`0` = no halving) |

### 6.4 Database

No schema changes. `piano_roll_notes_jobs.params_json` records the new
sidecar paths (`contours_path`, `ridges_path`) and a `n_contour_frames`
aggregate for observability. No Alembic migration.

### 6.5 Storage helpers

Add to `src/humpback/storage.py`:

- `event_encoder_ridges_path(storage_root, job_id, tokenizer_version)` →
  `event_encoders/{job_id}/event_ridges_{tokenizer_version}.parquet`
- `event_encoder_note_contours_path(storage_root, job_id, extractor_version)` →
  `event_encoders/{job_id}/event_note_contours_{extractor_version}.parquet`

`event_encoder_notes_path()` and `event_encoder_midi_export_path()` are
unchanged; their existing `extractor_version` arg accepts `"v3"` verbatim.

### 6.6 Versioning

`DEFAULT_EXTRACTOR_VERSION` in `src/humpback/models/piano_roll_notes.py`
moves from `"v2"` to `"v3"`. v1 and v2 sidecars remain readable; the
notes-version-resolver in the export worker picks the highest `complete`
notes-job version, which under `"v1" < "v2" < "v3"` string comparison
returns `"v3"` once any v3 notes job is complete.

## 7. API surface

All under `/sequence-models/event-encoders/{job_id}/...`:

| method | path | change |
|---|---|---|
| GET | `.../notes` | Response gains `note_uid`, `f0_track_id`, `contour_frame_count` per note row. Existing fields unchanged. |
| POST | `.../notes/contours` | **New.** Body: `{note_uids: [str], extractor_version?: str}`. Response: `{contours: {note_uid: [{time_offset_s, cents_from_pitch, harmonic_strength, subharmonic_octave}, ...]}}`. Cap of 2000 note_uids per request (413 above cap). POST (not GET) because the UUID-shaped uid list at viewport scale exceeds typical 8 KB HTTP-header limits when sent as repeated query parameters. |
| GET | `.../notes-status` | Unchanged. |
| POST | `.../notes-jobs` | Accepts `extractor_version: "v3"` in addition to legacy values. |
| GET | `.../midi-export` | Streams MPE SMF when `extractor_version = "v3"`. `Content-Disposition` filename suffix becomes `notes_v3.mid`. |
| POST | `.../midi-exports` | Unchanged. |
| GET | `.../midi-export-status` | Unchanged. |
| GET | `.../timeline` | Unchanged. |

Pydantic schemas in `src/humpback/schemas/piano_roll_notes.py` extend
`PianoRollNote` with three optional fields (so legacy v1/v2 responses still
deserialize cleanly). A new `PianoRollNoteContour` and
`PianoRollNoteContourResponse` model the contour endpoint.

## 8. MPE MIDI synthesis

Replaces `notes_table_to_midi_bytes()` in
`src/humpback/processing/midi_synthesis.py` when the input parquet carries the
v3 columns (`note_uid` present). When called with a v2-shape table, the
function falls back to the existing slim 7-channel layout — preserves
backward callable behavior for unmodified callers during the dual-version
window.

### 8.1 MPE configuration

**Lower Zone, full member span.** SMF Type 1, 480 PPQ, 120 BPM written once on
the tempo track. The first non-tempo track is the **MPE Master**:

1. RPN 6 with payload `15` declaring 15 member channels (`channels 2..16`,
   0-indexed `1..15`).
2. Per-member RPN 0/0 followed by Data Entry MSB = `24` → pitch-bend range
   ±24 semitones (~0.29 cent resolution at 14-bit bend, ample for whale
   excursion).
3. `track_name = "MPE Master"`.
4. Per-note `MetaMessage("text", text=f"p{partial_index}")` events
   immediately before each note's start tick, on the master track. Round-trip
   preserved; not auditioned.

### 8.2 Per-note channel allocator

A pure deterministic allocator:

```
allocate_channels(notes) -> {note_uid: channel}
  sort notes by (start_utc, note_uid)
  maintain free-pool of channels 1..15 (0-indexed)
  on note start: pop channel with longest idle interval
                 (tie-break by channel index ascending)
  if pool empty: steal channel of currently-sounding note with earliest
                 start_utc (FIFO voice steal); emit explicit note_off
                 for stolen note
  on note end: return channel to pool
```

Steal count per export is recorded in `params_json` for observability.

### 8.3 Partial identification

Since channels rotate by voice, partial identity is preserved through:

1. **Per-note `program_change`** at one tick before each `note_on` on the
   note's allocated channel. GM mapping mirrors ADR-067:
   - F0 → 0 (Acoustic Grand Piano)
   - H2 → 11 (Vibraphone)
   - H3 → 12 (Marimba)
   - H4 → 10 (Music Box)
   - H5 → 8 (Celesta)
   - H6..H16 → 88 (New Age Pad)
2. **Per-note CC 74** at the same tick: `partial_index * 16` clamped to
   `[0, 127]`. Conventional MPE timbre control; most DAWs map it to filter
   cutoff for visual distinction.
3. **Master-track text meta-event** per note (see §8.1).

### 8.4 Pitch-bend stream

For each note:

- Emit `pitch_bend` events for each contour frame whose `cents_from_pitch`
  differs from the last-emitted bend value by ≥ `bend_quantize_cents`
  (default `4¢` ≈ 14 bend units at ±24-semitone range). Flat segments do
  not generate redundant bend traffic.
- Bend value: `8192 + round(cents_from_pitch / 2400 · 8192)`. Standard
  14-bit; `8192` = center.
- First bend event fires at the note's `note_on` tick. Final bend event
  fires at the `note_off` tick.
- For harmonic notes (`partial_index ≥ 1`), the bend stream in cents is
  shared with the parent F0 (cents-conservation; §5.4). The implementation
  reuses the F0's already-computed bend events offset to the harmonic note's
  channel and timestamps.

### 8.5 SMF structure

```
Track 0:  tempo                  (set_tempo, end_of_track)
Track 1:  MPE Master             (RPN 6, per-member bend range, track_name,
                                  per-note text meta-events, end_of_track)
Track 2:  Voice 1 (channel 1)    (program_change, CC 74, note_on,
                                  pitch_bend stream, note_off, …)
Track 3:  Voice 2 (channel 2)
...
Track 16: Voice 15 (channel 15)
```

17 tracks total. Empty voice tracks still emit `track_name` and
`end_of_track` so the layout is portable across jobs.

### 8.6 Determinism

Allocator deterministic on `(start_utc, note_uid)`. Bend decimation
deterministic on `(cents_from_pitch, bend_quantize_cents)`. Identical parquet
+ contour sidecar → byte-identical SMF.

### 8.7 Backward compatibility

None. v1/v2 `.mid` files remain valid SMFs but are not regenerated; users
delete them via existing UI. The download `Content-Disposition` derives from
`extractor_version`, so v3 downloads carry the `notes_v3.mid` suffix.

## 9. Frontend

### 9.1 Data fetching

Add `usePianoRollNoteContours(jobId, noteUids, enabled)` to
`frontend/src/api/sequenceModels.ts`. The hook batches `note_uid` requests
against `POST .../notes/contours` and caches by `note_uid` via React Query so
panning doesn't re-fetch already-loaded contours.

The page computes `visibleNotes` (existing logic) and asks the contours hook
for `note_uids` that aren't yet cached. Cached note_uids are stable across
viewport changes because note_uid is deterministic.

### 9.2 Y-axis extension

`EventEncoderPianoRollPage.tsx` Y-axis range becomes MIDI 12–120. The 88-key
band (MIDI 21–108) retains normal black/white shading; extended bands
(12–20, 109–120) render with a desaturated background tint so out-of-piano
content is visible at a glance. Octave labels add C0, C9, and G9.

### 9.3 Ribbon rendering

A new `drawNoteRibbon(ctx, note, contour, transform, tokenColor, velocity)`
helper runs alongside the existing `drawNotesCanvas`:

1. Build polyline `[(time_offset_s + start_utc, midi_pitch + cents_from_pitch / 100), …]`
   over the note's frames.
2. Transform to canvas `(x, y)`.
3. Stroke a 2 px-wide ribbon path with rounded caps in full-alpha token color.
4. Fill a 4 px-tall ribbon body centered on the polyline in 30 %-alpha token
   color so the contour reads as a colored ridge rather than a thin line.
5. Velocity modulates stroke opacity (existing convention).

Hit-testing uses distance from cursor to the polyline (≤ 6 px in canvas
space).

### 9.4 Fallback ladder

Notes with no fetched contour render as the existing flat bar at `midi_pitch`.
They "hydrate" into ribbons when the contour fetch resolves. If
`/notes/contours` returns 500 for a note, the flat-bar rendering is the
terminal state for that session — a non-blocking toast fires once: "Some
notes are showing flat — contour fetch failed."

### 9.5 Tooltips and status

- Hover tooltip gains `Δpitch: ±N¢` summary (max absolute `cents_from_pitch`
  across the note's contour).
- "Download MIDI" link tooltip: "MPE Standard MIDI File (Type 1). Best in
  DAWs that support MIDI Polyphonic Expression — Logic Pro, Ableton Live
  11+, Cubase, Bitwig, Reaper-with-MPE."
- Below the file-size line in the export status panel, a "Format: MPE v3"
  string indicates the export version.
- `PianoRollNotesStatusPill` gains a "v3 available" badge when the encoder
  has a v2 sidecar but no v3. Clicking enqueues a v3 notes job.

### 9.6 Performance

Job `2679ab0d` rendered at the worst-case viewport (60 s span, 5–10k visible
notes × ~10 contour frames each ≈ 50–100k polyline points) stays within
Canvas-2D's redraw budget. The existing `rafBudgetMs` cap protects against
edge cases. A new `frontend/e2e/sequence-models/event-encoder-piano-roll-perf.spec.ts`
asserts ≥ 30 fps during a pan gesture on a synthetic-backend fixture.

## 10. Versioning, migration, and the stale-params bug

### 10.1 Default bump

`DEFAULT_EXTRACTOR_VERSION = "v3"` after Phase 4 lands.

### 10.2 Coexistence

No automated migration of v1 or v2 artifacts. The user deletes them via the
job-admin UI. The notes worker writes whatever version the job row names;
the export worker resolves the highest `complete` notes version per
`max("v1", "v2", "v3")` string comparison.

### 10.3 Encoder ridge sidecar backfill

New encoder runs always emit the ridge sidecar. Old encoder jobs without it
fall through to in-process ridge recomputation in the notes worker. No
backfill job is required; one can be added later if frequent.

### 10.4 Stale-params bug, retired by construction

A bug in `_resolve_params()` in `src/humpback/workers/piano_roll_notes_worker.py`
silently downgrades the v2 harmonic parameters (`max_harmonic = 8`,
`cents_tolerance = 50.0`) when a job has no explicit `params_json` payload —
contradicting the `HarmonicParams` dataclass defaults (`16` / `75.0`)
established in ADR-067. Every "v2" job created via the auto-enqueue hook
therefore ran the v2 algorithm structure with v1 thresholds.

The v3 architecture has no harmonic prior step: harmonics are derived
structurally from `n · f₀(t)` (§5.4), and the `HarmonicParams` dataclass is
retired. The buggy code path ceases to exist. No separate v2 patch is
shipped; users wanting correct v2 quality re-run jobs at v3.

The bug is mentioned here for the historical record and to explain why the
production data on job `2679ab0d` (66.2% `partial_index = 0`) skews
oddly relative to ADR-067's predicted 17–25% unmatched-rate.

### 10.5 ADR-069 and documentation

Append to `DECISIONS.md`:

> ADR-069: Ridge-aligned F0 + harmonics extractor and MPE Piano Roll MIDI export
> — STFT ridge as canonical F0 source, encoder ridge sidecar, coherent-contour
> note model, MPE Lower Zone export replacing slim 7-channel, MIDI 12–120
> pitch range, no backward compatibility for v1/v2, ribbon rendering,
> retirement of `HarmonicParams`.

Capsule + reference doc updates per the existing pattern (see §13).

## 11. Phase breakdown

Five phases, each individually shippable.

### Phase 1 — Encoder ridge sidecar (foundation)

Extract `compute_ridge_path()` into `humpback.processing.ridge_path`; extend
the Event Encoder worker to persist `event_ridges_{tokenizer_version}.parquet`.
No user-visible change.

### Phase 2 — v3 notes extractor (backend, dual-write)

Add `humpback.processing.note_extractor_v3` implementing §5.1–§5.5. Branch
the notes worker on `extractor_version`. `DEFAULT_EXTRACTOR_VERSION` stays
`"v2"`. v3 runs only when explicitly POST'd. Manual round-trip on job
`2679ab0d` confirms note-count reduction and reasonable partial distribution
before Phase 3 lands.

### Phase 3 — MPE synthesis (backend)

Rewrite `notes_table_to_midi_bytes()` to emit MPE when v3 input is detected.
v2 path preserved as fallback (regression guard). Manual smoke in Logic Pro.

### Phase 4 — Default bump and frontend ribbon rendering

`DEFAULT_EXTRACTOR_VERSION = "v3"`. Add `/notes/contours` endpoint and
`usePianoRollNoteContours`. Wire `drawNoteRibbon`. Extend Y-axis to MIDI
12–120. Add status-pill / tooltip updates. Playwright suite + perf test.

### Phase 5 — Documentation and ADR-069

ADR-069, capsule and reference doc updates, superseded headers on the three
prior specs, `current-state.md` line. Ships at the same time as Phase 4 in
practice.

## 12. Testing strategy

### 12.1 Pure-DSP unit tests

- `tests/processing/test_ridge_path.py`: empty input, constant tone, linear
  sweep, multi-candidate Viterbi preference, parquet round-trip.
- `tests/processing/test_note_extractor_v3.py`: synthetic sine sweep (one
  note, contour traces sweep within 5¢), harmonic stack (F0 + H2..H5
  emitted), subharmonic refinement on loud-H2 input, refinement smoothing,
  coherent-contour split on energy gap, split on register jump, short-event
  skip, velocity calibration over 20 dB range, determinism.
- `tests/processing/test_midi_synthesis.py` (extended): MPE Configuration
  Message at tick 0, per-member bend-range setup, deterministic allocator,
  voice-stealing on 17-simultaneous-notes fixture, per-note `program_change`
  / CC 74 / text meta-events on right tracks/ticks, harmonic bend stream in
  cents equals parent F0's, round-trip parses via `mido.MidiFile`, v2 input
  still produces slim 7-channel layout (regression guard).

### 12.2 Worker integration tests

- `tests/workers/test_piano_roll_notes_worker.py` (extended): v3 job on
  synthetic-audio encoder; idempotency on `(job_id, "v3")`; ridge sidecar
  fallback to in-process recompute; partial-failure cleanup of both
  parquets.
- `tests/workers/test_piano_roll_midi_export_worker.py` (extended): v3
  export reads v3 parquet + contour sidecar; missing contour sidecar fails
  with a specific error; atomic write + cleanup; `force=true` reset.
- `tests/sequence_models/test_event_encoder_worker_ridges.py` (new):
  encoder run writes `event_ridges_*.parquet` atomically; partial-failure
  cleanup; `event_id` set matches `event_vectors.parquet`.

### 12.3 API tests

- `tests/api/test_sequence_models_notes_contours.py` (new): contour fetch
  by `note_uid`, 404 on unknown uid, 422 on missing sidecar, 413 above
  request cap.
- `tests/api/test_sequence_models_midi_export.py` (extended): v3 export
  response carries `notes_v3.mid` filename, parses as MPE SMF (probe with
  `mido.MidiFile`).

### 12.4 Frontend Playwright

- `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts`
  (extended): Notes mode renders ribbons not bars; viewport-scoped contour
  fetch dedupes; fallback flat-bar + toast on 500; Y-axis MIDI 12–120 with
  desaturated extended-band tint; MPE Format tooltip on download link;
  "v3 available" badge click → notes-job status flips queued → complete.
- `frontend/e2e/sequence-models/event-encoder-piano-roll-perf.spec.ts`
  (new): synthetic fixture with 10k visible notes × 10 frames each; pan
  gesture maintains ≥ 30 fps.

### 12.5 Manual acceptance on job `2679ab0d`

Before Phase 4 merges:

1. Run Phase 1 + 2 on the job. Verify the ridge sidecar exists, the v3
   notes parquet exists, `n_notes` falls in the 30–50k range (down from
   80,561), `partial_index` distribution shows F0 below ~40 % with a clean
   monotonic harmonic tail.
2. Run Phase 3 export. Open `notes_v3.mid` in Logic Pro. MPE indicator
   illuminates. A known sweep event auditions with audible pitch bend.
   CC 74 visible on the per-voice automation lane.
3. Run Phase 4 frontend. The ribbons follow the spectrogram ridges
   visually — the staircase artifact is gone, sweeps render as single
   colored ribbons. Hover shows `Δpitch: ±N¢`. The "Export view" round-trip
   produces a v3 bundle that auditions correctly.

### 12.6 Final gates per CLAUDE.md §6

Run on every phase merge:

1. `uv run ruff format --check` on modified Python files.
2. `uv run ruff check` on modified Python files.
3. `uv run pyright` on modified Python files; full run when schemas or DSP
   modules change.
4. `uv run pytest tests/`.
5. `cd frontend && npx tsc --noEmit` when frontend files changed.
6. `cd frontend && npx playwright test e2e/sequence-models/event-encoder-piano-roll*`
   when UI files changed.

## 13. Future work

- **Compact (non-expressive) MIDI export** for non-MPE DAWs. The
  slim-7-channel layout could return as a parallel export mode selected
  per request — same parquet inputs, simpler synthesizer.
- **Multi-F0 in a single event** (concurrent overlapping callers). Requires
  a graph-cluster F0 model and per-cluster contour partitioning.
- **Ridge sidecar backfill job** so existing encoder jobs can be promoted
  to cached-ridge status without re-encoding.
- **QA visualization for subharmonic refinement.** A dev-only overlay
  showing where `subharmonic_octave > 0` engaged, for tuning the refinement
  thresholds.
- **Frame-by-frame partial relabeling.** A note that changes harmonic
  identity mid-life would carry per-frame `partial_index` rather than a
  single value.
- **Continuous note-bend mode in the renderer.** A toggle to play the bend
  stream as a `requestAnimationFrame`-driven Web Audio synth in-browser,
  removing the FLAC dependency for short previews.

## 14. Open questions

None at spec time. All design decisions were resolved during the brainstorm
session captured in `docs/plans/2026-05-22-piano-roll-mpe-ridge-aligned.md`
preconditions.
