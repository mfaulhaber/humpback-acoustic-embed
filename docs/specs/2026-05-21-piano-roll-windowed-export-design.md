# Piano Roll Windowed Export — Design

**Date:** 2026-05-21
**Status:** Approved (brainstorming)
**Primary domain:** sequence-models
**Neighbor domains:** core-platform, signal-timeline, frontend-shell

## Summary

Modify the Event Encoder Piano Roll's MIDI export so that it exports only the
currently viewed time window in the player, and additionally produces a `.flac`
audio clip of the same window. The MIDI's time origin is the window start so
that when both files are imported into a DAW (e.g. Logic Pro) at the same
project position with project tempo matching the MIDI's tempo, the notes align
with the audio.

A single export per `(event_encoder_job_id, extractor_version)` is retained
("Approach A — Bundled Piano Roll Export"); re-exporting overwrites both the
MIDI file and the FLAC file atomically.

## Motivation

The current MIDI export ships the entire Event Encoder job's notes as one file
with auto-shifted time origin. Users working in DAWs want to study a specific
zoomed/panned region of a job alongside the actual audio. Re-exporting the full
job for every region is wasteful, and there is no companion audio export.

This change makes the export workflow:

1. Visible — what you see in the player is what you get.
2. DAW-ready — the audio file and the MIDI file align when imported together
   at the same project position.
3. Atomic — the MIDI and FLAC always agree (one window, two files).

## Scope

In scope:

- New windowed-export behavior on the existing
  `POST /event-encoders/{job_id}/midi-exports` endpoint.
- New companion `.flac` artifact produced by the same worker run.
- New `GET /event-encoders/{job_id}/audio-export` endpoint.
- Schema change to `piano_roll_midi_exports` (new fields, NOT NULL).
- Alembic migration that drops legacy rows and their `.mid` files.
- Frontend updates: pass viewer's `timeRange` into the export request; render
  a download panel with separate "Download MIDI" and "Download audio (FLAC)"
  controls; show the exported window in the UI.
- Backend safety cap on window size.

Out of scope:

- Window selection separate from the viewer's `timeRange` (e.g. drag-select).
- Bundling MIDI + FLAC as a single download archive (zip/tar).
- SMPTE-timecode MIDI for tempo-independent alignment.
- Pitch-bend / MPE export of frequency contours within a note.
- Multi-format audio export (only FLAC).
- Multi-window history (only one rolling artifact per job/version).
- Auto-enqueue at Event Encoder completion (the existing auto-enqueue of the
  Piano Roll Notes sidecar still runs; the MIDI export remains user-initiated).
- DAW workflow hint or import help in the UI.

## User-facing behavior

- The Piano Roll page exposes a single export action. When no export exists for
  the job/version, label is **"Export view"**. When an export exists, label is
  **"Re-export view"**.
- Clicking the button captures the viewer's current `timeRange`
  (`{ start, end }` in UTC seconds), posts the create request, and shows a
  progress indicator (existing `queued → running` polling unchanged).
- On `complete`, the page renders a compact panel:
  - Last-exported window text:
    `Exported window: <start UTC>  →  <end UTC>  (<duration> s)`.
  - **Download MIDI** button with size.
  - **Download audio (FLAC)** button with size.
- When the viewer's current `timeRange` differs from the persisted exported
  window beyond a small tolerance (~50 ms), the **"Re-export view"** button
  uses its emphasized (default) styling. When it matches, the button is
  non-emphasized — clicking still works.
- If the requested window exceeds a 30-minute soft cap, the button is disabled
  with a tooltip; if the user bypasses (e.g. via stale state), the API responds
  with a clear 400 surfaced as a toast.
- Empty windows (zero notes in range) still succeed — a valid MIDI with a
  tempo event and no notes is produced; the FLAC contains the corresponding
  audio (possibly silent if the source has no coverage).
- Failures surface the error message and offer retry, matching today's
  failure UI.

## Backend architecture

### Schema (Alembic migration)

Extend the `piano_roll_midi_exports` table with new NOT NULL columns. Approach:
delete all pre-existing rows (and their on-disk `.mid` files) in the migration,
then add the new columns without server defaults. The pre-existing artifacts
correspond to non-windowed exports that the UI will no longer surface, so a
clean break is preferable to keeping an obsolete cache.

New columns:

| Column              | Type    | NOT NULL | Description                                    |
|---------------------|---------|----------|------------------------------------------------|
| `window_start_utc`  | REAL    | yes      | UTC epoch seconds, inclusive lower bound       |
| `window_end_utc`    | REAL    | yes      | UTC epoch seconds, exclusive upper bound       |
| `audio_path`        | TEXT    | yes      | Absolute storage path to the FLAC file         |
| `audio_size_bytes`  | INTEGER | yes      | FLAC file size on disk                         |
| `audio_sample_rate` | INTEGER | yes      | Stored sample rate (currently always 32000)    |
| `audio_duration_s`  | REAL    | yes      | Actual concatenated audio duration             |

Use `op.batch_alter_table()` for SQLite compatibility. Add a CHECK constraint
or row-level validation that `window_end_utc > window_start_utc`.

Existing columns are preserved.

### Storage paths

Under `<storage_root>/exports/event_encoders/{job_id}/`:

- `notes_{extractor_version}.mid` (existing).
- `audio_{extractor_version}.flac` (new sibling).

Both files written via a `.tmp` suffix and atomically renamed. The worker
constructs both temp files first and only renames them when both succeed.

A new helper in `src/humpback/storage.py`:

- `event_encoder_audio_export_path(storage_root, job_id, extractor_version)`
  returns `<storage_root>/exports/event_encoders/{job_id}/audio_{version}.flac`.

The existing `event_encoder_midi_export_path()` helper is unchanged.

### Model & schema

- `src/humpback/models/piano_roll_midi_export.py` gains the new columns as
  required (non-Optional) `Mapped[...]` fields.
- `src/humpback/schemas/piano_roll_midi_export.py`:
  - `PianoRollMidiExportRead` gains `window_start_utc`, `window_end_utc`,
    `audio_path`, `audio_size_bytes`, `audio_sample_rate`, `audio_duration_s`.
  - `PianoRollMidiExportCreateRequest` gains required `window_start_utc:
    float`, `window_end_utc: float`. Existing fields stay.

### API

- `POST /event-encoders/{job_id}/midi-exports`:
  - Validate `window_end_utc > window_start_utc`.
  - Validate `(window_end_utc - window_start_utc) <= 1800.0` (30 minutes).
  - Validate the window overlaps the job's data range (resolved via the event
    encoder → event segmentation → detection job chain).
  - On invalid inputs, return 400 with a descriptive `detail`.
  - Behavior: if no row exists, insert; if a row exists with the same
    requested window AND `status == complete` AND `force=false`, return the
    existing row unchanged (cache hit); otherwise re-queue (reset status to
    `queued`, clear `started_at`/`finished_at`/`error_message`/`midi_path`/
    `audio_path`/sizes, update `window_*`).
- `GET /event-encoders/{job_id}/midi-export-status`: unchanged endpoint shape;
  the embedded `PianoRollMidiExportRead` now includes the new fields.
- `GET /event-encoders/{job_id}/midi-export`: unchanged; streams the `.mid`.
- `GET /event-encoders/{job_id}/audio-export`: new endpoint. Streams the
  `.flac` with `Content-Type: audio/flac` and an `attachment` Content-Disposition
  header. Returns 404 with a clear `detail` when no completed export exists
  for the job's default version.

### Worker

`src/humpback/workers/piano_roll_midi_export_worker.py` is extended:

1. Claim the row; transition `queued → running`.
2. Load the Piano Roll Notes parquet for `(event_encoder_job_id,
   extractor_version)`.
3. Filter notes to those overlapping `[window_start_utc, window_end_utc)`:
   - Keep a note if `note.start_utc < window_end_utc` AND
     `(note.start_utc + note.duration_s) > window_start_utc`.
   - Clip the kept note's effective `start_utc` to
     `max(note.start_utc, window_start_utc)`.
   - Clip the kept note's effective end to
     `min(note.start_utc + note.duration_s, window_end_utc)`.
   - Drop notes whose clipped duration is < 1 ms.
4. Synthesize MIDI bytes via `notes_table_to_midi_bytes()`. Pass a
   `time_origin_utc=window_start_utc` argument so that a note at exactly
   `window_start_utc` lands at tick 0 (instead of the existing
   earliest-note-shift behavior).
5. Resolve the source audio:
   - Walk the event encoder job → event segmentation job → detection job →
     hydrophone identifier and job timestamp bounds.
   - Call `resolve_timeline_audio(hydrophone_id, local_cache_path,
     job_start_timestamp, job_end_timestamp, start_sec=window_start_utc,
     duration_sec=(window_end_utc - window_start_utc), target_sr=32000)`.
     Gap regions are silence-filled by the helper.
6. Write artifacts atomically:
   - Write the MIDI bytes to `notes_{version}.mid.tmp`.
   - Write the FLAC via the new `write_flac_clip()` helper (see DSP section)
     to `audio_{version}.flac.tmp`.
   - On success of both writes, rename both temp files to their final paths,
     then update the DB row in a single transaction with status `complete`,
     populated `midi_path`, `audio_path`, `n_notes`, `n_bytes`,
     `audio_size_bytes`, `audio_sample_rate=32000`, and `audio_duration_s =
     samples.size / 32000`.
   - On any write failure, delete the temp files, leave the previous final
     artifacts and the DB row untouched (status transitions to `failed` with
     a captured error_message). The previous successful export remains
     downloadable.

### Service

`src/humpback/services/piano_roll_midi_export_service.py` is updated so that
the create/upsert path:

- Stores the requested window on the row at insert / re-queue time.
- Considers a request a "cache hit" only when the request window matches the
  persisted window (within 1 ms tolerance) AND status is `complete` AND
  `force=false`. Otherwise the row is reset to `queued` with the new window
  bounds.

### DSP / encoding helpers

In `src/humpback/processing/audio_encoding.py` (the module already housing
`encode_wav`/`encode_mp3` from the timeline-export work):

- Add `write_flac_clip(samples: np.ndarray, sr: int, path: Path) -> None`:
  - Writes float32 mono samples as 16-bit PCM FLAC via `soundfile`.
  - No normalization. (Loudness must match what the player rendered.)
  - Creates the parent directory; uses an atomic write via temp suffix +
    rename (mirror the MIDI write pattern).
  - Asserts the input is 1-D and finite.

`src/humpback/processing/midi_synthesis.py`:

- Add `time_origin_utc: float | None = None` parameter to
  `notes_table_to_midi_bytes()`. When provided, use it as the tick-0 anchor
  instead of `min(note.start_utc)`. The 120 BPM tempo, 480 PPQ, and 7-channel
  layout remain unchanged (per ADR-066, ADR-067).

## Frontend architecture

`frontend/src/components/sequence-models/MidiExportButton.tsx` is extended (not
replaced). It now:

- Receives `windowStartUtc: number` and `windowEndUtc: number` props from
  `EventEncoderPianoRollPage.tsx`, threaded from the page's existing
  `timeRange` state.
- Computes whether the current window matches the persisted exported window
  within a 50 ms tolerance.
- Renders one of three states:
  - **No export yet** → "Export view" button enabled when notes status is
    `complete` and the window is valid (>0, ≤30 min).
  - **Export in progress** (`queued`/`running`) → spinner + "Exporting MIDI
    and audio…".
  - **Export complete** → exported-window text + two download buttons
    ("Download MIDI", "Download audio (FLAC)") + "Re-export view" action.
    "Re-export view" is emphasized when the current window differs from the
    persisted window, non-emphasized when they match.
- Disables itself with a tooltip when the requested window exceeds the
  30-minute cap.

The mutation function (`frontend/src/api/sequenceModels.ts` or equivalent
hook file) is updated so its POST body carries `window_start_utc` and
`window_end_utc`.

Generated TypeScript types from the Pydantic schemas pick up the new fields
through the existing codegen step (`npm run gen:api` or the project's
equivalent — verify and use that).

## Error handling

- 400 on `window_end_utc <= window_start_utc`, window > 30 min, or window
  outside job range — toast in the UI.
- 404 on `GET …/audio-export` when no completed export exists.
- Worker failures roll back gracefully: prior `.mid` and `.flac` (if present)
  remain downloadable; the row's status becomes `failed` with an error
  message; the UI shows the message and a retry control.
- `resolve_timeline_audio()` silence-fills coverage gaps automatically; no
  special handling required at the worker.

## Testing strategy

Backend:

- Migration test: legacy fixture row + its `.mid` are removed by the
  migration; the new columns exist with the expected types and NOT NULL.
- Service unit tests: window-match cache hit, window-mismatch re-queue, force
  flag behavior.
- Worker unit tests: note filter + clip, sub-ms drop, empty-window valid SMF
  + correct-duration FLAC, time-origin correctness against `mido`-parsed
  output, FLAC sample rate / duration / non-normalized sample correctness,
  atomic rollback on FLAC write failure.
- API tests: window validation 400s, the success path for both create and
  GET `…/audio-export`, the absent-export 404 case, the schema round-trip
  including new fields.

Frontend:

- Vitest unit/component tests for `MidiExportButton` state transitions and
  payload computation (window pulled from props, tolerance comparison,
  disabled state for over-cap windows).
- Playwright e2e in
  `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts`:
  pan/zoom the viewer, click Export, wait for `complete`, assert both
  download links appear and trigger downloads; re-export with a different
  window and assert sizes / exported-window text update; assert disabled
  state for an over-cap window.

Manual verification (one-off, recorded in the PR description, not in CI):

- Drop an exported `.mid` + `.flac` pair into Logic Pro at the same project
  position with project tempo set to (or accepted as) 120 BPM; visually
  confirm notes align with the audio's waveform.

## Open follow-ups (not in this change)

- Add a UI affordance to re-anchor the viewer to the persisted window so the
  user can quickly compare the export against their current view.
- Optionally promote `write_flac_clip()` to a generic export utility if other
  features need lossless audio output.
- Revisit the 30-minute cap when there's user demand for longer windows.
