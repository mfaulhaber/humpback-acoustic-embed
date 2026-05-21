# Piano Roll MIDI Notes — Design Spec

**Status:** Approved (superseded in part)
**Date:** 2026-05-20
**Primary domain:** sequence-models

> §6.5 (harmonic labeling) is superseded by
> [2026-05-20-event-encoder-midi-channelized-design.md](2026-05-20-event-encoder-midi-channelized-design.md)
> §5. The v2 labeler replaces the median-bin ratio metric with per-frame
> ratio matching, bumps `max_harmonic` 8 → 16, widens `cents_tolerance`
> 50 → 75, adds `min_overlap_frames = 3`, and leaves tracks that fail
> the harmonic check eligible to anchor their own clusters.
**Neighbor domains:** core-platform, signal-timeline, frontend-shell

## 1. Goal

Render the Event Encoder timeline as a traditional MIDI-style piano roll. Each
whale event is decomposed into one or more semitone-quantized notes (F0 plus
visible harmonics) drawn on a log-frequency, piano-keyed Y-axis. Notes carry
`(midi_pitch, start_time, duration, velocity)` and inherit the event's existing
token color. Per-event note data is persisted in a format that supports later
MIDI/MPE export with no recomputation.

## 2. Non-goals

- Displaying pitch bends or micro-pitch deviations within a note. Notes are
  flat semitones.
- Vocalization-type-based coloring. Token color from the existing Event Encoder
  palette is the source.
- Building the MIDI/MPE export UI or endpoint. The on-disk schema is designed
  so export later is a thin read-and-convert.
- Re-segmenting events or changing event identity. Event boundaries from
  segmentation (ADR-062) are inputs.
- Extending the pitch range beyond the 88-key piano range. A future ADR
  captures the option.
- Replacing the Sequence Models embedding pipeline (ADR-056, ADR-057) or the v3
  descriptor block (ADR-063). This spec adds an independent sidecar; the
  descriptor block is untouched.

## 3. Constraints inherited

- Idempotent worker keying on `(event_encoder_job_id, extractor_version)`,
  mirroring ADR-056-style canonical keys.
- UTC end-to-end (CLAUDE.md §5).
- SQLite-safe worker claims via atomic status updates (core-platform invariant).
- Backup-before-migrate (CLAUDE.md §4) before applying the new Alembic table.
- Token ids and `Txx` labels remain job-local and k-local (sequence-models
  invariant).
- Audio resolution and chunk timing semantics defer to signal-timeline.

## 4. Architecture

A new dispatcher branch, the Piano Roll Notes worker, runs independently of and
idempotently after the Event Encoder job. It reads the Event Encoder's
descriptor block and the source audio, and writes a per-job parquet sidecar.
The UI surfaces both the Event Encoder status and the Piano Roll Notes status
on the same job pages.

```
Event Encoder job  ──▶  descriptor block + token assignments  (unchanged)
                                │
                                ▼
                  ┌─ Piano Roll Notes worker ─┐
                  │  CQT + peak-pick +        │
                  │  tracking + harmonic      │
                  │  prior + MIDI quantize    │
                  └─────────────┬─────────────┘
                                ▼
       event_encoders/{job_id}/event_notes_v1.parquet
                  + piano_roll_notes_jobs SQL row
                                │
                                ▼
            API: timeline (extended with notes_status),
                 notes (parquet rows for viewport),
                 notes-jobs (enqueue / re-run),
                 notes-status (latest job row)
                                │
                                ▼
            Frontend: piano roll renders notes by default when
                      sidecar ready; status chip on job card;
                      "Generate notes" backfill action.
```

### 4.1 Trigger paths

1. Auto-after-encoding. When an Event Encoder job transitions to `completed`,
   enqueue a notes job for it at the current `extractor_version`.
2. Backfill or re-run. A `POST .../notes-jobs` endpoint enqueues for already-
   completed Event Encoder jobs. Same `(event_encoder_job_id,
   extractor_version)` is a no-op if `completed`, refuses with 409 if
   `running`, replaces if `failed` or `canceled`.

### 4.2 Independence from Event Encoder

The notes worker does not modify Event Encoder outputs. It only reads them and
the source audio. Failure or re-run of the notes worker never invalidates the
descriptor block. Event Encoder jobs without a notes sidecar render with the
existing rectangle modes.

## 5. Data model

### 5.1 New SQL table: `piano_roll_notes_jobs`

| column | type | notes |
|---|---|---|
| `id` | str (uuid) PK | |
| `event_encoder_job_id` | str FK, indexed | |
| `extractor_version` | str | `"v1"` for this spec |
| `status` | str enum | `queued` / `running` / `completed` / `failed` / `canceled` |
| `started_utc` | float, nullable | epoch seconds |
| `finished_utc` | float, nullable | epoch seconds |
| `error` | text, nullable | first ~2 KB of failure message |
| `notes_path` | str, nullable | absolute path to parquet sidecar on success |
| `n_events` | int, nullable | events scanned |
| `n_notes` | int, nullable | total emitted notes |
| `compute_seconds` | float, nullable | wallclock cost |
| `params_json` | text | serialized extractor params for reproducibility |
| `created_utc` | float | |
| `updated_utc` | float | |

Unique constraint on `(event_encoder_job_id, extractor_version)`. Alembic
migration uses `op.batch_alter_table` per CLAUDE.md §4. Backup-before-migrate
is the first acceptance criterion of the migration task.

### 5.2 Parquet sidecar: `event_notes_v1.parquet`

Path: `event_encoders/{job_id}/event_notes_v1.parquet`. One row per note (not
per event). Sorted by `(start_utc, midi_pitch)`.

| column | type | notes |
|---|---|---|
| `event_id` | str | FK to event |
| `event_token` | int | duplicated for fast color lookup |
| `partial_index` | int | `0` = F0, `1` = 2×F0, …; `-1` if no harmonic prior match |
| `midi_pitch` | uint8 | `21`–`108` (A0–C8) |
| `start_utc` | float | absolute UTC epoch seconds |
| `start_offset_s` | float | seconds from event start (redundant, cheap) |
| `duration_s` | float | |
| `velocity` | uint8 | `0`–`127`, job-level dynamic range |
| `peak_magnitude` | float | raw CQT log-magnitude at median bin |
| `track_id` | uint32 | unique within event |

### 5.3 Versioning

`extractor_version = "v1"` for this spec. Future tracker or range changes bump
to `"v2"`, etc. The notes-jobs table allows multiple completed runs at
different versions for the same encoder job; the UI serves the latest
`completed` row (newest `finished_utc`).

## 6. Extraction algorithm

For each event in the source Event Encoder job:

### 6.1 Audio slice

- Resolve the source audio path via existing detection/event lineage; no copy.
- Read samples from `start_utc − pad` to `end_utc + pad` with `pad = 50 ms`.
- Resample to **22050 Hz mono** (downmix multi-channel).
- Skip events shorter than 30 ms; emit zero notes.

### 6.2 CQT

Compute magnitude CQT via `librosa.cqt` with:

- `sr = 22050`
- `hop_length = 256` (≈11.6 ms, ~86 fps)
- `fmin = 27.5` Hz (A0)
- `n_bins = 264` (88 semitones × 3 bins/semitone)
- `bins_per_octave = 36`
- `filter_scale = 1.0`

Log-magnitude: `L = log(C + 1e-6)`. Per-frame noise floor:
`noise[t] = median(L[:, t]) + k_noise · MAD(L[:, t])`, `k_noise = 3.0`.

### 6.3 Peak-pick per frame

Per frame `t`:
1. Find local maxima along the bin axis (3-bin neighborhood).
2. Drop maxima below `noise[t]`.
3. Keep top-K maxima by magnitude, `K = 8`.

### 6.4 Cross-frame tracking

Greedy nearest-neighbor:

- For each new frame's peaks, match to the nearest open track within ±3 bins
  (±1 semitone).
- Unmatched peaks open a new track. Unmatched tracks close after 2 missed
  frames.
- After all frames, drop tracks shorter than 50 ms (~4 frames), and drop
  tracks whose median magnitude is below an event-relative amplitude floor.

### 6.5 Harmonic prior (labeling only)

- Sort surviving tracks by `start_frame`, then by ascending median bin.
- The lowest-bin track that overlaps a candidate window is the candidate F0.
- For each other overlapping track, if its median frequency is within ±50
  cents of an integer multiple (`2×`…`8×`) of the F0 track's median frequency,
  set `partial_index` to the harmonic number minus one.
- Tracks that do not match any integer multiple keep `partial_index = -1`.
- The prior does not filter. It only labels. Inharmonic energy stays visible.
- `harmonic_prior_enabled` is a parameter (default `true`).

### 6.6 MIDI quantization and velocity

- `midi_pitch = 21 + round(median_bin / 3)`. Clamp to `[21, 108]`.
- `start_utc = event_start_utc + (start_frame · hop / sr)`.
- `duration_s = (end_frame − start_frame + 1) · hop / sr`.
- Velocity is computed in a second pass: per-track `median_log_magnitude` is
  linearly mapped from job-level [p5, p99] log-magnitude percentiles into
  `[1, 127]`. Job-level percentiles are accumulated in a first cheap pass
  over events (per-frame max log-mag) so cross-event loudness is comparable
  in the UI and in any future export.

### 6.7 Reproducibility

All defaults are written into `params_json` per job:

```
cqt:        {sr, hop_length, fmin, n_bins, bins_per_octave, filter_scale}
peak:       {k_noise, top_k}
tracker:    {bin_tolerance, miss_tolerance_frames, min_duration_s}
harmonic:   {enabled, max_harmonic, cents_tolerance}
midi:       {min_pitch=21, max_pitch=108}
velocity:   {job_percentiles=[5, 99], floor=1, ceiling=127}
```

The tracker is deterministic for a given input.

### 6.8 Compute budget

Per event: ~30 ms CQT plus ~5 ms tracking on CPU. A job with 10k events ≈ 6
minutes single-threaded. Parallelization by event group follows existing
worker patterns.

## 7. API surface

All under `/sequence-models/event-encoders/{job_id}/...`:

| method | path | purpose |
|---|---|---|
| `GET` | `.../notes-status` | Latest `piano_roll_notes_jobs` row, or `{status: "absent"}`. |
| `POST` | `.../notes-jobs` | Enqueue or re-enqueue. Body: `{extractor_version?: str, params?: {...}}`. 409 on `running`. |
| `GET` | `.../notes` | Notes parquet streamed as JSON. Query: `start_utc`, `end_utc`, `event_ids?`. |
| `GET` | `.../timeline` (existing) | **Extended** payload includes `notes_status`. |

A global `notes-jobs` index endpoint is not added in this spec.

## 8. Frontend behavior

### 8.1 Piano roll page

`EventEncoderPianoRollPage.tsx` keeps its current modes (`Ridge`, `Median F0`,
`Peak Frequency`) and adds a new `Notes` mode that becomes the default when
`notes_status === "completed"`.

- On mount, render the existing rectangle view immediately from
  `descriptor_values`. No TTFB regression.
- In parallel, fetch `.../notes` for the viewport. When it resolves and notes
  are available, swap to Notes mode.
- Y-axis switches to log-frequency with semitone gridlines and labeled octaves
  (C0…C8). Black-key shading. The existing linear-Hz scale remains used for
  the rectangle modes only.
- Each note draws as a horizontal bar:
  - Y from `midi_pitch`.
  - X from `start_offset_s` and `duration_s`.
  - Fill from `tokenColor(event_token)`.
  - Stroke opacity proportional to `velocity / 127`.
- Tooltip on hover: `midi_pitch` (with note name), `velocity`, `duration_s`,
  `event_id`, `token`, `partial_index` (`F0`, `2x F0`, … or `–`).
- Toolbar:
  - View dropdown: `Notes (default)` / `Ridge` / `Median F0` / `Peak
    Frequency`. `Notes` is disabled when `notes_status !== "completed"`, with
    a tooltip explaining why.
  - Status chip showing notes-job state. On `failed`, click reveals the error
    and a `Re-run` action.

### 8.2 Job card and job list

`EventEncoderJobCard` shows two pills: `Encoding: <state>` and `Notes:
<state>`. Clicking the Notes pill navigates to the piano roll route.

### 8.3 Spectrogram strip

`EventEncoderSpectrogramStrip.tsx` is unchanged. It still syncs to the same
time window. No semitone grid added to the strip.

### 8.4 Backfill UX

From the Event Encoder job page, `Generate notes` is visible when
`notes_status` is `absent` or `failed`. Disabled and replaced with a progress
indicator while `queued` or `running`.

## 9. Error handling

- Audio file missing for an event → emit zero notes for that event, capture
  per-event failures in an in-memory aggregate, store a truncated summary in
  `piano_roll_notes_jobs.error`. Job still completes if at least one event
  succeeded.
- All events failed → job status `failed`; UI surfaces the error.
- Frontend `.../notes` fetch fails → keep the rectangle render with a
  non-blocking toast.

## 10. Testing strategy

### 10.1 Backend unit tests

- CQT extraction on synthetic sinusoids at known MIDI pitches and harmonic
  stacks; assert correct bins and `partial_index`.
- Tracker continuity gaps above and below the miss-tolerance.
- Harmonic prior labeling at exact, near-miss (±50 cents), and out-of-range
  ratios.
- Velocity mapping for events 20 dB apart.
- Short-event skip threshold.
- Deterministic parquet bytes for identical input.

### 10.2 Worker integration tests

- Idempotency on same key (no-op on `completed`, 409 on `running`, replace on
  `failed`).
- Independence: re-running notes does not modify Event Encoder outputs.
- Partial-event audio failure does not fail the whole job.
- Atomic SQLite claim under simulated concurrent workers.

### 10.3 API tests

- `GET .../notes-status` for each lifecycle state.
- `POST .../notes-jobs` default and explicit version, including 409.
- `GET .../notes` viewport filtering.
- `GET .../timeline` extended payload.

### 10.4 Frontend tests

Playwright in `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts`
(extended):

- `notes_status=completed` renders semitone gridlines and ≥1 note.
- `notes_status=absent` falls back to rectangle mode and exposes a `Generate
  notes` button.
- `Generate notes` posts and updates status chip to `queued` then `running`
  (mocked backend).
- View dropdown disabled state when `notes_status !== "completed"`.
- Note pixel color matches `tokenColor(token)` for a known event.

`npx tsc --noEmit` clean for new components.

### 10.5 Reference fixture

A small synthetic audio file (~10 s, 3 events with known F0 and 3 harmonics)
committed under `tests/fixtures/piano_roll/`. Used by CQT unit tests and by
one Playwright end-to-end test with a mocked backend.

### 10.6 Manual acceptance

Browser verification using preview tools per CLAUDE.md preview workflow:

- Run notes worker on a real Event Encoder job on disk.
- Open the page in preview; switch modes; screenshot each.
- Hover notes and verify tooltip matches parquet contents.
- Simulate failure path by removing the parquet; verify re-run path.

## 11. Future work

- MIDI/MPE export endpoint and UI (uses parquet directly).
- Extended pitch range (future ADR), motivated by social sounds above 4 kHz
  and very low moans.
- Vocalization-type-aware coloring or multi-track MIDI export.
- Per-frame pitch contours (would extend parquet schema or add a sidecar)
  for an eventual pitch-bend display mode.
