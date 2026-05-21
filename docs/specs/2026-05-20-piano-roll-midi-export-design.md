# Piano Roll MIDI Export — Design Spec

**Date:** 2026-05-20
**Status:** Approved (superseded in part)
**Owner:** Sequence Models / Event Encoder

> §10 (single-channel MIDI synthesis) is superseded by
> [2026-05-20-event-encoder-midi-channelized-design.md](2026-05-20-event-encoder-midi-channelized-design.md)
> §6. All partials no longer stack on MIDI channel 1; the export now uses
> a slim seven-channel layout (F0 + 2nd–5th + higher + unmatched) with
> one SMF track per channel and a distinct GM `program_change` and
> `track_name` per channel.

## 1. Summary

Add an "Export MIDI" affordance to the Event Encoder piano roll view that
synthesizes a Standard MIDI File from the Piano Roll Notes parquet sidecar.
File creation runs as a user-initiated asynchronous job that mirrors the
existing Piano Roll Notes worker pattern. The button shows export status and
exposes a download action once the file is on disk. A new top-level `exports/`
directory is added under `storage_root` to hold this and any future export
artifacts.

Pitch-bend information is **not** included in this spec — the notes parquet
schema does not yet carry sub-semitone pitch data. A follow-up project will
extend the notes pipeline and use MPE (MIDI Polyphonic Expression) for
expressive playback. The `mido` library chosen here supports MPE without
modification.

## 2. Motivation

The Piano Roll Notes worker (ADR-064) writes one parquet row per detected note,
keyed to MIDI pitch and velocity. The frontend renders these on a Notes-mode
canvas. Users have no path to export the underlying note data into a format
they can audition in a DAW or load into a MIDI viewer. A MIDI export closes
that gap and produces a portable, persistent artifact per event-encoder job.

## 3. Scope

**In scope:**

- New top-level storage directory: `<storage_root>/exports/`.
- New async job, table, worker, service, API endpoints, and frontend
  components for MIDI export.
- All-partials-stacked export onto a single MIDI channel (no pitch bend).
- User-initiated trigger via a button on the piano roll page; `force=True`
  re-export.
- Backend test suite and Playwright end-to-end test.

**Out of scope (tabled to future work):**

- Pitch bend / MPE support.
- Per-frame pitch contours in the notes parquet (`v2` extractor).
- Filtering exports by viewport, event subset, or partial index.
- Multiple MIDI tracks per event or per partial.
- Tempo other than the constant default.
- Auto-export on notes completion (auto-trigger after notes job finishes).
- Bulk export across multiple jobs.

## 4. Decisions

| Decision | Choice |
|---|---|
| Persistence | Persist `.mid` files on disk under `storage_root` |
| Export scope per click | Entire job, all partials |
| Channel layout | All notes on MIDI channel 1, single notes track |
| Trigger model | User-initiated async job with its own status pill |
| Re-export when complete | Allowed via explicit `force=True` flag |
| Button visibility | Always visible; disabled until notes status is `complete` |
| MIDI library | `mido` (pure Python, pip-installable, MPE-capable) |
| Time origin in MIDI | `t=0` = earliest note's `start_utc` across the job |
| Tempo | 120 BPM, constant |
| Ticks per quarter note | 480 |
| Pitch-bend support | Deferred; will use MPE when added |

## 5. Storage Layout

A new top-level directory is added under `storage_root`:

```
<storage_root>/exports/event_encoders/{event_encoder_job_id}/notes_v{extractor_version}.mid
```

- `exports/` is the home for future export artifacts of any kind; nothing
  outside MIDI uses it yet.
- The `event_encoders/{job_id}/` shape mirrors the existing sidecar parquet
  location for symmetry.
- File naming includes the extractor version so future schema bumps coexist
  with current files.

**Storage helper** added to `src/humpback/storage.py`:

```python
def event_encoder_midi_export_path(
    job_id: str, extractor_version: str
) -> Path:
    ...
```

**Atomic write** is required: synth to `…tmp` → fsync → rename.

## 6. Database

### 6.1 New table

`piano_roll_midi_exports`:

| Column | Type | Notes |
|---|---|---|
| `id` | String, PK | UUID |
| `event_encoder_job_id` | String, FK → `event_encoder_jobs.id` | cascade delete |
| `extractor_version` | String, default `"v1"` | matches notes parquet version |
| `status` | String, default `"queued"`, indexed | `JobStatus` enum values |
| `started_at` | DateTime, nullable | UTC |
| `finished_at` | DateTime, nullable | UTC |
| `error_message` | Text, nullable | populated on failed status |
| `midi_path` | Text, nullable | path under `storage_root` |
| `n_notes` | Integer, nullable | rows pulled from parquet |
| `n_bytes` | Integer, nullable | size of `.mid` on disk |
| `compute_seconds` | Float, nullable | wall-clock synth time |
| `params_json` | Text, default `"{}"` | reserved for future synth options |
| `created_at`, `updated_at` | inherited from `TimestampMixin` | UTC |

**Unique constraint:** `(event_encoder_job_id, extractor_version)` — one
canonical export per notes version.

### 6.2 Model

`src/humpback/models/piano_roll_midi_export.py`, class
`PianoRollMidiExport(UUIDMixin, TimestampMixin, Base)`. Reuses `JobStatus`
from `src/humpback/models/processing.py`.

### 6.3 Alembic migration

`alembic/versions/078_piano_roll_midi_exports.py` (next sequential number;
verify at implementation time). Uses `op.batch_alter_table()` for SQLite
compatibility. Foreign-key cascade deletes the export row when the parent
event-encoder job is deleted.

**Database backup is acceptance criterion #1** in the implementation plan,
per CLAUDE.md §4: back up the SQLite file at `HUMPBACK_DATABASE_URL` before
running the migration.

## 7. Service Layer

**File:** `src/humpback/services/piano_roll_midi_export_service.py`

```python
class PianoRollMidiExportConflict(Exception): ...

def enqueue_piano_roll_midi_export(
    session,
    event_encoder_job_id: str,
    extractor_version: Optional[str] = None,
    params: Optional[dict] = None,
    force: bool = False,
) -> tuple[PianoRollMidiExport, bool]:
    ...

def latest_for_encoder_job(
    session, event_encoder_job_id: str
) -> Optional[PianoRollMidiExport]: ...

def complete_for_encoder_job_version(
    session, event_encoder_job_id: str, extractor_version: str
) -> Optional[PianoRollMidiExport]: ...
```

`enqueue_piano_roll_midi_export` resolves the effective `extractor_version`
as follows:

- If the caller passes a non-null `extractor_version`, use it verbatim.
- If `None`, resolve to the `extractor_version` of the latest `complete`
  notes job for the event encoder. If no `complete` notes job exists, raise
  `ValueError`.

It then validates that the event encoder job exists and that a `complete`
notes job for `(event_encoder_job_id, extractor_version)` exists. If either
is missing, raises `ValueError` (mapped to HTTP 422 at the router).

Behavior on existing rows, keyed by `(event_encoder_job_id, extractor_version)`:

| Existing status | Behavior |
|---|---|
| `complete` and `force=False` | return existing row, `created=False` |
| `complete` and `force=True` | reset to `queued`, clear transient fields, return `(row, False)` |
| `queued` or `running` | raise `PianoRollMidiExportConflict` |
| `failed` or `canceled` | reset to `queued`, clear transient fields, return `(row, False)` |
| No row | insert new row, return `(row, True)` |

"Transient fields" means `started_at`, `finished_at`, `error_message`,
`midi_path`, `n_notes`, `n_bytes`, `compute_seconds`.

No automatic enqueue from the notes worker. Export only runs when triggered
through the API.

## 8. Worker

**File:** `src/humpback/workers/piano_roll_midi_export_worker.py`

```python
async def run_piano_roll_midi_export(session, job, settings) -> None:
    # queued → running: set started_at = utcnow(), commit
    # Resolve params (extractor_version, future synth options)
    # Load notes from event_encoder_notes_path(job.event_encoder_job_id, version)
    #   - if file missing, fail with "notes parquet not found"
    # Synthesize MIDI bytes (see Section 10)
    # Atomic write: tmp → fsync → rename to event_encoder_midi_export_path(...)
    # running → complete: set finished_at, midi_path, n_notes, n_bytes,
    #   compute_seconds, params_json
    # On exception:
    #   - remove partial .mid via cleanup helper
    #   - running → failed: set finished_at, error_message
```

**Queue claim** in `src/humpback/workers/queue.py`:

```python
async def claim_piano_roll_midi_export(session):
    for _ in range(3):
        job = await _claim_next_job(
            session,
            PianoRollMidiExport,
            status_attr=PianoRollMidiExport.status,
            queued_value=JobStatus.queued.value,
            running_value=JobStatus.running.value,
            order_attr=PianoRollMidiExport.created_at,
        )
        if job is not None:
            return job
    return None
```

**Stale recovery:** same 10-minute `running` → `queued` reset that other job
types use, extended in `queue.py` to include `PianoRollMidiExport`.

**Runner loop:** add a `claim_piano_roll_midi_export()` call alongside the
existing claim sequence in `src/humpback/workers/runner.py` and dispatch to
`run_piano_roll_midi_export()` when claimed.

## 9. API

**File:** `src/humpback/api/routers/sequence_models.py`

### 9.1 Endpoints

| Path | Method | Purpose |
|---|---|---|
| `/event-encoders/{job_id}/midi-export-status` | GET | status pill data |
| `/event-encoders/{job_id}/midi-exports` | POST | enqueue (or re-enqueue) |
| `/event-encoders/{job_id}/midi-export` | GET | stream `.mid` binary |

### 9.2 Schemas

`src/humpback/api/schemas/sequence_models.py`:

```python
class PianoRollMidiExportRead(BaseModel):
    id: str
    event_encoder_job_id: str
    extractor_version: str
    status: PianoRollMidiExportJobStatus  # Literal of JobStatus values
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error_message: Optional[str]
    midi_path: Optional[str]
    n_notes: Optional[int]
    n_bytes: Optional[int]
    compute_seconds: Optional[float]
    params_json: str
    created_at: datetime
    updated_at: datetime

class PianoRollMidiExportStatusAbsent(BaseModel):
    status: Literal["absent"]

PianoRollMidiExportStatusResponse = Union[
    PianoRollMidiExportRead, PianoRollMidiExportStatusAbsent
]

class PianoRollMidiExportCreateRequest(BaseModel):
    extractor_version: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    force: bool = False
```

### 9.3 Behavior

**GET `/midi-export-status`** — returns the latest row for the event encoder
job, or `{status: "absent"}` if none exists. Optional query param
`extractor_version` to pin a specific version.

**POST `/midi-exports`** — body `PianoRollMidiExportCreateRequest`. Returns
the upserted row. HTTP 201 on insert, HTTP 200 on reset of a terminal-state
row, HTTP 409 on `PianoRollMidiExportConflict`, HTTP 422 if the underlying
notes job is not yet `complete`.

**GET `/midi-export`** — returns a `StreamingResponse`:

```
media_type: audio/midi
Content-Disposition: attachment; filename="event_encoder_{job_id}_notes_v{version}.mid"
```

Reads from `midi_path` on disk. Returns HTTP 404 if no `complete` row exists
or the file is missing. Optional query param `extractor_version` for
version-pinning; defaults to the latest `complete` row for the job.

## 10. MIDI Synthesis

### 10.1 Library

`mido` (pure Python). Add to the main `[project] dependencies` in
`pyproject.toml`. `uv lock` regenerates `uv.lock`.

### 10.2 Function

`src/humpback/processing/midi_synthesis.py`:

```python
def notes_dataframe_to_midi_bytes(notes_df: pd.DataFrame) -> bytes:
    """Synthesize a Standard MIDI File from a notes parquet DataFrame.

    Returns the bytes of an SMF Type 1 file.
    """
```

Extracted from the worker for unit-testability.

### 10.3 Conventions

| Choice | Value | Rationale |
|---|---|---|
| SMF format | Type 1 (multi-track) | Tempo track + notes track is canonical |
| Track layout | Two tracks: tempo track, notes track | Standard layout for DAW import |
| Ticks per quarter note | 480 | High-resolution PPQ |
| Tempo | 120 BPM, written once at t=0 | Music-neutral default |
| Time origin | `t=0` = earliest note's `start_utc` across the parquet | MIDI has no absolute epoch concept |
| Channel | All notes on MIDI channel 1 | Decision in Section 4 |
| Note ordering | Sort by `start_utc`, then `event_id`, then `partial_index` | Deterministic output |
| Velocity | Use `velocity` column directly (already 1–127) | No re-quantization |
| Note On / Off | Paired per row; Off-time = On-time + `duration_s` → ticks | Standard |
| Pitches outside 0–127 | Clamped to `[0, 127]` silently | Defensive; upstream `MIDIQuantizeParams` already constrains to a sub-range |
| Zero-duration notes | Skipped silently | Defensive; would emit a malformed event |
| Empty parquet | Emit valid SMF with header + tempo track only | Defensive |

### 10.4 Determinism

The synthesizer is deterministic: same parquet rows → same bytes. This is
required for the `force=False` re-click no-op behavior and for byte-comparison
in tests.

## 11. Frontend

### 11.1 Hooks

**File:** `frontend/src/api/sequenceModels.ts`

```ts
export type PianoRollMidiExportJobStatus =
  | "queued" | "running" | "complete" | "failed" | "canceled";

usePianoRollMidiExportStatus(jobId: string | null)
  // GET /midi-export-status
  // refetchInterval = 3s when status ∈ {queued, running}, else false

useCreatePianoRollMidiExportJob(jobId: string)
  // POST /midi-exports
  // onSuccess: invalidate 'piano-roll-midi-export-status' query
```

The download itself is triggered by a direct browser GET to `/midi-export`
(an `<a download href>` or `window.open` call); the `Content-Disposition`
header drives the browser save dialog. No `useQuery` wraps the download.

### 11.2 Component

**File:** `frontend/src/components/sequence-models/MidiExportButton.tsx`

A single button placed by `EventEncoderPianoRollPage.tsx` immediately to the
left of `<NotesStatusControls />` (currently around line 985 of that file).

State machine driven by `(notesStatus, exportStatus)`:

| Notes status | Export status | Label | Enabled | Click action |
|---|---|---|---|---|
| Not `complete` | any | "Export MIDI" | disabled | tooltip explains the gate |
| `complete` | `absent` / `failed` / `canceled` | "Export MIDI" | enabled | POST `/midi-exports` |
| `complete` | `queued` | "Exporting…" | disabled | spinner |
| `complete` | `running` | "Exporting…" | disabled | spinner |
| `complete` | `complete` | "Download MIDI" | enabled | direct GET `/midi-export` |

**Re-export affordance:** when export status is `complete`, a small kebab/
overflow menu attached to the button exposes a single "Re-export" item that
POSTs with `force=true`. Re-export is one click deeper than the primary
download to avoid accidental re-runs.

**Failure surfacing:** on `failed`, the label reverts to "Export MIDI"
(enabled, will re-enqueue) and the `error_message` is shown in a tooltip on
hover, matching the existing "Notes failed" pattern in `NotesStatusControls`.

### 11.3 Notifications

A toast "Export queued" fires on POST success if a toast pattern exists in
the page (matching the notes-job enqueue UX). Otherwise the state transition
on the button is the only signal.

## 12. Testing

### 12.1 Backend

- `tests/processing/test_midi_synthesis.py` — pure synthesis function:
  three-note round-trip via `mido.MidiFile`, all-partials-stacked-on-one-
  channel preserves every note-on, determinism (byte-identical for repeated
  calls), empty DataFrame produces a valid SMF, pitch clamping, zero-duration
  skipping.
- `tests/services/test_piano_roll_midi_export_service.py` — idempotency matrix:
  one test per row of the table in Section 7.
- `tests/workers/test_piano_roll_midi_export_worker.py` — end-to-end run
  with a fixture parquet sidecar; failure cleanup (partial `.mid` removed
  on exception); missing parquet path produces `failed` status with helpful
  error message.
- `tests/workers/test_queue_midi_export.py` — atomic claim under simulated
  concurrent worker pickups.
- `tests/api/test_sequence_models_midi_export.py` — status / enqueue /
  download endpoints, 404 paths, 422 when notes job not complete,
  `force=true` reset behavior.

### 12.2 Frontend

- `frontend/e2e/sequence-models-midi-export.spec.ts` — open piano roll page
  for a job whose notes are `complete`; verify the export button is visible
  and enabled; click; observe transition to "Exporting…" then "Download
  MIDI" via the existing 3 s refetch interval; click "Download MIDI" and
  verify the response carries the expected `Content-Disposition` header via
  Playwright's `waitForResponse`.

### 12.3 Final gates

Per CLAUDE.md §6:

1. `uv run ruff format --check` on modified Python files
2. `uv run ruff check` on modified Python files
3. `uv run pyright` on modified Python files
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`

## 13. Documentation

- Update `docs/agent-context/domains/sequence-models/README.md` — add a
  "MIDI export" subsection under Piano Roll Notes describing the artifact
  path, the three endpoints, the worker, and the `force` re-export semantics.
- Update `docs/reference/storage-layout.md` — add
  `exports/event_encoders/{job_id}/notes_v{version}.mid` to the storage tree.
- Update `docs/reference/sequence-models-api.md` (or the equivalent owner
  file) — add the three new endpoints with request/response shapes.
- Update `docs/agent-context/current-state.md` — one-line note that the
  Piano Roll Notes feature now has MIDI export.
- Append **ADR-066: User-initiated async MIDI export for Piano Roll Notes**
  to `DECISIONS.md`. ADR captures: separate async job (not folded into the
  notes worker), single-channel all-partials layout, `force=True` re-export
  semantics, MIDI conventions (Type 1, 480 PPQ, 120 BPM, channel 1), and the
  forward-compatibility note that `mido` will support a future MPE-based
  pitch-bend extension without a library change.

## 14. Open Questions

None at spec time. All design decisions have been resolved through
brainstorming.

## 15. Future Work

- Extend the notes parquet schema with per-frame pitch contours (bumps the
  extractor to `v2`) so MIDI exports can include MPE pitch-bend data.
- Implement MPE-mode synthesis in `midi_synthesis.py` once the v2 schema
  lands. `mido` already supports every primitive MPE requires (multi-channel
  pitch bend, RPN sequences for the MPE Configuration Message and bend-range
  setup, CC 74 timbre, channel pressure).
- Optional viewport-scoped export (current `event_ids` / `start_utc` /
  `end_utc` filter), if users want to dump a single event window.
- Auto-export on notes completion as an opt-in setting on the encoder job,
  if the manual click proves to be friction.
