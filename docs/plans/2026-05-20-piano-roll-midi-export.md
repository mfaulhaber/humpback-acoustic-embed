# Piano Roll MIDI Export — Implementation Plan

**Goal:** Add a user-initiated async MIDI export action to the Event Encoder piano roll page, persisting `.mid` files under a new `<storage_root>/exports/` directory.
**Spec:** [docs/specs/2026-05-20-piano-roll-midi-export-design.md](../specs/2026-05-20-piano-roll-midi-export-design.md)
**Primary domain:** `sequence-models`
**Neighbor domains:** `core-platform` (Alembic, model, storage, queue/runner), `frontend-shell` (api client, query hooks)

---

### Task 1: Database migration and model for `piano_roll_midi_exports`

**Files:**
- Create: `alembic/versions/078_piano_roll_midi_exports.py`
- Create: `src/humpback/models/piano_roll_midi_export.py`
- Modify: `src/humpback/models/__init__.py` (export the new model)

**Acceptance criteria:**
- [ ] **Database backup performed** before running the migration: read `HUMPBACK_DATABASE_URL` from `.env`, copy the SQLite file to `<original_path>.YYYY-MM-DD-HH:mm.bak` with a UTC timestamp, and verify the backup exists with non-zero size. Do not proceed if the backup step fails.
- [ ] Migration `078_piano_roll_midi_exports.py` creates the table with columns matching spec §6.1 (`id`, `event_encoder_job_id`, `extractor_version`, `status` indexed, `started_at`, `finished_at`, `error_message`, `midi_path`, `n_notes`, `n_bytes`, `compute_seconds`, `params_json`, `created_at`, `updated_at`).
- [ ] Foreign key `event_encoder_job_id → event_encoder_jobs.id` with `ON DELETE CASCADE`.
- [ ] Unique constraint on `(event_encoder_job_id, extractor_version)`.
- [ ] Migration uses `op.batch_alter_table()` for SQLite compatibility wherever needed.
- [ ] `downgrade()` drops the table cleanly.
- [ ] `PianoRollMidiExport` SQLAlchemy model extends `UUIDMixin`, `TimestampMixin`, `Base`; `status` defaults to `JobStatus.queued.value`.
- [ ] Migration runs cleanly against the production database (`HUMPBACK_DATABASE_URL` from `.env`).

**Tests needed:**
- Smoke test in `tests/models/test_piano_roll_midi_export.py`: instantiate the model, persist a row, read it back, verify defaults.
- Verify the unique constraint by inserting two rows with the same `(event_encoder_job_id, extractor_version)` and asserting an integrity error.

---

### Task 2: MIDI synthesis module and `mido` dependency

**Files:**
- Modify: `pyproject.toml` (add `mido` to `[project] dependencies`)
- Modify: `uv.lock` (regenerated via `uv lock`)
- Create: `src/humpback/processing/midi_synthesis.py`
- Create: `tests/processing/test_midi_synthesis.py`

**Acceptance criteria:**
- [ ] `mido` listed in main dependencies (not behind an extra).
- [ ] `uv lock` regenerates `uv.lock` and `uv sync --group dev --extra tf-macos` succeeds locally.
- [ ] `notes_dataframe_to_midi_bytes(notes_df: pd.DataFrame) -> bytes` produces a deterministic SMF Type 1 file per spec §10.3:
  - 480 ticks per quarter, 120 BPM tempo track + one notes track on channel 1.
  - Notes sorted by `start_utc`, then `event_id`, then `partial_index`.
  - Time origin shifted so the earliest `start_utc` lands at MIDI tick 0.
  - `velocity` column used verbatim.
  - Pitches outside `[0, 127]` clamped silently.
  - Zero-duration notes skipped silently.
  - Empty DataFrame → valid SMF (header + tempo track only, no notes track is acceptable but emit one as a placeholder for consistency).
- [ ] Function is pure (no I/O, no global state) so it can be reused outside the worker.

**Tests needed:**
- Three-note hand-crafted DataFrame round-trips through `mido.MidiFile(...)` parse, recovering pitches/velocities/durations within one tick.
- All-partials-stacked case: multiple notes with the same `start_utc` and different `partial_index` all appear as note-on events on channel 1.
- Determinism: two calls with the same input produce byte-identical output.
- Empty DataFrame returns valid MIDI bytes (parseable by `mido`).
- Pitches at 200 and -10 clamp to 127 and 0 respectively.
- Zero-duration notes are dropped from the output.
- Time origin shift: a DataFrame whose earliest `start_utc` is `1_700_000_000.0` produces a file whose first note-on is at delta 0.

---

### Task 3: Storage helper and service layer

**Files:**
- Modify: `src/humpback/storage.py` (add `event_encoder_midi_export_path` and `exports_root` helpers)
- Create: `src/humpback/services/piano_roll_midi_export_service.py`
- Create: `tests/services/test_piano_roll_midi_export_service.py`

**Acceptance criteria:**
- [ ] `event_encoder_midi_export_path(job_id, extractor_version) -> Path` returns `<storage_root>/exports/event_encoders/{job_id}/notes_v{extractor_version}.mid`.
- [ ] Helper does not create directories itself; the worker is responsible for `mkdir(parents=True, exist_ok=True)` before writing.
- [ ] Module exports a custom exception `PianoRollMidiExportConflict`.
- [ ] `enqueue_piano_roll_midi_export(session, event_encoder_job_id, extractor_version=None, params=None, force=False) -> tuple[PianoRollMidiExport, bool]` implements the behavior matrix in spec §7:
  - `extractor_version=None` → resolve to the `extractor_version` of the latest `complete` `PianoRollNotesJob` for the encoder; raise `ValueError` if no `complete` notes job exists.
  - Validate event encoder exists; raise `ValueError` if not.
  - Validate notes job for the resolved `(event_encoder_job_id, extractor_version)` is `complete`; raise `ValueError` if not.
  - Existing row by `(event_encoder_job_id, extractor_version)`:
    - `complete` and `force=False` → return existing, `created=False`.
    - `complete` and `force=True` → reset to `queued`, clear transient fields, return `(row, False)`.
    - `queued` / `running` → raise `PianoRollMidiExportConflict`.
    - `failed` / `canceled` → reset to `queued`, clear transient fields, return `(row, False)`.
  - No row → insert new with `params_json` from caller (default `{}`), return `(row, True)`.
- [ ] `latest_for_encoder_job(session, event_encoder_job_id)` returns the most recent row (any status) for the encoder.
- [ ] `complete_for_encoder_job_version(session, event_encoder_job_id, extractor_version)` returns the `complete` row for that version, or `None`.

**Tests needed:**
- Behavior matrix: one test per row of spec §7.
- Version resolution: `extractor_version=None` resolves to the latest complete notes job's version; raises when no complete notes job exists.
- `ValueError` when event encoder does not exist; `ValueError` when notes job is not complete.
- `force=True` on a `complete` row resets transient fields (`started_at`, `finished_at`, `error_message`, `midi_path`, `n_notes`, `n_bytes`, `compute_seconds` all become `None`) and sets `status='queued'`.
- `latest_for_encoder_job` returns the most recent row by `created_at`.

---

### Task 4: Worker, queue claim, and runner integration

**Files:**
- Create: `src/humpback/workers/piano_roll_midi_export_worker.py`
- Modify: `src/humpback/workers/queue.py` (add `claim_piano_roll_midi_export`, extend stale-running recovery)
- Modify: `src/humpback/workers/runner.py` (dispatch the new claim/run)
- Create: `tests/workers/test_piano_roll_midi_export_worker.py`
- Create: `tests/workers/test_queue_midi_export.py`

**Acceptance criteria:**
- [ ] `run_piano_roll_midi_export(session, job, settings)` implements the lifecycle in spec §8:
  - Transition `queued → running`, set `started_at = utcnow()`, commit.
  - Resolve params from `params_json` (defaults to empty dict if not present).
  - Load notes parquet from `event_encoder_notes_path(job.event_encoder_job_id, job.extractor_version)`. If missing, transition `running → failed` with `error_message="notes parquet not found at <path>"` and return.
  - Call `notes_dataframe_to_midi_bytes()` on the parquet contents.
  - Atomically write the bytes: write to `…tmp` in the target directory, fsync, rename to the final path returned by `event_encoder_midi_export_path()`.
  - Transition `running → complete`, set `finished_at`, `midi_path` (path under `storage_root`), `n_notes`, `n_bytes`, `compute_seconds`, `params_json`.
  - On exception during synthesis or write: remove any partial `.mid` file at the temp path; transition `running → failed`; set `finished_at` and `error_message` (truncate to a sane length).
- [ ] `claim_piano_roll_midi_export(session)` mirrors `claim_piano_roll_notes_job` in `queue.py` (3-retry atomic claim ordered by `created_at`).
- [ ] Stale-running recovery in `queue.py` includes `PianoRollMidiExport` (10-minute `running → queued` reset).
- [ ] Runner loop in `runner.py` calls `claim_piano_roll_midi_export` in the standard claim sequence and dispatches to `run_piano_roll_midi_export` when claimed.

**Tests needed:**
- Successful run: fixture event encoder + complete notes job + valid parquet on disk → job transitions to `complete`, file exists at the expected path with correct byte count, `n_notes` matches parquet rows.
- Missing parquet → status `failed` with the expected error message; no `.mid` file left behind.
- Synthesis exception → partial file cleaned up; status `failed`; error message populated.
- Queue claim: two simulated concurrent claim calls → only one succeeds, the other returns `None`.
- Stale recovery: a row stuck in `running` for >10 minutes is reset to `queued` and reclaimable.

---

### Task 5: API endpoints and Pydantic schemas

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py` (or create `src/humpback/schemas/piano_roll_midi_export.py` — match the convention used by `piano_roll_notes.py`)
- Modify: `src/humpback/api/routers/sequence_models.py`
- Create: `tests/api/test_sequence_models_midi_export.py`

**Acceptance criteria:**
- [ ] Pydantic schemas defined per spec §9.2: `PianoRollMidiExportRead`, `PianoRollMidiExportStatusAbsent`, `PianoRollMidiExportStatusResponse` (union), `PianoRollMidiExportCreateRequest`.
- [ ] `GET /event-encoders/{job_id}/midi-export-status` returns the latest row as `PianoRollMidiExportRead`, or `{status: "absent"}` if no row exists. Supports optional `?extractor_version=` query param to pin a version.
- [ ] `POST /event-encoders/{job_id}/midi-exports`:
  - Body: `PianoRollMidiExportCreateRequest`.
  - Calls `enqueue_piano_roll_midi_export`.
  - Returns 201 on new insert; 200 on reset-from-terminal-state.
  - Returns 409 on `PianoRollMidiExportConflict`.
  - Returns 422 on `ValueError` (notes job not complete, encoder missing, etc.).
- [ ] `GET /event-encoders/{job_id}/midi-export`:
  - Streams the `.mid` binary from `midi_path` on disk.
  - Headers: `media_type="audio/midi"`, `Content-Disposition='attachment; filename="event_encoder_{job_id}_notes_v{version}.mid"'`.
  - Optional `?extractor_version=` query param; defaults to latest `complete` row.
  - Returns 404 if no `complete` row exists or the file is missing on disk.

**Tests needed:**
- GET status: absent (no row) returns `{status: "absent"}`; with a row returns full read schema.
- POST insert: new row created, 201 returned, body matches `PianoRollMidiExportRead`.
- POST reset of failed row: returns 200, status reset to `queued`, transient fields cleared.
- POST when queued/running: returns 409.
- POST when notes job is not complete: returns 422 with informative detail.
- POST with `force=true` on a complete row: returns 200, status reset to `queued`.
- GET download: 200 with correct headers and a valid MIDI byte stream (parseable by `mido` in test).
- GET download when no complete row: 404.
- GET download when file is missing on disk: 404.

---

### Task 6: Frontend hooks, button component, page integration, Playwright

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts` (types, fetch functions, hooks)
- Create: `frontend/src/components/sequence-models/MidiExportButton.tsx`
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx` (mount `<MidiExportButton />` immediately left of `<NotesStatusControls />`)
- Create: `frontend/e2e/sequence-models/midi-export.spec.ts`

**Acceptance criteria:**
- [ ] `PianoRollMidiExportJobStatus` and read/absent union types added to `sequenceModels.ts`.
- [ ] `fetchPianoRollMidiExportStatus(jobId)`, `createPianoRollMidiExportJob(jobId, body)` client functions implemented.
- [ ] `usePianoRollMidiExportStatus(jobId)` `useQuery` with `refetchInterval = 3000` when `status ∈ {queued, running}`, `false` otherwise.
- [ ] `useCreatePianoRollMidiExportJob(jobId)` `useMutation`; on success invalidates the `piano-roll-midi-export-status` query.
- [ ] `MidiExportButton` accepts `eventEncoderJobId` and `notesStatus` props and renders the state matrix from spec §11.2:
  - Disabled with tooltip when notes not `complete`.
  - "Export MIDI" + enabled when no row, failed, or canceled — click calls the mutation.
  - "Exporting…" + disabled + spinner when queued/running.
  - "Download MIDI" + enabled when complete — click navigates the browser to the GET endpoint (use `window.location.assign` or an `<a download>` element so the browser dialog fires).
  - Failure tooltip shows the error message.
  - When complete, an overflow menu next to the primary button exposes a single "Re-export" item that calls the mutation with `force: true`.
- [ ] Placement in `EventEncoderPianoRollPage.tsx`: the new component sits immediately to the left of `<NotesStatusControls />` (around line 985 at spec time; verify exact location in the current file).

**Tests needed:**
- Playwright `frontend/e2e/sequence-models/midi-export.spec.ts`:
  - Navigate to a piano roll page for a job with a complete notes job.
  - Verify the button is visible and enabled with label "Export MIDI".
  - Click the button; verify it transitions to "Exporting…" then "Download MIDI" via the existing 3-second poll.
  - Click "Download MIDI"; use `page.waitForResponse` to verify the response carries `Content-Disposition` matching `attachment; filename=...`.
  - Verify the button is disabled with tooltip text when notes status is not `complete`.
  - Verify the "Re-export" overflow item is present when status is `complete`, and clicking it transitions back to "Exporting…".

---

### Task 7: Documentation and ADR-066

**Files:**
- Modify: `docs/agent-context/domains/sequence-models/README.md` (add MIDI export subsection under Piano Roll Notes; list new artifact path, endpoints, worker, `force` semantics)
- Modify: `docs/reference/storage-layout.md` (add `exports/event_encoders/{job_id}/notes_v{version}.mid` to the storage tree)
- Modify: `docs/reference/sequence-models-api.md` (or the equivalent owner file; add the three new endpoints with shapes)
- Modify: `docs/agent-context/current-state.md` (one-line note that Piano Roll Notes now exports MIDI)
- Modify: `DECISIONS.md` (append ADR-066)

**Acceptance criteria:**
- [ ] Sequence Models domain capsule has a "MIDI export" subsection covering artifact path, the three API endpoints, and `force` re-export semantics.
- [ ] Storage layout reference includes the new `exports/event_encoders/{job_id}/notes_v{version}.mid` path.
- [ ] Sequence Models API reference lists the three new endpoints with method, path, request body shape, response shape, and status codes (200/201/404/409/422).
- [ ] `current-state.md` notes Piano Roll Notes can now be exported as MIDI.
- [ ] ADR-066 added to `DECISIONS.md`: title "User-initiated async MIDI export for Piano Roll Notes". Body captures (a) separate async job rather than folding into the notes worker, (b) single-channel all-partials layout, (c) `force=True` re-export semantics, (d) MIDI conventions (Type 1, 480 PPQ, 120 BPM, channel 1), (e) `mido` library choice and its forward compatibility with a future MPE-based pitch-bend extension.

**Tests needed:**
- None (documentation only).

---

### Verification

Run in order after all tasks. Database backup (Task 1) must have completed before any migration was applied.

1. `uv run ruff format --check src/humpback tests`
2. `uv run ruff check src/humpback tests`
3. `uv run pyright src/humpback tests`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test e2e/sequence-models/midi-export.spec.ts`
7. Manual sanity check: visit the Event Encoder piano roll page for a job whose notes are complete, click "Export MIDI", wait for the status to flip, click "Download MIDI", and load the resulting `.mid` file in any MIDI viewer to confirm playback.
