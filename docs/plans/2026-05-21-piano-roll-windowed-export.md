# Piano Roll Windowed Export Implementation Plan

**Goal:** Make the Piano Roll MIDI export operate on the viewer's current
time window and produce a co-exported `.flac` audio clip of the same window
so the MIDI and audio align when imported into a DAW.

**Spec:** [docs/specs/2026-05-21-piano-roll-windowed-export-design.md](../specs/2026-05-21-piano-roll-windowed-export-design.md)

**Primary domain:** sequence-models
**Neighbor domains:** core-platform, signal-timeline, frontend-shell

---

### Task 1: Alembic migration — add window/audio columns, drop legacy rows

**Files:**

- Create: `alembic/versions/079_piano_roll_midi_exports_window_and_audio.py`

**Acceptance criteria:**

- [ ] Database backup performed first per CLAUDE.md §4: read `HUMPBACK_DATABASE_URL` from `.env`, copy the database file to `<original_path>.YYYY-MM-DD-HH:mm.bak` using a UTC timestamp, verify the backup exists and has non-zero size. This step is a blocking prerequisite — no migration commands run until the backup is confirmed.
- [ ] Migration uses `op.batch_alter_table("piano_roll_midi_exports")` for SQLite compatibility.
- [ ] In `upgrade()`, before adding columns, delete all existing rows from `piano_roll_midi_exports` and remove any on-disk `.mid` files under `<storage_root>/exports/event_encoders/*/notes_*.mid` (the storage root is resolved from runtime config; the migration deletes files via filesystem operations within the storage root prefix and verifies path containment before deletion).
- [ ] Add NOT NULL columns: `window_start_utc` (Float), `window_end_utc` (Float), `audio_path` (Text), `audio_size_bytes` (Integer), `audio_sample_rate` (Integer), `audio_duration_s` (Float). No server defaults required because all rows were just deleted.
- [ ] Add a CHECK constraint `window_end_utc > window_start_utc` via `batch_op.create_check_constraint`.
- [ ] `downgrade()` drops the CHECK constraint, drops the six new columns. (Legacy rows are not restored; downgrade is a schema-only reverse.)
- [ ] Revision id chains from `078_piano_roll_midi_exports`.

**Tests needed:**

- Migration test in `tests/unit/test_migrations.py` (or wherever migration smoke lives) that applies the upgrade to a temporary copy of a fixture database containing a legacy row and an associated `.mid` file; asserts the row and file are gone and the new columns exist with the expected types and NOT NULL constraints.

---

### Task 2: Model and schema updates

**Files:**

- Modify: `src/humpback/models/piano_roll_midi_export.py`
- Modify: `src/humpback/schemas/piano_roll_midi_export.py`

**Acceptance criteria:**

- [ ] `PianoRollMidiExport` gains required (non-Optional) `Mapped` fields matching the new columns: `window_start_utc: float`, `window_end_utc: float`, `audio_path: str`, `audio_size_bytes: int`, `audio_sample_rate: int`, `audio_duration_s: float`.
- [ ] The model docstring at the top of the file is updated to describe the new "windowed bundled export" semantics (window bounds + co-exported FLAC).
- [ ] `PianoRollMidiExportRead` adds those six fields. Since they are NOT NULL in the database, they are required in the schema (no `Optional`).
- [ ] `PianoRollMidiExportCreateRequest` adds required `window_start_utc: float` and `window_end_utc: float`. Existing fields (`extractor_version`, `params`, `force`) remain.
- [ ] `PianoRollMidiExportCreateRequest` has a Pydantic validator (`@model_validator(mode="after")` or equivalent) that rejects requests where `window_end_utc - window_start_utc <= 0` or `> 1800.0`, raising a `ValueError` with a descriptive message.

**Tests needed:**

- `tests/unit/test_sequence_models_schemas.py`: validator rejects zero/negative duration and over-cap requests; accepts valid windows; round-trips through `PianoRollMidiExportRead` with the new fields populated.

---

### Task 3: Storage helper, FLAC writer, MIDI time-origin parameter

**Files:**

- Modify: `src/humpback/storage.py`
- Modify: `src/humpback/processing/audio_encoding.py`
- Modify: `src/humpback/processing/midi_synthesis.py`

**Acceptance criteria:**

- [ ] `storage.py` exports a new helper `event_encoder_audio_export_path(storage_root, job_id, extractor_version) -> Path` that returns `<storage_root>/exports/event_encoders/{job_id}/audio_{extractor_version}.flac`. The existing `event_encoder_midi_export_path()` is untouched.
- [ ] `audio_encoding.py` exports `write_flac_clip(samples, sr, path)`: writes 1-D float32 mono samples as 16-bit PCM FLAC via `soundfile`; no normalization; creates parent dir; writes through a `*.tmp` suffix and renames on success; raises on non-finite samples or non-1-D input.
- [ ] `midi_synthesis.py`'s `notes_table_to_midi_bytes()` gains a `time_origin_utc: float | None = None` parameter. When `None`, the function preserves the existing behavior (anchor to earliest note's `start_utc`). When provided, the function uses `time_origin_utc` as the tick-0 anchor and any note whose effective start equals `time_origin_utc` lands at tick 0.
- [ ] No other behavior change in `notes_table_to_midi_bytes()`: tempo (120 BPM), PPQ (480), and the 7-channel layout from ADR-067 remain identical.

**Tests needed:**

- `tests/unit/test_storage.py`: `event_encoder_audio_export_path` returns the expected path and stays under the storage root.
- `tests/unit/test_audio_encoding.py` (new or existing module): `write_flac_clip` produces a FLAC at the expected sample rate, channel count (1), and duration; samples round-trip within 16-bit quantization tolerance; non-finite or non-1-D inputs raise.
- `tests/unit/test_midi_synthesis.py` (new or existing module): with `time_origin_utc=t0`, a note at `start_utc == t0` lands at tick 0; a note at `t0 + 1.0` lands at the tick corresponding to 1.0 s at 120 BPM / 480 PPQ; behavior with `time_origin_utc=None` is unchanged for existing test fixtures.

---

### Task 4: Service — windowed upsert / cache-hit logic

**Files:**

- Modify: `src/humpback/services/piano_roll_midi_export_service.py`

**Acceptance criteria:**

- [ ] Service create/upsert accepts `window_start_utc` and `window_end_utc` as required parameters.
- [ ] On insert, the row is populated with the requested window; status starts at `queued`; the audio/midi/size/duration fields are left in their NOT-NULL-default placeholder state until the worker fills them (chosen approach: the worker is responsible for the final commit that sets all NOT NULL audio fields, and the insert path uses sentinel values like an empty `audio_path = ""` and zero sizes that the worker overwrites — OR adopt nullable transient state by making these fields nullable until completion. **Decision (consistent with Task 1 NOT NULL schema):** the create path uses non-null sentinel values: `audio_path = ""`, `audio_size_bytes = 0`, `audio_sample_rate = 0`, `audio_duration_s = 0.0`, and only the worker writes the real values on transition to `complete`).
- [ ] On a repeat create call for the same `(event_encoder_job_id, extractor_version)`:
  - If `force=true`, reset the row to `queued` with the new window (clear started/finished/error/midi_path and reset audio sentinels).
  - Else if the requested window matches the persisted window within 1 ms AND status is `complete`, return the existing row (cache hit).
  - Else (window differs OR status is `failed`/`canceled`), reset the row to `queued` with the new window.
- [ ] A clear service-level error is raised when the event encoder job is not in a state that supports export (e.g. notes status not `complete`) — preserving the current behavior.

**Tests needed:**

- `tests/services/test_piano_roll_midi_export_service.py`:
  - Cache hit on matching window + complete status.
  - Re-queue when window changes.
  - Re-queue when `force=true` regardless of window match.
  - Re-queue when previous status is `failed`.
  - Reject when notes sidecar is not complete.

---

### Task 5: Worker — note filtering, audio resolution, atomic dual-write

**Files:**

- Modify: `src/humpback/workers/piano_roll_midi_export_worker.py`

**Acceptance criteria:**

- [ ] Worker reads `window_start_utc`/`window_end_utc` from the claimed row.
- [ ] Loads the Piano Roll Notes parquet for `(event_encoder_job_id, extractor_version)` and filters notes to those overlapping `[window_start_utc, window_end_utc)`. Kept notes have their effective start clipped to `max(note.start_utc, window_start_utc)` and effective end clipped to `min(note.start_utc + note.duration_s, window_end_utc)`. Notes whose clipped duration is < 1 ms are dropped.
- [ ] Calls `notes_table_to_midi_bytes()` with `time_origin_utc=window_start_utc`.
- [ ] Resolves the source audio by walking event encoder job → event segmentation job → detection job to obtain the hydrophone identifier and job timestamp bounds, then calls `resolve_timeline_audio(hydrophone_id, local_cache_path, job_start_timestamp, job_end_timestamp, start_sec=window_start_utc, duration_sec=(window_end_utc - window_start_utc), target_sr=32000)`.
- [ ] Writes both artifacts atomically:
  - MIDI bytes to `<midi_path>.tmp`.
  - FLAC via `write_flac_clip()` to `<audio_path>.tmp`.
  - Only when both temp files exist on disk does the worker rename both to their final paths, then commit the DB row update.
  - On any write or audio-resolution failure, delete temp files (best-effort), leave the previous final artifacts and the row unchanged in terms of artifacts (only the status moves to `failed` with an error message and `finished_at` set).
- [ ] On success, the DB row update in a single transaction sets `status="complete"`, `midi_path`, `audio_path`, `n_notes`, `n_bytes` (MIDI size), `audio_size_bytes`, `audio_sample_rate=32000`, `audio_duration_s = samples.size / 32000`, `finished_at`, and `compute_seconds`.
- [ ] Empty-window case (zero notes after filter) succeeds: a valid SMF with the tempo meta-event and zero notes is produced; FLAC is produced for the same window (silence-filled where audio coverage is missing).

**Tests needed:**

- `tests/workers/test_piano_roll_midi_export_worker.py`:
  - Note filter/clip: overlapping kept and clipped, fully outside dropped, sub-ms clipped dropped.
  - Time origin: parse the resulting MIDI with `mido` and assert a known note's tick offset matches the expected tick = `delta_s * (120/60) * 480`.
  - FLAC verification: sample rate is 32000, channel count is 1, duration matches `window_end_utc - window_start_utc` within one sample at 32 kHz, samples are not normalized (max abs sample equals the input's max abs sample within 16-bit quantization).
  - Atomic rollback: mock `write_flac_clip` to raise; assert prior `.mid` and `.flac` (from a previously successful export) remain on disk unchanged and the row transitions to `failed` with an error message.
  - Empty-window: zero-note window produces a valid SMF (parseable by `mido`) and a FLAC of the correct duration.

---

### Task 6: API — validation + audio-export GET endpoint

**Files:**

- Modify: `src/humpback/api/routers/sequence_models.py`

**Acceptance criteria:**

- [ ] `POST /event-encoders/{job_id}/midi-exports` accepts the new `window_start_utc`/`window_end_utc` fields and threads them into the service call.
- [ ] Returns 400 with `detail` when:
  - `window_end_utc <= window_start_utc`.
  - `window_end_utc - window_start_utc > 1800.0`.
  - The window does not overlap the resolved job's data range. (Range resolution: same chain as the worker uses — event encoder → segmentation → detection job timestamps.)
- [ ] `GET /event-encoders/{job_id}/midi-export-status` continues to return either `PianoRollMidiExportRead` (with the new fields populated) or `PianoRollMidiExportStatusAbsent`. No signature change.
- [ ] `GET /event-encoders/{job_id}/midi-export` (download `.mid`) is unchanged.
- [ ] New endpoint `GET /event-encoders/{job_id}/audio-export`:
  - Looks up the latest export row for the job at the default `extractor_version`.
  - Returns 404 with `detail="No completed audio export for this job."` when no row exists, status is not `complete`, or the on-disk file is missing.
  - On success, returns the file with `Content-Type: audio/flac` and an `attachment; filename="..."` Content-Disposition. The filename should include the job id and exported window times for clarity, e.g. `event-encoder-<job_id_short>-<start_utc>-<end_utc>.flac`.

**Tests needed:**

- `tests/integration/test_sequence_models_api.py`:
  - POST 400s for each invalid window condition.
  - POST 202/200 for a valid window; response includes the new fields.
  - GET `/audio-export` 404 when no export exists.
  - GET `/audio-export` 200 with `audio/flac` content-type and correct `Content-Length` after a worker-produced fixture is in place.
  - GET `/midi-export-status` round-trips the new fields.

---

### Task 7: Frontend types, hooks, mutation payload

**Files:**

- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: any generated TypeScript types file (e.g. `frontend/src/api/generated.ts` — verify the project's actual codegen output location during implementation).
- Modify: `frontend/src/hooks/queries/...` for the Piano Roll MIDI export hook (locate via the existing `MidiExportButton` import path).

**Acceptance criteria:**

- [ ] Regenerated TypeScript types include `window_start_utc`, `window_end_utc`, `audio_path`, `audio_size_bytes`, `audio_sample_rate`, `audio_duration_s` on the read schema, and required `window_start_utc`/`window_end_utc` on the create request.
- [ ] The API client function for creating a Piano Roll MIDI export accepts and sends `windowStartUtc` and `windowEndUtc` (camelCase props mapped to snake_case body fields, matching project conventions).
- [ ] A new API client function exists for downloading the audio file — either as a `Blob` fetch or as a function that returns the absolute URL for the new `GET /event-encoders/{job_id}/audio-export` endpoint, following the same pattern the MIDI download uses.

**Tests needed:**

- `cd frontend && npx tsc --noEmit` passes.
- A small Vitest spec for the mutation hook asserts that calling the create mutation with `windowStartUtc`/`windowEndUtc` sends a POST body containing the corresponding snake_case fields.

---

### Task 8: Frontend — MidiExportButton + piano roll page integration

**Files:**

- Modify: `frontend/src/components/sequence-models/MidiExportButton.tsx`
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`

**Acceptance criteria:**

- [ ] `MidiExportButton` accepts new required props `windowStartUtc: number` and `windowEndUtc: number`.
- [ ] When the requested window is invalid (`end <= start`, `end - start > 1800`), the button is disabled with a tooltip explaining the constraint. The 30-minute cap text is explicit.
- [ ] Three render states based on the status query result:
  - **Absent** (no row) → "Export view" button calls the create mutation with the current window.
  - **Queued/Running** → spinner with "Exporting MIDI and audio…".
  - **Complete** → render the exported-window text and two download buttons ("Download MIDI", "Download audio (FLAC)") plus a "Re-export view" action. "Re-export view" uses emphasized styling when the current window differs from the persisted window by more than 50 ms (start or end); non-emphasized otherwise. The persisted window text is `Exported window: <start UTC>  →  <end UTC>  (<duration> s)` formatted with the project's existing UTC display utility.
  - **Failed/Canceled** → error message + retry control.
- [ ] The button group is positioned consistently with the existing layout (left of the Notes status badge per the domain README).
- [ ] `EventEncoderPianoRollPage.tsx` threads `timeRange.start` and `timeRange.end` into `MidiExportButton` as `windowStartUtc`/`windowEndUtc`.
- [ ] No "How to import into Logic Pro" hint or other DAW workflow guidance is rendered.

**Tests needed:**

- Vitest unit/component tests in `frontend/src/components/sequence-models/MidiExportButton.test.tsx` (new or existing): renders each of the three primary states; tolerance-based "differs from exported window" computation; disabled state for over-cap windows.

---

### Task 9: Playwright e2e

**Files:**

- Modify: `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts`

**Acceptance criteria:**

- [ ] Adds a scenario: open the piano roll page for a fixture event encoder job whose notes status is `complete`; pan/zoom to a sub-window; click "Export view"; wait for `complete`; assert the exported-window text matches the panned range within ~50 ms tolerance; click "Download MIDI" and assert a download is triggered (`page.waitForEvent('download')`); click "Download audio (FLAC)" and assert a download is triggered.
- [ ] Adds a scenario: after the first export completes, pan to a different window; assert the "Re-export view" button is emphasized; re-export; assert the sizes/exported-window text update.
- [ ] Adds a scenario: simulate an over-cap window (zoom to > 30 min) and assert the button is disabled with the expected tooltip.
- [ ] All assertions hit the new endpoints/fields without relying on internal selectors that the existing test wouldn't already use.

**Tests needed:**

- The e2e scenarios above. Run via `cd frontend && npx playwright test e2e/sequence-models/event-encoder-piano-roll.spec.ts`.

---

### Task 10: Docs and ADR

**Files:**

- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/invariants.md`
- Modify: `docs/agent-context/current-state.md`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `DECISIONS.md`

**Acceptance criteria:**

- [ ] Sequence Models capsule `README.md` updated to describe windowed bundled export: the export action takes the viewer's `timeRange`, produces a co-exported `.flac`, and overwrites the canonical pair on re-export. The capsule lists the new audio artifact path `<storage_root>/exports/event_encoders/{job_id}/audio_{extractor_version}.flac` alongside the MIDI path.
- [ ] Sequence Models `invariants.md` documents: one canonical bundled export per `(event_encoder_job_id, extractor_version)`; the MIDI's time origin equals the requested `window_start_utc`; the FLAC is 32 kHz mono 16-bit PCM with no normalization; the 30-minute cap.
- [ ] `current-state.md` notes the new behavior alongside the existing Piano Roll Notes / MIDI export state.
- [ ] `docs/reference/sequence-models-api.md` documents the updated `POST /event-encoders/{job_id}/midi-exports` request body and the new `GET /event-encoders/{job_id}/audio-export` endpoint.
- [ ] `docs/reference/storage-layout.md` adds the FLAC artifact path.
- [ ] `docs/reference/data-model.md` adds the new columns on `piano_roll_midi_exports`.
- [ ] `docs/reference/behavioral-constraints.md` adds the window/cap/empty-window/atomic-rollback rules.
- [ ] `DECISIONS.md` gains ADR-068 ("Piano Roll windowed bundled export") capturing: bundled MIDI + FLAC artifact pair, single-rolling artifact per `(job, version)`, time origin at window start, 30-minute cap, no normalization on FLAC. References ADR-066 and ADR-067.

**Tests needed:**

- Documentation-only change; no automated test. Doc updates land alongside code so the spec/plan/ADR triplet stays consistent.

---

### Verification

Run after all tasks, in order. Run from the repo root unless otherwise noted.

1. `uv run ruff format --check $(git diff --name-only main -- '*.py')`
2. `uv run ruff check $(git diff --name-only main -- '*.py')`
3. `uv run pyright $(git diff --name-only main -- '*.py')`
4. `uv run pytest tests/sequence_models tests/unit/test_sequence_models_schemas.py tests/unit/test_storage.py tests/unit/test_audio_encoding.py tests/unit/test_midi_synthesis.py tests/services/test_piano_roll_midi_export_service.py tests/workers/test_piano_roll_midi_export_worker.py tests/integration/test_sequence_models_api.py -q`
5. `uv run pytest tests/`
6. `cd frontend && npx tsc --noEmit`
7. `cd frontend && npx vitest run src/components/sequence-models/MidiExportButton.test.tsx`
8. `cd frontend && npx playwright test e2e/sequence-models/event-encoder-piano-roll.spec.ts`
9. Manual one-off: drop an exported `.mid` + `.flac` pair into Logic Pro at the same project position with project tempo accepted as 120 BPM and visually confirm note alignment with the audio waveform. Record the result in the PR description.
