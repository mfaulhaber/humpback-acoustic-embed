# Piano Roll MIDI Notes Implementation Plan

**Goal:** Add a Piano Roll Notes worker that extracts per-partial MIDI notes from Event Encoder jobs and a Notes-mode view in the existing piano roll page.
**Spec:** [docs/specs/2026-05-20-piano-roll-midi-notes-design.md](../specs/2026-05-20-piano-roll-midi-notes-design.md)
**Primary domain:** sequence-models
**Neighbor domains:** core-platform, signal-timeline, frontend-shell

## Phases

Work is grouped into four phases that can be picked up independently once their prerequisites are met. Each phase ends in a green state (tests pass, types clean, UI usable if frontend was touched).

- **Phase A — Storage & job lifecycle.** Tasks 1–3. Adds the SQL table, schemas, and dispatcher branch. No extraction yet. Phase A ends with the worker able to enqueue, claim, and short-circuit to `completed` with a zero-row parquet.
- **Phase B — Extraction algorithm.** Tasks 4–6. Implements CQT, peak-pick, tracker, harmonic prior, velocity mapping inside the worker. Phase B ends with real notes written to parquet for a real Event Encoder job, validated against synthetic fixtures.
- **Phase C — API & status surface.** Tasks 7–9. New endpoints, extended timeline payload, frontend status pill and backfill button. Phase C ends with the UI showing notes-job status and a working "Generate notes" action without yet rendering notes on the canvas.
- **Phase D — Notes-mode rendering.** Tasks 10–12. Canvas rewrite for the Notes mode, log-frequency Y-axis, tooltips, Playwright coverage. Phase D ends with the piano roll defaulting to Notes when sidecar is available.

Phases A and B are backend-only and can land in one PR or two. Phase C requires Phase A (status data exists) but not Phase B (status can be `failed`/`pending` and still surfaced). Phase D requires Phases B and C.

---

## Phase A — Storage & job lifecycle

### Task 1: Alembic migration for `piano_roll_notes_jobs`

**Files:**
- Create: `alembic/versions/077_piano_roll_notes_jobs.py`

**Acceptance criteria:**
- [x] Database backup performed before applying migration: read `HUMPBACK_DATABASE_URL` from `.env`, copy the SQLite file to `<original_path>.YYYY-MM-DD-HH:mm.bak` using a UTC timestamp, and verify the backup exists with non-zero size. Do not proceed if backup fails.
- [x] New table `piano_roll_notes_jobs` created with columns: `id`, `event_encoder_job_id` (FK to `event_encoder_jobs.id`, indexed), `extractor_version`, `status`, `started_at` / `finished_at` / `created_at` / `updated_at` (DateTime), `error_message`, `notes_path`, `n_events`, `n_notes`, `compute_seconds`, `params_json`. (Columns use the existing `_at` / `DateTime` convention rather than the float-epoch shape sketched in the spec — kept consistent with `event_encoder_jobs` and the `TimestampMixin`.)
- [x] Unique constraint on `(event_encoder_job_id, extractor_version)`.
- [x] Table created via plain `op.create_table` (additive, no batch needed for a new SQLite table; `op.batch_alter_table` is reserved for alterations of existing tables per the project convention).
- [x] `alembic upgrade head` succeeds against the production database resolved from `.env`. (Required a one-time `alembic stamp 076` first because the production DB had drifted: `event_encoder_jobs` was already present from `Base.metadata.create_all` at worker startup, but `alembic_version` was still `075`.)
- [x] `alembic downgrade -1` succeeds and removes the table.

**Tests needed:**
- Migration smoke test that upgrades and downgrades cleanly against a temp SQLite file.

---

### Task 2: ORM model and Pydantic schemas

**Files:**
- Create: `src/humpback/models/piano_roll_notes.py`
- Create: `src/humpback/schemas/piano_roll_notes.py`
- Modify: `src/humpback/models/__init__.py` (re-export)
- `src/humpback/schemas/__init__.py` — left empty, matching the existing module convention.

**Acceptance criteria:**
- [x] `PianoRollNotesJob` ORM model maps every column from Task 1's migration.
- [x] Pydantic response schema `PianoRollNotesJobRead` mirrors the ORM model with appropriate type narrowing (`status` as a `Literal` union).
- [x] `PianoRollNotesJobCreateRequest` accepts optional `extractor_version` (defaults to `"v1"` resolved at the service layer) and optional `params` (dict serialized to `params_json`).
- [x] `PianoRollNotesStatusResponse` is the timeline-piggyback shape: either a full `PianoRollNotesJobRead` or `PianoRollNotesStatusAbsent` (`{"status": "absent"}`).
- [x] Types pass `uv run pyright` clean.

**Tests needed:**
- Schema round-trip test (ORM → Pydantic → JSON) for one canonical row in each lifecycle state.

---

### Task 3: Worker scaffold and dispatcher branch

**Files:**
- Create: `src/humpback/workers/piano_roll_notes_worker.py`
- Create: `src/humpback/services/piano_roll_notes_service.py`
- Modify: `src/humpback/workers/runner.py` (register dispatcher branch)
- Modify: `src/humpback/workers/queue.py` (only if a new claim helper is required by the existing pattern)
- Modify: `src/humpback/workers/event_encoder_worker.py` (auto-enqueue hook on completion)

**Acceptance criteria:**
- [x] `piano_roll_notes_service.enqueue_piano_roll_notes_job(...)` returns the existing row if `status == "complete"` for the same key; raises `PianoRollNotesJobConflict` if `running` or `queued`; resets to `queued` if `failed` or `canceled`; otherwise inserts new. (Status enum is `JobStatus.complete`, not "completed" — aligned with the project enum.)
- [x] `piano_roll_notes_service.latest_for_encoder_job(...)` returns the most recent `complete` row, falling back to the most recent any-state row.
- [x] Worker uses the existing `_claim_next_job` compare-and-set pattern via a new `claim_piano_roll_notes_job` helper. Stale `running` rows older than `STALE_JOB_TIMEOUT` are recovered to `queued` in `recover_stale_jobs`.
- [x] Worker stub does not yet run extraction; it claims the job, writes an empty parquet sidecar at the canonical path, and marks the row `complete` with `n_events = 0`, `n_notes = 0`, and a populated `params_json`. This stub will be replaced in Phase B.
- [x] Auto-enqueue hook: when an Event Encoder job transitions to `complete` in `event_encoder_worker.py`, `auto_enqueue_after_encoder_complete` enqueues a notes job at the current default `extractor_version`. The hook swallows `PianoRollNotesJobConflict` and any other exception, so it cannot block the Event Encoder transition.

**Tests needed:**
- Idempotency: enqueue twice with the same key returns the same row on the second call.
- Lifecycle: simulate a successful claim path; assert the row ends `completed` and the parquet sidecar exists with zero rows.
- Auto-enqueue: simulate Event Encoder completion and assert a notes job row appears.

---

## Phase B — Extraction algorithm

### Task 4: CQT and peak-pick helpers

**Files:**
- Create: `src/humpback/processing/piano_roll_cqt.py`

**Acceptance criteria:**
- [x] `compute_event_cqt(audio, sr, *, params=CQTParams())` returns log-magnitude matrix with shape `(264, n_frames)` for the spec's default params, resampling input to 22050 Hz mono if necessary. Uses `res_type="polyphase"` to avoid the optional `resampy` dependency.
- [x] `pick_peaks_per_frame(log_mag, *, params=PeakParams())` returns a list of per-frame peak lists `(bin, log_magnitude)`, filtered by the per-frame noise floor and capped at `top_k`. Noise floor uses the **bottom-half** of each frame's bins for the median/MAD estimate, so strong signal peaks do not inflate MAD and push the floor above the harmonics — a known failure mode of full-frame MAD on clean sustained tones.
- [x] All parameters surfaced via keyword-only frozen dataclasses (`CQTParams`, `PeakParams`) with the defaults from the spec.
- [x] Pure functions; no I/O.

**Tests needed:**
- Sinusoid at 440 Hz (A4) → strong peak in CQT bin corresponding to MIDI 69.
- Harmonic stack at 100/200/300/400 Hz → peaks at expected bins.
- Pure noise → no peaks survive the noise floor.

---

### Task 5: Track builder, harmonic prior, MIDI quantizer

**Files:**
- Create: `src/humpback/processing/piano_roll_tracker.py`

**Acceptance criteria:**
- [x] `build_tracks(per_frame_peaks, *, cqt_params, params=TrackerParams())` produces `Track` objects with `start_frame`, `end_frame`, `bins`, `log_magnitudes` (median accessible via properties), using greedy nearest-neighbor matching within `bin_tolerance` and `miss_tolerance_frames`.
- [x] Tracks below `min_duration_s` (50 ms at default CQT hop) are dropped first; surviving tracks are then filtered by `amplitude_floor_percentile` (event-relative).
- [x] `label_harmonics(tracks, *, cqt_params, params=HarmonicParams())` mutates `partial_index` per spec §6.5. Lowest-bin overlapping track becomes F0 (`partial_index = 0`); harmonics that match within ±cents tolerance get their integer position; **all other tracks in the same overlap cluster are marked processed (partial_index stays at -1) so they do not get re-promoted to F0 on a later cluster iteration**. Default `harmonic_prior_enabled = True`; when `False`, the function is a no-op.
- [x] `quantize_to_midi(track, *, cqt_params, midi_params=MIDIQuantizeParams())` returns a `MidiNote` record with `midi_pitch` clamped to `[min_pitch, max_pitch]` when the raw quantized pitch is within ±1 semitone of the range; returns `None` for tracks further out of range. (Note: the worker also derives `start_utc` from `region_offset + event.start_sec + (start_offset_s − pad)`; the tracker only emits relative offsets.)
- [x] Deterministic given the same inputs.

**Tests needed:**
- Two-frame-gap continuity (gap below tolerance keeps one track; gap above splits it).
- Harmonic prior at exact, ±50 cents near-miss, and out-of-range ratios.
- Pitch clamping at A0 boundary.

---

### Task 6: Wire extraction into the worker

**Files:**
- Modify: `src/humpback/workers/piano_roll_notes_worker.py`
- Modify: `src/humpback/services/piano_roll_notes_service.py` (params resolution)

**Acceptance criteria:**
- [x] Replace the Task 3 stub with the real extraction:
  - **One pass with deferred velocity** rather than two physical passes. Per event: CQT → peak-pick → tracker → harmonic prior → quantize → emit notes with raw `peak_magnitude`. Per-frame max log-mag is accumulated as we go. After the loop, job-level [p5, p99] percentiles are computed once and each pending note's `peak_magnitude` is mapped to a uint8 velocity. Produces the same job-level velocity calibration as the spec's two-pass description but at half the CQT compute. Memory cost: ~8 bytes per frame across all events (a few MB on huge jobs).
  - Per-event extraction calls `compute_event_cqt`, `pick_peaks_per_frame`, `build_tracks`, `label_harmonics`, `quantize_to_midi`.
  - Audio is loaded via `build_event_audio_loader` at the CQT target sample rate (22050 Hz); the existing audio lineage (`audio_file_id` or `hydrophone_id`) is honored exactly as the Event Encoder uses it.
- [x] Output parquet `event_notes_v1.parquet` matches schema in spec §5.2, sorted by `(start_utc, midi_pitch)`. `event_token` is the token assigned at the **largest available k** in `event_tokens.parquet`; missing rows fall back to `-1`.
- [x] Per-event audio failure is captured into a `(event_id, message)` aggregate. First 10 failures land in `error_message` (truncated to 2 KB); the job still marks `complete` so long as at least one event yielded a note.
- [x] `params_json` is rewritten from the resolved `_ResolvedParams` dataclass at job completion, capturing every default that was effectively used.
- [x] `n_events` counts events scanned (including skipped-short and failed); `n_notes` counts notes emitted; `compute_seconds` recorded.

**Tests needed:**
- End-to-end on a synthetic 3-event fixture (`tests/fixtures/piano_roll/`): verify expected MIDI pitches and partial labels.
- Partial-failure: inject one missing-audio event; assert other events still produce notes and the job ends `completed` with the per-event failure summarized.
- Determinism: run twice on the same job; parquet byte-equal.

---

## Phase C — API and status surface

### Task 7: API endpoints

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`

**Acceptance criteria:**
- [x] `GET /sequence-models/event-encoders/{job_id}/notes-status` returns `PianoRollNotesStatusResponse`.
- [x] `POST /sequence-models/event-encoders/{job_id}/notes-jobs` enqueues a job; body accepts optional `extractor_version` and `params`. Returns `PianoRollNotesJobRead`. 409 when a `running` row exists for the same key. Also returns 409 when the encoder job is not `complete`.
- [x] `GET /sequence-models/event-encoders/{job_id}/notes` reads `event_notes_v1.parquet` and returns rows filtered by `start_utc`, `end_utc`, and optional `event_ids`. Defaults to the latest completed `extractor_version` for the job; an explicit `extractor_version` query param can pin to a specific version.
- [x] `GET /sequence-models/event-encoders/{job_id}/timeline` payload extended with a `notes_status` field (the same shape as `/notes-status`). The existing response shape is otherwise unchanged.
- [x] Pyright clean; new schemas re-exported.

**Tests needed:**
- One test per endpoint covering happy path and the 409 path for `POST`.
- Timeline payload contains `notes_status` when the encoder job exists and when no notes job exists yet.
- Viewport filtering on `GET /notes` returns only overlapping rows.

---

### Task 8: Frontend API client and React Query hooks

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/api/types.ts` (or local types module per existing convention)
- Create: `frontend/src/hooks/queries/usePianoRollNotesStatus.ts`
- Create: `frontend/src/hooks/queries/usePianoRollNotes.ts`
- Create: `frontend/src/hooks/queries/useGeneratePianoRollNotes.ts` (mutation)

**Acceptance criteria:**
- [x] Typed client functions for the three new endpoints and the extended timeline payload. Co-located in `frontend/src/api/sequenceModels.ts` per the existing Sequence Models convention (rather than the `hooks/queries/` location suggested in the plan, which is used for other domains).
- [x] React Query hooks for status, notes, and trigger. The trigger mutation invalidates both the status query and the timeline query on success. The status query auto-polls (3 s) while `queued` or `running`.
- [x] Notes query is windowed by `(start_utc, end_utc)`, with stable keys including encoder job id, viewport bounds, and `extractor_version`.
- [x] `npx tsc --noEmit` clean.

**Tests needed:**
- Type tests in existing TS test surface, where present.

---

### Task 9: Status surfaces on job card and piano roll page (without notes render)

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`
- Modify: the Event Encoder job card component (locate under `frontend/src/components/sequence-models/`)
- Possibly: small additions to a shared "status pill" primitive under `frontend/src/components/ui/` or `shared/`

**Acceptance criteria:**
- [x] Job card shows a `Notes: <state>` pill next to the existing encoding pill. The repo's Event Encoder UI uses `EventEncoderJobTable` rather than a card layout, so the pill landed as a new `Notes` column in that table. Clicking the pill navigates to the piano roll route. Non-complete encoder rows render `—` (the notes worker only runs for completed encoder jobs).
- [x] Piano roll page shows a notes status chip in the toolbar. `Re-run` is the Generate button's label when `failed`; clicking the pill itself toggles a small inline error message captured from `error_message`.
- [x] When `notes_status` is `absent` or `failed`, a `Generate notes` button is visible. When `queued` or `running`, the button is replaced with an inline `Generating…` label (the Generate button is removed so the user cannot double-fire the mutation).
- [x] No change yet to canvas rendering: existing rectangle modes remain the default.

**Tests needed:**
- Playwright spec covering all five status states (`absent` / `queued` / `running` / `completed` / `failed`) with mocked backend; assert the correct chip text and button visibility.

---

## Phase D — Notes-mode rendering

### Task 10: Log-frequency canvas substrate

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`
- Optionally factor a helper into `frontend/src/components/sequence-models/pianoRollAxis.ts` (new file).

**Acceptance criteria:**
- [ ] When the view mode is `Notes`, the Y-axis switches to a log-frequency / piano-key axis spanning MIDI 21–108 with semitone gridlines and labeled octaves (C0…C8).
- [ ] Black-key shading drawn behind the gridlines.
- [ ] Existing rectangle modes still use the current linear-Hz scale.
- [ ] Switching modes does not reset playback or selection state.

**Tests needed:**
- Playwright snapshot or DOM assertion that `Notes` mode renders 88 rows and labeled octave markers; non-Notes modes do not.

---

### Task 11: Note rendering, tooltips, view-mode toggle

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`
- Modify: the existing tooltip/legend primitives used by the page (resolved during implementation).

**Acceptance criteria:**
- [ ] When `notes_status === "completed"`, the page defaults to `Notes` mode and renders one bar per note row from `/notes`:
  - Y position from `midi_pitch`.
  - X from `start_offset_s` (or `start_utc` minus viewport origin) and `duration_s`.
  - Fill from existing `tokenColor(event_token)`.
  - Stroke opacity proportional to `velocity / 127`.
- [ ] Hover tooltip shows: pitch with note name (e.g., `MIDI 60 (C4)`), `velocity`, `duration_s`, `event_id`, `token`, `partial_index` rendered as `F0` / `2x F0` / `3x F0` / `–`.
- [ ] View toolbar adds `Notes` to the existing modes and makes it the default when available. `Notes` is disabled with an explanatory tooltip when `notes_status !== "completed"`.
- [ ] Fallback: if the `/notes` fetch fails, the canvas reverts to the previous rectangle mode with a non-blocking toast; the `Notes` selection remains visible but greys out.
- [ ] The spectrogram strip is unchanged.

**Tests needed:**
- Playwright extension to `event-encoder-piano-roll.spec.ts`: `notes_status=completed` job renders ≥1 note bar with the expected token color (pixel sample) and tooltip content.
- Playwright covers the fetch-failure fallback path.

---

### Task 12: Reference fixture and end-to-end verification

**Files:**
- Create: `tests/fixtures/piano_roll/synthetic_three_events.wav`
- Create: `tests/fixtures/piano_roll/synthetic_three_events.json` (event timing + expected notes)
- Modify: `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts` (mock-backed end-to-end case using the fixture)
- Modify: domain capsule `docs/agent-context/domains/sequence-models/README.md` to mention the piano-roll-notes artifact, and `docs/agent-context/domains/sequence-models/invariants.md` to record the canonical key.
- Modify: `DECISIONS.md` — append a new ADR ("Piano Roll Notes sidecar") capturing the worker / sidecar split, idempotent key, and `extractor_version` strategy. Append a second ADR placeholder for extended MIDI range as future work.
- Modify: `docs/agent-context/current-state.md` to add the Piano Roll Notes capability.

**Acceptance criteria:**
- [ ] Synthetic fixture committed; backend integration test uses it for end-to-end note extraction; frontend Playwright case uses it via mocked backend.
- [ ] Manual acceptance: run the worker against a real Event Encoder job; open the piano roll page; verify Notes mode renders, mode toggle works, tooltip data matches parquet contents, and the failure-and-rerun path works after deleting the parquet.
- [ ] ADRs and capsule updates merged in this phase's commit set.

**Tests needed:**
- One Playwright end-to-end test using the fixture, mocked at the API layer.

---

## Verification

Run in order after all tasks (or per-phase as work lands):

1. `uv run ruff format --check` on modified Python files.
2. `uv run ruff check` on modified Python files.
3. `uv run pyright` on modified Python files.
4. `uv run pytest tests/`.
5. `cd frontend && npx tsc --noEmit` (Phases C and D).
6. `cd frontend && npx playwright test frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts` (Phases C and D).
7. Manual acceptance per Task 12 against a real Event Encoder job.
