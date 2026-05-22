# Piano Roll MPE & Ridge-Aligned Note Extractor — Implementation Plan

**Goal:** Replace the independent CQT peak tracker in Piano Roll Notes with a
ridge-aware F0 + harmonics extractor that produces coherent-contour notes with
sub-semitone pitch trajectories, persisted in parquet sidecars and consumed
by both an MPE Standard MIDI File export and a curved-ribbon Piano Roll
renderer.

**Spec:** [docs/specs/2026-05-22-piano-roll-mpe-ridge-aligned-design.md](../specs/2026-05-22-piano-roll-mpe-ridge-aligned-design.md)
**Primary domain:** sequence-models
**Neighbor domains:** core-platform, frontend-shell

The plan is organized in five phases that mirror the spec's §11 phase
breakdown. Each phase produces a coherent, shippable change. Phases 1–3 are
backend-only with no user-visible change; Phase 4 is the user-visible cut;
Phase 5 ships the docs alongside Phase 4.

---

## Phase 1 — Encoder ridge sidecar

### Task 1: Extract STFT ridge tracker into a shared module

**Files:**
- Create: `src/humpback/processing/ridge_path.py`
- Modify: `src/humpback/sequence_models/event_encoder.py`
- Create: `tests/processing/test_ridge_path.py`

**Acceptance criteria:**
- [ ] `humpback.processing.ridge_path` exports `compute_ridge_path()` and
  `RidgePathResult` matching the spec §5.1 signature.
- [ ] `RidgePathResult` is a frozen dataclass with fields `log_frequencies`,
  `frame_times`, `strengths`, `energy_ratios`, `total_frames`.
- [ ] `event_encoder.py` imports `compute_ridge_path` from the new module
  and removes the in-file `_compute_ridge_path_result` helper.
- [ ] The encoder's existing descriptor outputs are byte-identical to those
  produced before extraction for any unchanged input.
- [ ] `compute_ridge_path()` accepts the new `max_frequency_hz` default of
  6000 Hz (encoder caller passes this) and a caller-overrideable ceiling.
- [ ] No public path in the encoder's API surface changes.

**Tests needed:**
- Empty `spectra` input returns an empty `RidgePathResult` with
  `total_frames == 0`.
- Single-frame input returns the empty result (matches existing behavior).
- Synthetic constant-frequency tone yields a ridge within ±1 STFT bin of
  the true frequency at every frame.
- Synthetic linear sweep yields a slope within tolerance of the input slope.
- Multi-candidate frame where the loudest candidate is far from the
  Viterbi-favored continuation: the smooth path wins, not the loud one.
- Regression: a fixture passing existing event-encoder ridge descriptor
  expectations produces identical output values via the new module.

---

### Task 2: Event Encoder persists per-event ridge sidecar

**Files:**
- Modify: `src/humpback/workers/event_encoder_worker.py`
- Modify: `src/humpback/storage.py`
- Create or modify: `tests/sequence_models/test_event_encoder_worker_ridges.py`

**Acceptance criteria:**
- [ ] `storage.event_encoder_ridges_path(storage_root, job_id, tokenizer_version)`
  returns
  `event_encoders/{job_id}/event_ridges_{tokenizer_version}.parquet`.
- [ ] The encoder worker accumulates per-event ridges as it runs and writes
  one parquet at job end with the schema in spec §6.1
  (`event_id`, `frame_index`, `frame_time_offset_s`, `log_frequency`,
  `strength`, `energy_ratio`).
- [ ] Rows are sorted by `(event_id, frame_index)`.
- [ ] Write is atomic (`.tmp` then rename); the `.tmp` is removed on
  exception.
- [ ] Encoder job's `params_json` (or manifest, whichever already lists
  artifact paths) gains the ridge sidecar path.
- [ ] An encoder job that previously completed without writing the sidecar
  is unaffected (no migration).

**Tests needed:**
- Worker run on a synthetic-audio fixture writes
  `event_ridges_{tokenizer_version}.parquet`; the file exists, has non-zero
  size, and its `event_id` set matches `event_vectors.parquet`.
- Per-event row counts match each event's frame count.
- Partial-failure simulated mid-run leaves no `.parquet.tmp` behind.
- Idempotent re-run produces the same bytes.

---

## Phase 2 — v3 notes extractor (dual-write)

### Task 3: Implement the v3 ridge-aware note extractor

**Files:**
- Create: `src/humpback/processing/note_extractor_v3.py`
- Create: `tests/processing/test_note_extractor_v3.py`

**Acceptance criteria:**
- [ ] Public entrypoint
  `extract_notes_v3(audio, sample_rate, *, params, ridge_sidecar_rows)` returns
  a `NotesV3Result` with `notes: list[NoteV3]` and
  `contours: list[ContourFrame]`. `notes` and `contours` are aligned by
  `note_uid`.
- [ ] When `ridge_sidecar_rows` is `None`, the extractor recomputes ridges
  in-process via `humpback.processing.ridge_path.compute_ridge_path()` using
  the wider 8500 Hz ceiling (spec §5.1).
- [ ] Subharmonic refinement (spec §5.2) is implemented with `k_sub = 2.0`
  and 5-frame majority-vote smoothing; per-frame `subharmonic_octave` is
  exposed on the contour rows.
- [ ] Coherent-contour segmentation (spec §5.3) splits the F0 contour into
  notes only on energy gaps (≥ 3-frame floor) or surviving octave jumps.
- [ ] Harmonic siblings (spec §5.4) emit at `n ∈ {2..16}` when the CQT peak
  near `n · f₀(t)` is within ±75 ¢ of the integer multiple; harmonic notes
  carry `partial_index = n - 1` and their contour reuses F0's `cents_from_pitch`.
- [ ] No `partial_index = -1` rows are ever produced.
- [ ] `midi_pitch` clamps to `[12, 120]`; cents are clamped to `[-9600,
  9600]`.
- [ ] Velocity calibration uses peak STFT magnitude per spec §5.5.
- [ ] `note_uid` is deterministic UUID v5 from
  `(job_id, event_id, partial_index, track_id, start_utc_rounded_ms)`.
- [ ] The extractor never depends on the `HarmonicParams` dataclass from
  `piano_roll_tracker.py`.

**Tests needed:**
- Synthetic sine sweep from C5 to E5 over 200 ms: produces exactly one F0
  note; the contour traces the sweep within ±5 cents at every frame.
- Synthetic harmonic stack (F0 = 220 Hz + H2..H5 at −6 dB each): produces
  exactly 5 notes with `partial_index ∈ {0, 1, 2, 3, 4}` and correct
  midi pitches.
- Loud-H2 / quiet-F0 input (220 Hz weak fundamental with 440 Hz dominant
  ridge): refined F0 sits one octave below the dominant ridge — note's
  `midi_pitch` corresponds to 220 Hz, not 440 Hz.
- Borderline subharmonic frames don't flip per-frame after smoothing.
- Two-event synthetic with a 100 ms silent gap → 2 notes, not 1.
- Single event containing a deliberate octave register shift surviving the
  5-frame smoothing → 2 notes.
- Sub-30 ms event yields zero notes.
- Velocity over a 20 dB-louder fixture event maps to higher velocity in the
  proportional range while remaining ≥ 1.
- Determinism: identical inputs → identical `note_uid` set and identical
  contour bytes.
- A track whose CQT peaks never align with any integer multiple of F0
  emits no harmonic note (no `partial_index = -1` regression).

---

### Task 4: Piano Roll Notes worker v3 branch and contour sidecar

**Files:**
- Modify: `src/humpback/workers/piano_roll_notes_worker.py`
- Modify: `src/humpback/storage.py`
- Modify: `src/humpback/schemas/piano_roll_notes.py`
- Modify: `tests/workers/test_piano_roll_notes_worker.py`

**Acceptance criteria:**
- [ ] `storage.event_encoder_note_contours_path(storage_root, job_id,
  extractor_version)` returns
  `event_encoders/{job_id}/event_note_contours_{extractor_version}.parquet`.
- [ ] The worker branches on `job.extractor_version`: `"v2"` (and older)
  continues to call the existing pipeline; `"v3"` calls
  `note_extractor_v3.extract_notes_v3()`.
- [ ] On a v3 run, the worker reads the encoder's ridge sidecar via
  `event_encoder_ridges_path()`; if absent, it passes `None` so the
  extractor recomputes.
- [ ] On a v3 run, the worker writes both
  `event_notes_v3.parquet` (one row per note with the new `note_uid`,
  `f0_track_id`, `contour_frame_count` columns per spec §6.2) and
  `event_note_contours_v3.parquet` (one row per frame per note per spec §6.3).
- [ ] Both files are written via `.tmp` + atomic rename; if either fails,
  both are cleaned up.
- [ ] `params_json` records `contours_path`, `ridges_path` (or the literal
  string `"absent"` when fallback used), and `n_contour_frames`.
- [ ] `PianoRollNote` schema gains `note_uid: str | None`,
  `f0_track_id: int | None`, `contour_frame_count: int | None` so existing
  v1/v2 responses still deserialize.
- [ ] `DEFAULT_EXTRACTOR_VERSION` remains `"v2"` until Phase 4 — v3 only
  runs when callers explicitly pass it.

**Tests needed:**
- v3 job on a synthetic event-encoder fixture writes both parquets with the
  correct schema and non-empty rows.
- Idempotency: `(job_id, "v3")` re-submit on `complete` is a no-op, on
  `running` raises 409, on `failed`/`canceled` resets.
- Ridge sidecar present → worker reads it (assert via a spy that
  `compute_ridge_path()` is not called).
- Ridge sidecar absent → worker recomputes in-process and still succeeds.
- Partial-failure simulation (raise inside the writer): both parquets
  removed; row status `failed`; no `.tmp` left behind.
- A v2 job submitted alongside a v3 job runs the legacy path unchanged.

---

## Phase 3 — MPE MIDI synthesis

### Task 5: MPE-aware MIDI synthesizer with v2 fallback

**Files:**
- Modify: `src/humpback/processing/midi_synthesis.py`
- Modify: `src/humpback/workers/piano_roll_midi_export_worker.py`
- Modify: `tests/processing/test_midi_synthesis.py`
- Modify: `tests/workers/test_piano_roll_midi_export_worker.py`

**Acceptance criteria:**
- [ ] `notes_table_to_midi_bytes()` accepts both v2 parquet shape (no
  `note_uid` column) and v3 shape (with `note_uid`). For v2 it emits the
  existing slim 7-channel layout (regression guard). For v3 it emits MPE
  per spec §8.
- [ ] When called for v3, the function takes a second positional argument
  `contour_table: pa.Table` (per spec §6.3) and a keyword
  `time_origin_utc: float | None = None` matching the existing windowed-
  export semantics.
- [ ] MPE Master track emits RPN 6 with payload `15` once at tick 0, plus
  per-member-channel RPN 0/0 + Data Entry MSB `24` once at tick 0.
- [ ] Per-note channel allocator is deterministic on `(start_utc, note_uid)`
  with longest-idle pick and FIFO voice steal (spec §8.2). Steal count is
  exposed on the return value (or via the worker's `params_json`).
- [ ] Each note emits `program_change`, `CC 74`, `note_on`, a
  bend-decimated `pitch_bend` stream, `note_off` on its allocated channel
  per spec §8.3 and §8.4.
- [ ] Harmonic notes' bend stream in cents equals their parent F0's bend
  stream in cents (cents conservation).
- [ ] Empty voice tracks still emit `track_name` + `end_of_track`.
- [ ] SMF parses cleanly via `mido.MidiFile()` without warnings.
- [ ] Identical input parquets + contour sidecars produce byte-identical
  output across repeated calls.
- [ ] The export worker reads the contour sidecar (via
  `event_encoder_note_contours_path()`) when synthesizing v3; missing sidecar
  causes a `failed` row with a clear `error_message`.
- [ ] Atomic write of the `.mid` artifact and partial-failure cleanup are
  preserved from ADR-066 / ADR-068.
- [ ] The export worker's window-clipping logic from ADR-068 still applies
  to v3 notes (a partial note whose `start_utc + duration_s` extends past
  `window_end_utc` is truncated; same sub-millisecond residual drop).

**Tests needed:**
- MPE Configuration Message present at master track tick 0 with the
  correct payload.
- Per-member channel bend-range setup messages emit before any note on
  that channel.
- Channel allocator: deterministic output on a 20-note fixture with mixed
  start times.
- Voice stealing: 17-simultaneous-notes fixture forces a steal; the
  stolen note has an explicit `note_off` emitted at the steal tick.
- Per-note `program_change`, CC 74, and master-track `MetaMessage("text",
  "pN")` land on the expected channels and ticks.
- Harmonic bend stream in cents equals parent F0's bend stream in cents.
- Round-trip: synthesized bytes parse via `mido.MidiFile()` with track
  count exactly `1 + 1 + 15`.
- v2-shape input (no `note_uid`) still produces the existing slim 7-channel
  layout (regression guard).
- Determinism: identical input → byte-identical output.
- Worker: v3 export reads contour sidecar; missing sidecar fails with a
  specific error message; `force=true` resets a `complete` row.
- Worker: windowed export of v3 notes clips notes correctly to
  `[window_start_utc, window_end_utc)`.

---

## Phase 4 — Default bump and frontend ribbon rendering

### Task 6: Contours API endpoint

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `src/humpback/services/piano_roll_notes_service.py`
- Modify: `src/humpback/schemas/piano_roll_notes.py`
- Create: `tests/api/test_sequence_models_notes_contours.py`

**Acceptance criteria:**
- [ ] `GET /sequence-models/event-encoders/{job_id}/notes/contours`
  accepts `note_uids` as a repeated query param or POST body (matching
  existing patterns in this router); responds with
  `PianoRollNoteContourResponse` per spec §7.
- [ ] Cap at 2000 `note_uids` per request; above cap returns HTTP 413 with
  a diagnostic body.
- [ ] When the requested job has no v3 contour sidecar, return HTTP 422
  with a clear message.
- [ ] Unknown `note_uid` in a request that's otherwise valid is omitted
  from the response (no 404 — partial misses are not errors).
- [ ] `GET .../notes` response payload gains the three new note-row fields;
  legacy v1/v2 responses keep returning `null` for them.

**Tests needed:**
- Contour fetch for known `note_uid`s returns the expected per-frame rows
  in `(frame_index)` ascending order.
- Missing v3 sidecar → 422.
- Request with > 2000 `note_uids` → 413.
- Mix of known and unknown `note_uids` → known ones populated, unknown
  silently dropped from the response.
- Existing `.../notes` endpoint regression suite still passes; v2 responses
  show `note_uid: null`.

---

### Task 7: Frontend ribbon renderer and Y-axis extension

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`
- Modify: `frontend/src/components/sequence-models/MidiExportButton.tsx`
- Modify: `frontend/src/components/sequence-models/PianoRollNotesStatusPill.tsx`
- Modify: `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts`
- Create: `frontend/e2e/sequence-models/event-encoder-piano-roll-perf.spec.ts`

**Acceptance criteria:**
- [ ] `usePianoRollNoteContours(jobId, noteUids, enabled)` hook batches
  requests against `/notes/contours` and caches per `note_uid` via React
  Query.
- [ ] `EventEncoderPianoRollPage.tsx` computes `noteUids` from
  `visibleNotes`; the hook only fetches uncached uids; panning never
  triggers a re-fetch of cached uids.
- [ ] `drawNoteRibbon` helper exists and renders polyline-ribbons per spec
  §9.3. Velocity modulates stroke opacity.
- [ ] Notes without a fetched contour render as the existing flat bar
  (fallback); they hydrate into ribbons when contours arrive.
- [ ] If `/notes/contours` returns 500 for a session, a non-blocking toast
  fires once: "Some notes are showing flat — contour fetch failed."
- [ ] Y-axis range becomes MIDI 12–120 with new octave labels C0, C9, G9;
  88-key range retains current shading; extended bands render with a
  desaturated background tint.
- [ ] Hover tooltip gains `Δpitch: ±N¢` summary.
- [ ] Hit-testing uses ≤ 6 px polyline distance.
- [ ] "Download MIDI" tooltip mentions MPE / DAW compatibility.
- [ ] Export status panel displays `Format: MPE v3` below the file size.
- [ ] `PianoRollNotesStatusPill` shows a "v3 available" badge when the
  encoder has a v2 sidecar but no v3 yet; clicking it POSTs a v3 notes job.
- [ ] `npx tsc --noEmit` clean.

**Tests needed:**
- Notes view renders ribbons for a v3 job; assert canvas pixel at a known
  ribbon midpoint matches token color.
- Pan / zoom never re-fetches a cached `note_uid` (assert via mocked
  fetch counter).
- Forcing 500 on `/notes/contours` falls back to flat bars and shows the
  toast exactly once.
- Y-axis: labels C0…G9 present; extended-band tint asserted on a sentinel
  pixel.
- MPE Format tooltip text on the Download MIDI link.
- "v3 available" badge click → notes-job status transitions queued →
  complete in the mocked backend.
- Perf spec: synthetic backend with 10k visible notes × 10 frames each;
  Playwright tracing shows ≥ 30 fps median during a pan gesture.

---

### Task 8: Default extractor version bump and auto-enqueue integration

**Files:**
- Modify: `src/humpback/models/piano_roll_notes.py`
- Modify: `src/humpback/workers/event_encoder_worker.py` (auto-enqueue
  call site, if it pins a version)
- Modify: `src/humpback/services/piano_roll_midi_export_service.py` (version
  resolver, if pinned)
- Modify: `tests/workers/test_piano_roll_notes_worker.py`

**Acceptance criteria:**
- [ ] `DEFAULT_EXTRACTOR_VERSION = "v3"`.
- [ ] The encoder's auto-enqueue hook now creates a v3 notes job for any
  newly-completing encoder job.
- [ ] The export-service's "latest complete notes version" resolver still
  returns the highest `complete` version per string comparison, which is
  `"v3"` once a v3 job is complete and `"v2"` (or earlier) otherwise.
- [ ] Existing v1/v2 notes parquets on disk remain readable; nothing
  rewrites them.
- [ ] Auto-enqueue still swallows conflicts so a re-completing encoder job
  doesn't block on an in-flight or completed notes job.

**Tests needed:**
- A completing encoder job auto-enqueues a notes job with
  `extractor_version = "v3"`.
- An existing v2 notes row remains visible via API and continues to
  resolve the v2 MIDI export.
- Once v3 completes for the same encoder job, the export resolver picks v3
  on the next export-creation call.
- `models.piano_roll_notes.DEFAULT_EXTRACTOR_VERSION` is referenced (not
  duplicated) by callers; no stray hard-coded `"v3"` literals outside the
  constant.

---

## Phase 5 — Documentation and ADR-069

### Task 9: ADR-069, capsule, and reference doc updates

**Files:**
- Modify: `DECISIONS.md`
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/invariants.md`
- Modify: `docs/agent-context/current-state.md`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/specs/2026-05-20-piano-roll-midi-notes-design.md` (add
  "Superseded in part" header).
- Modify: `docs/specs/2026-05-20-event-encoder-midi-channelized-design.md`
  (add "Superseded in part" header).
- Modify: `docs/specs/2026-05-20-piano-roll-midi-export-design.md` (add
  "Superseded in part" header).

**Acceptance criteria:**
- [ ] **ADR-069** appended to `DECISIONS.md` covering: STFT-ridge as
  canonical F0 source, encoder ridge sidecar, coherent-contour note model,
  MPE Lower Zone export, MIDI 12–120 pitch range, no v1/v2 backward
  compatibility, ribbon rendering, retirement of `HarmonicParams`.
- [ ] Sequence-models capsule README updated: artifact roots include
  `event_ridges_*.parquet`, `event_note_contours_v3.parquet`,
  `notes_v3.mid`; the slim 7-channel description is replaced with the MPE
  Lower Zone description; Notes view defaults to ribbon rendering at MIDI
  12–120.
- [ ] Sequence-models invariants updated: Piano Roll Notes pitch contours
  are sub-semitone and persisted; encoder ridge contours are persisted and
  consumed by downstream sidecar workers; harmonic notes inherit their
  parent F0's bend trajectory in cents; the `HarmonicParams` dataclass and
  the related labeling pass are retired.
- [ ] `current-state.md` gains a one-line update: "Piano Roll Notes now
  produces MPE-ready ridge-aligned contours; export uses MPE Lower Zone."
- [ ] `sequence-models-api.md` documents the new `/notes/contours`
  endpoint with request and response shapes and notes the MPE format on
  the existing `/midi-export` endpoint.
- [ ] `storage-layout.md` adds the two new sidecar paths and the v3 export
  filename.
- [ ] The three prior specs gain a "Superseded in part" header pointing
  back to this spec.

**Tests needed:**
- No automated tests; manual doc render check.
- Capsule grep for stale references to the slim 7-channel layout returns
  zero matches.
- `docs/agent-context/current-state.md` and `DECISIONS.md` parse cleanly
  (no broken markdown links).

---

## Verification

Run in order after all phases land on the feature branch (or per phase
before merge):

1. `uv run ruff format --check $(git diff --name-only main -- '*.py')`
2. `uv run ruff check $(git diff --name-only main -- '*.py')`
3. `uv run pyright $(git diff --name-only main -- '*.py')` — full run when
   `src/humpback/processing/`, `src/humpback/schemas/`, or
   `src/humpback/models/` files changed.
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit` (when frontend files changed)
6. `cd frontend && npx playwright test e2e/sequence-models/event-encoder-piano-roll*`
   (when UI files changed)

Manual acceptance on job `2679ab0d-4467-43f6-9113-a31439a89329`, in order,
before Phase 4 merges:

1. Phase 1 + 2 produce the ridge sidecar and the v3 notes parquet; the
   `partial_index` distribution shifts away from 66 % F0 dominance and
   `n_notes` falls into the 30–50k band.
2. Phase 3 export opens in Logic Pro with the MPE indicator illuminated;
   a known sweep auditions with audible pitch bend; CC 74 visible on the
   per-voice automation lane.
3. Phase 4 frontend draws ribbons that visibly track the spectrogram
   ridges; the staircase artifact from the original screenshot is gone.
