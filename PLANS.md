# Development Plans

---

## Active

(none)

---

## Recently Completed

# Plan: Detection Spectrogram Popup

## Outcome (2026-03-09)

- Added `matplotlib>=3.8` dependency and spectrogram config settings
  (`spectrogram_hop_length`, `spectrogram_dynamic_range_db`, `spectrogram_width_px`,
  `spectrogram_height_px`, `spectrogram_cache_max_items`) to `Settings`.
- Created `processing/spectrogram.py` (STFT via scipy, matplotlib Agg rendering to PNG)
  and `processing/spectrogram_cache.py` (FIFO disk cache with atomic writes).
- Refactored `get_detection_audio_slice` into shared `_resolve_detection_audio()` helper
  for both audio-slice and spectrogram endpoints.
- Added `GET /classifier/detection-jobs/{job_id}/spectrogram` endpoint returning cached PNG.
- Frontend: `SpectrogramPopup` component with loading spinner, viewport-aware positioning,
  click-to-dismiss overlay. Alt+click on detection rows in `HydrophoneContentTable` triggers popup.
- Added `detectionSpectrogramUrl()` to API client.

## Verification

- `uv run pytest tests/` — 440 passed.
- `cd frontend && npx tsc --noEmit` — passed.
- 8 unit tests (generator + cache) + 2 integration tests (404 + PNG response) added.

# Plan: Enhanced Classifier/Train Expanded Model Details

## Outcome (2026-03-09)

- Restructured expanded model row into separated Training Parameters (classifier type,
  class weight, L2 normalize, regularization C, effective weights) and Performance
  (accuracy, ROC AUC, precision, recall, F1 with ±std, n-fold CV, score separation
  with color hints, mean scores, 2x2 confusion matrix) sections.
- Added lazy-loaded Training Data section via `useTrainingDataSummary` hook showing
  positive/negative source folders grouped by top-level parent name with set/vector counts.
- Backend: enriched `/models/{id}/training-summary` response with `filename` and
  `folder_path` fields (joined from AudioFile).

## Verification

- `uv run pytest tests/` — 430 passed.
- `cd frontend && npx tsc --noEmit` — passed.

# Plan: Retrain Classifier Model

## Outcome (2026-03-09)

- Added `retrain_workflows` table (migration `014`) with backend state machine
  (`queued` → `importing` → `processing` → `training` → `complete`/`failed`).
- Implemented folder-tracing algorithm to resolve import roots from training job
  embedding set provenance; embedding set collection gathers ALL sets from folder
  hierarchies (includes newly added audio files).
- Added 4 API endpoints: `GET /retrain-info`, `POST /retrain`,
  `GET /retrain-workflows`, `GET /retrain-workflows/{id}`.
- Worker integration: retrain polling in main loop after extraction jobs,
  stale recovery for importing/processing/training states.
- Frontend: `RetrainPanel` component in expanded model rows with step indicator,
  progress tracking, and form for starting retrains.
- ADR-016 documents the design.

## Verification

- `uv run pytest tests/` — 430 passed.
- `cd frontend && npx tsc --noEmit` — passed.
- 12 unit tests + 8 integration tests + Playwright spec added.

# Plan: Popover Date Range Picker for Hydrophone Tab

## Outcome (2026-03-09)

- Replaced two manual text `<Input>` fields (Start/End Date/Time UTC) with a single
  `DateRangePickerUtc` popover component: dual-month calendar, HH:MM time inputs,
  Apply/Cancel buttons, UTC-only semantics via fake-local Date strategy.
- Added shadcn primitives `popover.tsx` and `calendar.tsx` (react-day-picker v9 +
  @radix-ui/react-popover); composed `DateRangePickerUtc.tsx` in `shared/`.
- Removed `parseDatetimeLocalAsUtcSeconds` from HydrophoneTab (replaced by epoch state).
- Removed 11 broken Detect-tab UI Playwright tests (API-level tests retained).
- Updated hydrophone UTC and progress-format Playwright tests for new picker interaction.

## Verification

- `cd frontend && npx tsc --noEmit` passed.
- `cd frontend && npx playwright test e2e/hydrophone*` — 16 passed.
- `uv run pytest tests/` — 410 passed.


# Plan: Hydrophone Tab Improvements + Detect Tab Removal

## Outcome (2026-03-09)

- Removed Classifier/Detect sub-tab; Classifier page now has Train + Hydrophone only.
- Added pause/resume/cancel controls for hydrophone jobs with `threading.Event` pause gate.
- Changed date picker to 24hr UTC text input (`YYYY-MM-DD HH:MM`), progress display to `Xh YYm`.
- Added `hydrophone_name` column to hydrophone detection TSV output.
- Made canceled jobs fully functional in Previous Jobs (expand, download, labels, extract).
- Added ADR-015 for the combined changes.

## Verification

- `uv run pytest tests/` passed (`410` passed).
- `cd frontend && npx tsc --noEmit` passed.
- `cd frontend && npx playwright test` — 15 hydrophone tests passed.

---

# Plan: Hydrophone Detection UI — UTC-Only Input + Date Range Semantics

## Outcome (2026-03-09)

- Interpreted Hydrophone `datetime-local` Start/End inputs as UTC at submit time
  (no local-time conversion), matching backend/API timestamp semantics.
- Updated Hydrophone date labels and displays to explicit UTC wording:
  `Start Date/Time (UTC)`, `End Date/Time (UTC)`, and `Date Range (UTC)`.
- Added a UTC regression Playwright spec that mocks Hydrophone APIs and verifies
  submitted epoch values for `2025-07-04T09:00`/`10:00` and UTC-rendered job range text.

## Verification

- `cd frontend && npx tsc --noEmit` passed.
- `cd frontend && npx playwright test e2e/hydrophone-utc-timezone.spec.ts` passed (`2` passed).
- `uv run pytest tests/` passed (`407` passed).

---

# Plan: Hydrophone Detection — Configurable Lookback + Explicit No-Audio Failure

## Outcome (2026-03-09)

- Replaced the fixed hydrophone folder-lookback heuristic with configurable
  timeline expansion (`4h` increment, `168h` max by default), so ranges that
  start hours after a folder timestamp can still load overlapping audio.
- Kept timeline clipping authoritative for `[start_timestamp, end_timestamp]`
  while switching true no-overlap outcomes to explicit failures.
- Propagated no-audio timeline misses as `FileNotFoundError` and updated the
  hydrophone worker to persist a clear `error_message` with hydrophone ID and
  requested UTC range.
- Added regression coverage for long-folder overlap, bounded clipping, explicit
  no-audio failure, and hydrophone worker failure/success progress behavior.

## Verification

- `uv run pytest tests/unit/test_s3_stream.py -q` passed.
- `uv run pytest tests/unit/test_extractor.py -q` passed.
- `uv run pytest tests/unit/test_classifier_worker.py -q` passed.
- `uv run pytest tests/integration/test_hydrophone_api.py -q` passed.
- `uv run pytest tests/` passed (`407` passed).

---

# Plan: Hydrophone Detection UI — Clip Timing Consistency (Display + Playback + Extract)

## Outcome (2026-03-09)

- Added a single Hydrophone clip-timing resolver with precedence:
  `extract_filename` -> snapped window bounds -> raw detection span.
- Switched Hydrophone Detection Range primary display to extraction-aligned UTC range,
  with raw detection range retained as secondary audit text/tooltip.
- Updated Hydrophone playback to use resolved extraction-aligned clip start/duration so
  play length now matches displayed Duration and extraction output.
- This supersedes the earlier Hydrophone UI behavior that showed raw range as the
  primary Detection Range value.

## Verification

- `cd frontend && npx tsc --noEmit` passed.
- `cd frontend && npx playwright test e2e/hydrophone-extract.spec.ts` passed (`1` passed).
- `uv run pytest tests/` passed (`399` passed).

---

# Plan: Hydrophone Detection UI Label Clarity + Extract Filename Metadata

## Outcome (2026-03-09)

- Kept Hydrophone `Detection Range` as raw UTC span while adding extraction basename
  (`extract_filename`) to hydrophone TSV rows and content API responses.
- Updated Hydrophone content table to replace `Start (s)` + `End (s)` with
  `Duration (s)` computed from snapped extraction bounds; tooltip now shows
  extraction filename (with legacy fallback derivation when missing).
- Made label-save TSV rewrites preserve unknown/extra columns so `extract_filename`
  is not dropped during annotation updates.

## Verification

- `uv run pytest tests/` passed (`399` passed).
- `cd frontend && npx tsc --noEmit` passed.
- `cd frontend && npx playwright test e2e/hydrophone-extract.spec.ts` passed (`1` passed).

---

# Plan: Hydrophone Detection — Timeline-Correct Segment Assembly + End-Bounded Range

## Outcome (2026-03-09)

- Replaced lexicographic `.ts` ordering with numeric segment-suffix ordering across
  S3/local/cache clients to prevent mixed-width key jumps (e.g. `live100` vs `live1000`).
- Added playlist (`live.m3u8`) duration parsing and unified stream timeline construction
  for hydrophone detection, playback (`/audio-slice`), and extraction.
- Bounded timeline consumption to each job's `[start_timestamp, end_timestamp]` range so
  detection/playback/extraction no longer spill past requested end times.
- Kept legacy playback compatibility via `job.start_timestamp` anchor fallback for older jobs.
- Added regression coverage for mixed-width ordering, end-bound clipping, and resolver continuity.

## Verification

- `uv run pytest tests/unit/test_s3_stream.py -q` passed (`15` passed).
- `uv run pytest tests/integration/test_hydrophone_api.py -q` passed (`14` passed).
- `uv run pytest tests/` passed (`395` passed).

---

# Plan: Classifier/Hydrophone Extract — Hydrophone-Partitioned Output Paths

## Outcome (2026-03-09)

- Updated hydrophone labeled-sample extraction paths to include hydrophone short label
  (`hydrophone_id`) under both positive and negative roots:
  `{positive|negative}_root/{hydrophone_id}/{label}/YYYY/MM/DD/*.wav`.
- Preserved local (non-hydrophone) extraction path behavior.
- Added unit coverage for hydrophone positive/negative path routing, including a guard
  that old non-partitioned hydrophone negative paths are no longer used.

## Verification

- `uv run pytest tests/unit/test_extractor.py -q` passed (`21` passed).
- `uv run pytest tests/` passed (`390` passed).

---

# Plan: Hydrophone Tab — Playback Timestamp Mapping + Saved-Label Extract Activation

## Outcome (2026-03-08)

- Implemented shared hydrophone stream-offset audio-slice resolver with anchor order:
  first available folder timestamp, then legacy `job.start_timestamp`.
- Switched hydrophone extraction to the same resolver path and passed detection
  job stream bounds (`start_timestamp`, `end_timestamp`) through worker plumbing.
- Updated Hydrophone tab extract enablement to use saved labels on the expanded
  completed job only; Extract dialog now targets that single job.

## Verification

- `uv run pytest tests/` passed (`389` passed).
- `cd frontend && npx tsc --noEmit` passed.
- `cd frontend && npx playwright test` ran:
  - New hydrophone regression test passed (`frontend/e2e/hydrophone-extract.spec.ts`).
  - Existing unrelated failures remain in classifier training selectors
    (`frontend/e2e/classifier-training.spec.ts`) and one slider-fill test
    (`frontend/e2e/detection-hysteresis.spec.ts`).

---

# Plan: HydrophoneTab — Live Detection Content + Save/Extract Labels


---

### Verification

1. **Live content during running job:**
   - Start a hydrophone detection job
   - Confirm the active job panel shows detection rows once `segments_processed > 0`
   - Confirm rows update every 3s as new detections arrive
   - Confirm sort switches from filename/asc → confidence/desc on completion

2. **Save Labels:**
   - Expand a completed job, toggle label checkboxes
   - Confirm Save Labels button enables (dirty state)
   - Click Save Labels, confirm it saves and button disables
   - Confirm Save Labels is disabled while a running job is expanded

3. **Extract Labeled Samples:**
   - Select completed jobs via checkboxes
   - Click Extract, confirm dialog opens with path fields
   - Submit extraction, confirm extract_status badge appears

4. **Type-check:** `cd frontend && npx tsc --noEmit`
5. **Existing tests:** `cd frontend && npx playwright test`

---

# Plan: Classifier/Hydrophone Tab — S3 HLS Streaming Detection

---

## Verification

1. **Unit test S3 client** — mock boto3, verify folder listing, segment fetching, retry on 503
2. **Unit test hydrophone detector** — mock S3 client + fake embedding model, verify detections, cancel support, alert propagation
3. **Integration test API** — create hydrophone job, verify DB state, list/cancel endpoints
4. **Manual E2E** — start backend + worker, create hydrophone job via UI with real Orcasound data, verify progress updates, flash alerts, detection results, audio playback, label auto-save, stop button
5. **Playwright test** — verify Hydrophone tab renders, form submits, active job panel shows progress, previous jobs panel expandable with content


---

## Backlog

- Explore GPU-accelerated batch processing for large audio libraries
- Add WebSocket push for real-time job status updates (replace polling)
- Investigate multi-model ensemble clustering
- Optimize `/audio/{id}/spectrogram` window fetch path to avoid materializing all windows when only one index is requested (reduce memory/time on long files)
- Optimize hydrophone incremental lookback discovery to avoid repeated full S3
  folder scans at each lookback step (reduce first-segment startup latency)

---

## Completed

- Overlap-back windowing (ADR-001)
- In-place folder import (ADR-002)
- Balanced class weights for detection (ADR-003)
- Negative embedding sets for training (ADR-004)
- Multi-agent memory framework migration
- Overlapping window inference + hysteresis event detection (ADR-005)
- Incremental detection rendering with per-file progress (ADR-006)
- Fix escalating false positives: MLP classifier + diagnostics (ADR-007)
- Queue claim hardening + API validation pass (P0-P2): atomic compare-and-set claims for all worker job types, strict input validation, robust Range parsing, overlap-back-aligned spectrogram offsets (ADR-009)
- Multi-model support: grouping, filtering & validation
- Fix TFLite/TF2 vector_dim mismatch: auto-detect from model output
- Optimize TFLite encoding: batch inference via resize_tensor_input + multi-threading + timing instrumentation
- Optimize spectrogram extraction: vectorized STFT via batched `np.fft.rfft` (10.9x feature speedup) + TFLite inference batch_size tuned to 64 (6.2x vs sequential)
- S3 HLS Streaming Detection (Hydrophone Tab): Orcasound hydrophone integration with in-memory S3 streaming, cancel support, flash alerts, auto-save labels
- S3 Caching, UTC Display & WAV Export: CachingS3Client with write-through cache + 404 markers, UTC range display in detection table, WAV export for hydrophone jobs
