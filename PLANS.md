# Development Plans

---

## Active

- None

---

## Recently Completed

# Plan: Hydrophone Detection Ordered S3 Prefetch + Timing Telemetry

## Outcome (2026-03-11)

- Added ordered bounded concurrent segment prefetch to hydrophone streaming
  (`iter_audio_chunks`) with configurable worker/in-flight limits while preserving
  timeline order and existing warning/retry behavior.
- Enabled prefetch for S3-backed hydrophone detection runs only (direct S3 and
  write-through cache clients), with local-cache detection behavior unchanged.
- Added hydrophone run-summary timing fields (`fetch_sec`, `decode_sec`,
  `features_sec`, `inference_sec`, `pipeline_total_sec`) plus prefetch metadata.
- Added new runtime settings:
  `hydrophone_prefetch_enabled`, `hydrophone_prefetch_workers`,
  `hydrophone_prefetch_inflight_segments`.
- Added unit regressions for prefetch ordering, in-flight bound enforcement, and
  error continuation behavior.

## Verification

- `uv run pytest tests/unit/test_s3_stream.py tests/unit/test_hydrophone_resume.py tests/unit/test_classifier_worker.py -q` — 46 passed.
- `uv run pytest tests/` — 476 passed.

# Plan: Paused Hydrophone Playback + Spectrogram Resolution Fix

## Outcome (2026-03-11)

- Fixed `/classifier/detection-jobs/{id}/content` status gating to allow paused jobs
  with partial TSV output (`running|paused|complete|canceled`).
- Fixed sparse local-cache timeline reconstruction in `s3_stream.py` by preserving
  playlist-derived segment offsets (instead of assuming first cached segment starts at
  folder timestamp), restoring local playback/spectrogram resolution for paused jobs.
- Updated Hydrophone active table behavior to treat paused jobs as non-running in UI
  content polling/sorting mode, and added unit/integration regressions.

## Verification

- `uv run pytest tests/unit/test_s3_stream.py tests/integration/test_hydrophone_api.py -q` — 49 passed.
- `uv run pytest tests/` — 473 passed.
- `cd frontend && npx tsc --noEmit` — passed.
- `cd frontend && npx playwright test e2e/hydrophone-pause-resume.spec.ts` — 6 passed.

# Plan: Snapped Canonical Detection Ranges

## Outcome (2026-03-10)

- Canonicalized detection ranges to snapped window-aligned bounds before labeling for local
  and hydrophone workflows, while preserving unsnapped audit metadata
  (`raw_start_sec`, `raw_end_sec`, `merged_event_count`).
- Aligned preview/label/extract behavior by using canonical bounds end-to-end, with
  hydrophone legacy normalization precedence:
  `detection_filename -> extract_filename -> snapped-from-range`.
- Fixed download scalability regression by streaming normalized TSV rows instead of buffering
  full files in memory.

## Verification

- `uv run pytest tests/` — 471 passed.
- `cd frontend && npx tsc --noEmit` — passed.

# Plan: Hydrophone Detection Range Collision Fix (Exact-Range Canonicalization)

## Outcome (2026-03-10)

- Added canonical hydrophone `detection_filename` (exact event UTC range) and preserved
  `extract_filename` as a compatibility alias for new rows.
- Updated Hydrophone UI Detection Range to a single exact value and aligned playback +
  extraction to the same exact clip bounds used for labeling.
- Added legacy-read fallback to derive `detection_filename` from `filename + start_sec/end_sec`
  when older TSV rows do not contain the new field.
- Updated tests (unit/integration/playwright) and docs (`CLAUDE.md`, `MEMORY.md`,
  `README.md`, `STATUS.md`) and added ADR-017 in `DECISIONS.md`.

## Verification

- `uv run pytest tests/` — 466 passed.
- `cd frontend && npx tsc --noEmit` — passed.
- `cd frontend && npx playwright test e2e/hydrophone-extract.spec.ts e2e/hydrophone-utc-timezone.spec.ts e2e/hydrophone-canceled-job.spec.ts` — 7 passed.

# Plan: Hydrophone Timeline Start-Boundary Lookback Fix

## Outcome (2026-03-10)

- Fixed `_build_stream_timeline()` so lookback does not stop on first in-range overlap;
  it now continues until overlap at the requested start boundary is found (or max
  lookback is reached), preventing undercounted hydrophone coverage windows.
- Added a regression test for the boundary-coverage failure mode where an in-range
  folder exists but earlier overlap is only discovered after additional lookback.
- Updated workflow semantics in `CLAUDE.md`, `MEMORY.md`, `README.md`, and `STATUS.md`
  to match the corrected start-boundary lookback behavior.

## Verification

- `uv run pytest tests/unit/test_s3_stream.py -q` — 26 passed.
- `uv run pytest tests/` — 463 passed.

# Plan: Hydrophone Detection Job Resume After Worker Restart

## Outcome (2026-03-09)

- Hydrophone detection jobs now resume from `segments_processed` checkpoint after worker
  restart: reads prior detections from TSV, skips already-processed segments, appends new
  detections. Guards against timeline changes (clears prior detections if skip is invalidated).
- Added `read_detections_tsv()` to `detector.py` for reading existing TSV into memory.
- Added `skip_segments` parameter to `iter_audio_chunks()` in `s3_stream.py`.
- Added cache invalidation on decode failure: `invalidate_cached_segment()` on
  `LocalHLSClient` and `CachingS3Client` deletes corrupted cached `.ts` files;
  `iter_audio_chunks()` retries once after invalidation.
- Fixed local detection job TSV stale-append: existing TSV deleted before reprocessing.
- Added periodic stale job recovery every 60s in worker loop (previously startup-only).
- Fixed `UnboundLocalError` from redundant local `write_detections_tsv` imports in
  `run_hydrophone_detection_job`.

## Verification

- `uv run pytest tests/` — 462 passed.
- `cd frontend && npx tsc --noEmit` — passed.
- 12 unit tests added: TSV read, skip segments, resume detection, skip invalidation,
  cache invalidation + retry, TSV cleanup.

# Plan: Retry Transient S3 Errors in Hydrophone Segment Fetching

## Outcome (2026-03-09)

- Added segment-level retry with exponential backoff (3 attempts, 1s/2s/4s) to
  `OrcasoundS3Client.fetch_segment()` for transient errors: `IncompleteRead`,
  `ReadTimeoutError`, `ConnectionError`, `EndpointConnectionError`, `ConnectionResetError`, `OSError`.
- Non-retryable errors (`NoSuchKey`, `404`, `AccessDenied`) raise immediately.
- Set explicit boto3 timeouts: `connect_timeout=10`, `read_timeout=30`.
- `CachingS3Client` benefits automatically (delegates to `OrcasoundS3Client.fetch_segment`).

## Verification

- `uv run pytest tests/` — 450 passed.
- 7 unit tests added covering retry success, exhausted retries, and no-retry for non-transient errors.

# Plan: Humpback Label Indicator ("Whale" Badge)

## Outcome (2026-03-09)

- Added `has_humpback_labels` nullable Boolean column to `detection_jobs` (migration `015`).
- SQLAlchemy model, Pydantic schema, and API response converter updated to expose the field.
- Label save endpoint (`PUT /detection-jobs/{job_id}/labels`) now computes the flag from the
  full merged TSV state and persists it to the DB on each save.
- Frontend `DetectionJob` type extended with `has_humpback_labels`.
- `useSaveDetectionLabels` now also invalidates `hydrophoneDetectionJobs` query on success.
- "Whale" outline badge rendered next to job status in `HydrophoneJobRow` when flag is true.

## Verification

- `uv run alembic upgrade head` — migration applied cleanly.
- `uv run pytest tests/` — 443 passed.
- `cd frontend && npx tsc --noEmit` — passed.
- 3 integration tests added: set flag true, clear flag to false, partial save preserves flag.

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
- Add integration/perf harness for hydrophone S3 prefetch (verify worker-level
  prefetch settings on real S3-backed runs and tune default worker/in-flight values)

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
