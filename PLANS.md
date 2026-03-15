# Development Plans

---

## Active

(none)

---

## Recently Completed

# Plan: Sidecar Spectrogram PNGs for Extracted Detection Clips

[Full plan](/Users/michael/.claude/plans/extracted-audio-sidecar-spectrograms.md)

## Outcome (2026-03-15)

- Extract labeled samples now write a same-basename `.png` beside each extracted
  `.flac` for local and hydrophone jobs, using the same shared spectrogram
  renderer/settings as the UI popup base image and the actual extracted clip
  window rather than the full detection span.
- Extraction reruns now backfill missing PNG sidecars without rewriting existing
  audio, and stale positive-output cleanup removes sibling PNGs when selected
  clip filenames change.
- Added extractor/worker/API regression coverage for positive-window parity,
  negative-label PNG generation, settings plumbing, and saved-PNG parity with
  the `/classifier/detection-jobs/{id}/spectrogram` endpoint.

## Verification

- `uv run ruff format --check src/humpback/classifier/extractor.py src/humpback/workers/classifier_worker.py tests/integration/test_classifier_api.py tests/unit/test_classifier_worker.py tests/unit/test_extractor.py` — passed.
- `uv run ruff check src/humpback/classifier/extractor.py src/humpback/workers/classifier_worker.py tests/integration/test_classifier_api.py tests/unit/test_classifier_worker.py tests/unit/test_extractor.py` — passed.
- `uv run pyright src/humpback/classifier/extractor.py src/humpback/workers/classifier_worker.py tests/integration/test_classifier_api.py tests/unit/test_classifier_worker.py tests/unit/test_extractor.py` — passed.
- `cd frontend && npx tsc --noEmit` — passed.
- `uv run pytest tests/` — 618 passed.

# Plan: Classifier/Detection Data Flow + Spectrogram Windowing

[Full plan](/Users/michael/.claude/plans/classifier-detection-data-flow-windowing.md)

## Outcome (2026-03-15)

- Detection jobs now persist canonical editable row state in
  `detection_rows.parquet`, compute default positive-selection windows during
  detection, and treat TSV as a synchronized download/export adapter.
- The Hydrophone spectrogram popup now supports 5-second window editing with
  Apply/Cancel, opposite-edge compensation at clip boundaries, and immediate
  persistence through the row-state API without requiring extraction first.
- Follow-up fixes in this session kept resumed hydrophone jobs from serving
  stale row-store state, allowed row-store-backed content/download/edit flows
  when the TSV adapter is missing, and corrected manual-bound validation so any
  in-clip `5 * N` duration such as `3..13` is accepted.

## Verification

- `uv run ruff format --check` on touched Python files — passed.
- `uv run ruff check` on touched Python files — passed.
- `uv run pyright` on touched Python files — passed.
- `cd frontend && ./node_modules/.bin/tsc --noEmit` — passed.
- `cd frontend && ./node_modules/.bin/playwright test e2e/detection-spectrogram.spec.ts -g 'positive rows can adjust the window|edge expansion promotes|right-edge expansion borrows'` — passed.
- `uv run pytest tests/` — 615 passed.

# Plan: UI Changes for Classifier/Detection Spectrogram

[Full plan](/Users/michael/.claude/plans/classifier-detection-spectrogram-client-overlay.md)

## Outcome (2026-03-15)

- Detection Play clicks now open the spectrogram popup alongside playback, while
  `Alt`/`Option`-click still opens spectrogram-only view.
- The popup now renders client-side black extraction markers using persisted
  `positive_selection_*` bounds when available, unsaved live positive-label
  edits immediately, and legacy extracted/clip bounds as fallback for older
  rows without selection metadata.
- Spectrogram PNG rendering now uses deterministic plot margins and cache-key
  versioning so the client-side overlay aligns consistently with cached images.

## Verification

- `uv run ruff format --check src/humpback/processing/spectrogram.py src/humpback/processing/spectrogram_cache.py tests/unit/test_spectrogram.py` — passed.
- `uv run ruff check src/humpback/processing/spectrogram.py src/humpback/processing/spectrogram_cache.py tests/unit/test_spectrogram.py` — passed.
- `uv run pyright src/humpback/processing/spectrogram.py src/humpback/processing/spectrogram_cache.py tests/unit/test_spectrogram.py` — passed.
- `cd frontend && npx tsc --noEmit` — passed.
- `cd frontend && npx playwright test e2e/detection-spectrogram.spec.ts` — 4 passed, 1 skipped.
- `uv run pytest tests/` — 608 passed.

# Plan: Relax Positive Extraction Length With 5-Second Chunk Growth

[Full plan](/Users/michael/.claude/plans/relaxed-positive-extraction-length.md)

## Outcome (2026-03-14)

- Positive extraction now seeds from the best 5-second window but can widen in
  adjacent 5-second chunks when neighboring smoothed scores stay above the new
  `positive_selection_extend_min_score` threshold.
- Added extraction API/worker/frontend config support for the new extension
  threshold, kept legacy rescoring fallback aligned with the same widening
  behavior, and appended ADR-027.
- Added selector/extractor regression coverage including the NOAA-style widened
  fallback case and stale positive-output replacement.

## Verification

- `uv run ruff format --check` on touched files — passed.
- `uv run ruff check` on touched files — passed.
- `uv run pyright` — passed.
- `cd frontend && npx tsc --noEmit` — passed.
- `uv run pytest tests/` — 606 passed.

# Plan: Positive Window Selection From Stored Detection Scores

[Full plan](/Users/michael/.claude/plans/stored-positive-window-selection-from-detection-scores.md)

## Outcome (2026-03-14)

- Positive extraction now chooses one 5-second training clip from stored 1-second-hop
  detection diagnostics using smoothing plus a minimum-score threshold, with rescoring kept
  as a legacy fallback only.
- Hydrophone detection now persists incremental diagnostics shards, and detection TSV/content
  rows carry `positive_selection_*` provenance plus `positive_extract_filename`.
- Added extraction API defaults for smoothing/threshold, updated frontend API typings,
  expanded extractor/worker/API coverage, and appended ADR-026.

## Verification

- `uv run ruff format` on touched files — passed.
- `uv run ruff check` on touched files — passed.
- `uv run pyright` — passed.
- `cd frontend && npx tsc --noEmit` — passed.
- `uv run pytest tests/` — 602 passed.


# Plan: UI Changes for Classifier/Detection Page

[Full plan](/Users/michael/.claude/plans/cheeky-growing-liskov.md)

## Outcome (2026-03-14)

- Added `type="search"`, `autoComplete="off"`, and `data-lpignore="true"` to Previous Jobs
  filter input to suppress browser autocomplete and password managers.
- Added sortable "Created" column (local time display) to both Active and Previous Jobs
  tables; default sort changed from "date" to "created" descending.
- Hidden Hydrophone dropdown when Local Cache source is selected; Classifier Model takes
  full width; relaxed frontend validation to not require hydrophone for local mode.
- Changed slider defaults: Confidence 0.50→0.90, Start 0.70→0.80, Continue 0.45→0.70.
  Restructured layout: all three sliders in a single row, Hop Size moved below.
- Fixed `created_at` UTC parsing (append `Z` to naive ISO strings from API).

## Verification

- `npx tsc --noEmit` — passed.
- `uv run pytest tests/` — 594 passed.


# Plan: UI Refactor for Classifier/Detection Page

[Full plan](/Users/michael/.claude/plans/dynamic-painting-glacier.md)

## Outcome (2026-03-14)

- Added `provider_kind` field to `HydrophoneInfo` API schema and `/classifier/hydrophones`
  endpoint response; frontend `HydrophoneInfo` type updated to match.
- Replaced 2-way S3/Local source toggle with 3-way Orcasound/NOAA/Local Cache selector,
  moved above the hydrophone dropdown, and filtered the dropdown by `provider_kind`.
- Simplified Local Cache UI (generic placeholder, removed HLS-specific hint text).
- Added Previous Jobs table: text filter (hydrophone name substring), sortable column
  headers (status, hydrophone, date, threshold, results), client-side pagination, and
  preferences dialog (page size 10/20/50/100, column visibility toggles).
- Updated 5 Playwright specs with `provider_kind` in mocked HYDROPHONE data.

## Verification

- `uv run ruff format --check` on modified files — passed.
- `uv run ruff check` on modified files — passed.
- `uv run pyright` on modified files — 0 errors.
- `cd frontend && npx tsc --noEmit` — passed.
- `uv run pytest tests/` — 594 passed.


# Plan: Fix NOAA GCS Playback/Spectrogram — Interval Estimation Bug

[Full plan](/Users/michael/.claude/plans/happy-crunching-crayon.md)

## Outcome (2026-03-14)

- Fixed `estimate_noaa_interval_sec()` to use `median(intervals)` instead of
  minimum-based filtering, making it robust to outlier file gaps (3 anomalous
  25s gaps among 24K files at 300s intervals).
- Made `read_noaa_manifest()` re-estimate interval from file timestamps,
  auto-healing existing cached manifests with wrong `default_interval_sec`.
- Added 7 new tests for estimator robustness and manifest re-estimation.

## Verification

- `uv run ruff format --check` on modified files — passed.
- `uv run ruff check` on modified files — passed.
- `uv run pyright` on modified files — 0 errors.
- `uv run pytest tests/` — 594 passed.

# Plan: NOAA GCS Local Cache with GCS Fallback

[Full plan](/Users/michael/.claude/plans/synthetic-splashing-prism.md)

## Outcome (2026-03-13)

- Added `CachingNoaaGCSProvider` with local filesystem cache + GCS fallback for
  NOAA metadata manifests (JSON) and `.aif` segment files, using atomic writes
  (tmpfile + `os.replace`).
- Added `noaa_cache_path` setting (defaults to `{storage_root}/noaa-gcs-cache`),
  factory helpers (`build_noaa_detection_provider`, `build_noaa_playback_provider`),
  and threaded `noaa_cache_path` through detection worker, extraction worker, and
  playback router.
- Updated docs (ADR-025, STATUS.md, README.md) and added 16 new tests.

## Verification

- `uv run ruff format --check` on modified files — passed.
- `uv run ruff check` on modified files — passed.
- `uv run pyright` on modified files — 0 errors.
- `uv run pytest tests/` — 587 passed.

# Plan: ArchiveProvider Abstraction — Phase 4: Promote NOAA GCS provider

[Full plan](/Users/michael/.claude/plans/peaceful-herding-rossum.md)

## Outcome (2026-03-13)

- Promoted the NOAA Glacier Bay Bartlett Cove GCS POC into
  `src/humpback/classifier/providers/noaa_gcs.py` as a production
  `ArchiveProvider`, and rewired `scripts/noaa_gcs_poc.py` to reuse the same
  implementation.
- Added a static archive-source registry so the legacy hydrophone service/router/worker
  flow can build either Orcasound HLS or NOAA GCS providers without changing the
  database schema or public endpoint names in this phase.
- Kept Orcasound playback/extraction local-cache-authoritative, while allowing NOAA
  detection/playback/extraction to use direct anonymous GCS fetch; rejected
  `local_cache_path` for NOAA jobs.
- Promoted `google-cloud-storage` to a runtime dependency, added NOAA provider/API/worker
  tests, updated docs/status text, and appended ADR-024.

## Verification

- `uv run ruff check` on modified backend/tests/scripts — passed.
- `uv run pyright` — passed.
- `uv run pytest tests/` — 571 passed.
- `uv run python scripts/noaa_gcs_poc.py --skip-download` — passed against the live
  NOAA public bucket.

# Plan: ArchiveProvider Abstraction — Phase 3: Adapt Upstream Consumers

[Full plan](/Users/michael/.claude/plans/peaceful-herding-rossum.md)

## Outcome (2026-03-13)

- Migrated `hydrophone_detector.py`, `extractor.py`, `classifier_worker.py`, and the
  classifier API router to construct and pass `ArchiveProvider` instances instead of
  raw clients plus `hydrophone_id`.
- Added Orcasound provider factory helpers so detection uses the existing
  local-cache -> S3-cache -> direct-S3 priority while playback and extraction remain
  local-cache-authoritative.
- Removed the temporary backward-compat wrappers from `s3_stream.py`
  (`build_hydrophone_stream_timeline`, `resolve_hydrophone_audio_slice`,
  `_ClientAdapter`, and the legacy `iter_audio_chunks` signature).
- Updated provider, stream, extractor, worker, and hydrophone API tests to exercise
  the provider-only path end to end.

## Verification

- `uv run ruff check` on modified backend/tests — passed.
- `uv run pyright` — passed.
- `uv run pytest tests/unit/test_archive_providers.py tests/unit/test_s3_stream.py tests/unit/test_hydrophone_resume.py tests/unit/test_extractor.py tests/unit/test_classifier_worker.py tests/integration/test_hydrophone_api.py tests/integration/test_classifier_api.py -q` — 159 passed.
- `uv run pytest tests/` — 552 passed.

# Plan: Refactor Classifier/Detection Active Job UI and Backend

[Full plan](/Users/michael/.claude/plans/fizzy-seeking-valley.md)

## Outcome (2026-03-13)

- Replaced single-job "Active Job" card with multi-row "Active Jobs" table showing all
  running/queued/paused jobs with per-row Pause/Resume/Cancel controls.
- Extended backend status gates: `"paused"` now allowed for save-labels, download, and
  extract endpoints; `"queued"` now allowed for cancel endpoint.
- Extended `HydrophoneJobRow` with `isActive` prop for dual-context rendering (active
  table: progress + actions columns; previous table: results + extract + error columns).
- Extended `expandedContentJobId` and `extractTargetIds` to include paused jobs.
- Added 5 backend integration tests: save labels on paused, download paused, extract
  paused, cancel queued, reject labels on running.
- Added Playwright spec (`hydrophone-active-queue.spec.ts`) with 4 tests for active
  table rendering, action buttons, paused content expansion, and queued cancel.
- Updated `hydrophone-pause-resume.spec.ts` locators from "Active Job" card to
  "Active Jobs" table.

## Verification

- `uv run pytest tests/integration/test_hydrophone_api.py -q` — 27 passed.
- `uv run pytest tests/` — 548 passed.
- `cd frontend && npx tsc --noEmit` — passed.
- `uv run pyright` on modified backend files — 0 errors.


# Plan: Expand Pyright Enforcement to `scripts/`, Then `tests/`

[Full plan](/Users/michael/.claude/plans/expand-pyright-enforcement-scripts-tests.md)

## Outcome (2026-03-13)

- Cleared the remaining Pyright backlog in `scripts/` and `tests/`, including
  the progress-total typing issue in `scripts/stage_s3_epoch_cache.py`,
  async-fixture annotations, optional-value narrowing in tests, and a few
  narrow source annotations in hydrophone stream/stability helpers.
- Expanded repo Pyright enforcement from `src/humpback` only to
  `src/humpback`, `scripts`, and `tests` in both `pyproject.toml` and the
  pre-commit hook trigger.
- Updated tooling lock tests plus docs/status text in `CLAUDE.md`,
  `README.md`, and `STATUS.md` to reflect the widened enforcement scope.

## Verification

- `uv run pyright` — passed.
- `uv run pyright scripts tests` — passed.
- `uv run pytest tests/unit/test_pyproject_metadata.py tests/unit/test_stage_s3_epoch_cache.py tests/unit/test_s3_stream.py tests/unit/test_detector.py tests/unit/test_diagnostics.py tests/unit/test_clustering_pipeline.py tests/unit/test_model_registry.py tests/unit/test_retrain.py tests/unit/test_inference.py tests/unit/test_detection_spans.py tests/unit/test_stability.py tests/integration/test_hydrophone_api.py tests/unit/test_archive_providers.py tests/unit/test_config.py -q` — 235 passed.
- `uv run pre-commit run --all-files` — passed.
- `uv run pytest tests/` — 543 passed.

# Plan: Add Pyright to the Python Tooling Chain

[Full plan](/Users/michael/.claude/plans/pyright-tooling-integration.md)

## Outcome (2026-03-13)

- Added Pyright to the `uv` dev toolchain, configured it in `pyproject.toml`,
  and enforced it via a local `uv run pyright` pre-commit hook alongside Ruff.
- Cleaned the initial `src/humpback` type-checking baseline with narrow typing
  fixes so Pyright passes without changing intended runtime behavior.
- Updated docs (`CLAUDE.md`, `README.md`, `STATUS.md`), lockfile metadata, and
  repo tests that lock the packaging/tooling contract.

## Verification

- `uv sync --group dev --extra tf-macos` — passed.
- `uv run pyright` — passed.
- `uv run pre-commit run --all-files` — passed.
- `uv run pytest tests/` — 543 passed.


## Backlog

- Expand Pyright enforcement beyond `src/humpback` to `scripts/`, then `tests/`,
  after clearing the remaining type-checking backlog in those areas.
- Smoke-test `tf-linux-gpu` on a real Ubuntu/NVIDIA host — verify `uv sync --extra tf-linux-gpu`, TensorFlow import, and GPU device visibility/runtime behavior.
- Generalize the legacy hydrophone API/frontend naming to archive-source terminology now that NOAA Glacier Bay is exposed through the same backend surfaces.
- Add shared NOAA archive metadata caching for playback/extraction so each
  `/classifier/detection-jobs/{id}/audio-slice` request does not re-list the full
  Bartlett Cove GCS prefix before resolving a clip.
- Explore GPU-accelerated batch processing for large audio libraries
- Add WebSocket push for real-time job status updates (replace polling)
- Investigate multi-model ensemble clustering
- Optimize `/audio/{id}/spectrogram` window fetch path to avoid materializing all windows when only one index is requested (reduce memory/time on long files)
- Optimize hydrophone incremental lookback discovery to avoid repeated full S3
  folder scans at each lookback step (reduce first-segment startup latency)
- Add integration/perf harness for hydrophone S3 prefetch (verify worker-level
  prefetch settings on real S3-backed runs and tune default worker/in-flight values)
- Make `hydrophone_id` optional for local-cache detection jobs: update backend API
  schema, service layer, and worker to allow local-cache jobs without a hydrophone
  selection (frontend already hides the dropdown for local mode)

---

## Completed

- Deployment Config via Env + Trusted Hosts
- Selective Merge for Cross-Platform TensorFlow `pyproject.toml`
- Switch Detection Extraction to FLAC + Conversion Script
- Add Orca Detection Label + Extraction Path Reorder
- ArchiveProvider Abstraction — Phase 1 & 2
- Stage S3 Epoch Cache Progress + Dry-Run + README
- POC: NOAA GCS Passive Bioacoustic Client
- Hydrophone Detection Ordered S3 Prefetch + Timing Telemetry
- Paused Hydrophone Playback + Spectrogram Resolution Fix
- Snapped Canonical Detection Ranges
- Hydrophone Detection Range Collision Fix (Exact-Range Canonicalization)
- Hydrophone Timeline Start-Boundary Lookback Fix
- Hydrophone Detection Job Resume After Worker Restart
- Retry Transient S3 Errors in Hydrophone Segment Fetching
- Humpback Label Indicator ("Whale" Badge)
- Detection Spectrogram Popup
- Enhanced Classifier/Train Expanded Model Details
- Retrain Classifier Model
- Popover Date Range Picker for Hydrophone Tab
- Hydrophone Tab Improvements + Detect Tab Removal
- Hydrophone Detection UI — UTC-Only Input + Date Range Semantics
- Hydrophone Detection — Configurable Lookback + Explicit No-Audio Failure
- Hydrophone Detection UI — Clip Timing Consistency (Display + Playback + Extract)
- Hydrophone Detection UI Label Clarity + Extract Filename Metadata
- Hydrophone Detection — Timeline-Correct Segment Assembly + End-Bounded Range
- Classifier/Hydrophone Extract — Hydrophone-Partitioned Output Paths
- Hydrophone Tab — Playback Timestamp Mapping + Saved-Label Extract Activation
- HydrophoneTab — Live Detection Content + Save/Extract Labels
- Classifier/Hydrophone Tab — S3 HLS Streaming Detection
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
