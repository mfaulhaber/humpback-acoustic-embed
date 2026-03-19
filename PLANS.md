# Development Plans

---

## Active

(none)

---

## Recently Completed

# Plan: Side + Top Navigation with Breadcrumbs

[Full plan](/Users/michael/.claude/plans/effervescent-soaring-comet.md)

## Outcome (2026-03-19)

- Replaced flat horizontal tab bar with persistent top nav (branding), icon side nav
  (expandable Classifier group), and shadcn/ui breadcrumbs
- Classifier sub-views are now direct routes (`/app/classifier/training`,
  `/app/classifier/hydrophone`); removed `ClassifierTab.tsx` state-based sub-tabs
- Deleted `Header.tsx` and `TabNav.tsx`; created `TopNav.tsx`, `SideNav.tsx`,
  `Breadcrumbs.tsx`, and `breadcrumb.tsx` (shadcn/ui)
- Updated 10 Playwright e2e test files for direct sub-route navigation

## Verification

- `cd frontend && npx tsc --noEmit` — passed
- `cd frontend && npm run build` — passed
- `uv run pytest tests/` — 744 passed, 1 skipped

# Plan: Derive Detection Output Paths from Storage Root, Not DB

[Full plan](/Users/michael/.claude/plans/sparkling-jingling-axolotl.md)

## Outcome (2026-03-19)

- Replaced all reads of `job.output_tsv_path` / `job.output_row_store_path` in
  API router and extraction worker with runtime path derivation via `storage.py`
  helpers (`detection_tsv_path()`, `detection_row_store_path()`,
  `detection_diagnostics_path()`, `detection_embeddings_path()`)
- Worker no longer writes `output_tsv_path`/`output_row_store_path` to DB;
  columns are now vestigial (cleanup backlog item added)
- Fixes 33 of 52 detection jobs that stored stale relative paths after storage
  root changed to an absolute external drive path

## Verification

- `uv run ruff format --check` / `uv run ruff check` — passed
- `uv run pyright` — 0 errors
- `cd frontend && npx tsc --noEmit` — passed
- `uv run pytest tests/` — 744 passed, 1 skipped

# Plan: Search by Audio — Worker-Encoded Detection Search

[Full plan](/Users/michael/.claude/plans/greedy-tickling-orbit.md)

## Outcome (2026-03-19)

- Added `SearchJob` model + `019_search_jobs.py` migration for ephemeral worker-encoded search jobs
- `POST /search/similar-by-audio` queues a search job; worker encodes detection audio via
  the classifier model; `GET /search/jobs/{id}` polls and returns similarity search results
  (row deleted after results returned)
- Search jobs prioritized first in worker loop (sub-second interactive work)
- Frontend detection mode uses StrictMode-safe direct fetch loop with AbortController cleanup
- Added `/search` to Vite proxy config
- ADR-034: Worker-encoded detection search via ephemeral SearchJob

## Verification

- `uv run ruff format --check` / `uv run ruff check` — passed
- `uv run pyright` — 0 errors
- `cd frontend && npx tsc --noEmit` — passed
- `uv run pytest tests/` — 744 passed, 1 skipped

# Plan: Agile Modeling Phase 2 — Search Results UI

[Full plan](/Users/michael/.claude/plans/polished-strolling-hare.md)

## Outcome (2026-03-18)

- Stored per-detection peak-window embeddings in `detection_embeddings.parquet`
  during local + hydrophone + subprocess detection paths
- Added `POST /search/similar-by-vector` (raw vector search), `GET /classifier/
  detection-jobs/{id}/embedding` (retrieve stored embedding), `GET /audio/{id}/
  spectrogram-png` (cached PNG spectrogram)
- Built Search tab with standalone mode (embedding set + window picker) and
  detection-sourced mode ("Search Similar" button on detection rows)
- Results table with spectrogram thumbnails, inline playback, score ranking
- ADR-033: Detection embedding storage for similarity search

## Verification

- `uv run ruff format --check` / `uv run ruff check` — passed
- `uv run pyright` — 0 errors
- `cd frontend && npx tsc --noEmit` — passed
- `uv run pytest tests/` — 736 passed, 1 skipped

# Plan: Agile Modeling Phase 1 — Embedding Similarity Search

[Full plan](/Users/michael/.claude/plans/replicated-snacking-goblet.md)

## Outcome (2026-03-18)

- Added `POST /search/similar` endpoint with brute-force cosine/euclidean search across all embedding sets for the same model version
- LRU cache (128 entries) for loaded parquet embeddings avoids repeated I/O
- Standard cosine similarity (not mean-centered) for stable cross-corpus search
- Supports `top_k`, `exclude_self`, `embedding_set_ids` filter, and `metric` selection
- Unit + integration tests (541 lines across 2 test files)
- ADR-032: Standard cosine similarity for cross-corpus embedding search

## Verification

- `uv run pytest tests/` — 688+ passed
- `uv run ruff format --check` / `uv run ruff check` — passed
- `uv run pyright` — 0 errors

# Plan: DB Load Logging + UI Error Flash Bar

[Full plan](/Users/michael/.claude/plans/staged-greeting-waterfall.md)

## Outcome (2026-03-18)

- Added structured logging to `database.py` (dir creation INFO, engine URL/pragma
  DEBUG) and `app.py` (startup INFO/ERROR with `exc_info=True`).
- Added `/health` GET endpoint in `app.py` that returns `ok`/`starting`/`error`
  (503) based on `app.state.db_healthy`; registered before routers, no DB session
  required.
- Added `DatabaseErrorBanner` React component polling `/health` every 15 s via
  `useHealth` TanStack Query hook; shows full-width red flash bar with dismiss
  button that auto-redisplays on the next poll while the API remains unhealthy.
- Added 3 unit tests (`test_health.py`): ok, 503-error, and starting-state paths.

## Verification

- `uv run ruff format --check src/humpback/api/app.py src/humpback/database.py tests/unit/test_health.py` — passed.
- `uv run ruff check src/humpback/api/app.py src/humpback/database.py tests/unit/test_health.py` — passed.
- `uv run pyright src/humpback/api/app.py src/humpback/database.py tests/unit/test_health.py` — 0 errors.
- `cd frontend && npx tsc --noEmit` — passed.
- `uv run pytest tests/` — 688 passed.

# Plan: Local Dev Stack Startup

[Full plan](/Users/michael/.claude/plans/playful-scribbling-eclipse.md)

## Outcome (2026-03-18)

- Added `Procfile.dev` (api/worker/frontend) and `Makefile` with `make dev`
  as the single-command startup target, plus individual targets for `api`,
  `worker`, `frontend-dev`, `build`, `test`, `test-watch`, `lint`,
  `typecheck`, and `playwright`.
- Added `honcho>=1.1.0` to the dev group in `pyproject.toml` so `uv run
  honcho` starts all three processes with colored prefixed output and clean
  Ctrl+C group shutdown.
- Updated README Development Mode section and `test_pyproject_metadata.py`
  fixture.

## Verification

- `uv sync --group dev --extra tf-macos` — honcho 2.0.0 resolved.
- `make help` — all targets print with descriptions.
- `uv run ruff format --check` / `uv run ruff check` — passed.
- `uv run pyright` — 0 errors.
- `uv run pytest tests/` — 685 passed.

# Plan: Fix Slow NOAA SanctSound Playback — Process-Level Provider Registry

[Full plan](/Users/michael/.claude/plans/imperative-wiggling-pascal.md)

## Outcome (2026-03-17)

- Added `_NoaaPlaybackProviderRegistry` singleton to `classifier.py` router that
  caches `CachingNoaaGCSProvider` instances keyed by `(source_id, noaa_cache_path)`.
- `_resolve_detection_audio` now reuses the warm cached provider for NOAA sources
  when `noaa_cache_path` is configured, avoiding repeated manifest JSON disk reads
  on every audio-slice/spectrogram request; Orcasound and unknown sources fall through
  to the existing `build_archive_playback_provider` path unchanged.
- Added 5 unit tests in `TestNoaaPlaybackProviderRegistry` covering: same-key identity,
  distinct source_ids, distinct cache paths, warm `_files_by_prefix` preservation, and
  20-thread concurrent safety.

## Verification

- `uv run ruff format --check src/humpback/api/routers/classifier.py tests/unit/test_archive_providers.py` — passed.
- `uv run ruff check src/humpback/api/routers/classifier.py tests/unit/test_archive_providers.py` — passed.
- `uv run pyright src/humpback/api/routers/classifier.py tests/unit/test_archive_providers.py` — 0 errors.
- `uv run pytest tests/` — 685 passed.

# Plan: Refactor noaa_detection_metadata.py — CSV URL Input

[Full plan](/Users/michael/.claude/plans/sleepy-swinging-wren.md)

## Outcome (2026-03-17)

- Replaced hardcoded CI01 GCS URL with `--csv-url` CLI arg accepting any HTTPS detection
  CSV URL; added `--deployment` arg (default `"01"`); made `--hydrophone-id` required in
  generation mode.
- Fixed `parse_noaa_detection_csv()` to normalize DictReader fieldnames to lowercase,
  resolving "No presence days found" for OC01 CSVs that use `IsoStartTime` (mixed case)
  vs CI01's `ISOStartTime`.
- Added 3 new tests: mixed-case header, no-Z timestamp format, `--csv-url` arg parsing.

## Verification

- `uv run ruff format --check` — passed.
- `uv run ruff check` — passed.
- `uv run pyright` — 0 errors.
- `uv run pytest tests/` — 680 passed.

# Plan: Enable Multi-Site NOAA SanctSound Sources (Channel Islands + Olympic Coast)

[Full plan](/Users/michael/.claude/plans/shimmying-napping-avalanche.md)

## Outcome (2026-03-17)

- Expanded Channel Islands (`sanctsound_ci01`) from 8 ci01-only deployments to
  27 deployments spanning sub-sites ci01, ci02, ci03, ci04 by raising base
  prefix to `sanctsound/audio/` and nesting sub-site paths in child_folder_hints.
- Populated Olympic Coast (`sanctsound_oc01`) with 12 deployments spanning
  oc01, oc02, oc03, oc04 and enabled in the detection UI (7 total UI sources).
- Fixed `_candidate_prefixes()` fallback: when child_folder_hints exist but none
  match the time range, return empty list instead of scanning the entire archive
  root prefix (prevented accidental full-SanctSound tree scans for multi-site
  sources).

## Verification

- `uv run ruff format --check` on modified Python files — passed.
- `uv run ruff check` on modified Python files — passed.
- `uv run pyright` on modified Python files — 0 errors.
- `uv run pytest tests/` — 677 passed.

# Plan: Fix Slow NOAA SanctSound Audio Preview

[Full plan](/Users/michael/.claude/plans/misty-dreaming-pascal.md)

## Outcome (2026-03-16)

- Switched `resolve_audio_slice` from full-file decode to chunked ffmpeg-seek
  decode for NOAA providers, reducing playback latency from 10-30s to ~100-200ms.
- Added `is_segment_cached()` check to skip redundant multi-hundred-MB file
  reads when the segment is already in the local cache.
- Added an in-memory LRU cache (`_DecodedAudioCache`, 64 entries) so the
  spectrogram request immediately following audio-slice reuses the decoded array.

## Verification

- `uv run ruff format --check` on modified Python files — passed.
- `uv run ruff check` on modified Python files — passed.
- `uv run pyright` on modified Python files — passed.
- `uv run pytest tests/` — 676 passed.

# Plan: NOAA Hydrophone Detection Metadata Job Generator

[Full plan](/Users/michael/.claude/plans/parsed-pondering-curry.md)

## Outcome (2026-03-16)

- Added `scripts/noaa_detection_metadata.py` — CLI utility that fetches the NOAA
  SanctSound CI01 deployment 01 daily humpback presence CSV from GCS, filters for
  Presence=1 days (25 out of 46), groups consecutive presence days into job ranges
  respecting the 7-day API limit, and outputs a JSON file of detection job payloads
  ready to POST to the hydrophone detection API.
- Supports `--classifier-model-name` for API-based name-to-UUID resolution,
  `--days-per-job` for job granularity control (default 1 day), and
  `--post --job-index N` for submitting one job at a time from the generated file.
- Handles real-world NOAA CSV quirks (BOM prefix, `\r\n` line endings).

## Verification

- `uv run ruff format --check` on modified Python files — passed.
- `uv run ruff check` on modified Python files — passed.
- `uv run pyright` on modified Python files — passed.
- `uv run pytest tests/` — 670 passed.

# Plan: Windowed Detection Mode (Fixed 5-Second Detections)

[Full plan](/Users/michael/.claude/plans/adaptive-wiggling-firefly.md)

## Outcome (2026-03-16)

- Added `detection_mode` column to detection jobs with "windowed" option that
  keeps 1-sec hop + hysteresis merge for sensitivity, then applies NMS peak
  selection within each merged event to output only non-overlapping fixed 5-sec
  detections — eliminating manual positive-selection during labeling.
- Fixed cross-event duplicate peaks by deduplicating NMS output per
  `(start_sec, end_sec)`, keeping the higher-confidence entry.
- Frontend defaults to windowed mode; spectrogram window-editing controls and
  positive-selection markers are hidden for windowed jobs.

## Verification

- `uv run ruff format --check` on modified Python files — passed.
- `uv run ruff check` on modified Python files — passed.
- `uv run pyright` on modified Python files — passed.
- `cd frontend && npx tsc --noEmit` — passed.
- `uv run pytest tests/` — 643 passed.

# Plan: NOAA Hydrophone Metadata

[Full plan](/Users/michael/.claude/plans/noaa-hydrophone-metadata.md)

## Outcome (2026-03-16)

- NOAA archive sources are now metadata-driven from the packaged
  `src/humpback/data/noaa_archive_sources.json` registry, including verified
  SanctSound and Glacier Bay records, UI visibility controls, and
  child-folder hints for Bartlett Cove's partitioned archive layout.
- The detection UI now exposes the supported NOAA sources
  `sanctsound_ci01` and legacy `noaa_glacier_bay`, while preserving the
  legacy Glacier Bay ID for compatibility and loading the rest of the verified
  NOAA metadata as reference/runtime config.
- NOAA long-object detection no longer stalls before first progress on
  multi-hour SanctSound files because decode now streams in chunks, and raw
  segment prefetch behavior is controlled per NOAA source in metadata
  (disabled for SanctSound, enabled for Glacier Bay).

## Verification

- `uv run ruff format --check` on modified Python files — passed.
- `uv run ruff check` on modified Python files — passed.
- `uv run pyright` on modified Python files — passed.
- `cd frontend && npx tsc --noEmit` — passed.
- `uv run pytest tests/` — 634 passed.

# Plan: Isolate TF2 Hydrophone Detection in a Subprocess

[Full plan](/Users/michael/.claude/plans/tf2-hydrophone-subprocess-isolation.md)

## Outcome (2026-03-16)

- TF2 SavedModel hydrophone detection now runs inside a spawned subprocess, so
  TensorFlow/Metal state is released between jobs while the parent worker keeps
  progress updates, diagnostics shards, alerts, and pause/resume/cancel
  orchestration.
- Hydrophone run summaries now record provider/runtime metadata including
  `provider_mode`, `execution_mode`, `avg_audio_x_realtime`,
  `peak_worker_rss_mb`, and `child_pid` to make cache/runtime regressions easier
  to diagnose.
- Follow-up review work corrected `avg_audio_x_realtime` to use end-to-end
  measured time (`fetch_sec + decode_sec + pipeline_total_sec`) and removed the
  completed P2 backlog item.

## Verification

- `uv run ruff format --check src/humpback/workers/classifier_worker.py tests/unit/test_classifier_worker.py` — passed.
- `uv run ruff check src/humpback/workers/classifier_worker.py tests/unit/test_classifier_worker.py` — passed.
- `uv run pyright src/humpback/workers/classifier_worker.py tests/unit/test_classifier_worker.py` — passed.
- `uv run pytest tests/` — 622 passed.


## Backlog

- Agile Modeling Phase 1b: Search by uploaded audio clip — embed on-the-fly
  using specified model, then search existing embedding sets
- Agile Modeling Phase 3: Classifier integration — label search results to
  create training sets, connect into the retrain loop
- Agile Modeling Phase 4: Active learning — prioritize labeling suggestions
  based on model uncertainty (entropy, margin)
- Smoke-test `tf-linux-gpu` on a real Ubuntu/NVIDIA host — verify `uv sync --extra tf-linux-gpu`, TensorFlow import, and GPU device visibility/runtime behavior.
- Generalize the legacy hydrophone API/frontend naming to archive-source terminology now that NOAA Glacier Bay is exposed through the same backend surfaces.
- Explore GPU-accelerated batch processing for large audio libraries
- Add WebSocket push for real-time job status updates (replace polling)
- Investigate multi-model ensemble clustering
- Optimize `/audio/{id}/spectrogram` window fetch path to avoid materializing all windows when only one index is requested (reduce memory/time on long files)
- Optimize hydrophone incremental lookback discovery to avoid repeated full S3
  folder scans at each lookback step (reduce first-segment startup latency)
- Add integration/perf harness for hydrophone S3 prefetch (verify worker-level
  prefetch settings on real S3-backed runs and tune default worker/in-flight values)
- Investigate a lower-overhead Orcasound decode path that reduces per-segment
  `ffmpeg` startup cost, likely via chunk-level or persistent-stream decode;
  treat this as a signal-processing/runtime change that needs validation and an
  ADR before implementation.
- Make `hydrophone_id` optional for local-cache detection jobs: update backend API
  schema, service layer, and worker to allow local-cache jobs without a hydrophone
  selection (frontend already hides the dropdown for local mode)
- Remove vestigial `output_tsv_path` and `output_row_store_path` columns from
  `DetectionJob` model and DB schema (Alembic migration); remove corresponding
  fields from `DetectionJobOut` schema and frontend `types.ts`. These columns are
  no longer read or written after the runtime path derivation change.

---

## Completed

- Rewrite README Overview for Research-First Positioning
- Add Year-Jump Buttons to the Hydrophone UTC Date Picker
- Sidecar Spectrogram PNGs for Extracted Detection Clips
- Classifier/Detection Data Flow + Spectrogram Windowing
- UI Changes for Classifier/Detection Spectrogram
- Relax Positive Extraction Length With 5-Second Chunk Growth
- Positive Window Selection From Stored Detection Scores
- UI Changes for Classifier/Detection Page
- UI Refactor for Classifier/Detection Page
- Fix NOAA GCS Playback/Spectrogram — Interval Estimation Bug
