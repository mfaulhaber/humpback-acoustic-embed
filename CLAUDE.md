# Humpback Acoustic Embedding & Clustering Platform

## 1. Purpose

This system processes humpback whale audio recordings into reusable embedding vectors
using a Perch-compatible TFLite model, then performs clustering with optional
behavioral/ecological metadata.

The system must:
- Support asynchronous, resumable workflows
- Persist workflow state in SQL
- Prevent reprocessing of already-encoded audio for the same configuration
- Allow separate queuing of processing and clustering jobs
- Provide a web UI for job management and inspection

This document defines behavioral rules, engineering constraints, project reference
material, and workflow integration. For architecture decisions, see `DECISIONS.md`.
For Codex-specific workflow, see `AGENTS.md`.

---

## 2. High-Level Architecture

Components:
1. Web UI
2. API Server
3. Workflow Queue (SQL-backed)
4. Worker Processes (processing + clustering)
5. SQL Database
6. Object/File Storage (audio + embeddings + cluster outputs)
7. Clustering Engine

All components run locally for MVP but should be designed so workers can scale horizontally.

---

## 3. Core Development Rules

### 3.1 Package Management
*   **ONLY** use `uv` for all Python package operations. **NEVER** use `pip`, `pip-tools`, `poetry`, or `conda`.
*   Dependencies are managed via `pyproject.toml` and `uv.lock` files. The lock file should be committed to version control for reproducible builds.
*   TensorFlow is selected via mutually-exclusive extras: `tf-macos`, `tf-linux-cpu`, or `tf-linux-gpu`.
*   Do **NOT** use `uv sync --all-extras` in this project because the TensorFlow extras intentionally conflict.

### 3.2 Environment Commands
Use these commands for managing dependencies:
*   Install/synchronize Apple Silicon macOS dependencies (including dev tools): `uv sync --group dev --extra tf-macos`
*   Install/synchronize Linux CPU dependencies (including dev tools): `uv sync --group dev --extra tf-linux-cpu`
*   Install/synchronize Linux GPU/CUDA dependencies (including dev tools): `uv sync --group dev --extra tf-linux-gpu`
*   Install production Linux GPU/CUDA dependencies (no dev tools): `uv sync --extra tf-linux-gpu`
*   Add a new package (e.g., `requests`): `uv add requests`
*   Remove a package: `uv remove <package>`
*   Refresh the lock file after dependency changes: `uv lock`
*   Upgrade a specific package: `uv lock --upgrade-package <package>`

### 3.3 Running Python Code and Tools
*   Run a Python script: `uv run <script-name>.py`
*   Run Python tools/tests (e.g., `pytest`): `uv run pytest tests/`
*   Run the Python type checker: `uv run pyright`
*   Install pre-commit hooks once per clone: `uv run pre-commit install`
*   Run all pre-commit hooks manually: `uv run pre-commit run --all-files`

### 3.4 Best Practices
*   Prefer `uv run` over manually activating a virtual environment and running commands directly.
*   Python edits must pass pre-commit Ruff and Pyright hooks before commit.
*   Pyright enforcement covers `src/humpback`, `scripts/`, and `tests/`; expand deliberately after cleaning any new areas.
*   When troubleshooting, use `uv cache clean` as a last resort.

### 3.5 Database Migrations
*   Honor database file location defined with HUMPBACK_DATABASE_URL, check .env override.  
*   When a change adds, removes, or renames columns on any SQL table, **always** create an Alembic migration in `alembic/versions/` and run it with `uv run alembic upgrade head` before verifying the change works.
*   Migration files follow the naming convention `NNN_short_description.py` (e.g., `007_negative_embedding_set_ids.py`), incrementing from the latest revision.
*   Use `op.batch_alter_table()` for SQLite compatibility.

### 3.6 Documentation
*   When a change adds, removes, or modifies API endpoints, data models, configuration options, architecture, or workflows, update the relevant files:
    *   `CLAUDE.md` — rules, reference material, project state (this file)
    *   `DECISIONS.md` — append new ADR for significant architecture changes
    *   `README.md` — user-facing API endpoints, configuration, feature list
    *   `docs/specs/` — design specs (written during brainstorming phase)
    *   `docs/plans/` — implementation plans (written during planning phase)
    *   `docs/reference/` — detailed reference material (data model, signal processing, storage, frontend, hydrophone rules, testing)

## 3.7 Frontend Stack & Development
React 18 + Vite + TypeScript + Tailwind + shadcn/ui SPA in `frontend/`. Use `npm` for frontend package ops.
See [docs/reference/frontend.md](docs/reference/frontend.md) for full stack details, file structure, and dev workflow.

### 3.8 Timezone and Timestamp Standard (UTC-Only)
All operational timestamps in this project must use UTC end-to-end.

*   Backend must compute, compare, persist, and serialize timestamps in UTC.
*   API timestamp fields should be UTC epoch seconds or ISO-like values with `Z` semantics.
*   Frontend must parse and submit timestamp inputs as UTC for project workflows unless an endpoint explicitly requires local time.
*   Frontend displays for operational time ranges must be labeled as UTC; avoid locale-time rendering for these values.
*   Time-derived filenames/identifiers must use compact UTC format (`%Y%m%dT%H%M%SZ`), including detection/extraction naming paths.
*   Tests touching timestamp behavior must assert UTC semantics explicitly.

---

## 4. Core Design Principles

### 4.1 Idempotent Encoding (No Reprocessing)
Each audio file is encoded once per (model_version, window_size, target_sample_rate, feature_config).

A ProcessingJob MUST:
- compute a stable "encoding_signature"
- check for an existing completed embedding set with that signature
- skip work if the embedding set exists

### 4.2 Resumable Workflow
All steps are recorded in SQL. Workers must be restart-safe:
- jobs can resume after crash/restart
- partial artifacts should be either:
  - safely overwritten, or
  - written to temp and atomically promoted on completion

### 4.3 Asynchronous, Observable Jobs
Jobs are queued and executed in the background by workers.
UI can monitor via polling or a push channel.

### 4.4 Hydrophone Rules
Hydrophone detection, extraction, playback, and timeline assembly follow detailed conventions.
See [docs/reference/hydrophone-rules.md](docs/reference/hydrophone-rules.md) for extraction paths, timeline assembly, TSV metadata, and job lifecycle.

---

## 5. Testing Requirements (MANDATORY)

Testing is not optional. Every meaningful change must include tests.
*   Run all tests: `uv run pytest tests/`
*   Run frontend tests: `cd frontend && npx playwright test`
*   Type-check frontend: `cd frontend && npx tsc --noEmit`

See [docs/reference/testing.md](docs/reference/testing.md) for unit test guidelines, E2E smoke test spec, Playwright patterns, and model stub strategy.

---

## 6. Definition of Done (Engineering)
A PR/change is "done" only if:
- unit tests added/updated for changed behavior
- test suite passes locally
- E2E smoke test passes locally
- Playwright tests added/updated for UI changes (`cd frontend && npx playwright test`)
- idempotency rules preserved (no duplicate embedding sets)

---

## 7. Non-Goals (MVP)
- Model fine-tuning
- Real-time streaming inference
- Multi-tenant support
- Distributed GPU execution

---

## 8. Project Reference

### 8.1 Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11-3.12 |
| Package Manager | uv (pyproject.toml + uv.lock, explicit TensorFlow extras by platform) |
| Web Framework | FastAPI |
| Database | SQLite (via SQLAlchemy) |
| Migrations | Alembic |
| Embedding Format | Apache Parquet |
| ML Models | TFLite (Perch), TF2 SavedModel |
| Clustering | HDBSCAN, K-Means, Agglomerative |
| Dim Reduction | UMAP, PCA |
| Metric Learning | PyTorch (triplet loss MLP) |
| Classifier | scikit-learn LogisticRegression |
| Frontend | React 18 + Vite + TypeScript + Tailwind + shadcn/ui |
| Charts | react-plotly.js |
| Server State | TanStack Query |
| Testing | pytest + pre-commit/Ruff (backend), Playwright (frontend) |

### 8.2 Repository Layout

```
humpback-acoustic-embed/
├── CLAUDE.md              (rules, reference, project state — auto-loaded)
├── AGENTS.md              (Codex entry point)
├── DECISIONS.md           (architecture decision log)
├── pyproject.toml         (Python dependencies)
├── uv.lock                (lockfile)
├── alembic.ini            (migration config)
├── alembic/versions/      (migration scripts, 001–037)
├── src/humpback/
│   ├── api/               (FastAPI routes)
│   ├── classifier/        (training, detection, embedding)
│   ├── clustering/        (HDBSCAN, K-Means, metrics, refinement)
│   ├── config.py          (settings)
│   ├── data/              (packaged metadata assets such as NOAA archive sources)
│   ├── database.py        (SQLAlchemy models + session)
│   ├── models/            (TFLite + TF2 model runners)
│   ├── processing/        (audio decode, windowing, features, parquet)
│   ├── schemas/           (Pydantic request/response models)
│   ├── services/          (business logic layer)
│   ├── static/            (built frontend SPA)
│   ├── storage.py         (file path helpers)
│   └── workers/           (background job processing)
├── frontend/              (React SPA — see §3.7)
├── tests/                 (pytest suite)
├── models/                (ML model files)
├── scripts/               (utility scripts)
├── docs/
│   ├── specs/             (design specs from brainstorming)
│   ├── plans/             (implementation plans + backlog)
│   └── reference/         (detailed reference: data model, signal processing, storage, frontend, hydrophone, testing)
└── data/                  (runtime data)
```

### 8.3 Data Model Summary
See [docs/reference/data-model.md](docs/reference/data-model.md) for condensed model reference. Full field lists in `src/humpback/database.py`.

### 8.4 Signal Processing Parameters
See [docs/reference/signal-processing.md](docs/reference/signal-processing.md) for parameter defaults, windowing rules, and processing pipeline diagram.

### 8.5 Storage Layout
See [docs/reference/storage-layout.md](docs/reference/storage-layout.md) for full directory tree of all artifact types.

### 8.6 Runtime Configuration

- `Settings` reads `HUMPBACK_`-prefixed environment variables.
- API and worker entrypoints load the repo-root `.env`; direct `Settings()` does not.
- `api_host` defaults to `0.0.0.0`, `api_port` to `8000`.
- `allowed_hosts` defaults to `*`. `HUMPBACK_ALLOWED_HOSTS` uses Starlette wildcard syntax.
- `positive_sample_path`, `negative_sample_path`, `s3_cache_path` derive from `storage_root` when unset.
- `timeline_cache_max_jobs` defaults to `15`. `HUMPBACK_TIMELINE_CACHE_JOBS` controls how many detection jobs keep fully cached timeline tiles on disk (~8-16 GB at default). LRU eviction removes the oldest job when exceeded.
- `timeline_prepare_workers` defaults to `2`; startup/full tile batches may render through a bounded worker pool.
- `timeline_startup_radius_tiles` defaults to `2`; the Timeline button now triggers startup-scoped cache warming around the initial viewport rather than a full all-zoom warmup.
- `timeline_startup_coarse_levels` defaults to `1`, `timeline_neighbor_prefetch_radius` defaults to `1`, `timeline_tile_memory_cache_items` defaults to `256`, `timeline_manifest_memory_cache_items` defaults to `8`, and `timeline_pcm_memory_cache_mb` defaults to `128` for bounded in-memory timeline reuse.
- `replay_metric_tolerance` defaults to `0.01`. `HUMPBACK_REPLAY_METRIC_TOLERANCE` controls the absolute tolerance for rate metrics (precision, recall, fp_rate, high_conf_fp_rate) during replay verification of candidate-backed training. Count metrics (tp, fp, fn, tn) must match exactly.
- `pcen_time_constant_sec` (`2.0`), `pcen_gain` (`0.98`), `pcen_bias` (`2.0`), `pcen_power` (`0.5`), `pcen_eps` (`1e-6`): PCEN parameters applied per timeline tile via `librosa.pcen`. Overridable via `HUMPBACK_PCEN_*`. The per-bin low-pass filter state is pre-initialized to the first STFT frame's magnitude (scaled `lfilter_zi`), eliminating librosa's default unit-step cold-start transient that would otherwise paint a dark strip at the left edge of every tile.
- `pcen_warmup_sec` defaults to `2.0`. Each tile's audio fetch is extended backward by this amount so the PCEN low-pass filter can settle before the first rendered frame; the warm-up frames are trimmed off the output. Redundant with warm-zi initialization for stationary signals, but still useful when the signal changes across the tile boundary.
- `pcen_vmin` (`0.0`) / `pcen_vmax` (`1.0`): fixed colormap range for PCEN-normalized tiles. Because PCEN's output is bounded, there is no per-job `ref_db` computation.
- `playback_target_rms_dbfs` defaults to `-20.0`. `HUMPBACK_PLAYBACK_TARGET_RMS_DBFS` controls the RMS level that timeline playback chunks are scaled to before MP3/WAV encoding.
- `playback_ceiling` defaults to `0.95`. `HUMPBACK_PLAYBACK_CEILING` is the `tanh` soft-clip ceiling applied after RMS scaling to prevent harsh clipping on transients.

### 8.7 Behavioral Constraints

Non-obvious constraints that are not immediately derivable from code:

- **Worker priority order**: search -> processing -> clustering -> classifier training -> detection -> extraction -> detection embedding generation -> label processing -> retrain -> vocalization training -> vocalization inference -> region detection -> event segmentation -> event classification -> segmentation feedback training -> classifier feedback training -> manifest generation -> hyperparameter search
- **Job claim semantics**: Workers claim queued jobs via atomic compare-and-set (`WHERE id=:candidate AND status='queued'`). SQLite has no true row-level locks; correctness relies on atomic status updates, not `SELECT ... FOR UPDATE`.
- **Job status transitions**: `queued -> running -> complete`, `queued -> running -> failed`, `queued -> canceled`
- **Processing concurrency**: prevent two running ProcessingJobs for same encoding_signature; allow multiple clustering jobs in parallel
- **Prefetch semantics**: `time_covered_sec` tracks summed processed audio duration rather than wall-clock range coverage
- **Parquet row-store**: detection jobs write directly to Parquet row store during detection; TSV is generated on-the-fly for download only; legacy jobs with only TSV are lazily upgraded on first access
- **Timeline label editing**: enforces single-label-per-row (mutual exclusivity of humpback/orca/ship/background); batch edits via `PATCH /classifier/detection-jobs/{id}/labels` with overlap validation
- **Timeline vocalization label editing**: when in Vocalization overlay mode, the Label button enters vocalization label editing. Click any detection window to open a popover for adding/removing vocalization type labels. Changes accumulate locally with explicit Save/Cancel. Batch save via `PATCH /labeling/vocalization-labels/{id}/batch` with atomic transaction. `"(Negative)"` mutual exclusivity enforced server-side.
- **Vocalization training negatives**: training data assembly from detection jobs only includes explicitly labeled windows; unlabeled windows are excluded. `"(Negative)"` labels are converted to empty set (negative for all types). `"(Negative)"` is mutually exclusive with type labels on the same window.
- **Vocalization type name guard**: `"(Negative)"` is a reserved label string and cannot be used as a vocalization type name
- **Detection job deletion guard**: detection jobs with vocalization labels or training dataset references cannot be deleted; returns HTTP 409 with dependency details
- **Detection–vocalization consistency**: detection rows and vocalization labels are linked by stable `row_id` (UUID4). Deleting a detection row via batch edit cascade-deletes associated vocalization labels. No version tracking or reconciliation needed.
- **Candidate-backed replay training**: `source_mode="autoresearch_candidate"` training jobs use exact replay via `promoted_config` and the shared `replay.py` module — they bypass `train_binary_classifier` and `map_autoresearch_config_to_training_parameters`. Replay verification compares produced metrics against imported candidate metrics and persists the result in `training_summary` and `source_comparison_context`.
- **Timeline spectrogram normalization (PCEN)**: timeline tile rendering applies `librosa.pcen` to the STFT magnitude as a per-tile, per-frequency AGC. PCEN output is bounded, so tiles render with a fixed colormap range (`pcen_vmin`/`pcen_vmax`) and there is **no** per-job `ref_db` computation and no `.ref_db.json` cache artifact. Each tile is rendered independently from its own audio fetch extended backward by `pcen_warmup_sec` so the PCEN low-pass filter can settle; the warm-up frames are trimmed off the output before the PNG is generated. Tiles at the very start of a job use a shorter warm-up. The detection / classifier feature extraction pipeline is intentionally untouched — PCEN applies only to the timeline viewer and static export.
- **Timeline audio playback normalization**: playback audio (`/audio` endpoint and timeline export chunks) is RMS-scaled to `playback_target_rms_dbfs` inside `normalize_for_playback`, then soft-clipped with `tanh` at `playback_ceiling` before encoding. This equalizes gain jumps within a playback chunk without tracking per-job gain state; source audio is never modified. The spectrogram and audio paths deliberately use different normalizations because a good visualization filter is not a good listening filter.
- **Timeline cache version marker**: `TimelineTileCache.ensure_job_cache_current(job_id)` reads `.cache_version` from each job's cache directory and migrates legacy caches (pre-PCEN) on first access by deleting stale tile PNGs and legacy sidecars (`.ref_db.json`, `.gain_profile.json`) and writing the current version. `.prepare_plan.json`, `.audio_manifest.json`, and `.last_access` are preserved. Callers in `_prepare_tiles_sync` and `get_tile` invoke it; the migration is a one-shot no-op after the first call per job.
- **Window selection modes**: detection jobs have a `window_selection` parameter (`"nms"` default, `"prominence"`, or `"tiling"`). NMS produces non-overlapping peak windows via greedy suppression. Prominence transforms raw confidence scores to logit (log-odds) space before peak finding and prominence computation, amplifying meaningful dips in high-confidence regions where probability scores saturate. When `window_selection="prominence"`, `min_prominence` (default 1.0, in logit units) controls the minimum dip required to distinguish two peaks as separate vocalizations. If no peaks pass the prominence filter but the event has windows above threshold, a fallback emits the single highest-scoring window. After peak selection, a recursive gap-filling pass scans for uncovered regions (gaps > 5.0 seconds between consecutive peaks or from event edges to the nearest peak) and emits additional windows at the gap midpoint. Gap-filling is always-on in prominence mode. When `window_selection="tiling"`, a greedy multi-pass algorithm seeds from the highest-scoring uncovered window and tiles contiguously outward in both directions until the logit-space drop from the seed exceeds `max_logit_drop` (default 2.0). Uncovered windows above threshold become seeds for subsequent passes. This covers flat plateau regions without relying on peak finding or gap-filling heuristics.
- **Call parsing Pass 1 source contract**: `CallParsingRun` and `RegionDetectionJob` rows carry source identity as exactly one of (`audio_file_id`) or the hydrophone triple (`hydrophone_id` + `start_timestamp` + `end_timestamp`). The exactly-one-of invariant is enforced by the `CreateRegionJobRequest` Pydantic validator and the service layer's FK checks, not by a DB CHECK constraint — matching the `DetectionJob` pattern.
- **Call parsing Pass 1 streaming chunk alignment**: the region detection worker's hydrophone path fetches audio in chunks whose edges are whole multiples of `config.window_size_seconds` (via `_aligned_chunk_edges`), so no Perch window ever straddles a chunk boundary. Each per-chunk `score_audio_windows` call receives `time_offset_sec = chunk_start - range_start`, and the concatenated trace is hysteresis-merged in one pass. This guarantees the streaming path is mathematically equivalent to a single-buffer call on the full range.
- **Call parsing Pass 1 chunk artifacts (hydrophone only)**: the region detection worker writes per-chunk parquet files under `chunks/` and a `manifest.json` tracking planned vs completed chunks. DB columns `chunks_total`/`chunks_completed` are updated after each chunk for API polling. On crash, final artifacts (`trace.parquet`, `regions.parquet`) and `.tmp` files are cleaned up, but completed chunk parquets and manifest are preserved. A re-queued failed job resumes from the last completed chunk. File-based jobs bypass chunk artifacts entirely (ADR-052).
- **Call parsing Pass 1 single job-level timeline**: the region detection worker builds the HLS timeline once for the full `[start_ts, end_ts]` range before entering the chunk loop, then filters per chunk. Chunks with no overlapping segments are skipped immediately without calling `iter_audio_chunks`. The provider is constructed with `force_refresh=False` so all S3 listing operations prefer local cache.
- **Hydrophone S3 cache local-first mode**: `CachingS3Client` methods (`list_hls_folders`, `list_segments`) accept `force_refresh: bool = True`. With `force_refresh=False`, folder listings come from the local filesystem only, and segment listings come from a cached `.segments.json` manifest (written on first S3 query). `fetch_playlist` always checks for `live.m3u8.404.json` markers before calling S3. `CachingHLSProvider` stores `force_refresh` as instance state and threads it through `_build_stream_timeline`. Batch workloads (region detection) pass `force_refresh=False`; live streaming callers use the default `True`.
- **Call parsing Pass 2 framewise α supervision**: the segmentation CRNN's training target is binary framewise presence inside/outside event bounds, with masked `BCEWithLogitsLoss` and auto-computed `pos_weight`. Bootstrap row bounds are too loose for onset/offset point targets; α is the simplest target that gives the hysteresis decoder a clean frame probability vector.
- **Call parsing Pass 2 per-audio-file train/val split**: `split_by_audio_source` groups training samples by `audio_file_id` (or `hydrophone_id`) and ensures no audio source appears in both train and val. When only one source group exists (common when all labels come from a single hydrophone), the function falls back to a temporal split: samples are sorted by `start_timestamp` and the last `val_fraction` go to val, giving temporal separation within a single source. Per-sample random splits leak background noise signature in bioacoustic ML.
- **Call parsing Pass 2 inherits source from upstream**: the event segmentation worker resolves the audio source from the upstream Pass 1 `RegionDetectionJob`'s `audio_file_id` or `hydrophone_id` columns, never from the Pass 2 `EventSegmentationJob` row itself. Pass 2 rows carry no source identity columns.
- **Call parsing Pass 3 model family validation**: `POST /classification-jobs` validates that the `vocalization_model_id` has `model_family='pytorch_event_cnn'` and `input_mode='segmented_event'` (422 if wrong). Existing `sklearn_perch_embedding` models are not usable through the call parsing classification job flow — they remain exclusive to detection-job rescoring and embedding-set inference.
- **Call parsing Pass 3 frequency-only pooling**: the `EventClassifierCNN` uses `MaxPool2d((2,1))` after each convolutional block, pooling only in the frequency axis and preserving the time dimension. This ensures short events (~6 time frames at 0.2s) survive all 4 conv blocks without collapsing. `AdaptiveAvgPool2d((1,1))` handles the final collapse to a fixed-size vector regardless of input time length.
- **Call parsing Pass 3 bootstrap single-label filter**: the event classifier bootstrap script (`scripts/bootstrap_event_classifier_dataset.py`) only uses detection windows with exactly one distinct vocalization type label (excluding `(Negative)`). Multi-label windows are ambiguous at the event level — a 5-second detection window may contain multiple events of different types, so assigning all types to all contained events would be incorrect. `(Negative)` windows produce no positive training samples.
- **Call parsing Pass 3 inherits source transitively**: the event classification worker resolves the audio source from the Pass 1 `RegionDetectionJob` through Pass 2's `region_detection_job_id` column. Pass 3 rows carry no source identity columns, matching the Pass 2 pattern.
- **Feedback training correction tables**: human corrections are stored in separate SQL tables (`event_boundary_corrections`, `event_type_corrections`), not by amending parquet output. Original inference parquet files (`events.parquet`, `typed_events.parquet`) remain immutable. Corrections reference events by `event_id`.
- **Feedback training implicit approval**: regions/events in source jobs that have no corrections are included in training as-is. The absence of a correction constitutes implicit approval of the inference output. No explicit "approve" action is needed.
- **Feedback training single-label Pass 3**: event type corrections enforce single-label semantics — each event has exactly one `type_name` or null (negative). This differs from the multi-label vocalization labeling system. Unique constraint on `(event_classification_job_id, event_id)`.
- **Feedback training hydrophone-only**: all audio sources in the feedback training workflow are hydrophone-based. Audio is resolved through the job chain (classification → segmentation → region detection → hydrophone) using `resolve_timeline_audio`. File-based audio is not supported in feedback training workers.
- **Bootstrap training scripts**: bootstrap scripts (`scripts/bootstrap_segmentation_dataset.py`, `scripts/bootstrap_event_classifier_dataset.py`) call trainer library functions directly — they do not queue worker jobs. The `vocalization_worker.py` only handles `sklearn_perch_embedding` model family; `pytorch_event_cnn` training is handled exclusively by feedback training workers.

### 8.8 Classifier API Surface

Classifier training currently has three distinct flows:

- **Embedding-set training**: `POST /classifier/training-jobs` creates the original positive/negative embedding-set-backed training jobs.
- **Hyperparameter tuning** (UI-driven manifest generation + random search):
  - `POST /classifier/hyperparameter/manifests` — create and queue manifest generation
  - `GET /classifier/hyperparameter/manifests` — list manifests
  - `GET /classifier/hyperparameter/manifests/{id}` — manifest detail
  - `DELETE /classifier/hyperparameter/manifests/{id}` — delete (409 if referenced by search)
  - `POST /classifier/hyperparameter/searches` — create and queue search job
  - `GET /classifier/hyperparameter/searches` — list searches
  - `GET /classifier/hyperparameter/searches/{id}` — search detail
  - `GET /classifier/hyperparameter/searches/{id}/history` — trial history
  - `DELETE /classifier/hyperparameter/searches/{id}` — delete search + artifacts
  - `GET /classifier/hyperparameter/search-space-defaults` — default search space
  - `POST /classifier/hyperparameter/searches/{id}/import-candidate` — import as candidate
- **Autoresearch candidate review and promotion** (relocated under `/classifier/hyperparameter/candidates/*`; old `/classifier/autoresearch-candidates/*` paths still work):
  - `POST /classifier/hyperparameter/candidates/import`
  - `GET /classifier/hyperparameter/candidates`
  - `GET /classifier/hyperparameter/candidates/{candidate_id}`
  - `DELETE /classifier/hyperparameter/candidates/{candidate_id}` — delete candidate
  - `POST /classifier/hyperparameter/candidates/{candidate_id}/training-jobs`
- **Legacy retrain workflow**:
  - `GET /classifier/models/{id}/retrain-info`
  - `POST /classifier/retrain`
  - `GET /classifier/retrain-workflows`
  - `GET /classifier/retrain-workflows/{id}`

Candidate-backed promotion imports reviewed autoresearch artifacts, persists a durable `AutoresearchCandidate`, and creates manifest-backed training jobs that keep source candidate and comparison provenance on both the training job and resulting classifier model. After candidate-backed training completes, the training job's `source_comparison_context` and the model's `training_summary` include a `replay_verification` dict with status (`"verified"`/`"mismatch"`), per-split metric comparisons, and effective config. The candidate detail endpoint (`GET /classifier/autoresearch-candidates/{id}`) also exposes `replay_verification` when the linked model exists.

- **Timeline export**:
  - `POST /classifier/detection-jobs/{id}/timeline/export` — export a completed detection job's timeline as a self-contained static bundle (tiles, MP3 audio, JSON manifest) for hosting as a readonly viewer on S3. Also available as `scripts/export_timeline.py` CLI.

### 8.9 Call Parsing Pipeline API Surface

Four-pass pipeline under `/call-parsing/*`. Passes 1–3 are fully functional; Pass 4 still returns HTTP 501 for the sequence export endpoint. Per-pass job status transitions follow the standard `queued → running → complete|failed|canceled` pattern.

- **Parent runs** (functional):
  - `POST /call-parsing/runs` — create a parent run and its queued Pass 1 job in one transaction; accepts the same source + model + config shape as `POST /region-jobs`
  - `GET /call-parsing/runs` — list runs with pagination
  - `GET /call-parsing/runs/{id}` — parent run with nested Pass 1/2/3 status summaries
  - `DELETE /call-parsing/runs/{id}` — cascade deletes all three child jobs and their parquet directories
  - `GET /call-parsing/runs/{id}/sequence` — **501** (Pass 4 sequence export)
- **Pass 1 — region detection** (functional):
  - `POST /call-parsing/region-jobs` — create a queued Pass 1 job from an audio-file source or a hydrophone range; validates every FK and the `source` XOR invariant
  - `GET /call-parsing/region-jobs`, `GET /call-parsing/region-jobs/{id}`, `DELETE /call-parsing/region-jobs/{id}` — list / detail / delete
  - `GET /call-parsing/region-jobs/{id}/trace` — stream `trace.parquet` as JSON `{time_sec, score}` rows; 409 while the job is not `complete`, 404 if the parquet file is missing
  - `GET /call-parsing/region-jobs/{id}/regions` — return `regions.parquet` sorted by `start_sec`; same 409/404 guards
- **Pass 2 — event segmentation** (functional):
  - `POST /call-parsing/segmentation-jobs` — create a queued Pass 2 job; validates `region_detection_job_id` (404) and `segmentation_model_id` (404); 409 if upstream Pass 1 job is not `complete`
  - `GET /call-parsing/segmentation-jobs`, `GET /call-parsing/segmentation-jobs/{id}`, `DELETE /call-parsing/segmentation-jobs/{id}` — list / detail / delete
  - `GET /call-parsing/segmentation-jobs/{id}/events` — return `events.parquet` as JSON; 409 while job is not `complete`, 404 if parquet file is missing
- **Pass 2 — segmentation training datasets** (functional):
  - `GET /call-parsing/segmentation-training-datasets` — list training datasets with sample counts (used for bootstrap dataset inspection; bootstrap scripts call trainers directly)
- **Pass 2 — segmentation models** (functional):
  - `GET /call-parsing/segmentation-models` — list models with condensed metrics
  - `GET /call-parsing/segmentation-models/{id}` — full detail
  - `DELETE /call-parsing/segmentation-models/{id}` — removes row + checkpoint directory on disk; 409 if referenced by an in-flight segmentation job
- **Pass 2 — boundary corrections** (functional):
  - `POST /call-parsing/segmentation-jobs/{id}/corrections` — batch upsert boundary corrections; validates job exists (404) and is complete (409); returns correction count
  - `GET /call-parsing/segmentation-jobs/{id}/corrections` — list all corrections for a segmentation job
  - `DELETE /call-parsing/segmentation-jobs/{id}/corrections` — clear all corrections; 204
- **Pass 3 — event classification** (functional):
  - `POST /call-parsing/classification-jobs` — create a queued Pass 3 job; validates `vocalization_model_id` exists (404) and has `model_family='pytorch_event_cnn'` + `input_mode='segmented_event'` (422); validates `event_segmentation_job_id` exists (404) and is `complete` (409)
  - `GET /call-parsing/classification-jobs`, `GET /call-parsing/classification-jobs/{id}`, `DELETE /call-parsing/classification-jobs/{id}` — list / detail / delete
  - `GET /call-parsing/classification-jobs/{id}/typed-events` — return `typed_events.parquet` as JSON sorted by `start_sec`; 409 while job is not `complete`, 404 if parquet file is missing
- **Pass 3 — type corrections** (functional):
  - `POST /call-parsing/classification-jobs/{id}/corrections` — batch upsert type corrections; unique on `(job_id, event_id)`; validates job exists (404) and is complete (409); returns correction count
  - `GET /call-parsing/classification-jobs/{id}/corrections` — list all corrections for a classification job
  - `DELETE /call-parsing/classification-jobs/{id}/corrections` — clear all corrections; 204
- **Pass 2 — segmentation feedback training** (functional):
  - `POST /call-parsing/segmentation-feedback-training-jobs` — create queued job; validates all source segmentation job IDs exist and are complete (404/409); 201
  - `GET /call-parsing/segmentation-feedback-training-jobs` — list jobs
  - `GET /call-parsing/segmentation-feedback-training-jobs/{id}` — detail; 404 if not found
  - `DELETE /call-parsing/segmentation-feedback-training-jobs/{id}` — deletes job row; produced models managed via segmentation model endpoints; 204; 404 if not found
- **Pass 3 — classifier feedback training** (functional):
  - `POST /call-parsing/classifier-training-jobs` — create queued job; validates all source classification job IDs exist and are complete (404/409); 201
  - `GET /call-parsing/classifier-training-jobs` — list jobs
  - `GET /call-parsing/classifier-training-jobs/{id}` — detail; 404 if not found
  - `DELETE /call-parsing/classifier-training-jobs/{id}` — deletes job row; produced models managed via classifier model endpoints; 204; 404 if not found
- **Pass 3 — classifier model management** (functional):
  - `GET /call-parsing/classifier-models` — list `pytorch_event_cnn` models
  - `DELETE /call-parsing/classifier-models/{id}` — deletes model + checkpoint directory; 409 if referenced by in-flight classification or training jobs; 204; 404 if not found
- **Pass 3 — event classifier training** (bootstrap only):
  - Bootstrap scripts call `train_event_classifier()` directly; no API endpoint or worker queue for bootstrap training
  - `scripts/bootstrap_event_classifier_dataset.py` generates training samples from vocalization-labeled detection windows

---

## 9. Current State

### 9.1 Implemented Capabilities

- Audio upload, folder import, metadata editing
- Processing pipeline: TFLite + TF2 SavedModel, overlap-back windowing, incremental Parquet
- Embedding similarity search (cosine/euclidean, cross-set, detection-sourced, score calibration with percentile ranks and distribution histogram, pluggable projector for future classifier-projected search)
- Clustering: HDBSCAN/K-Means/Agglomerative, UMAP/PCA, parameter sweeps, metric learning
- Binary classifier training (LogisticRegression/MLP) + local/hydrophone detection with configurable window selection (NMS non-overlapping or prominence-based overlapping)
- Hydrophone streaming: Orcasound HLS + NOAA archives, pause/resume/cancel, subprocess isolation
- Label processing: score-based + sample-builder workflows
- Vocalization labeling: per-window type labels on detection rows linked by stable row_id, cascade deletion on row removal
- Multi-label vocalization classifier: managed vocabulary, binary relevance training (per-type sklearn pipeline), per-type threshold optimization, multi-source inference (detection job / embedding set / rescore), paginated results with client-side threshold filtering, TSV export
- Vocalization labeling workspace: source abstraction (detection jobs / embedding sets / local folders), progressive pipeline (source → embeddings → inference → labeling), local-state label accumulation with batch Save/Cancel, three visual label states (suggested/saved/pending), score-sorted results by default, click-to-expand spectrogram popup, one-click retrain loop
- Training dataset review: unified editable snapshot of training data (from embedding sets and detection jobs), type-filtered positive/negative browsing with large inline spectrograms, batch label editing with save/cancel, dataset extend for incremental source addition, retrain from edited labels
- Hyperparameter tuning: UI-driven manifest generation from training/detection jobs, random hyperparameter search with configurable search space, production model comparison, candidate import from search results
- Autoresearch candidate promotion: import reviewed manifest-backed autoresearch bundles, inspect split deltas versus production, and create candidate-backed training jobs with persisted provenance. AR-v1 exact replay: candidates with PCA, probability calibration (platt/isotonic), context pooling (mean3/max3), and linear SVM (wrapped in `CalibratedClassifierCV` for probability output) are promotable via shared replay module with replay verification against imported candidate metrics
- Retrain workflow: reimport -> reprocess -> retrain
- Timeline viewer: zoomable spectrogram with startup-scoped background tile pre-caching plus bounded in-memory manifest/PCM reuse, interactive species labeling (add/move/delete/change-type with batch save at 1m and 5m zoom), warm/cool color-coded detection label bars with hover tooltips, toggleable vocalization type overlay (inference suggestions + manual labels as purple bars with colored type badges), vocalization label editing via popover (add/remove type labels on detection windows with batch save), audio-authoritative playhead sync, gapless double-buffered MP3 playback, embedding sync button (diff row store vs embeddings, generate missing, remove orphans), static export for readonly S3-hosted viewer, PCEN spectrogram normalization and RMS-targeted audio playback
- Web UI: routed SPA with Audio, Processing, Clustering, Classifier (Training, Hydrophone, Embeddings, Tuning), Vocalization, Call Parsing (Detection, Segment, Segment Training), Search, Label Processing, Admin
- Four-pass call parsing pipeline — Phase 0 scaffold + Pass 1 region detection + Pass 2 event segmentation + Pass 3 event classification + human feedback training loop: streaming Perch inference on uploaded files or hydrophone ranges, hysteresis + padded-overlap merge, `trace.parquet` + `regions.parquet` outputs; hydrophone jobs write per-chunk parquet artifacts with manifest-based resume and DB progress tracking (`chunks_total`/`chunks_completed`); PyTorch CRNN framewise segmentation training + inference (both file-based and hydrophone-sourced via `resolve_timeline_audio`) with hysteresis decoder producing `events.parquet`; bootstrap scripts for building training datasets and event classifier samples from vocalization-labeled hydrophone detection rows (hydrophone-only, context-padded audio fetch for z-score normalization); temporal train/val split fallback for single-source datasets; PyTorch CNN event classifier training on variable-length event crops with per-type threshold optimization, `typed_events.parquet` output; human-in-the-loop correction storage (boundary corrections for Pass 2, type corrections for Pass 3) with feedback training workers that assemble corrected + implicitly-approved training data and retrain models; correction and feedback training CRUD API endpoints; bootstrap training paths removed from backend (vocalization worker is sklearn-only, segmentation training worker deleted). Pass 4 still deferred.

### 9.2 Database Schema

- **Engine**: SQLite via SQLAlchemy
- **Latest migration**: `046_feedback_training_tables.py`
- **Tables**: model_configs, audio_files, audio_metadata, processing_jobs, embedding_sets, clustering_jobs, clusters, cluster_assignments, classifier_models, classifier_training_jobs, autoresearch_candidates, detection_jobs, retrain_workflows, label_processing_jobs, vocalization_labels, vocalization_types, vocalization_models, vocalization_training_jobs, vocalization_inference_jobs, detection_embedding_jobs, training_datasets, training_dataset_labels, hyperparameter_manifests, hyperparameter_search_jobs, call_parsing_runs, region_detection_jobs, event_segmentation_jobs, event_classification_jobs, segmentation_models, segmentation_training_datasets, segmentation_training_samples, segmentation_training_jobs, event_boundary_corrections, event_type_corrections, event_segmentation_training_jobs, event_classifier_training_jobs

### 9.3 Sensitive Components

| Component | Risk | Why |
|-----------|------|-----|
| `processing/windowing.py` | Signal integrity | Affects all downstream embeddings |
| `processing/features.py` | Signal integrity | Spectrogram shape must be 128x128 |
| `processing/parquet_writer.py` | Data integrity | Atomic write semantics |
| `database.py` | Schema | Must match Alembic migrations |
| `encoding_signature` computation | Idempotency | Duplicate prevention depends on this |
| `clustering/engine.py` | Correctness | Metrics and assignments must be consistent |
| `classifier/trainer.py` | Model quality | Class weight balance, CV splits |

### 9.4 Known Constraints

- SQLite has no true row-level locking; worker claims rely on `UPDATE` plus status checks.
- The UI remains polling-based rather than real-time.
- Deployment is still single-machine MVP infrastructure.
- Exactly one TensorFlow extra must be selected per environment; `uv sync --all-extras` is invalid.
- Linux GPU installs assume a modern glibc baseline compatible with TensorFlow CUDA wheels.
- Model files must be present on disk; there is no remote model registry.
- Pyright enforcement covers `src/humpback`, `scripts/`, and `tests/`.
- `HUMPBACK_ALLOWED_HOSTS` uses Starlette wildcard syntax such as `*.example.com`, not `.example.com`.
- Audio shorter than `window_size_seconds` (5 seconds) is skipped entirely.
- Imported audio must remain at its original path for in-place reads.

---

## 10. Workflow

### 10.1 Session Workflow

This project uses lightweight session workflow skills stored in `docs/workflows/`.

**Canonical flow for every task:**

```
session-begin -> brainstorm -> session-plan -> session-implement -> [session-debug]* -> session-review -> session-end
```

- `brainstorm` uses the superpowers brainstorming skill (unchanged)
- `[session-debug]*` means zero or more rounds of debugging after manual testing
- All other steps use `docs/workflows/session-*.md`
- In Claude Code, invoke as `/session-begin`, `/session-plan`, etc.

**Brainstorming overrides:**
- Spec path: `docs/specs/YYYY-MM-DD-<topic>-design.md` (not `docs/superpowers/specs/`)
- Spec is NOT written to disk during brainstorming — `session-plan` writes and commits it on the feature branch
- Brainstorming skill steps 6–8 (write doc, self-review, user review file) are skipped; the approved design lives in conversation context until `session-plan` writes it
- After brainstorming, control passes to `session-plan` (not superpowers `writing-plans`)

**Artifact locations:**
- Workflow skills: `docs/workflows/session-*.md`
- Design specs: `docs/specs/YYYY-MM-DD-<topic>-design.md`
- Implementation plans: `docs/plans/YYYY-MM-DD-<feature>.md`

### 10.2 Feature Branch Lifecycle

All work for a task — spec, plan, and implementation — lives on a single feature branch.
Nothing is committed to `main` directly; `main` only advances via squash-merge of PRs.

**Branch creation (session-plan):**
1. Brainstorming writes the spec on main (uncommitted)
2. `session-plan` creates `feature/<feature-name>` from main
3. Spec and plan are committed as the first commit on the feature branch

**Implementation (session-implement):**
- Work directly on the feature branch (no worktrees or subagents)
- Single batched commit after all plan tasks complete

**Completion (session-end):**
1. Push feature branch, create PR against main
2. Squash-merge the PR
3. Checkout main, `git pull --ff-only origin main`
4. Delete local feature branch

**Early exit (session ends before implementation is complete):**
1. Commit any uncommitted work to the feature branch
2. Push feature branch to remote
3. Checkout main (leave working directory clean on main)
4. Next session detects the feature branch and resumes — see §10.3

### 10.3 Session Start Checklist

Use `session-begin` (`/session-begin` in Claude Code) at the start of every session.
See `docs/workflows/session-begin.md` for full steps. Summary:

1. Normalize the repo onto local `main` (fast-forward from origin; stop if dirty or detached)
2. Read CLAUDE.md and DECISIONS.md
3. Check for active local feature branches with in-progress work:
   - Look for local `feature/*` branches
   - Check `docs/plans/` and `docs/specs/` on those branches for active specs/plans
4. If an active feature branch exists, offer to resume on it
5. If no active work, summarize current state and begin brainstorming for the next task

### 10.4 Project Verification Gates

Before claiming work is complete, run these in order:
1. `uv run ruff format --check` on modified Python files
2. `uv run ruff check` on modified Python files
3. `uv run pyright` on modified Python files (full run if pyproject.toml changed)
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit` (if frontend files changed)

**Doc-update matrix:**

| Change type | Update |
|---|---|
| API endpoints added/changed | CLAUDE.md §8, README.md |
| Data model changed | `docs/reference/data-model.md`, Alembic migration |
| Signal processing changed | `docs/reference/signal-processing.md`, DECISIONS.md |
| New capability | CLAUDE.md §9.1 |
| Constraint changed | CLAUDE.md §9.4 |
| Architecture decision | DECISIONS.md |
| Frontend routes/components | `docs/reference/frontend.md` |
| Hydrophone behavior | `docs/reference/hydrophone-rules.md` |
| Storage paths | `docs/reference/storage-layout.md` |

### 10.5 Codex Compatibility

Codex follows the same session workflow phases but uses only Codex-available
tools (file read/write, bash, grep, glob). The workflow files in `docs/workflows/`
are the shared source of truth for both Claude Code and Codex. See AGENTS.md for
Codex-specific phase mapping.
