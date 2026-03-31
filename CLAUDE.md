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
├── alembic/versions/      (migration scripts, 001–025)
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
- `timeline_prepare_workers` defaults to `2`; startup/full tile batches share one per-job `ref_db` and may render through a bounded worker pool.
- `timeline_startup_radius_tiles` defaults to `2`; the Timeline button now triggers startup-scoped cache warming around the initial viewport rather than a full all-zoom warmup.
- `timeline_startup_coarse_levels` defaults to `1`, `timeline_neighbor_prefetch_radius` defaults to `1`, `timeline_tile_memory_cache_items` defaults to `256`, `timeline_manifest_memory_cache_items` defaults to `8`, and `timeline_pcm_memory_cache_mb` defaults to `128` for bounded in-memory timeline reuse.

### 8.7 Behavioral Constraints

Non-obvious constraints that are not immediately derivable from code:

- **Worker priority order**: search -> processing -> clustering -> classifier training -> detection -> extraction -> detection embedding generation -> label processing -> retrain -> vocalization training -> vocalization inference
- **Job claim semantics**: Workers claim queued jobs via atomic compare-and-set (`WHERE id=:candidate AND status='queued'`). SQLite has no true row-level locks; correctness relies on atomic status updates, not `SELECT ... FOR UPDATE`.
- **Job status transitions**: `queued -> running -> complete`, `queued -> running -> failed`, `queued -> canceled`
- **Processing concurrency**: prevent two running ProcessingJobs for same encoding_signature; allow multiple clustering jobs in parallel
- **Prefetch semantics**: `time_covered_sec` tracks summed processed audio duration rather than wall-clock range coverage
- **Parquet row-store**: detection jobs write directly to Parquet row store during detection; TSV is generated on-the-fly for download only; legacy jobs with only TSV are lazily upgraded on first access
- **Timeline label editing**: enforces single-label-per-row (mutual exclusivity of humpback/orca/ship/background); batch edits via `PATCH /classifier/detection-jobs/{id}/labels` with overlap validation
- **Vocalization training negatives**: training data assembly from detection jobs only includes explicitly labeled windows; unlabeled windows are excluded. `"(Negative)"` labels are converted to empty set (negative for all types). `"(Negative)"` is mutually exclusive with type labels on the same window.
- **Vocalization type name guard**: `"(Negative)"` is a reserved label string and cannot be used as a vocalization type name

---

## 9. Current State

### 9.1 Implemented Capabilities

- Audio upload, folder import, metadata editing
- Processing pipeline: TFLite + TF2 SavedModel, overlap-back windowing, incremental Parquet
- Embedding similarity search (cosine/euclidean, cross-set, detection-sourced, score calibration with percentile ranks and distribution histogram, pluggable projector for future classifier-projected search)
- Clustering: HDBSCAN/K-Means/Agglomerative, UMAP/PCA, parameter sweeps, metric learning
- Binary classifier training (LogisticRegression/MLP) + local/hydrophone detection
- Hydrophone streaming: Orcasound HLS + NOAA archives, pause/resume/cancel, subprocess isolation
- Label processing: score-based + sample-builder workflows
- Vocalization labeling: per-window type labels on detection rows
- Multi-label vocalization classifier: managed vocabulary, binary relevance training (per-type sklearn pipeline), per-type threshold optimization, multi-source inference (detection job / embedding set / rescore), paginated results with client-side threshold filtering, TSV export
- Vocalization labeling workspace: source abstraction (detection jobs / embedding sets / local folders), progressive pipeline (source → embeddings → inference → labeling), local-state label accumulation with batch Save/Cancel, three visual label states (suggested/saved/pending), score-sorted results by default, click-to-expand spectrogram popup, one-click retrain loop
- Training dataset review: unified editable snapshot of training data (from embedding sets and detection jobs), type-filtered positive/negative browsing with large inline spectrograms, batch label editing with save/cancel, dataset extend for incremental source addition, retrain from edited labels
- Retrain workflow: reimport -> reprocess -> retrain
- Timeline viewer: zoomable spectrogram with startup-scoped background tile pre-caching plus bounded in-memory manifest/PCM reuse, interactive species labeling (add/move/delete/change-type with batch save at 1m and 5m zoom), warm/cool color-coded detection label bars with hover tooltips, audio-authoritative playhead sync, gapless double-buffered MP3 playback
- Web UI: routed SPA with Audio, Processing, Clustering, Classifier, Vocalization, Search, Label Processing, Admin

### 9.2 Database Schema

- **Engine**: SQLite via SQLAlchemy
- **Latest migration**: `032_training_datasets.py`
- **Tables**: model_configs, audio_files, audio_metadata, processing_jobs, embedding_sets, clustering_jobs, clusters, cluster_assignments, classifier_models, classifier_training_jobs, detection_jobs, retrain_workflows, label_processing_jobs, vocalization_labels, vocalization_types, vocalization_models, vocalization_training_jobs, vocalization_inference_jobs, detection_embedding_jobs, training_datasets, training_dataset_labels

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
