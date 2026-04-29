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

**⚠️ MANDATORY: Back up the production database BEFORE any database change.**
Any operation that modifies the database — migrations, data backfills, manual SQL, schema changes, or destructive deletes — **must** be preceded by a backup. No exceptions.

1. Read the production database path from the `HUMPBACK_DATABASE_URL` entry in `.env`.
2. Copy the database file to `<original_path>.YYYY-MM-DD-HH:mm.bak` (UTC timestamp).
   Example: `cp "$DB_PATH" "${DB_PATH}.2026-04-24-18:30.bak"`
3. Confirm the backup file exists and has a non-zero size before proceeding.

If the backup step fails or is skipped, **stop** — do not apply the migration or modification.

*   Honor database file location defined with HUMPBACK_DATABASE_URL, check .env override.  
*   When a change adds, removes, or renames columns on any SQL table, **always** create an Alembic migration in `alembic/versions/` and run it with `uv run alembic upgrade head` before verifying the change works.
*   Migration files follow the naming convention `NNN_short_description.py` (e.g., `007_negative_embedding_set_ids.py`), incrementing from the latest revision.
*   Use `op.batch_alter_table()` for SQLite compatibility.

### 3.6 Documentation
*   When a change adds, removes, or modifies API endpoints, data models, configuration options, architecture, or workflows, update the relevant files:
    *   `CLAUDE.md` — rules, project state summary, pointers to reference docs (this file)
    *   `DECISIONS.md` — append new ADR for significant architecture changes
    *   `README.md` — user-facing API endpoints, configuration, feature list
    *   `docs/specs/` — design specs (written during brainstorming phase)
    *   `docs/plans/` — implementation plans (written during planning phase)
    *   `docs/reference/` — detailed reference material (data model, signal processing, storage, frontend, hydrophone rules, testing, behavioral constraints, API surfaces, runtime config)
*   **Keep CLAUDE.md lean.** CLAUDE.md is auto-loaded into every conversation. New behavioral constraints, API endpoint listings, runtime config details, and other reference material must go into the appropriate `docs/reference/` file — not inline in CLAUDE.md. CLAUDE.md §8 should contain only one-line pointers to reference docs. See §10.2 doc-update matrix for which file gets which content.

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

### 4.1 Idempotent Derived Artifacts
Retained artifact producers must not create duplicate canonical outputs.

Current guarantees:
- `detection_embedding_jobs` are unique per `(detection_job_id, model_version)`
- `continuous_embedding_jobs` are unique per `encoding_signature`
- bootstrap/training-dataset loaders preserve sample-level uniqueness through their retained source identifiers

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

During implementation sessions, per-task verification uses targeted inline tests plus a background sub-agent for the full suite. See `docs/workflows/session-implement.md` step 5 (Per-task testing) for details.

---

## 6. Definition of Done (Engineering)
A PR/change is "done" only if:
- unit tests added/updated for changed behavior
- test suite passes locally
- E2E smoke test passes locally
- Playwright tests added/updated for UI changes (`cd frontend && npx playwright test`)
- retained idempotency rules preserved (for example, no duplicate detection embeddings or continuous-embedding jobs)

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
│   └── reference/         (detailed reference: data model, signal processing, storage, frontend, hydrophone, testing, behavioral constraints, API surfaces, runtime config)
└── data/                  (runtime data)
```

### 8.3 Data Model Summary
See [docs/reference/data-model.md](docs/reference/data-model.md) for condensed model reference. Full field lists in `src/humpback/database.py`.

### 8.4 Signal Processing Parameters
See [docs/reference/signal-processing.md](docs/reference/signal-processing.md) for parameter defaults, windowing rules, and processing pipeline diagram.

### 8.5 Storage Layout
See [docs/reference/storage-layout.md](docs/reference/storage-layout.md) for full directory tree of all artifact types.

### 8.6 Runtime Configuration
See [docs/reference/runtime-config.md](docs/reference/runtime-config.md) for all `HUMPBACK_`-prefixed settings, defaults, and PCEN/playback parameters.

### 8.7 Behavioral Constraints
See [docs/reference/behavioral-constraints.md](docs/reference/behavioral-constraints.md) for non-obvious constraints covering job system, detection/labeling, classifier/training, timeline rendering, window selection, call parsing pipeline, and feedback training.

### 8.8 Classifier API Surface
See [docs/reference/classifier-api.md](docs/reference/classifier-api.md) for endpoint listings (embedding-set training, hyperparameter tuning, autoresearch candidates, legacy retrain, timeline export).

### 8.9 Call Parsing Pipeline API Surface
See [docs/reference/call-parsing-api.md](docs/reference/call-parsing-api.md) for endpoint listings (parent runs, Pass 1–3 CRUD, corrections, feedback training, model management).

### 8.10 Timeline Compound-Component Architecture
See [docs/reference/behavioral-constraints.md](docs/reference/behavioral-constraints.md) § "Timeline Compound-Component Architecture" for composability rules: ref-based playback handle, keyboard shortcut opt-out, zoom persistence, overlay context.

### 8.11 Sequence Models API Surface
See [docs/reference/sequence-models-api.md](docs/reference/sequence-models-api.md) for endpoint listings (continuous embedding producer; HMM + interpretation visualizations land in subsequent PRs).

---

## 9. Current State

### 9.1 Implemented Capabilities

- Detection-job-based classifier training (LogisticRegression/MLP), hyperparameter tuning, autoresearch candidate promotion, and legacy-model visibility; perch_v2 is a first-class embedding model family with detection-manifest training and model-versioned re-embedding (ADR-055)
- Hydrophone streaming (Orcasound HLS + NOAA), detection with configurable window selection (NMS/prominence/tiling)
- Vocalization labeling workspace, managed multi-label vocalization classifier, training dataset review, and Vocalization / Clustering on detection-job embeddings
- Timeline viewer with PCEN spectrograms, interactive labeling, gapless playback, static export
- Four-pass call parsing pipeline (Passes 1–3 functional, Pass 4 deferred): region detection, event segmentation, event classification, human feedback training loop
- Window classification sidecar: standalone enrichment scoring cached Pass 1 Perch embeddings through vocalization classifiers, producing dense per-window probability vectors with event-level review workspace (requires Pass 2 segmentation) and unified vocalization corrections shared with Classify review
- Classify review boundary editing: adjust/add/delete event boundaries in Pass 3 review, corrections flow to both classification inference and classifier feedback training via read-time overlay (ADR-054); type corrections use unified vocalization_corrections table shared with Window Classify review
- Pass 2 (event segmentation) and Pass 3 (event classification) inference run on MPS/CUDA when available with per-job load-time validation against CPU output and a fallback to CPU on validation failure; the chosen device and any fallback reason are persisted on the job row and surfaced as a UI badge
- Sequence Models track (parallel to Call Parsing): continuous embedding producer that emits event-scoped, hydrophone-only 1-second-hop SurfPerch embeddings padded around Pass-2 segmentation events (each event is an independent span — no merging), idempotent on `encoding_signature` (ADR-056). HMM training (PCA + GaussianHMM via hmmlearn) with Viterbi decode, state timeline visualization with dual region/event navigation (A/D keyboard shortcuts), transition matrix heatmap, and dwell-time histograms. Interpretation visualizations: PCA/UMAP overlay scatter colored by HMM state, state-to-label distribution via center-time join with vocalization labels, per-state exemplar gallery (high-confidence, nearest-centroid, boundary picks). Motif mining lands in subsequent PRs.
- Web UI: Classifier (Training, Hydrophone Detection, Embeddings, Tuning, Labeling), Vocalization (Training, Labeling, Training Data, Clustering), Call Parsing (Detection, Segment, Segment Training, Classify, Classify Training, Window Classify), Sequence Models (Continuous Embedding, HMM Sequence), Admin

### 9.2 Database Schema

- **Engine**: SQLite via SQLAlchemy
- **Latest migration**: `060_legacy_workflow_removal.py`
- **Tables**: model_configs, audio_files, clustering_jobs, clusters, cluster_assignments, classifier_models, classifier_training_jobs, autoresearch_candidates, detection_jobs, retrain_workflows, vocalization_labels, vocalization_types, vocalization_models, vocalization_training_jobs, vocalization_inference_jobs, detection_embedding_jobs, training_datasets, training_dataset_labels, hyperparameter_manifests, hyperparameter_search_jobs, call_parsing_runs, segmentation_models, region_detection_jobs, event_segmentation_jobs, event_classification_jobs, window_classification_jobs, vocalization_corrections, event_boundary_corrections, segmentation_training_datasets, segmentation_training_samples, segmentation_training_jobs, region_boundary_corrections, event_classifier_training_jobs, continuous_embedding_jobs, hmm_sequence_jobs

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
| `processing/region_windowing.py` | Sequence integrity | Merge geometry + window-center membership rules feed every downstream HMM sequence consumer |

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

Canonical flow: `session-begin -> brainstorm -> session-plan -> session-implement -> [session-debug]* -> session-review -> session-end`

- Detailed steps for each phase live in `docs/workflows/session-*.md`
- In Claude Code, invoke as `/session-begin`, `/session-plan`, etc.
- Codex uses the same workflow files — see `AGENTS.md` for tool mapping

**Brainstorming overrides:** spec path is `docs/specs/YYYY-MM-DD-<topic>-design.md`; spec is NOT written during brainstorming (session-plan writes it on the feature branch); skip brainstorming steps 6–8; control passes to session-plan (not superpowers writing-plans).

**Branch lifecycle:** all work on `feature/<name>` branches, never commit to main directly. Main advances only via squash-merge of PRs.

### 10.2 Project Verification Gates

Before claiming work is complete, run these in order:
1. `uv run ruff format --check` on modified Python files
2. `uv run ruff check` on modified Python files
3. `uv run pyright` on modified Python files (full run if pyproject.toml changed)
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit` (if frontend files changed)

**Doc-update matrix:**

| Change type | Update |
|---|---|
| API endpoints added/changed | relevant `docs/reference/*-api.md`, README.md |
| Data model changed | `docs/reference/data-model.md`, Alembic migration |
| Signal processing changed | `docs/reference/signal-processing.md`, DECISIONS.md |
| New capability | CLAUDE.md §9.1 |
| Constraint changed | CLAUDE.md §9.4 |
| Architecture decision | DECISIONS.md |
| Behavioral constraint added | `docs/reference/behavioral-constraints.md` |
| Frontend routes/components | `docs/reference/frontend.md` |
| Hydrophone behavior | `docs/reference/hydrophone-rules.md` |
| Storage paths | `docs/reference/storage-layout.md` |
| Runtime setting added | `docs/reference/runtime-config.md` |
