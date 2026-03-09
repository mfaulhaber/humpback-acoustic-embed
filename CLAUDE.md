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

This document defines behavioral rules and engineering constraints.
For reference material (data models, workflows, parameters), see `MEMORY.md`.

## Memory Files

| File | When to read |
|------|-------------|
| `MEMORY.md` | Working on data models, workflows, signal processing, storage layout |
| `DECISIONS.md` | Before making architecture changes (check for prior decisions) |
| `STATUS.md` | At session start — understand current capabilities and constraints |
| `PLANS.md` | At session start — check active and backlog plans |
| `AGENTS.md` | Codex entry point (not needed for Claude Code) |

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

### 3.2 Environment Commands
Use these commands for managing dependencies:
*   Install/synchronize all dependencies: `uv sync`
*   Add a new package (e.g., `requests`): `uv add requests`
*   Remove a package: `uv remove <package>`
*   Compile a new lock file: `uv pip compile pyproject.toml -o uv.lock`
*   Upgrade a specific package: `uv lock --upgrade-package <package>`

### 3.3 Running Python Code and Tools
*   Run a Python script: `uv run <script-name>.py`
*   Run Python tools/tests (e.g., `pytest`): `uv run pytest tests/`
*   Run pre-commit hooks: `uv run pre-commit install`

### 3.4 Best Practices
*   Prefer `uv run` over manually activating a virtual environment and running commands directly.
*   When troubleshooting, use `uv cache clean` as a last resort.

### 3.5 Database Migrations
*   When a change adds, removes, or renames columns on any SQL table, **always** create an Alembic migration in `alembic/versions/` and run it with `uv run alembic upgrade head` before verifying the change works.
*   Migration files follow the naming convention `NNN_short_description.py` (e.g., `007_negative_embedding_set_ids.py`), incrementing from the latest revision.
*   Use `op.batch_alter_table()` for SQLite compatibility.

### 3.6 Documentation
*   When a change adds, removes, or modifies API endpoints, data models, configuration options, architecture, or workflows, update the relevant files:
    *   `CLAUDE.md` — behavioral rules, development constraints (this file)
    *   `MEMORY.md` — data models, workflows, signal parameters, storage layout
    *   `README.md` — user-facing API endpoints, configuration, feature list
    *   `STATUS.md` — current capabilities, schema version, known constraints
    *   `DECISIONS.md` — append new ADR for significant architecture changes
*   `CLAUDE.md` is the authoritative spec for rules; `MEMORY.md` is authoritative for reference material.

## 3.7 Frontend Stack & Development

The web UI is a React SPA in the `frontend/` directory, built with:

| Layer | Technology |
|-------|-----------|
| Build | Vite + TypeScript |
| UI Framework | React 18 |
| Styling | Tailwind CSS |
| Component Library | shadcn/ui (Radix primitives, copy-paste model in `frontend/src/components/ui/`) |
| Server State | TanStack Query (polling, caching, mutations) |
| Charts | react-plotly.js (wraps Plotly.js basic dist) |
| Icons | lucide-react |
| API Client | Hand-rolled typed fetch wrapper (`frontend/src/api/client.ts`) |

**No React Router** — the UI uses tab-based navigation managed via React state, not URL routing.

#### Frontend Package Management
*   Use `npm` for all frontend package operations. Run commands from the `frontend/` directory.
*   `npm install` — install dependencies
*   `npm run dev` — start Vite dev server on `:5173` (proxies API calls to `:8000`)
*   `npm run build` — production build to `src/humpback/static/dist/`
*   `npx tsc --noEmit` — type-check without emitting

#### Frontend File Structure
```
frontend/
├── package.json, vite.config.ts, tsconfig.json, tailwind.config.ts
├── playwright.config.ts         (Playwright test config)
├── components.json              (shadcn/ui config)
├── index.html
├── e2e/                         (Playwright test specs)
└── src/
    ├── main.tsx                 (QueryClientProvider + App mount)
    ├── App.tsx                  (tab state + tab content switching)
    ├── index.css                (Tailwind directives + shadcn CSS vars)
    ├── lib/utils.ts             (cn() helper)
    ├── api/
    │   ├── client.ts            (typed fetch wrapper, all endpoints)
    │   └── types.ts             (TS interfaces mirroring Pydantic schemas)
    ├── hooks/queries/           (TanStack Query hooks per domain)
    ├── components/
    │   ├── ui/                  (shadcn primitives)
    │   ├── layout/              (AppShell, Header, TabNav)
    │   ├── audio/               (AudioTab, AudioUpload, AudioList, AudioDetail, AudioPlayerBar, SpectrogramPlot, SimilarityMatrix)
    │   ├── processing/          (ProcessingTab, QueueJobForm, ProcessingJobsList, EmbeddingSetsList)
    │   ├── clustering/          (ClusteringTab, EmbeddingSetSelector, ClusteringParamsForm, ClusteringJobCard, ClusterTable, UmapPlot, EvaluationPanel, ExportReport)
    │   ├── classifier/          (ClassifierTab, TrainingTab, DetectionTab, BulkDeleteDialog)
    │   ├── admin/               (AdminTab, ModelRegistry, ModelScanner, DatabaseAdmin)
    │   └── shared/              (FolderTree, FolderBrowser, StatusBadge, MessageToast)
    └── utils/                   (format.ts, audio.ts)
```

#### Dev Workflow
```bash
# Terminal 1: Backend
uv run humpback-api          # API on :8000
uv run humpback-worker       # Worker process

# Terminal 2: Frontend dev server
cd frontend && npm run dev   # Vite on :5173, proxies to :8000
```

#### Production Build & Serving
```bash
cd frontend && npm run build  # outputs to src/humpback/static/dist/
uv run humpback-api           # serves SPA at / and API on :8000
```

The FastAPI backend detects `static/dist/index.html` at startup. When present, it serves the built SPA at `/` and mounts `/assets` for JS/CSS bundles. When absent, it falls back to the legacy `static/index.html`.

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

### 4.4 Hydrophone Extraction Path Convention
Hydrophone labeled-sample extraction writes outputs under the hydrophone short label:
- positives: `{positive_output_path}/{hydrophone_id}/humpback/YYYY/MM/DD/*.wav`
- negatives: `{negative_output_path}/{hydrophone_id}/{ship|background}/YYYY/MM/DD/*.wav`

### 4.5 Hydrophone Timeline Assembly
Hydrophone detection, playback, and extraction must use the same bounded stream timeline:
- segment ordering must be numeric by segment suffix (never plain lexicographic)
- playlist (`live.m3u8`) duration metadata should be used when available
- processing/playback/extraction must stay within `[start_timestamp, end_timestamp]`
- legacy playback compatibility for older jobs may fall back to `job.start_timestamp`
- hydrophone extraction is local-cache-authoritative (same as playback): resolve from local HLS cache only, with no S3 listing/fetch fallback
- hydrophone extraction should build/reuse timeline metadata once per extraction run (avoid rebuilding per labeled row)

### 4.6 Hydrophone Detection TSV Metadata
Hydrophone detection TSV output should carry extraction-oriented metadata:
- include `extract_filename` for hydrophone rows (`{start_utc}_{end_utc}.wav`)
- `extract_filename` must be derived from snapped extraction bounds (window-size aligned)
- local (non-hydrophone) detection TSV output remains unchanged

---

## 5. Testing Requirements (MANDATORY)

Testing is not optional. Every meaningful change must include:
- unit tests for new logic
- integration tests for API endpoints
- at least one end-to-end smoke test path that exercises the real workflows

### 5.1 Unit Tests
Add unit tests for:
- encoding_signature computation (idempotency)
- audio window slicing logic
- feature extraction shape correctness
- TFLite runner batching (mock interpreter acceptable)
- Parquet writer behavior (temp + atomic promote)
- clustering pipeline (small synthetic embeddings)

Guidelines:
- prefer deterministic tests
- isolate file I/O behind temp directories
- mock external dependencies when appropriate (e.g., TFLite interpreter)

### 5.2 Running Tests Locally
The repo must include:
- `pytest` configuration
- a single command to run unit+integration tests
- a command to run tests continuously on file changes

Required commands:
- `pytest` (all tests)
- `pytest -q` (quiet)
- `pytest -k <pattern>` (focused)
- "watch mode" (choose one):
  - `pytest-watch` (`ptw`) OR
  - `watchexec -r pytest` OR
  - `entr -r pytest`

Document the chosen tool in README and add it to dev dependencies.

### 5.3 End-to-End Smoke Test (E2E)
Add a minimal E2E test that:
1. Starts API + worker (in-process for tests or via subprocess)
2. Uploads a small fixture audio file
3. Queues a ProcessingJob
4. Polls until job completes
5. Verifies EmbeddingSet exists and parquet file is readable
6. Queues a ClusteringJob on that embedding set
7. Polls until complete
8. Verifies clusters and assignments exist and are consistent

Constraints:
- E2E must run in under a few minutes locally
- Use a tiny audio fixture (e.g., 10–20 seconds)
- Use a tiny embedding model stub if needed (see below)

### 5.4 Frontend Tests (Playwright)
When changing UI components, add or update Playwright tests in `frontend/e2e/`.

**When to add tests:**
- Any new interactive feature (buttons, forms, expandable rows, audio playback)
- Changes to data flow between frontend and backend (API calls, query hooks)
- Bug fixes for UI behavior (regression tests)

**Test patterns:**
- **API-level tests** — use `request` fixture to hit backend endpoints directly and validate response content (e.g., WAV duration, JSON shape). These are fast and don't need a browser page.
- **UI interaction tests** — use `page` fixture to navigate, click, and assert DOM state. Verify that user actions produce correct side effects (e.g., audio element src, table expansion, form submission).
- **Hydrophone regressions** — include timestamp-mapping playback checks and Extract-button activation checks when fixing Hydrophone tab playback/label workflows.
- Skip gracefully when preconditions aren't met (e.g., no completed jobs) using `test.skip()`.

**Running:**
```bash
cd frontend
npx playwright test                    # all tests
npx playwright test e2e/some.spec.ts   # specific file
npx playwright test -g "test name"     # by name pattern
npx playwright test --headed           # see the browser
```

**Requirements:**
- Tests run against `localhost:5173` (frontend dev server) proxying to `localhost:8000` (backend)
- Backend must be running with real or fixture data
- Config is in `frontend/playwright.config.ts`; tests go in `frontend/e2e/*.spec.ts`
- Install browsers once: `cd frontend && npx playwright install chromium`

### 5.5 Model Stub Strategy (So Tests Are Fast)
For unit/integration/E2E tests:
- Provide a "FakeTFLiteModel" implementation that returns deterministic embeddings
  (e.g., sine/cosine transforms of window index)
- Gate real Perch model execution behind an environment flag
  - default tests use FakeTFLiteModel
  - optional manual run uses real model if available

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
