# Humpback Acoustic Embedding & Clustering Platform

This is the authoritative project rulebook. It is loaded globally, so keep it
small. Domain-specific details belong in `docs/agent-context/` and
`docs/reference/`.

For Codex-specific workflow, see `AGENTS.md`. For architecture decisions, scan
ADR headings in `DECISIONS.md` and read full entries only when relevant.

---

## 1. Purpose

The system processes humpback whale audio into reusable embedding vectors,
supports retained detection and review workflows, and exposes a local web UI
for job management and inspection.

The system must:

- Support asynchronous, resumable workflows.
- Persist workflow state in SQL.
- Prevent duplicate canonical derived artifacts for the same retained
  configuration.
- Provide inspectable API and frontend workflows for detection, labeling,
  call parsing, sequence-model embedding production, and review.

---

## 2. Global Context Loading

Use the domain-local agent context layer for detailed work:

1. Read `docs/agent-context/domain-map.md`.
2. Select one primary domain and any directly affected neighbor domains.
3. Load those capsules from `docs/agent-context/domains/*/`.
4. Open `docs/reference/` files only when the capsule points to them for the
   current task.

Do not add detailed API listings, schema inventories, storage trees, or product
state narratives to this file.

---

## 3. Package And Environment Rules

- Use `uv` for all Python package operations. Never use `pip`, `pip-tools`,
  `poetry`, or `conda`.
- Dependencies are managed through `pyproject.toml` and `uv.lock`.
- TensorFlow is selected through exactly one extra: `tf-macos`,
  `tf-linux-cpu`, or `tf-linux-gpu`.
- Do not use `uv sync --all-extras`; the TensorFlow extras intentionally
  conflict.
- Use `npm` from `frontend/` for frontend package operations.

Common commands:

- macOS Apple Silicon dev sync: `uv sync --group dev --extra tf-macos`
- Linux CPU dev sync: `uv sync --group dev --extra tf-linux-cpu`
- Linux GPU dev sync: `uv sync --group dev --extra tf-linux-gpu`
- Run Python tools/tests: `uv run <tool>`
- Run backend tests: `uv run pytest tests/`
- Run Pyright: `uv run pyright`
- Run frontend typecheck: `cd frontend && npx tsc --noEmit`
- Run frontend Playwright: `cd frontend && npx playwright test`

---

## 4. Database And Migration Rules

Back up the production database before any database change: migrations, schema
changes, data backfills, manual SQL, or destructive deletes.

1. Read `HUMPBACK_DATABASE_URL` from `.env`.
2. Copy the database file to `<original_path>.YYYY-MM-DD-HH:mm.bak` using a UTC
   timestamp.
3. Verify the backup exists and has non-zero size.

If backup fails or is skipped, stop before applying the database change.

Schema rules:

- Use Alembic for table/column changes.
- Migration files live in `alembic/versions/`.
- Use `op.batch_alter_table()` for SQLite compatibility.
- Honor the database path from `HUMPBACK_DATABASE_URL`.

---

## 5. Universal Invariants

- Operational timestamps are UTC end to end.
- Detection embedding output is canonical per `(detection_job_id, model_version)`.
- Continuous Embedding output is canonical per `encoding_signature`.
- Worker state is persisted in SQL and must be restart-safe.
- Partial artifacts must be safely overwritten or written to temporary paths and
  atomically promoted.
- SQLite worker claims rely on atomic status updates, not row locks.
- Imported audio must remain readable from its original path.
- Model files must be present on disk; there is no remote model registry.

For domain-local invariants, read `docs/agent-context/global-invariants.md` and
the relevant `docs/agent-context/domains/*/invariants.md`.

---

## 6. Testing Requirements

Testing is required for every meaningful change.

Fast feedback:

- Use `docs/agent-context/test-map.md` and the relevant domain `tests.md` file
  for targeted verification during implementation.

Final gates:

1. `uv run ruff format --check` on modified Python files.
2. `uv run ruff check` on modified Python files.
3. `uv run pyright` on modified Python files, or full run when config changed.
4. `uv run pytest tests/`.
5. `cd frontend && npx tsc --noEmit` when frontend files changed.
6. `cd frontend && npx playwright test` when UI behavior changed.

---

## 7. Documentation Rules

When a change adds, removes, or modifies API endpoints, data models,
configuration options, architecture, workflows, storage paths, frontend routes,
or behavioral constraints, update the smallest relevant docs.

Use this order:

1. Update the matching `docs/agent-context/domains/*/` capsule if agent
   operating context changes.
2. Update detailed `docs/reference/` files for API, schema, storage, signal
   processing, runtime config, frontend, testing, or behavioral detail.
3. Update `docs/agent-context/current-state.md` for active capability changes.
4. Update `DECISIONS.md` only for significant architecture decisions or
   non-obvious trade-offs.
5. Update this file only for universal rules that every agent must load.

Detailed reference routing:

| Change type | Primary doc target |
|---|---|
| API endpoints | relevant `docs/reference/*-api.md` and owning domain capsule |
| Data model | `docs/reference/data-model.md`, Alembic migration, owning domain capsule |
| Signal processing | `docs/reference/signal-processing.md`, `signal-timeline` capsule |
| Storage paths | `docs/reference/storage-layout.md`, owning domain capsule |
| Runtime settings | `docs/reference/runtime-config.md`, owning domain capsule |
| Behavioral constraints | `docs/reference/behavioral-constraints.md`, owning domain capsule |
| Frontend routes/components | `docs/reference/frontend.md`, `frontend-shell` plus owning domain |
| Hydrophone behavior | `docs/reference/hydrophone-rules.md`, `ingest-detection` capsule |
| Workflow or agent context | `docs/workflows/`, `AGENTS.md`, `docs/agent-context/` |

---

## 8. Workflow

Canonical flow:

`session-begin -> design -> session-plan -> session-implement -> [session-debug]* -> session-review -> session-end`

Workflow files:

- `docs/workflows/session-begin.md`
- `docs/workflows/session-plan.md`
- `docs/workflows/session-implement.md`
- `docs/workflows/session-debug.md`
- `docs/workflows/session-review.md`
- `docs/workflows/session-end.md`

Branch lifecycle:

- Work happens on `feature/<name>` branches.
- Do not commit directly to `main`.
- `session-plan` commits the spec and plan first.
- `session-implement` commits implementation as a single batch.
- `session-end` pushes the branch and creates the PR.

---

## 9. Project Map

Use `docs/agent-context/current-state.md` for current product state by domain.
Use `docs/agent-context/domain-map.md` for path-to-domain routing.

Top-level layout:

- `src/humpback/`: backend application code.
- `frontend/`: React SPA.
- `tests/`: backend pytest suite.
- `frontend/e2e/`: Playwright specs.
- `alembic/versions/`: database migrations.
- `docs/agent-context/`: agent-local domain capsules.
- `docs/reference/`: detailed reference docs.
- `docs/specs/`: design specs.
- `docs/plans/`: implementation plans.
- `docs/workflows/`: session workflow definitions.

---

## 10. Non-Goals

- Model fine-tuning.
- Real-time streaming inference.
- Multi-tenant support.
- Distributed GPU execution.
