# Global Invariants

Read this before implementation when a task crosses domains. Domain-local
details live under `domains/*/invariants.md`.

## Package And Tooling

- Use `uv` for Python package operations and Python tools.
- Use `npm` from `frontend/` for frontend package operations.
- Do not use `uv sync --all-extras`; TensorFlow extras are mutually exclusive.
- Pyright covers `src/humpback`, `scripts`, and `tests`.

## Database Safety

- Back up the production database before migrations, schema changes, data
  backfills, manual SQL, or destructive deletes.
- Read the production database path from `HUMPBACK_DATABASE_URL` in `.env`.
- Use Alembic for schema changes and `op.batch_alter_table()` for SQLite.

## Time

- Operational timestamps are UTC end to end.
- API timestamp fields should be UTC epoch seconds or `Z` semantics.
- Time-derived identifiers and filenames use UTC.

## Retained Artifact Idempotency

- Detection embedding output is canonical per `(detection_job_id, model_version)`.
- Continuous Embedding output is canonical per `encoding_signature`.
- Training/bootstrap sample loaders preserve retained source uniqueness.

## Resumable Workflows

- Worker state is persisted in SQL.
- Workers must be restart-safe.
- Partial artifacts must be safely overwritten or written to temp paths and
  atomically promoted on completion.
- SQLite worker claims rely on atomic status updates, not row locks.

## Verification

- Use targeted domain tests for fast feedback during implementation.
- The full backend gate remains `uv run pytest tests/`.
- Run frontend TypeScript and Playwright checks when frontend files change.
