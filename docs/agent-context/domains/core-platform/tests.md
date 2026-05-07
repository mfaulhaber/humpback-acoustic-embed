# Core Platform Tests

Use these commands for targeted feedback. The final backend gate remains
`uv run pytest tests/`.

## Smoke

- `uv run pytest tests/unit/test_config.py tests/unit/test_storage.py tests/unit/test_queue.py -q`

## Database And Migrations

- `uv run pytest tests/unit/test_migration_* tests/db -q`
- For a specific migration, run the matching migration test file directly.

## Workers And Queue

- `uv run pytest tests/unit/test_worker_runner.py tests/unit/test_queue.py -q`
- Include the owning domain's worker tests when a claim function or runner block
  changes domain behavior.

## Docs And Workflow Only

- `git diff --check`
- Run full backend tests during final verification unless the plan explicitly
  narrows verification for docs-only changes.
