---
name: session-review
description: Systematic review checklist for changes before committing.
---

## Checklist

1. **Architecture violations?**
   - Idempotent encoding preserved? (no duplicate embedding sets)
   - Resumable workflows intact? (atomic writes, restart-safe)
   - Signal processing semantics unchanged? (unless intentional — needs ADR)

2. **Missing tests?**
   - Unit tests for new logic?
   - Integration tests for new/changed API endpoints?
   - Playwright tests for UI changes?

3. **Missing migrations?**
   - Any new/changed/removed DB columns without Alembic migration?
   - Uses `op.batch_alter_table()` for SQLite?

4. **Stale documentation?**
   - CLAUDE.md, MEMORY.md, README.md, STATUS.md reflect the changes?
   - New ADR needed in DECISIONS.md?

5. **Code quality?**
   - No security vulnerabilities (injection, XSS)?
   - No unnecessary complexity or over-engineering?
   - Follows project conventions (uv, npm, file structure)?

6. **Collect staged review scope**
   - Run `git diff --name-only --cached --diff-filter=ACMR`
   - If no files are staged, report "no staged files to review" and stop
   - Treat staged `.py` files under `src/humpback/`, `scripts/`, and `tests/` as Ruff/Pyright targets

7. **Run validation in order**
   - Ruff: if staged Python targets exist, run `uv run ruff check <staged_python_files>`
   - Pyright: if staged files include `pyproject.toml` or `.pre-commit-config.yaml`, run `uv run pyright`
   - Pyright: otherwise, if staged Python targets exist, run `uv run pyright <staged_python_files>`
   - Pytest: for any non-empty staged review scope, run `uv run pytest tests/` after Ruff/Pyright

## Output
- List any issues found with file:line references
- If no files are staged, report that there is no review scope
- Confirm ready to commit only if Ruff/Pyright checks (when applicable) and `uv run pytest tests/` all pass
