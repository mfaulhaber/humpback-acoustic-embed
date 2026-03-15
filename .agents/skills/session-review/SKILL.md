---
name: session-review
description: Required validation gate before session-end and any git mutation.
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

6. **Collect modified review scope**
   - Collect tracked working-tree changes with `git diff --name-only HEAD --diff-filter=ACMR`
   - Collect untracked files with `git ls-files --others --exclude-standard`
   - Review the union of those paths as the current modified-file scope
   - If the modified-file scope is empty, report "no modified files to review" and stop
   - Treat modified `.py` files under `src/humpback/`, `scripts/`, and `tests/` as Ruff/Pyright targets

7. **Run validation in order**
   - Ruff format: if modified Python targets exist, run `uv run ruff format --check <modified_python_files>`
   - Ruff lint: if modified Python targets exist, run `uv run ruff check <modified_python_files>`
   - Pyright: if modified files include `pyproject.toml` or `.pre-commit-config.yaml`, run `uv run pyright`
   - Pyright: otherwise, if modified Python targets exist, run `uv run pyright <modified_python_files>`
   - Pytest: for any non-empty modified review scope, run `uv run pytest tests/` after Ruff/Pyright

## Rules
- Do NOT commit, push, or create/update a pull request in this skill.
- If any repo-tracked file changes after this review, rerun `session-review`
  before `session-end`.

## Output
- List any issues found with file:line references
- If no modified files are present, report that there is no review scope
- If validation fails or any findings remain, explicitly state `Ready for session-end: no`
- Confirm ready to commit only if Ruff format, Ruff lint, Pyright checks (when applicable), and `uv run pytest tests/` all pass
- If all checks pass with no blocking findings, explicitly state `Ready for session-end: yes`
