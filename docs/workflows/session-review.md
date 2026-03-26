# session-review

Validation gate that must pass before `session-end` can proceed.

## Steps

1. **Collect modified file scope**
   - Tracked changes: `git diff --name-only HEAD --diff-filter=ACMR`
   - Untracked files: `git ls-files --others --exclude-standard`
   - Review the union as the current scope
   - If scope is empty, report "no modified files to review" and stop

2. **Architecture checks**
   - Idempotent encoding preserved? (no duplicate embedding sets)
   - Resumable workflows intact? (atomic writes, restart-safe)
   - Signal processing semantics unchanged? (unless intentional — needs ADR)

3. **Completeness checks**
   - Missing tests for new logic?
   - Missing Alembic migration for schema changes?
   - Missing doc updates per CLAUDE.md §3.6 doc-update matrix?

4. **Run verification gates in order**
   - `uv run ruff format --check` on modified Python files
   - `uv run ruff check` on modified Python files
   - `uv run pyright` on modified Python files (full run if `pyproject.toml` changed)
   - `uv run pytest tests/`
   - `cd frontend && npx tsc --noEmit` (if frontend files changed)

5. **Report findings** with file:line references where applicable

6. **Output verdict**
   - If all checks pass with no blocking findings: `Ready for session-end: yes`
   - If any validation fails or findings remain: `Ready for session-end: no`

## Rules

- If any repo-tracked file changes after this review, rerun `session-review` before `session-end`

## Does NOT

- Commit, push, or create/update pull requests
- Fix issues itself (report them for `session-debug` or manual fix)

## Output

Verdict with any blocking issues listed.

## Next Step

`session-end` if yes, fix issues and re-review if no.
