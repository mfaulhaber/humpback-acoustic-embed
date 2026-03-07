# Review Workflow

Systematic review checklist for changes before committing.

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

6. **Run final check**: `uv run pytest tests/`

## Output
- List any issues found with file:line references
- If clean, confirm ready to commit
