# Humpback Acoustic Embed — Codex Agent Instructions

CLAUDE.md is the authoritative project rulebook — read it first.

## Codex Workflow

Follow these phases in order for any task. Each phase references a workflow file
in `docs/workflows/` — read the full file for detailed steps.

### Phase 1: Context (`docs/workflows/session-begin.md`)
- Read CLAUDE.md (rules + reference)
- Read DECISIONS.md (recent ADRs)
- Check for active feature branches and in-progress plans
- Understand what's being asked before acting

### Phase 2: Design
- For new features or significant changes:
  - Explore affected code
  - Identify 2-3 approaches with trade-offs
  - Write a design spec to `docs/specs/YYYY-MM-DD-<topic>-design.md`
  - Do not commit the spec yet (Phase 3 handles this)
- For bug fixes: skip to Phase 3 after root-cause investigation

### Phase 3: Plan (`docs/workflows/session-plan.md`)
- Create feature branch: `feature/<feature-name>`
- Write an implementation plan to `docs/plans/YYYY-MM-DD-<feature>.md`
- Tasks include: file paths, acceptance criteria, test requirements (no code blocks)
- Commit spec and plan as the first commit on the feature branch

### Phase 4: Implement (`docs/workflows/session-implement.md`)
- Work through plan tasks sequentially
- Tests are required but ordering is flexible (alongside or after implementation)
- Single batched commit after all tasks complete
- Follow all CLAUDE.md rules (uv, migrations, doc updates)

### Phase 5: Debug (`docs/workflows/session-debug.md`)
- If manual testing reveals issues, debug with structured root-cause analysis
- Minimal fixes, regression tests where appropriate
- Repeatable — as many rounds as needed

### Phase 6: Verify (`docs/workflows/session-review.md`)
- Run the project verification gates (CLAUDE.md §10.4)
- All must pass before proceeding
- Explicit verdict: `Ready for session-end: yes/no`

### Phase 7: Finish (`docs/workflows/session-end.md`)
- Push branch, create PR targeting main
- PR body includes: summary, test plan, verification results
- Squash-merge, return to clean main

## Key Constraints

- Package manager: uv only (never pip/conda/poetry)
- Frontend: npm from frontend/ directory
- DB migrations: Alembic with op.batch_alter_table() for SQLite
- Testing: every change needs tests (uv run pytest tests/)
- Idempotency: never create duplicate embedding sets for same encoding_signature
