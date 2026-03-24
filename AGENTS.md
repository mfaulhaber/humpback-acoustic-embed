# Humpback Acoustic Embed — Codex Agent Instructions

CLAUDE.md is the authoritative project rulebook — read it first.

## Codex Workflow

Follow these phases in order for any task:

### Phase 1: Context
- Read CLAUDE.md (rules + reference)
- Read DECISIONS.md (recent ADRs)
- Check docs/plans/ for active work
- Understand what's being asked before acting

### Phase 2: Design
- For new features or significant changes:
  - Explore affected code
  - Identify 2-3 approaches with trade-offs
  - Write a design spec to docs/specs/YYYY-MM-DD-<topic>-design.md
  - Commit the spec before implementing
- For bug fixes: skip to Phase 3 after root-cause investigation

### Phase 3: Plan
- Write an implementation plan to docs/plans/YYYY-MM-DD-<feature>.md
- Break into small tasks (2-5 minutes each)
- Include affected files, test strategy, verification commands
- Commit the plan before implementing

### Phase 4: Implement
- Create a feature branch: codex/<slug>
- TDD: write failing test -> implement -> verify green -> refactor
- One task at a time, commit after each
- Follow all CLAUDE.md rules (uv, migrations, doc updates)

### Phase 5: Verify
- Run the project verification gates (CLAUDE.md §10.3)
- All must pass before proceeding

### Phase 6: Finish
- Push branch, create PR targeting main
- PR body includes: summary, test plan, verification results

## Key Constraints

- Package manager: uv only (never pip/conda/poetry)
- Frontend: npm from frontend/ directory
- DB migrations: Alembic with op.batch_alter_table() for SQLite
- Testing: every change needs tests (uv run pytest tests/)
- Idempotency: never create duplicate embedding sets for same encoding_signature
