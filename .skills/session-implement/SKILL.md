---
name: implement
description: Checklist for implementing planned changes safely.
---


## Steps

1. **Restate the task** — validate current active plan in PLANS.md and confirm what you're building/fixing in one sentence

2. **Identify affected files** — list files that need changes before editing

3. **Check DECISIONS.md** — verify no prior decision conflicts with the approach

4. **Implement with minimal diff**:
   - Read existing code before modifying
   - Change only what's necessary
   - Follow conventions in CLAUDE.md (uv, migrations, testing)
   - If adding/changing DB columns, create Alembic migration

5. **Run tests**: `uv run pytest tests/`

6. **Update documentation** (per CLAUDE.md section 3.6):
   - CLAUDE.md — if behavioral rules changed
   - MEMORY.md — if data models, workflows, or parameters changed
   - README.md — if user-facing APIs or features changed
   - STATUS.md — if capabilities or constraints changed
   - DECISIONS.md — if a significant architecture decision was made (append new ADR)

7. **Verify** — re-run tests, confirm no regressions

## Rules
- Prefer editing existing files over creating new ones
- Keep changes focused — don't refactor surrounding code
- Test before declaring done (see Definition of Done in CLAUDE.md)
