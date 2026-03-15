---
name: session-implement
description: Checklist for implementing planned changes safely after session-transition.
---


## Steps

1. **Confirm implementation readiness**
   - Validate the current Active plan in `PLANS.md`.
   - Confirm `session-transition` already prepared the branch state. If you are
     still on the protected branch, stop and run `session-transition`.

2. **Restate the task** — confirm what you're building/fixing in one sentence

3. **Identify affected files** — list files that need changes before editing

4. **Check DECISIONS.md** — verify no prior decision conflicts with the approach

5. **Implement with minimal diff**:
   - Read existing code before modifying
   - Change only what's necessary
   - Follow conventions in CLAUDE.md (uv, migrations, testing)
   - If adding/changing DB columns, create Alembic migration

6. **Run tests**: `uv run pytest tests/`

7. **Update documentation** (per CLAUDE.md section 3.6):
   - CLAUDE.md — if behavioral rules changed
   - MEMORY.md — if data models, workflows, or parameters changed
   - README.md — if user-facing APIs or features changed
   - STATUS.md — if capabilities or constraints changed
   - DECISIONS.md — if a significant architecture decision was made (append new ADR)

8. **Verify** — re-run tests, confirm no regressions

9. **Hand off to review**
   - After implementation, documentation, and tests are complete, run
     `session-review`.
   - Do NOT commit, push, or open/update a pull request in this skill. That
     work belongs to `session-end`.

## Rules
- Prefer editing existing files over creating new ones
- Keep changes focused — don't refactor surrounding code
- Test before declaring done (see Definition of Done in CLAUDE.md)
