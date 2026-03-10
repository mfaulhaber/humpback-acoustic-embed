---
name: debug
description: Checklist for structured root-cause debugging for issues.
---

## Steps

1. **Describe the symptom** — what's happening vs what's expected?

2. **Reproduce** — find the minimal reproduction path (test, command, or UI action)

3. **Identify root cause**:
   - Read the relevant code (don't guess)
   - Check recent commits that may have introduced the issue
   - Check DECISIONS.md for relevant context

4. **Implement minimal fix**:
   - Change only what's necessary to fix the root cause
   - Don't refactor surrounding code

5. **Add regression test** — a test that fails without the fix and passes with it

6. **Run full test suite**: `uv run pytest tests/`

## Rules
- Don't apply workarounds — fix root causes
- If the fix changes signal processing or data models, it needs an ADR
- Update STATUS.md if the fix changes known constraints
