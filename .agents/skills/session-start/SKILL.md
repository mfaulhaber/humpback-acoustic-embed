---
name: session-start
description: Checklist for starting every session by loading project context before choosing the next plan.
---


## Steps

1. **Read context files** (in order):
   - `STATUS.md` — current capabilities, schema version, constraints
   - `PLANS.md` — active plans and backlog
   - `DECISIONS.md` — recent architecture decisions

2. **Summarize for the user**:
   - Current project state (what's implemented, what's in progress)
   - Active plans and their status
   - Recent decisions that may affect upcoming work
   - Known constraints or risks

3. **Check git status**:
   - Current branch and uncommitted changes
   - Recent commits (last 5)

4. **Ask** what the user wants to work on, or confirm the active plan.
   - If the next task is implementation work, route it through `session-transition`
     before any code changes begin.

## Rules
- Do NOT start coding or making changes
- Do NOT read MEMORY.md unless the user's task requires reference material
- Keep the summary concise — bullet points, not paragraphs
