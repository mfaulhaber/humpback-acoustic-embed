---
name: session-start
description: Checklist for starting every session from synced local main before choosing the next plan.
---


## Steps

1. **Normalize the repo onto local `main`**
   - Record the current branch and `git status --porcelain`.
   - If `HEAD` is detached, stop and tell the user to recover before continuing.
   - If the working tree is dirty, stop and tell the user to recover before
     switching branches.
   - Confirm local `main` exists. If it does not, stop and report that
     explicitly.
   - If you are not already on local `main`, switch to it.
   - Fetch `origin/main` and fast-forward local `main` when possible.
   - If the fast-forward cannot be completed cleanly, stop and report instead
     of merging, rebasing, or stashing.

2. **Read context files** (in order):
   - `STATUS.md` — current capabilities, schema version, constraints
   - `PLANS.md` — active plans and backlog
   - `DECISIONS.md` — recent architecture decisions

3. **Summarize for the user**:
   - Current project state (what's implemented, what's in progress)
   - Active plans and their status
   - Recent decisions that may affect upcoming work
   - Known constraints or risks

4. **Check git status on normalized `main`**:
   - Confirm you are now on local `main`
   - Confirm whether local `main` is up to date with `origin/main`
   - Recent commits (last 5) from `main`

5. **Ask** what the user wants to work on, or confirm the active plan.
   - If the next task is implementation work, route it through `session-transition`
     before any code changes begin.

## Rules
- Do NOT start coding or making product changes
- Do NOT read MEMORY.md unless the user's task requires reference material
- Keep the summary concise — bullet points, not paragraphs
