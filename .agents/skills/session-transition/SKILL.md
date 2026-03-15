---
name: session-transition
description: Checklist for activating the current plan and enforcing feature-branch readiness before implementation.
---

## Steps

1. **Activate the current plan in `PLANS.md`**
   - If the current plan is not already listed under `Active`, add it by path
     reference.
   - If there is no clear current plan, stop and resolve that before proceeding.

2. **Validate the Active plan reference**
   - Confirm the `PLANS.md` link resolves to a real plan file.
   - If the link is stale or invalid, fix `PLANS.md` before implementation.

3. **Check git readiness**
   - Determine the protected/default branch from `origin/HEAD` when available;
     otherwise treat `main` as protected.
   - Record the current branch and `git status --porcelain`.

4. **Enforce feature-branch readiness before local changes**
   - If you are on the protected branch and the working tree is clean,
     create or switch to `codex/<slug>`, where `<slug>` is a concise kebab-case
     summary of the active plan title.
   - If you are on the protected branch and the working tree is dirty, stop and
     tell the user to recover before implementation. Do not auto-carry dirty
     protected-branch changes onto a new branch.
   - If you are already on a non-protected branch, continue on that branch.
   - Treat detached `HEAD` like a protected-branch state: only branch from it
     when the tree is clean.

5. **Hand off to implementation**
   - Reconfirm the Active plan link in `PLANS.md`.
   - Tell the user the next step is `session-implement`.
   - Prompt the user to clear context or start a fresh implementation session if
     needed.

## Rules
- Do NOT start implementing product changes in this skill.
- New branches created by this skill must use the `codex/` prefix.
