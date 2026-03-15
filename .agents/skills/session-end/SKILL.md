---
name: session-end
description: Post-review wrap-up checklist with commit, push, PR creation, and squash-merge handling.
---

## Steps

1. **Confirm review gate before any git mutation**
   - Locate the latest `session-review` result for the current work.
   - If it did not explicitly say `Ready for session-end: yes`, stop and run
     `session-review`.
   - If repo-tracked files changed after that review, stop and rerun
     `session-review` before committing, pushing, or opening a PR.

2. **Confirm git safety**
   - Determine the protected/default branch from `origin/HEAD` when available;
     otherwise treat `main` as protected.
   - Record the current branch and working tree status.
   - If you are on the protected branch or detached `HEAD`, stop and go back to
     `session-transition`.

3. **Summarize work completed** — what changed and why (1-3 bullets)

4. **Update STATUS.md** if:
   - New capabilities were added
   - Known constraints changed
   - Schema version changed (new migration)

5. **Update PLANS.md** if:
   - An active plan was completed (move to Completed)
   - New backlog items were identified
   - A plan's scope changed

6. **Append to DECISIONS.md** if:
   - A significant architecture decision was made during this session

7. **Re-check review status after session-end edits**
   - If steps 4-6 changed any repo-tracked files, rerun `session-review`.
   - Do not continue until the latest review again says
     `Ready for session-end: yes`.

8. **Prepare the commit scope**
   - Build the candidate scope from the latest reviewed files only.
   - If unexpected or unrelated files are present, stop and resolve that before
     staging.
   - Never use `git add -A`; stage only the reviewed task scope.

9. **Create the session commit**
   - If there are no reviewed changes to commit, report that and stop after the
     session summary.
   - Create one concise commit that matches the completed session work.

10. **Push the branch**
   - Use `git push -u origin <branch>` when publishing the branch for the first
     time.
   - Otherwise use `git push`.

11. **Handle the pull request against `main`**
   - Check for existing PRs for the current branch before creating anything.
   - If an open PR already exists, reuse it and report the URL. Do not create a
     duplicate or overwrite manual PR content.
   - If the branch already has a merged PR, report that and stop rather than
     creating a duplicate PR for the same branch.
   - If no PR exists and `gh` is installed plus authenticated, create a ready
     PR targeting `main`. Use the active plan title as the PR title when
     possible, falling back to the commit subject.
   - Build the PR body from the session summary, validation results, and next
     steps.
   - If `gh` is unavailable or unauthenticated, stop after push and report that
     PR automation could not run.

12. **Attempt the squash merge**
   - After creating or reusing the PR, attempt an immediate squash merge via
     GitHub.
   - If merge is blocked by conflicts, missing permissions, required checks,
     required review, or future branch protection, stop and report the specific
     blocker. Do not bypass GitHub policy.
   - Do not delete the local or remote feature branch after merge.
   - Report the PR URL and whether the squash merge succeeded.

13. **Report next steps**
   - Note that the next `session-start` should return the repo to synced local
     `main`.
   - Call out any follow-up work that did not make this session's merge.

## Rules
- `session-end` happens only after a clean `session-review`.
- Do NOT skip directly from implementation to commit/push/PR handling.
- Do NOT commit or push from a protected branch.
- Target `main` when creating a PR from this skill.
- Prefer squash merge when GitHub allows it.
- Keep updates concise
- Don't rewrite entire files — use targeted edits
- Verify memory files are consistent with each other
