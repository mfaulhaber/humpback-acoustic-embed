# session-begin

Start of every session — normalize repo state, read context, ask what to work on.

## Steps

1. **Check repo state**
   - Record current branch and `git status --porcelain`
   - If HEAD is detached, stop and tell the user to recover
   - If the working tree is dirty, stop and tell the user to recover

2. **Normalize onto main**
   - If not already on `main`, switch to it
   - Fetch `origin/main` and fast-forward local `main`
   - If fast-forward fails, stop and report (do not merge, rebase, or stash)

3. **Acknowledge context already loaded**
   - `CLAUDE.md` is auto-loaded into every conversation — do NOT read it again
   - Read only the ADR titles from `DECISIONS.md` (e.g., `grep '^## ADR-' DECISIONS.md`) — read full ADR text only when relevant to the task at hand
   - Do not load every domain capsule during session-begin
   - Use `docs/agent-context/current-state.md` only when a lightweight project-state summary is needed
   - Use `docs/agent-context/domain-map.md` once the task domain is known

4. **Check for active feature branches**
   - Look for local `feature/*` branches only, ignore remote feature/* branches
   - If found, check for in-progress specs in `docs/specs/` or plans in `docs/plans/` on those branches
   - If active work exists, offer to resume on that branch

5. **Summarize for the user**
   - Current project state (what's implemented, what's in progress), summarized from `docs/agent-context/current-state.md` when needed rather than from global instructions
   - Recent commits on main (last 5)
   - Active feature branches if any

6. **Ask what to work on**
   - If resuming, confirm the active plan
   - If new work, proceed to brainstorming

## Does NOT

- Create a feature branch (deferred to `session-plan`)
- Start any implementation work
- Read memory files unless the task requires it
- Re-read CLAUDE.md (it is already in the system prompt)
- Read all domain capsules

## Output

Summary of project state, prompt for next action.

## Next Step

- New feature → brainstorm (superpowers brainstorming skill)
- Resuming with existing spec → `session-plan`
- Resuming with existing plan → `session-implement`
