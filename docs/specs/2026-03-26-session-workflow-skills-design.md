# Session Workflow Skills — Design Spec

## Problem

The superpowers skill system (adopted in PR #36) slows feature development significantly:
- **Subagent-driven development** dispatches 3 subagents per task (implementer + spec reviewer + code quality reviewer), consuming excessive tokens and time
- **Strict TDD** mandates writing failing tests before any implementation code, with rigid red-green-refactor ceremony
- **Bite-sized plans** require complete code blocks in every step, inflating plan documents
- **Worktree isolation** adds complexity without proportional benefit for a single-developer project

The previous session-* skills (removed in PR #36) were lightweight and effective but lacked integration with a spec/plan artifact workflow.

## Solution

6 lightweight workflow skills that combine the directness of the old session-* approach with the spec-driven planning from brainstorming. Implementation happens directly in the main conversation — no subagent dispatching, no worktrees, no rigid TDD.

### Canonical Flow

```
session-begin -> brainstorm -> session-plan -> session-implement -> [session-debug]* -> session-review -> session-end
```

- `brainstorm` is the existing superpowers brainstorming skill (unchanged)
- `[session-debug]*` means zero or more rounds of debugging after manual testing
- All other skills are new

### Design Principles

- **Single-session execution** — no subagent dispatching, no worktree isolation
- **Tests required, not TDD** — write tests alongside or after implementation, but every change must include tests
- **Plans describe intent, not code** — tasks have file paths and acceptance criteria, no code blocks
- **One commit per phase** — batched commits, not micro-commits per step
- **Verification gates enforced** — ruff, pyright, pytest, tsc must pass before merge

## File Layout

### Canonical Content

```
docs/workflows/
├── session-begin.md
├── session-plan.md
├── session-implement.md
├── session-debug.md
├── session-review.md
└── session-end.md
```

### Claude Code Entry Points

```
.claude/commands/
├── session-begin.md      → "Read and execute the workflow defined in docs/workflows/session-begin.md"
├── session-plan.md       → "Read and execute the workflow defined in docs/workflows/session-plan.md"
├── session-implement.md  → "Read and execute the workflow defined in docs/workflows/session-implement.md"
├── session-debug.md      → "Read and execute the workflow defined in docs/workflows/session-debug.md"
├── session-review.md     → "Read and execute the workflow defined in docs/workflows/session-review.md"
└── session-end.md        → "Read and execute the workflow defined in docs/workflows/session-end.md"
```

### Codex Entry Point

`AGENTS.md` references the workflow files by phase, replacing the current inline phase descriptions.

## Skill Specifications

### 1. session-begin

**Purpose:** Start of every session — normalize repo state, read context, ask what to work on.

**Steps:**
1. Record current branch and `git status --porcelain`
2. If HEAD is detached or working tree is dirty, stop and tell the user to recover
3. If not on `main`, switch to it
4. Fetch `origin/main` and fast-forward local `main`; if fast-forward fails, stop and report
5. Read CLAUDE.md and DECISIONS.md
6. Check for active feature branches (local + remote `feature/*`)
   - If found, check for in-progress specs/plans on those branches
   - Offer to resume if active work exists
7. Summarize current state for the user
8. Ask what to work on

**Does NOT:**
- Create a feature branch (deferred to `session-plan`)
- Start any implementation work
- Read memory files unless the task requires it

**Outputs:** Summary of project state, prompt for next action.

**Next step:** Brainstorm (superpowers brainstorming skill) for new features, or `session-plan` if resuming with an existing spec.

### 2. session-plan

**Purpose:** Create an implementation plan from a spec, create the feature branch, and commit both artifacts.

**Preconditions:**
- A spec exists in `docs/specs/` (written by brainstorming, uncommitted on main)
- Currently on `main` with the spec as an unstaged file

**Steps:**
1. Read the spec from `docs/specs/`
2. Derive a feature name from the spec topic (e.g., `session-workflow-skills`)
3. Create `feature/<feature-name>` branch from main
4. Break the spec into discrete implementation tasks
5. Write the plan to `docs/plans/YYYY-MM-DD-<feature>.md`
6. Commit both the spec and the plan as the first commit on the feature branch

**Plan format:**
```markdown
# <Feature Name> Implementation Plan

**Goal:** One sentence describing what this builds
**Spec:** Link to docs/specs/ file

---

### Task N: <Task Title>

**Files:**
- Create: `exact/path/to/file.ext`
- Modify: `exact/path/to/existing.ext`

**Acceptance criteria:**
- [ ] Criterion 1
- [ ] Criterion 2

**Tests needed:**
- Description of what to test (not test code)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check <files>`
2. `uv run ruff check <files>`
3. `uv run pyright <files>`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit` (if frontend changed)
```

**Does NOT:**
- Include code blocks in tasks
- Enforce TDD ordering
- Dispatch subagents

**Outputs:** Plan file committed on feature branch.

**Next step:** `session-implement`

### 3. session-implement

**Purpose:** Work through the plan tasks sequentially, then commit all changes as a single batch.

**Preconditions:**
- On a `feature/*` branch
- A plan exists in `docs/plans/`

**Steps:**
1. Confirm on feature branch; read the plan
2. Restate the current task scope in one sentence
3. Identify all affected files before editing
4. Check DECISIONS.md for conflicting prior decisions
5. Work through tasks sequentially:
   - Read existing code before modifying
   - Implement the change
   - Write tests (alongside or after — not required before)
   - Check off acceptance criteria in the plan
6. After all tasks, run verification gates:
   - `uv run ruff format --check` on modified Python files
   - `uv run ruff check` on modified Python files
   - `uv run pyright` on modified Python files
   - `uv run pytest tests/`
   - `cd frontend && npx tsc --noEmit` (if frontend changed)
7. Fix any verification failures
8. Update documentation per CLAUDE.md §3.6 doc-update matrix
9. Single batched commit covering all tasks

**Does NOT:**
- Commit after each individual task
- Enforce test-before-implementation ordering
- Dispatch subagents
- Push to remote (that's `session-end`)

**Outputs:** All tasks implemented, tests passing, one commit on feature branch.

**Next step:** Manual testing, then `session-debug` if issues found, or `session-review` if clean.

### 4. session-debug

**Purpose:** Structured root-cause debugging for issues found during manual testing. Repeatable — invoke as many times as needed.

**Preconditions:**
- On a `feature/*` branch with implementation work present

**Steps:**
1. Describe the symptom — what's happening vs. what's expected
2. Reproduce minimally (test, command, or UI action)
3. Read the relevant code — don't guess at the cause
4. Check recent commits and DECISIONS.md for context
5. Identify root cause
6. Implement minimal fix — change only what's necessary
7. Add regression test if the bug is non-trivial
8. Run `uv run pytest tests/` to confirm fix doesn't break anything

**Does NOT:**
- Create a separate commit (fixes accumulate on the working branch)
- Apply workarounds — fix root causes
- Refactor surrounding code

**Outputs:** Fix applied to working branch, tests passing.

**Next step:** More `session-debug` rounds if needed, or `session-review` when manual testing passes.

### 5. session-review

**Purpose:** Validation gate that must pass before `session-end` can proceed.

**Steps:**
1. Collect modified file scope:
   - `git diff --name-only HEAD` (staged + unstaged changes)
   - `git ls-files --others --exclude-standard` (untracked)
2. Architecture checks:
   - Idempotent encoding preserved?
   - Resumable workflows intact?
   - Signal processing semantics unchanged (unless intentional + ADR)?
3. Completeness checks:
   - Missing tests for new logic?
   - Missing Alembic migration for schema changes?
   - Missing doc updates per CLAUDE.md §3.6?
4. Run verification gates in order:
   - `uv run ruff format --check` on modified Python files
   - `uv run ruff check` on modified Python files
   - `uv run pyright` on modified Python files (full run if pyproject.toml changed)
   - `uv run pytest tests/`
   - `cd frontend && npx tsc --noEmit` (if frontend changed)
5. Report findings with file:line references
6. Output explicit verdict: `Ready for session-end: yes` or `Ready for session-end: no`

**Does NOT:**
- Commit, push, or create PRs
- Fix issues itself (report them for `session-debug` or manual fix)

**Outputs:** Verdict with any blocking issues listed.

**Next step:** `session-end` if yes, fix issues if no.

### 6. session-end

**Purpose:** Submit PR, squash-merge, return to clean main.

**Preconditions:**
- `session-review` passed with `Ready for session-end: yes`
- On a `feature/*` branch

**Steps:**
1. **Gate check:** Verify session-review passed. If not, stop and run `session-review`.
2. **Commit** any remaining uncommitted changes (debug fixes added after implementation commit)
3. **Push** feature branch (`git push -u origin <branch>` if first push, else `git push`)
4. **Create PR** targeting main:
   - Check for existing PR first (reuse if found)
   - Title from plan/feature name
   - Body: summary bullets, test plan, verification results
5. **Squash-merge** the PR via `gh pr merge --squash`
   - If blocked (conflicts, required checks, permissions), report the blocker and stop
6. **Return to main:**
   - `git checkout main`
   - `git pull --ff-only origin main`
   - Delete local feature branch (`git branch -d feature/<name>`)
7. Report PR URL and any follow-up work

**Does NOT:**
- Force-push or bypass branch protections
- Delete remote branches (GitHub handles this on merge)
- Skip the review gate

**Outputs:** Merged PR URL, clean local main.

## Brainstorming Integration

The superpowers brainstorming skill is used unchanged, with two overrides documented in the workflow:

1. **Spec path:** `docs/specs/YYYY-MM-DD-<topic>-design.md` (not `docs/superpowers/specs/`)
2. **No commit:** The spec is written but left uncommitted on main. `session-plan` handles the commit on the feature branch.

After brainstorming completes, control passes to `session-plan` (not superpowers `writing-plans`).

## CLAUDE.md Updates

Section 10.1 (Workflow) will be rewritten to reference the new session skills:
- Remove superpowers integration details (subagent-driven-development, TDD enforcement, worktrees)
- Document the new canonical flow
- Keep verification gates (§10.4) unchanged
- Keep doc-update matrix unchanged

Section 10.2 (Feature Branch Lifecycle) updated:
- Branch creation moves from session-begin to session-plan
- Branch naming: `feature/<feature-name>` (not `codex/<slug>`)

## AGENTS.md Updates

Replace inline phase descriptions with references to `docs/workflows/` files:
- Phase 1 (Context) → `session-begin.md`
- Phase 2 (Design) → brainstorm (superpowers) + `session-plan.md`
- Phase 3 (Plan) → `session-plan.md`
- Phase 4 (Implement) → `session-implement.md`
- Phase 5 (Debug) → `session-debug.md` (new)
- Phase 6 (Verify) → `session-review.md`
- Phase 7 (Finish) → `session-end.md`

## Removed Artifacts

- `.claude/agents/task-implementer.md` — no longer needed without subagent dispatching
- References to superpowers skills: `writing-plans`, `subagent-driven-development`, `test-driven-development`, `executing-plans`, `using-git-worktrees`, `finishing-a-development-branch` removed from CLAUDE.md

## What Stays

- Superpowers brainstorming skill (unchanged)
- Superpowers systematic-debugging skill (available but not required — `session-debug` is lighter)
- Verification gates in CLAUDE.md §10.4 (unchanged)
- Doc-update matrix in CLAUDE.md §10.4 (unchanged)
- All testing requirements in CLAUDE.md §5 (unchanged — tests required, just not TDD-ordered)
