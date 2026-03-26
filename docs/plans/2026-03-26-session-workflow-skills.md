# Session Workflow Skills Implementation Plan

**Goal:** Replace superpowers-driven workflow with 6 lightweight session skills stored in `docs/workflows/`, invocable from both Claude Code and Codex.

**Spec:** `docs/specs/2026-03-26-session-workflow-skills-design.md`

---

### Task 1: Create workflow files

**Files:**
- Create: `docs/workflows/session-begin.md`
- Create: `docs/workflows/session-plan.md`
- Create: `docs/workflows/session-implement.md`
- Create: `docs/workflows/session-debug.md`
- Create: `docs/workflows/session-review.md`
- Create: `docs/workflows/session-end.md`

**Acceptance criteria:**
- [ ] Each file contains the full skill specification from the design spec
- [ ] Steps are numbered and actionable
- [ ] Preconditions, outputs, and "next step" are documented per skill
- [ ] session-review outputs explicit `Ready for session-end: yes/no` verdict
- [ ] session-end includes gate check for session-review
- [ ] session-plan documents brainstorming overrides (spec path, no commit)
- [ ] session-debug is documented as repeatable (zero or more rounds)

---

### Task 2: Create Claude Code slash command wrappers

**Files:**
- Create: `.claude/commands/session-begin.md`
- Create: `.claude/commands/session-plan.md`
- Create: `.claude/commands/session-implement.md`
- Create: `.claude/commands/session-debug.md`
- Create: `.claude/commands/session-review.md`
- Create: `.claude/commands/session-end.md`

**Acceptance criteria:**
- [ ] Each file is a one-liner referencing the corresponding `docs/workflows/` file
- [ ] Format: `Read and execute the workflow defined in docs/workflows/<skill>.md`

---

### Task 3: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

**Acceptance criteria:**
- [ ] Phases reference `docs/workflows/` files instead of inline descriptions
- [ ] Phase mapping: Context → session-begin, Design → brainstorm + session-plan, Plan → session-plan, Implement → session-implement, Debug → session-debug (new), Verify → session-review, Finish → session-end
- [ ] TDD references removed; tests-required language retained
- [ ] Codex branch prefix updated from `codex/<slug>` to `feature/<name>`

---

### Task 4: Update CLAUDE.md §10 (Workflow)

**Files:**
- Modify: `CLAUDE.md` (sections 10.1 through 10.5)

**Acceptance criteria:**
- [ ] §10.1 rewritten: canonical flow is `session-begin -> brainstorm -> session-plan -> session-implement -> [session-debug]* -> session-review -> session-end`
- [ ] §10.1 references `docs/workflows/` as skill location
- [ ] §10.1 documents brainstorming overrides (spec path `docs/specs/`, no commit on main)
- [ ] §10.1 removes all references to: subagent-driven-development, TDD enforcement, worktrees, writing-plans, executing-plans, finishing-a-development-branch, requesting-code-review, verification-before-completion
- [ ] §10.2 updated: branch creation moves to session-plan (not start of brainstorming)
- [ ] §10.2 removes worktree references
- [ ] §10.3 updated to reference session-begin skill
- [ ] §10.4 (Verification Gates) unchanged
- [ ] §10.5 updated to reference `docs/workflows/` and AGENTS.md

---

### Task 5: Remove obsolete artifacts

**Files:**
- Delete: `.claude/agents/task-implementer.md`

**Acceptance criteria:**
- [ ] File removed
- [ ] No remaining references to `task-implementer` in CLAUDE.md or AGENTS.md

---

### Verification

Run in order after all tasks:
1. Confirm all 6 files exist in `docs/workflows/`
2. Confirm all 6 files exist in `.claude/commands/`
3. Verify no references to removed superpowers skills in CLAUDE.md (grep for `subagent-driven-development`, `writing-plans`, `executing-plans`, `test-driven-development`, `finishing-a-development-branch`, `using-git-worktrees`)
4. Verify no references to `task-implementer` in CLAUDE.md or AGENTS.md
5. Verify AGENTS.md references resolve to real files in `docs/workflows/`
