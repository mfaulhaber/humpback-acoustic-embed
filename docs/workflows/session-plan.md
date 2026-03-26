# session-plan

Create an implementation plan from a spec, create the feature branch, and commit both artifacts.

## Brainstorming Overrides

When using the superpowers brainstorming skill before this step:
- **Spec path:** `docs/specs/YYYY-MM-DD-<topic>-design.md` (not `docs/superpowers/specs/`)
- **No file on disk:** The spec is NOT written to disk during brainstorming. The approved design exists only in conversation context. This skill writes it to disk on the feature branch.
- **Skip brainstorming steps 6–8:** The brainstorming skill's "write doc", "self-review", and "user reviews file" steps are skipped — this skill handles all of that.
- **No writing-plans handoff:** After brainstorming, control passes here (not superpowers `writing-plans`)

## Preconditions

- A design has been approved in conversation context (via brainstorming)
- Currently on `main`

## Steps

1. **Read the spec** from `docs/specs/`

2. **Derive feature name** from the spec topic (e.g., `session-workflow-skills`)

3. **Create feature branch** — `feature/<feature-name>` from main

4. **Break the spec into discrete implementation tasks**
   - Each task has: title, file paths (create/modify), acceptance criteria (checkboxes), test requirements
   - No code blocks in tasks — describe what to do, not how
   - Include a verification section at the end with the exact commands to run

5. **Write the plan** to `docs/plans/YYYY-MM-DD-<feature>.md`

6. **Commit both spec and plan** as the first commit on the feature branch

## Plan Format

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

## Does NOT

- Include code blocks in plan tasks
- Enforce TDD ordering
- Dispatch subagents

## Output

Plan file committed on feature branch.

## Next Step

`session-implement`
