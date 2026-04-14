# Test Execution Optimization Implementation Plan

**Goal:** Replace full-suite per-task test runs with targeted inline tests plus background sub-agent full suite runs, reducing blocking time and context window consumption during implementation sessions.
**Spec:** [docs/specs/2026-04-14-test-optimization-design.md](../specs/2026-04-14-test-optimization-design.md)

---

### Task 1: Add per-task testing sub-steps to session-implement workflow

**Files:**
- Modify: `docs/workflows/session-implement.md`

Add a new sub-section under step 4 ("Work through tasks sequentially") that describes the per-task testing procedure:

- After completing each task, identify the source files modified in that task
- Map modified source files to test files using the path convention:
  - `src/humpback/<module>.py` maps to `tests/unit/test_<module>*.py`
  - `src/humpback/<subdir>/<module>.py` maps to `tests/unit/test_<subdir>_<module>*.py` and `tests/unit/test_<module>*.py`
  - API routes in `src/humpback/api/` also include matching `tests/integration/` files
  - Union all mapped test files into a single pytest invocation
- Run the targeted tests inline
- Spawn a background sub-agent (`run_in_background: true`) to run the full suite (`uv run pytest tests/ -q`), reporting only a summary (pass/fail counts, failure names if any)
- Only one background test agent at a time — skip spawning if one is already in flight
- If a background agent reports failures, pause and fix before continuing
- If no matching test files are found, skip inline testing (background agent still covers it)

**Acceptance criteria:**
- [ ] Step 4 includes per-task testing sub-steps with the path mapping convention
- [ ] Background sub-agent behavior is documented (spawn, concurrency guard, failure handling)
- [ ] Existing rules in the workflow (import ordering, no ruff check between tasks) are preserved unchanged

**Tests needed:**
- None (documentation-only change)

---

### Task 2: Add conditional skip logic to verification gate

**Files:**
- Modify: `docs/workflows/session-implement.md`

Update step 5 ("Run verification gates") to add conditional skip logic for the final pytest run:

- If the last background test agent completed successfully after the final task finished (no source code changed since that green run), the full pytest run in step 5 can be skipped
- Ruff format, ruff check, pyright, and frontend tsc checks always run regardless (they are fast)
- If the last background run hasn't completed yet, or if any code was changed after it started, run the full suite fresh

**Acceptance criteria:**
- [ ] Step 5 documents the conditional skip for pytest based on last background agent status
- [ ] Linting, formatting, type-checking gates remain unconditional
- [ ] The condition is stated precisely (no ambiguity about what "after the final task" means)

**Tests needed:**
- None (documentation-only change)

---

### Task 3: Add tiered testing note to CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

Add a brief note to section 5 ("Testing Requirements") pointing to the tiered testing strategy documented in session-implement. This should be 1-2 lines only — CLAUDE.md stays lean, with session-implement holding the details.

The note should mention:
- Per-task verification uses targeted tests inline plus background sub-agent for full suite
- Full details are in `docs/workflows/session-implement.md`

**Acceptance criteria:**
- [ ] Section 5 of CLAUDE.md includes a brief pointer to the tiered testing strategy
- [ ] No duplication of the full strategy details (those stay in session-implement.md)
- [ ] The addition is 1-2 lines, keeping CLAUDE.md lean

**Tests needed:**
- None (documentation-only change)

---

### Verification

This plan modifies only documentation files, so Python linting/type-checking verification is not applicable. Verification is:
1. Review `docs/workflows/session-implement.md` for internal consistency and completeness
2. Review `CLAUDE.md` section 5 for correct pointer to session-implement
3. Confirm no unintended changes to other sections of either file
