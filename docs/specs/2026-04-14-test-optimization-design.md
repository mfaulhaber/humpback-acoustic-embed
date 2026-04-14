# Test Execution Optimization Design

## Problem

During implementation sessions using the superpowers `executing-plans` and `verification-before-completion` skills, the full test suite (1,945 tests, ~100s) runs between every task in a plan. For a typical 6-9 task plan, this means 10-15 minutes of test execution time and thousands of lines of pytest output consuming the main conversation's context window. The project's own `session-implement` workflow prescribes testing once at the end, but the superpowers skills override this with per-task verification.

## Solution

A tiered testing strategy that preserves per-task feedback while minimizing both wall time and token consumption:

1. **Targeted inline tests** after each task (fast, minimal output)
2. **Background sub-agent full suite** after each task (non-blocking, output stays out of main context)
3. **Conditional final gate** at end of plan (skip if last background run already confirmed green)

## Design

### Targeted inline tests

After completing each task, identify which source files were modified and run only the corresponding test files.

**Path mapping convention:**
- Modified `src/humpback/<module>.py` maps to `tests/unit/test_<module>*.py`
- Modified `src/humpback/<subdir>/<module>.py` maps to `tests/unit/test_<subdir>_<module>*.py` and `tests/unit/test_<module>*.py`
- Modified API routes in `src/humpback/api/` also include matching `tests/integration/` files
- When multiple source files are modified in one task, union their mapped test files into a single pytest invocation
- If no matching test files are found for a given source file, skip inline testing for that task

This is a convention followed by Claude based on workflow instructions, not an automated tool or pytest plugin. Claude already knows which files it modified during a task and can derive the test file paths.

**Expected cost:** ~2-10s per task, minimal output in context.

### Background sub-agent full suite

After each task completes, spawn a background agent to run the full test suite in parallel with continued implementation work.

**Agent behavior:**
- Spawned with `run_in_background: true`
- Runs `uv run pytest tests/ -q`
- Reports back a summary only: pass count, fail count, and if failures exist, the specific test names and short error descriptions
- Does not dump full pytest output into its report

**Concurrency guard:** Only one background test agent runs at a time. If a previous agent is still in flight when the next task completes, skip spawning a new one. The in-flight run covers the changes.

**On failure notification:**
- Pause current task work
- Investigate and fix the regression before continuing to the next task

**Token economics:** Full pytest output (~1,945 test lines plus warnings) stays in the sub-agent's context and never enters the main conversation. The notification back is ~2-5 lines for a pass, ~10-20 lines for failures. Over a 9-task plan, this eliminates roughly 9 full pytest outputs from the main context.

### End-of-plan verification gate

After all tasks complete, the existing verification sequence runs with one optimization:

1. `uv run ruff format --check` on modified Python files (inline)
2. `uv run ruff check` on modified Python files (inline)
3. `uv run pyright` on modified Python files (inline)
4. `uv run pytest tests/` — **skip if** the last background agent completed successfully after the final task finished (no code changed since last green run); **run fresh** otherwise
5. `cd frontend && npx tsc --noEmit` if frontend files changed (inline)

### Concurrency safety

Background agents run pytest against the same working directory. This is safe because:
- Each test gets its own temporary SQLite database and storage path via conftest fixtures
- No shared mutable state between the main conversation's file edits and the test runner's read operations
- The only risk is if Claude is mid-edit when pytest tries to import a partially-written file, but background agents are spawned after task completion (all edits finished)

## Implementation

This is a workflow documentation change only. No code changes, no pytest plugins, no test file modifications.

**Files to modify:**
- `docs/workflows/session-implement.md` — add per-task testing sub-steps to step 4, add conditional skip logic to step 5
- `CLAUDE.md` — brief addition to testing section (section 5) noting the tiered strategy and pointing to session-implement for details

**Files that do NOT change:**
- `pyproject.toml` — no pytest configuration changes
- Superpowers plugin skills — project workflow instructions take precedence per the skill priority rules (user instructions > skills > defaults)
- Test files — no restructuring, markers, or categorization

## Expected impact (9-task plan)

| Metric | Before | After |
|--------|--------|-------|
| Full suite runs in main context | ~9 | 0-1 |
| Blocking test time per task | ~100s | ~2-10s |
| Total blocking test time | ~15 min | ~1-2 min |
| Background test time | 0 | ~100s per run (non-blocking) |
| Pytest output in main context | ~9 full outputs | ~9 short summaries |

## Constraints

- No pytest-xdist or parallel test runner. The suite already runs at ~575% CPU for unit tests; adding parallelism would add complexity for marginal gain.
- No test markers or categorization system. Convention-based path mapping is sufficient and zero-maintenance.
- No persistent test timing database or profiling.
- Relies on existing test file naming conventions being consistent. When no matching test file is found, inline testing is skipped for that task (background agent covers it).
