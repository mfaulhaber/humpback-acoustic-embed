# session-implement

Work through the plan tasks sequentially, then commit all changes as a single batch.

## Preconditions

- On a `feature/*` branch
- A plan exists in `docs/plans/`

## Steps

1. **Confirm branch and read plan**
   - Verify you are on a `feature/*` branch; if on main, stop
   - Read the plan from `docs/plans/`
   - Restate the task scope in one sentence

2. **Identify affected files** before editing anything

3. **Check DECISIONS.md** for prior decisions that may conflict with the approach

4. **Work through tasks sequentially**
   - Read existing code before modifying
   - Implement the change
   - Write tests alongside or after implementation (tests are required, ordering is not)
   - Check off acceptance criteria in the plan as you go
   - **Per-task testing** — after completing each task:
     1. Identify source files modified in this task
     2. Map them to test files using path conventions:
        - `src/humpback/<module>.py` → `tests/unit/test_<module>*.py`
        - `src/humpback/<subdir>/<module>.py` → `tests/unit/test_<subdir>_<module>*.py` and `tests/unit/test_<module>*.py`
        - API routes in `src/humpback/api/` → also include matching `tests/integration/` files
        - Union all mapped test files into a single `pytest` invocation
     3. Run the targeted tests inline (skip if no matching test files found)
     4. Spawn a background sub-agent (`run_in_background: true`) to run the full suite (`uv run pytest tests/ -q`). The agent reports only a summary: pass/fail counts and, on failure, test names with short error descriptions. Do not include full pytest output in the summary. Only one background test agent at a time — skip spawning if one is already in flight.
     5. If a background agent reports failures, pause and fix the regression before continuing to the next task

5. **Run verification gates** after all tasks complete:
   - `uv run ruff format --check` on modified Python files
   - `uv run ruff check` on modified Python files
   - `uv run pyright` on modified Python files
   - `uv run pytest tests/` — **skip if** the last background test agent completed green after the final task finished and no source files were modified since that run; **run fresh** otherwise
   - `cd frontend && npx tsc --noEmit` (if frontend files changed)

6. **Fix any verification failures**

7. **Update documentation** per CLAUDE.md §3.6 doc-update matrix

8. **Single batched commit** covering all tasks and test additions

## Rules

- Prefer editing existing files over creating new ones
- Keep changes focused — don't refactor surrounding code
- Follow all CLAUDE.md conventions (uv, migrations, file structure)
- **Import ordering**: when adding new imports, include them in the same edit as the code that uses them. The PostToolUse hook runs `ruff format` after every file edit, which may reorder the import block. Adding import + usage together keeps edits self-contained and avoids unnecessary churn.
- Do NOT run `ruff check` or `pyright` between individual tasks — only at the end (step 5). Intermediate `ruff check --fix` runs will strip imports that are needed by later edits.

## Does NOT

- Commit after each individual task (one batch commit at the end)
- Enforce test-before-implementation ordering
- Dispatch subagents for implementation work (background test agents are the exception — see step 4)
- Push to remote (that's `session-end`)

## Output

All tasks implemented, tests passing, one commit on feature branch.

## Next Step

Manual testing, then `session-debug` if issues found, or `session-review` if clean.
