# session-implement

Work through the plan tasks sequentially, then commit all changes as a single batch.

## Preconditions

- On a `feature/*` branch
- A plan exists in `docs/plans/`

## Steps

1. **Confirm branch and read plan**
   - Verify you are on a `feature/*` branch; if on main, stop
   - Read the plan from `docs/plans/`
   - Read the selected domain capsule(s) recorded in the plan; if the plan predates domain capsules, select domains with `docs/agent-context/domain-map.md`
   - Restate the task scope in one sentence

2. **Identify affected files** before editing anything

3. **Check DECISIONS.md** for prior decisions that may conflict with the approach

4. **Back up the production database** if ANY task involves database changes (migrations, data backfills, manual SQL, schema changes, destructive deletes):
   - Read `HUMPBACK_DATABASE_URL` from `.env` to locate the database file
   - Copy it to `<path>.YYYY-MM-DD-HH:mm.bak` (UTC timestamp)
   - Verify the backup exists and has non-zero size before continuing
   - If the backup fails, **stop** — do not proceed with database modifications

5. **Work through tasks sequentially**
   - Read existing code before modifying
   - Implement the change
   - Write tests alongside or after implementation (tests are required, ordering is not)
   - Check off acceptance criteria in the plan as you go
   - **Per-task testing** — after completing each task:
     1. Identify source files modified in this task
     2. Use `docs/agent-context/test-map.md` and the selected domain `tests.md` files to choose targeted tests
     3. If the domain map does not cover the modified source files, map them to test files using path conventions:
        - `src/humpback/<module>.py` → `tests/unit/test_<module>*.py`
        - `src/humpback/<subdir>/<module>.py` → `tests/unit/test_<subdir>_<module>*.py` and `tests/unit/test_<module>*.py`
        - API routes in `src/humpback/api/` → also include matching `tests/integration/` files
        - Union all mapped test files into a single `pytest` invocation
     4. Run the targeted tests inline (skip only when no matching test command or file exists, as with docs-only tasks)
     5. If the current agent tooling supports explicitly delegated background testing, run one background full suite (`uv run pytest tests/ -q`) at a time and consume only its short pass/fail summary. If not, rely on the final full-suite gate.
     6. If any targeted or background test reports failures, pause and fix the regression before continuing to the next task

6. **Run verification gates** after all tasks complete:
   - `uv run ruff format --check` on modified Python files
   - `uv run ruff check` on modified Python files
   - `uv run pyright` on modified Python files
   - `uv run pytest tests/` — **skip if** the last background test agent completed green after the final task finished and no source files were modified since that run; **run fresh** otherwise
   - `cd frontend && npx tsc --noEmit` (if frontend files changed)

7. **Fix any verification failures**

8. **Update documentation** per CLAUDE.md documentation rules
   - Also update the relevant `docs/agent-context/domains/*/` capsule when the change affects domain-local agent operating context

9. **Single batched commit** covering all tasks and test additions

## Rules

- Prefer editing existing files over creating new ones
- Keep changes focused — don't refactor surrounding code
- Follow all CLAUDE.md conventions (uv, migrations, file structure)
- **Import ordering**: when adding new imports, include them in the same edit as the code that uses them. The PostToolUse hook runs `ruff format` after every file edit, which may reorder the import block. Adding import + usage together keeps edits self-contained and avoids unnecessary churn.
- Do NOT run `ruff check` or `pyright` between individual tasks — only at the end (step 6). Intermediate `ruff check --fix` runs will strip imports that are needed by later edits.

## Does NOT

- Commit after each individual task (one batch commit at the end)
- Enforce test-before-implementation ordering
- Dispatch subagents for implementation work (background test agents are the exception — see step 5)
- Push to remote (that's `session-end`)

## Output

All tasks implemented, tests passing, one commit on feature branch.

## Next Step

Manual testing, then `session-debug` if issues found, or `session-review` if clean.
