# session-debug

Structured root-cause debugging for issues found during manual testing. Repeatable — invoke as many times as needed between `session-implement` and `session-review`.

## Preconditions

- On a `feature/*` branch with implementation work present

## Steps

1. **Describe the symptom** — what's happening vs. what's expected

2. **Reproduce minimally** — find the minimal reproduction path (test, command, or UI action)

3. **Read the relevant code** — don't guess at the cause
   - Check recent commits for context
   - Check DECISIONS.md for relevant decisions

4. **Identify root cause** — explain why the bug happens

5. **Implement minimal fix** — change only what's necessary

6. **Add regression test** if the bug is non-trivial

7. **Run test suite** — `uv run pytest tests/` to confirm fix doesn't break anything

## Rules

- Don't apply workarounds — fix root causes
- Don't refactor surrounding code
- If the fix changes signal processing or data models, it needs an ADR in DECISIONS.md

## Does NOT

- Create a separate commit (fixes accumulate on the working branch)
- Dispatch subagents

## Output

Fix applied to working branch, tests passing.

## Next Step

More `session-debug` rounds if needed, or `session-review` when manual testing passes.
