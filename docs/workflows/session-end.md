# session-end

Submit PR, squash-merge, and return to clean main.

## Preconditions

- `session-review` passed with `Ready for session-end: yes`
- On a `feature/*` branch

## Steps

1. **Gate check**
   - Verify `session-review` was run and passed
   - If not, stop and run `session-review` first
   - If files changed since the last review, stop and rerun `session-review`

2. **Commit remaining changes** (if any)
   - If there are uncommitted changes (e.g., debug fixes after implementation commit), create a commit
   - Stage only the reviewed scope — never use `git add -A`

3. **Push feature branch**
   - `git push -u origin <branch>` if first push
   - `git push` otherwise

4. **Create or reuse pull request**
   - Check for existing PR on this branch first (reuse if found)
   - If no PR exists, create one targeting `main`
   - Title: from plan/feature name
   - Body: summary bullets, test plan, verification results

5. **Squash-merge the PR**
   - `gh pr merge --squash`
   - If blocked (conflicts, required checks, permissions), report the specific blocker and stop
   - Do not bypass GitHub policy

6. **Return to clean main**
   - `git checkout main`
   - `git pull --ff-only origin main`
   - Delete local feature branch: `git branch -d feature/<name>`

7. **Report**
   - PR URL and merge status
   - Any follow-up work that didn't make this session

## Does NOT

- Force-push or bypass branch protections
- Delete remote branches (GitHub handles this on merge)
- Skip the review gate

## Output

Merged PR URL, clean local main.
