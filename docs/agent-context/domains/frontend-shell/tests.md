# Frontend Shell Tests

Use these commands for targeted feedback. The final backend gate remains
`uv run pytest tests/` when backend behavior changes.

## Typecheck

- `cd frontend && npx tsc --noEmit`

## Shell Smoke

- `cd frontend && npx playwright test e2e/navigation-retired-workflows.spec.ts e2e/compute-device-badge.spec.ts`

## Expansion

- Route or query hook changes: run the owning feature domain's Playwright smoke.
- Timeline shared component changes: run Signal Timeline frontend tests.
- API client type changes: run TypeScript and the matching backend integration
  tests when backend behavior changed.
