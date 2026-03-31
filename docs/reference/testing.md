# Testing Requirements

> Read this when adding tests, setting up test infrastructure, or working on CI.

Testing is not optional. Every meaningful change must include:
- unit tests for new logic
- integration tests for API endpoints
- at least one end-to-end smoke test path that exercises the real workflows

## Unit Tests
Add unit tests for:
- encoding_signature computation (idempotency)
- audio window slicing logic
- feature extraction shape correctness
- TFLite runner batching (mock interpreter acceptable)
- Parquet writer behavior (temp + atomic promote)
- clustering pipeline (small synthetic embeddings)

Guidelines:
- prefer deterministic tests
- isolate file I/O behind temp directories
- mock external dependencies when appropriate (e.g., TFLite interpreter)

## Running Tests Locally
The repo must include:
- `pytest` configuration
- a single command to run unit+integration tests
- a command to run tests continuously on file changes

Required commands:
- `pytest` (all tests)
- `pytest -q` (quiet)
- `pytest -k <pattern>` (focused)
- "watch mode" (choose one):
  - `pytest-watch` (`ptw`) OR
  - `watchexec -r pytest` OR
  - `entr -r pytest`

Document the chosen tool in README and add it to dev dependencies.

## End-to-End Smoke Test (E2E)
Add a minimal E2E test that:
1. Starts API + worker (in-process for tests or via subprocess)
2. Uploads a small fixture audio file
3. Queues a ProcessingJob
4. Polls until job completes
5. Verifies EmbeddingSet exists and parquet file is readable
6. Queues a ClusteringJob on that embedding set
7. Polls until complete
8. Verifies clusters and assignments exist and are consistent

Constraints:
- E2E must run in under a few minutes locally
- Use a tiny audio fixture (e.g., 10-20 seconds)
- Use a tiny embedding model stub if needed (see below)

## Frontend Tests (Playwright)
When changing UI components, add or update Playwright tests in `frontend/e2e/`.

**When to add tests:**
- Any new interactive feature (buttons, forms, expandable rows, audio playback)
- Changes to data flow between frontend and backend (API calls, query hooks)
- Bug fixes for UI behavior (regression tests)

**Test patterns:**
- **API-level tests** — use `request` fixture to hit backend endpoints directly and validate response content (e.g., WAV duration, JSON shape). These are fast and don't need a browser page.
- **UI interaction tests** — use `page` fixture to navigate, click, and assert DOM state. Verify that user actions produce correct side effects (e.g., audio element src, table expansion, form submission).
- **Hydrophone regressions** — include timestamp-mapping playback checks and Extract-button activation checks when fixing Hydrophone tab playback/label workflows.
- Skip gracefully when preconditions aren't met (e.g., no completed jobs) using `test.skip()`.

**Running:**
```bash
cd frontend
npx playwright test                    # all tests
npx playwright test e2e/some.spec.ts   # specific file
npx playwright test -g "test name"     # by name pattern
npx playwright test --headed           # see the browser
```

**Requirements:**
- Tests run against `localhost:5173` (frontend dev server) proxying to `localhost:8000` (backend)
- Backend must be running with real or fixture data
- Config is in `frontend/playwright.config.ts`; tests go in `frontend/e2e/*.spec.ts`
- Install browsers once: `cd frontend && npx playwright install chromium`

## Model Stub Strategy (So Tests Are Fast)
For unit/integration/E2E tests:
- Provide a "FakeTFLiteModel" implementation that returns deterministic embeddings
  (e.g., sine/cosine transforms of window index)
- Gate real Perch model execution behind an environment flag
  - default tests use FakeTFLiteModel
  - optional manual run uses real model if available
