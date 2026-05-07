# Codebase Cleanup And Consolidation Design

**Status:** Draft for discussion
**Date:** 2026-05-07
**Track:** Code health, frontend shell, tests, and agent documentation

---

## 1. Goal

Reduce dead code, repeated helper logic, and stale documentation without
changing user-facing behavior.

The cleanup should make the retained runtime easier to reason about after the
legacy workflow and retired Sequence Models removals:

- Remove confirmed dead frontend and backend compatibility leftovers.
- Prune reference docs that still describe removed or production-unused
  components.
- Consolidate repeated test scaffolding where the duplication is mechanical and
  low risk.
- Identify runtime duplication that deserves a small shared abstraction, while
  avoiding broad refactors that would mix cleanup with behavior changes.
- Keep tests and documentation aligned with the current active domains.

This design is intentionally conservative. The first implementation should
favor deletions and narrow extractions over changing domain behavior.

---

## 2. Review Inputs

The review used static checks, unused-code tooling, duplicate scans, and manual
inspection.

Commands and tools used during review:

- `uv run ruff check src tests scripts`
- `uv run pyright`
- strict frontend unused-symbol check:
  `cd frontend && npm exec tsc -- --noEmit --noUnusedLocals --noUnusedParameters`
- `uvx vulture src tests scripts --min-confidence 80`
- `cd frontend && npx --yes knip --no-exit-code --reporter compact`
- custom AST/window duplicate scans over Python and TypeScript
- manual inspection of `AGENTS.md`, `CLAUDE.md`, `docs/workflows/`,
  `docs/agent-context/`, and `docs/reference/`

Backend `ruff` and `pyright` are currently clean. Most actionable findings are
frontend dead code, stale reference docs, repeated test helpers, and a few
backend compatibility leftovers.

---

## 3. Current Findings

### 3.1 Confirmed Or High-Confidence Dead Frontend Code

The strongest dead-code candidates are production-unreferenced files tied to
retired embedding-set/audio workflows or obsolete region-editing components:

- `frontend/src/components/shared/EmbeddingSetPanel.tsx`
- `frontend/src/hooks/queries/useProcessing.ts`
- `frontend/src/hooks/queries/useAudioFiles.ts`
- `frontend/src/components/shared/ModelFilter.tsx`
- `frontend/src/components/shared/FolderTree.tsx`
- `frontend/src/hooks/useCollapseState.ts`
- `frontend/src/components/call-parsing/RegionEditOverlay.tsx`
- `frontend/src/components/call-parsing/RegionJobSummary.tsx`
- `frontend/src/utils/audio.ts`

`RegionEditOverlay.tsx` under `call-parsing/` duplicates the newer
provider-based overlay under `timeline/overlays/`. Production code imports the
timeline overlay, not the call-parsing-local copy.

`EmbeddingSetPanel` and its supporting hooks/types align with the already
completed legacy workflow removal plan, which explicitly expected no
embedding-set source UI dependency to remain.

### 3.2 Frontend Symbols That Need Product Decisions

Strict TypeScript identified unused callbacks and props that are likely dead
but may also represent disconnected intended UI:

- `markNegative` and `handleReclassify` in
  `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`
- `eventCount` in `ReviewToolbar`
- `onEmbed` / `isEmbedding` in `DetectionSourcePicker`
- assorted unused locals in classifier and window-classify components

These should not all be deleted blindly. For review workspaces especially, the
right question is whether the workflow still needs the action. If yes, restore
the control and cover it. If no, remove the callback and related state.

### 3.3 Retained Generic Visualization Primitives

Knip reports several Sequence Models visualization primitives as unused in
production but covered by tests:

- `SpanNavBar`
- `DiscreteSequenceBar`
- `MotifTimelineLegend`
- `MotifHighlightOverlay`
- `CollapsiblePanelCard`
- `frontend/src/components/timeline/index.ts` barrel exports

These should remain in the tree as generic visualization primitives for future
analysis and review surfaces. They should not be described as active
user-facing Sequence Models workflows. The cleanup should document them as a
small retained visualization toolkit and keep their focused tests as the
contract for future reuse.

### 3.4 Backend Compatibility Leftovers

Backend dead-code signals are weaker because the codebase keeps a few explicit
compatibility stubs:

- `src/humpback/api/app.py` still imports
  `humpback.models.label_processing` for table registration even though the
  module is an empty retired stub.
- `src/humpback/models/label_processing.py` and
  `src/humpback/models/search.py` are empty retired stubs.
- `src/humpback/models/processing.py` still carries `JobStatus`, which is
  actively imported by current Continuous Embedding and worker code. It should
  not be removed until `JobStatus` is moved to an active shared model module.
- `src/humpback/classifier/label_processor.py` is production-unreferenced and
  only reached by `tests/unit/test_synthesis.py`.

The `label_processor.py` case should be handled carefully. It may contain
useful synthesis helpers that are now test-only. Options are to move those
helpers into a test fixture module, move still-useful audio synthesis code into
an active domain module, or delete the module and tests together if the
functionality is truly retired.

### 3.5 Repeated Test Code

Duplication is concentrated in test scaffolding:

- Alembic helpers repeated across migration tests:
  `_db_url`, `_create_db`, `_alembic_config`, `_columns`, `_indexes`,
  `_tables`, and table-existence probes.
- WAV writers repeated in integration tests.
- embedding parquet writers repeated across classifier worker, trainer, and API
  tests.
- session/session_factory fixtures repeated in several domain tests.

This duplication is mechanical and safe to consolidate if done in small batches
with targeted tests.

### 3.6 Repeated Runtime Code

The runtime has several repeated patterns that may warrant shared helpers:

- Atomic write helpers and in-memory LRU behavior in
  `processing/timeline_repository.py` and `processing/timeline_cache.py`.
- Nearly identical window slicing logic in `processing/windowing.py`.
- NOAA provider timeline/count/fetch factory duplication in
  `classifier/providers/noaa_gcs.py`.
- HLS provider wrapper boilerplate in `classifier/providers/orcasound_hls.py`.
- Tooltip placement logic duplicated between timeline overlays.
- Job table state repeated across active/previous job tables in frontend
  feature domains.

These should be second-phase refactors, after obvious deletions have reduced
the surface area.

### 3.7 Documentation Drift

Agent-local docs are mostly current. The clearest drift is in detailed
reference material:

- `docs/reference/frontend.md` lists removed or unused components such as
  `FeedbackTrainingJobTable`, `DetectionTab`, `RegionJobSummary`,
  `FolderTree`, and retained Sequence Models primitives as active component
  inventory.
- The reference should distinguish active user-facing components from retained
  generic/tested primitives.
- `docs/reference/storage-layout.md` correctly marks retired roots as cleanup
  targets, but its cleanup manifest wording should remain aligned with the
  active cleanup scripts if those scripts change.

The `.claude/commands/*.md` files are intentionally small command shims that
point at `docs/workflows/*.md`; they are not stale copies of the workflow
content.

---

## 4. Non-Goals

- Do not change API behavior.
- Do not change database schema or run migrations.
- Do not alter artifact storage contracts.
- Do not remove migration files or historical specs/plans/ADRs.
- Do not remove legacy database compatibility behavior that is intentionally
  still readable, such as historical classifier training provenance.
- Do not introduce a new frontend table framework or broad UI redesign.
- Do not require production database backup unless implementation later expands
  into schema/data changes.

---

## 5. Approaches

### Approach A: Deletion-First, Then Narrow Consolidation

Remove confirmed dead files and stale docs first. Then extract repeated helpers
only where the remaining duplication is obvious and stable.

Pros:

- Lowest behavioral risk.
- Shrinks the codebase before introducing abstractions.
- Keeps review diffs understandable.
- Lets tests prove that production imports do not depend on the deleted files.

Cons:

- Requires a second pass for runtime consolidation.
- Some repeated code remains until later implementation tasks.

### Approach B: Abstraction-First Cleanup

Create shared helpers for job tables, timeline cache writes, test migrations,
and overlay tooltips before deleting stale code.

Pros:

- Directly addresses repeated code.
- May create useful long-term primitives sooner.

Cons:

- Higher risk of abstraction churn around dead code.
- Larger diffs make regressions harder to isolate.
- Shared frontend abstractions could accidentally encode domain behavior.

### Approach C: One Broad Cleanup PR

Delete dead code, consolidate test helpers, refactor runtime duplication, and
update docs in a single implementation.

Pros:

- Fastest path to a visibly cleaner tree.
- One branch carries all cleanup context.

Cons:

- High review burden.
- Harder to bisect failures.
- Easy to mix mechanical cleanup with behavioral changes.

### Recommended Approach

Use Approach A.

Phase 1 should remove dead code and stale references. Phase 2 should
consolidate repeated test helpers. Phase 3 should handle runtime/frontend
abstractions only after the retained surface is smaller.

---

## 6. Proposed Design

### 6.1 Cleanup Pass

Remove production-unreferenced frontend files that are clearly tied to retired
or obsolete workflows:

- `EmbeddingSetPanel.tsx`
- `useProcessing.ts`
- `useAudioFiles.ts`
- `ModelFilter.tsx`
- `FolderTree.tsx`
- `useCollapseState.ts`
- `RegionJobSummary.tsx`
- `utils/audio.ts`
- `call-parsing/RegionEditOverlay.tsx`

Before deleting, run `rg` and TypeScript to confirm there are no production
imports. If a file is only referenced from a stale doc, update the doc in the
same change.

For `ClassifyReviewWorkspace.tsx`, make an explicit choice per unused action:

- If `markNegative` is still a desired review workflow, wire it to a visible
  control and add or update frontend tests.
- If it is no longer desired, remove the callback and any dead state.
- If `handleReclassify` is still desired after training, restore the UI affordance
  and cover the path.
- Otherwise remove the unused callback and mutation dependency.

Remove the stale `humpback.models.label_processing` import in `api/app.py` and
confirm startup/router tests still pass.

For `label_processor.py`, split the decision:

- If the synthesis helpers are still meaningful as test-only utilities, move
  them under `tests/helpers/` and delete the production module.
- If they are still domain-useful, move the helpers to an active domain module
  with a clear owner.
- If the old synthesis behavior is retired, delete the module and the tests.

The default recommendation is to move only the tested synthesis helpers into
test helpers and delete the production module.

### 6.2 Documentation Sync

Update `docs/reference/frontend.md` so the component inventory describes the
current production surface:

- Remove `FeedbackTrainingJobTable`, `DetectionTab`, `RegionJobSummary`,
  `FolderTree`, and any other deleted components.
- Remove or reclassify production-unused Sequence Models primitives.
- Preserve documentation for active routes:
  Classifier, Vocalization, Call Parsing, Sequence Models Continuous Embedding,
  and Admin.

Document retained generic visualization primitives in a short subsection rather
than listing them as active route components. This subsection should include
`SpanNavBar`, `DiscreteSequenceBar`, `MotifTimelineLegend`,
`MotifHighlightOverlay`, and `CollapsiblePanelCard` as future-use primitives
with existing tests.

Keep `docs/agent-context/current-state.md` unchanged unless implementation
changes active capability. A pure cleanup of dead files should normally only
touch reference docs and perhaps the frontend-shell capsule if agent operating
context changes.

### 6.3 Test Helper Consolidation

Create a small `tests/helpers/` package for repeated scaffolding:

- `tests/helpers/migrations.py`
  - database URL construction
  - Alembic config creation
  - SQLite schema introspection helpers
  - current-schema creation helper when needed
- `tests/helpers/audio.py`
  - sine WAV writer
  - in-memory WAV bytes builder
- `tests/helpers/embeddings.py`
  - embedding-set parquet writer
  - detection-embedding parquet writer

Migrate tests in batches. Start with one migration-test cluster and one audio
helper cluster rather than touching every test file at once.

Keep helpers boring and test-oriented. They should not import application
services unless the existing repeated code already did.

### 6.4 Runtime Consolidation Candidates

After deletion and test-helper consolidation, evaluate these narrow runtime
extractions:

- Add a storage or processing utility for atomic file writes used by timeline
  cache/repository code.
- Refactor `slice_windows` to delegate to one internal iterator used by
  `slice_windows_with_metadata`.
- Share overlay tooltip placement logic for detection and vocalization
  overlays.
- Consider a small NOAA provider base/mixin only if it reduces duplication
  without hiding cache-specific behavior.

Each runtime consolidation should be its own small task with focused tests.

### 6.5 Frontend Job Table Consolidation Candidates

Repeated table state exists in Region, Segmentation, Continuous Embedding, and
Vocalization Clustering tables:

- active/previous mode branching
- selected IDs
- bulk delete dialog state
- filter text
- sort key/direction
- pagination
- stale selection cleanup

Do not create a generic table component. Prefer a hook such as
`useJobTableState` that owns only state mechanics and lets each domain render
its own table markup and domain-specific columns.

This should happen after dead files are removed, because the remaining table
surface will be clearer.

---

## 7. Acceptance Criteria

The cleanup implementation is successful when:

- Confirmed dead frontend files are removed.
- Stale frontend reference docs no longer list deleted components as active.
- Retained generic visualization primitives are documented as future-use
  primitives rather than active workflow components.
- `api/app.py` no longer imports the empty label-processing model stub.
- `label_processor.py` has an explicit retained owner or is removed from
  production code.
- Strict frontend unused-symbol checks have fewer or no real findings in the
  touched areas.
- `knip` no longer reports the removed files, or remaining reports are
  documented as intentionally retained.
- Repeated test helpers have at least one migrated batch proving the helper
  pattern.
- Existing production behavior remains unchanged.

---

## 8. Test Plan

For cleanup-only changes:

- `uv run ruff check` on changed Python files.
- `uv run pyright` if Python imports or test helpers change.
- `uv run pytest` on affected test files.
- `cd frontend && npx tsc --noEmit` for frontend deletions/import changes.
- `cd frontend && npm exec tsc -- --noEmit --noUnusedLocals --noUnusedParameters`
  as a cleanup-specific signal.
- `cd frontend && npx --yes knip --no-exit-code --reporter compact` as a
  cleanup-specific signal.

For final verification, retain the project gates from `CLAUDE.md`, scaled to
the actual implementation scope.

Suggested targeted tests:

- `uv run pytest tests/unit/test_health.py tests/integration/test_trusted_hosts.py -q`
  for app startup/import cleanup.
- `uv run pytest tests/unit/test_synthesis.py -q` if `label_processor.py`
  synthesis helpers move.
- `uv run pytest tests/unit/test_migration_* tests/db -q` for migration helper
  consolidation.
- `cd frontend && npx tsc --noEmit` for all frontend deletions.
- Frontend shell smoke if navigation or shared components change:
  `cd frontend && npx playwright test e2e/navigation-retired-workflows.spec.ts e2e/compute-device-badge.spec.ts`

---

## 9. Risks And Guardrails

- **Test-only imports can mask dead production modules.** Check production and
  test imports separately before deciding whether a file is truly runtime dead.
- **Retained legacy readability is not dead code by default.** Historical
  classifier provenance and migration compatibility should stay unless a
  specific removal task owns them.
- **Generic frontend abstractions can overfit.** Prefer deleting and using
  hooks for repeated mechanics over introducing shared domain-rendering
  components.
- **Knip false positives exist.** Query hooks and API helpers can be exported
  for tests or near-term use. Treat knip as a review queue, not an automatic
  deletion list.
- **Migration helpers must stay SQLite-specific where needed.** Do not hide
  `op.batch_alter_table()` or raw `PRAGMA` behavior behind helpers that obscure
  what each migration test proves.

---

## 10. Open Questions

1. Should `markNegative` and `handleReclassify` in Classify Review be restored
   as visible actions, or deleted as abandoned workflow code?
2. Should `JobStatus` move out of the retired `models/processing.py` stub into
   an active shared model/status module?
3. Should `label_processor.py` synthesis helpers become test helpers, active
   domain helpers, or be deleted with their tests?
4. Should cleanup tooling such as strict unused-symbol TypeScript and knip
   become documented maintenance commands, or remain ad hoc review tools?
