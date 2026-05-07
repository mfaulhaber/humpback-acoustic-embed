# Codebase Cleanup And Consolidation Implementation Plan

**Goal:** Remove confirmed dead code, sync stale docs, and consolidate repeated test helpers without changing active user-facing behavior.
**Spec:** [docs/specs/2026-05-07-codebase-cleanup-consolidation-design.md](../specs/2026-05-07-codebase-cleanup-consolidation-design.md)
**Primary domain:** `frontend-shell`
**Neighbor domains:** `core-platform`, `call-parsing`, `ingest-detection`, `sequence-models`

---

### Task 1: Remove Confirmed Dead Frontend Legacy Files

**Files:**
- Delete: `frontend/src/components/shared/EmbeddingSetPanel.tsx`
- Delete: `frontend/src/hooks/queries/useProcessing.ts`
- Delete: `frontend/src/hooks/queries/useAudioFiles.ts`
- Delete: `frontend/src/components/shared/ModelFilter.tsx`
- Delete: `frontend/src/components/shared/FolderTree.tsx`
- Delete: `frontend/src/hooks/useCollapseState.ts`
- Delete: `frontend/src/components/call-parsing/RegionEditOverlay.tsx`
- Delete: `frontend/src/components/call-parsing/RegionJobSummary.tsx`
- Delete: `frontend/src/utils/audio.ts`

**Acceptance criteria:**
- [ ] Production import search confirms every deleted file has no remaining frontend import.
- [ ] The provider-based `frontend/src/components/timeline/overlays/RegionEditOverlay.tsx` remains the only active region edit overlay.
- [ ] Legacy embedding-set and audio-file query hooks are removed rather than returning empty placeholder data.
- [ ] `cd frontend && npx tsc --noEmit` passes after deletions.
- [ ] Knip no longer reports the deleted files.

**Tests needed:**
- Frontend TypeScript check.
- Knip cleanup signal.
- Frontend shell navigation smoke if route or shared component imports change unexpectedly.

---

### Task 2: Resolve Unused Frontend Symbols Without Restoring Retired UI

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`
- Modify: `frontend/src/components/call-parsing/ReviewToolbar.tsx`
- Modify: `frontend/src/components/call-parsing/WindowClassifyReviewWorkspace.tsx`
- Modify: `frontend/src/components/classifier/DetectionSourcePicker.tsx`
- Modify: `frontend/src/components/classifier/HydrophoneTab.tsx`
- Modify: `frontend/src/components/classifier/LabelingTab.tsx`
- Modify: `frontend/src/components/classifier/TrainingTab.tsx`
- Modify: `frontend/src/components/classifier/TuningTab.tsx`
- Modify: `frontend/src/components/timeline/provider/TimelineProvider.tsx`

**Acceptance criteria:**
- [ ] Strict TypeScript unused-symbol mode no longer reports the touched files.
- [ ] `markNegative` and `handleReclassify` are removed if no visible control consumes them.
- [ ] No removed callback changes visible review behavior or route behavior.
- [ ] Any prop removed from a component is also removed from all callers.
- [ ] Retained generic visualization primitives are not removed as part of this task.

**Tests needed:**
- `cd frontend && npm exec tsc -- --noEmit --noUnusedLocals --noUnusedParameters`
- `cd frontend && npx tsc --noEmit`
- Call Parsing frontend smoke if review workspace behavior is touched.

---

### Task 3: Clean Backend Compatibility Leftovers And Test-Only Synthesis

**Files:**
- Modify: `src/humpback/api/app.py`
- Delete: `src/humpback/classifier/label_processor.py` if no active production owner remains
- Create: `tests/helpers/synthesis.py`
- Modify: `tests/unit/test_synthesis.py`

**Acceptance criteria:**
- [ ] `src/humpback/api/app.py` no longer imports the empty `humpback.models.label_processing` stub for table registration.
- [ ] Synthesis helpers currently tested through `label_processor.py` either move to `tests/helpers/synthesis.py` or get an explicitly active production owner.
- [ ] If `label_processor.py` is deleted, no production, script, or test import still references it.
- [ ] `models/processing.py` remains because current code still imports `JobStatus`.
- [ ] Empty retired model stubs are not removed unless import search proves they are unused and no compatibility purpose remains.

**Tests needed:**
- `uv run pytest tests/unit/test_health.py tests/integration/test_trusted_hosts.py -q`
- `uv run pytest tests/unit/test_synthesis.py -q`
- Backend lint/type checks on modified Python files.

---

### Task 4: Consolidate Repeated Migration Test Helpers

**Files:**
- Create: `tests/helpers/__init__.py`
- Create: `tests/helpers/migrations.py`
- Modify: `tests/unit/test_migration_048_compute_device.py`
- Modify: `tests/unit/test_migration_053_window_classification.py`
- Modify: `tests/db/test_migration_061.py`

**Acceptance criteria:**
- [ ] Shared helper covers Alembic config creation from repo root.
- [ ] Shared helper covers async current-schema database creation when tests need it.
- [ ] Shared helper covers SQLite table, column, index, and table-existence introspection without hiding each test's migration-specific assertions.
- [ ] At least one migration-test cluster is migrated to prove the helper pattern.
- [ ] No migration behavior or revision target changes.

**Tests needed:**
- Targeted migrated migration tests.
- `uv run pytest tests/unit/test_migration_* tests/db -q` before final review if helper adoption spans multiple migration files.

---

### Task 5: Consolidate Repeated Audio And Embedding Test Fixtures

**Files:**
- Create: `tests/helpers/audio.py`
- Create: `tests/helpers/embeddings.py`
- Modify: `tests/integration/test_region_detection_worker.py`
- Modify: `tests/integration/test_call_parsing_router.py`
- Modify: `tests/integration/test_detection_reembedding_worker.py`
- Modify: `tests/integration/test_embedding_sync_worker.py`
- Modify: `tests/unit/test_classifier_worker.py`
- Modify: `tests/unit/test_trainer.py`
- Modify: `tests/integration/test_classifier_api.py`

**Acceptance criteria:**
- [ ] Sine WAV writing is provided by one test helper and reused by at least two existing tests.
- [ ] Detection embedding parquet writing is provided by one test helper and reused by classifier worker, trainer, or API tests.
- [ ] Embedding-set parquet writing is provided by one test helper only where historical fixture coverage still requires it.
- [ ] Helper names make legacy fixture intent explicit where embedding-set fixtures are retained for historical compatibility tests.
- [ ] Test helper consolidation does not broaden runtime imports or change fixture data semantics.

**Tests needed:**
- Targeted tests for each migrated fixture group.
- Ingest Detection smoke if classifier API or worker fixtures are changed.
- Call Parsing smoke if region/call-parsing WAV fixtures are changed.

---

### Task 6: Sync Frontend And Agent Reference Documentation

**Files:**
- Modify: `docs/reference/frontend.md`
- Modify: `docs/agent-context/domains/sequence-models/README.md`

**Acceptance criteria:**
- [ ] `docs/reference/frontend.md` no longer lists deleted components as active.
- [ ] Retained generic visualization primitives are documented as future-use primitives, not active Sequence Models workflows.
- [ ] `SpanNavBar`, `DiscreteSequenceBar`, `MotifTimelineLegend`, `MotifHighlightOverlay`, and `CollapsiblePanelCard` remain documented as retained generic primitives with tests.
- [ ] Active routes remain documented for Classifier, Vocalization, Call Parsing, Sequence Models Continuous Embedding, and Admin.
- [ ] Agent-context docs are touched only if the cleanup changes what agents need to load or know.

**Tests needed:**
- Documentation review and path spot-checks.
- `git diff --check`.

---

### Task 7: Clean Frontend Dependencies And Intentional Knip Findings

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Modify: `docs/reference/frontend.md` if dependency notes need updating

**Acceptance criteria:**
- [ ] `@radix-ui/react-toast` is removed if confirmed unused by imports and build.
- [ ] `@types/uuid` is removed if `uuid` continues to typecheck through bundled types.
- [ ] `plotly.js-basic-dist-min` is removed only if `react-plotly.js` and the UMAP plot still build without relying on it.
- [ ] Remaining knip reports for retained generic primitives are documented as intentional.
- [ ] `npm install` or the equivalent npm lockfile update is run from `frontend/` after package changes.

**Tests needed:**
- `cd frontend && npx tsc --noEmit`
- `cd frontend && npm exec tsc -- --noEmit --noUnusedLocals --noUnusedParameters`
- `cd frontend && npx --yes knip --no-exit-code --reporter compact`

---

### Verification

Run in order after all tasks:

1. `git diff --check`
2. `uv run ruff format --check src tests`
3. `uv run ruff check src tests scripts`
4. `uv run pyright`
5. `uv run pytest tests/unit/test_health.py tests/integration/test_trusted_hosts.py tests/unit/test_synthesis.py -q`
6. `uv run pytest tests/unit/test_migration_* tests/db -q`
7. `uv run pytest tests/`
8. `cd frontend && npx tsc --noEmit`
9. `cd frontend && npm exec tsc -- --noEmit --noUnusedLocals --noUnusedParameters`
10. `cd frontend && npx --yes knip --no-exit-code --reporter compact`
11. `cd frontend && npx playwright test e2e/navigation-retired-workflows.spec.ts e2e/compute-device-badge.spec.ts`
