# Sequence Models UI Alignment Implementation Plan

**Goal:** Align Sequence Models pages with Call Parsing Segment page patterns — add delete, Review buttons, table layouts, Active panel, and fix breadcrumbs.
**Spec:** [docs/specs/2026-04-28-sequence-models-ui-alignment-design.md](../specs/2026-04-28-sequence-models-ui-alignment-design.md)

---

### Task 1: Backend delete endpoints

**Files:**
- Modify: `src/humpback/services/continuous_embedding_service.py`
- Modify: `src/humpback/services/hmm_sequence_service.py`
- Modify: `src/humpback/api/routers/sequence_models.py`

**Acceptance criteria:**
- [ ] `delete_continuous_embedding_job(session, job_id, settings)` added to continuous embedding service — fetches job, removes `continuous_embedding_dir(settings.storage_root, job_id)` via `shutil.rmtree` (ignore missing), deletes DB row, commits, returns bool
- [ ] `delete_hmm_sequence_job(session, job_id, settings)` added to HMM sequence service — same pattern using `hmm_sequence_dir`
- [ ] `DELETE /sequence-models/continuous-embeddings/{job_id}` route added, returns 204 on success, 404 if not found
- [ ] `DELETE /sequence-models/hmm-sequences/{job_id}` route added, returns 204 on success, 404 if not found
- [ ] Both service functions accept `Settings` for `storage_root` access, matching the call_parsing delete pattern

**Tests needed:**
- Service-level tests for both delete functions: successful delete (DB row removed), delete of nonexistent job returns False, disk directory removal
- API integration tests: DELETE returns 204, DELETE of missing job returns 404

---

### Task 2: Frontend API client and query hooks for delete

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`

**Acceptance criteria:**
- [ ] `deleteContinuousEmbeddingJob(jobId)` function sends DELETE to `/sequence-models/continuous-embeddings/{jobId}`
- [ ] `deleteHMMSequenceJob(jobId)` function sends DELETE to `/sequence-models/hmm-sequences/{jobId}`
- [ ] `useDeleteContinuousEmbeddingJob()` mutation hook invalidates `continuous-embedding-jobs` query key on success
- [ ] `useDeleteHMMSequenceJob()` mutation hook invalidates `hmm-sequence-jobs` query key on success

**Tests needed:**
- TypeScript type-check passes (verified via `npx tsc --noEmit`)

---

### Task 3: ContinuousEmbeddingJobTable component

**Files:**
- Create: `frontend/src/components/sequence-models/ContinuousEmbeddingJobTable.tsx`

**Acceptance criteria:**
- [ ] `ContinuousEmbeddingJobTable` component with `mode: "active" | "previous"` prop
- [ ] Active mode: table with columns Status, Created, Region Job (short ID), Model Version, Spans, Windows, Actions (Cancel button with X icon); no checkboxes, no sort/filter/pagination
- [ ] Previous mode: table with checkbox column, sortable headers (Status, Created, Region Job, Spans, Windows), filter input, pagination (20/page), bulk Delete button in toolbar with `BulkDeleteDialog`
- [ ] Per-row actions in previous mode: Review link to `/app/sequence-models/continuous-embedding/{jobId}`, Delete button (red text ghost variant)
- [ ] `ContinuousEmbeddingJobTablePanel` wrapper with bordered panel, title, Badge count; hides when empty in active mode
- [ ] Uses existing `StatusBadge`, `BulkDeleteDialog`, `Badge`, `Checkbox` shared components

**Tests needed:**
- TypeScript type-check passes

---

### Task 4: HMMSequenceJobTable component

**Files:**
- Create: `frontend/src/components/sequence-models/HMMSequenceJobTable.tsx`

**Acceptance criteria:**
- [ ] `HMMSequenceJobTable` component with `mode: "active" | "previous"` prop
- [ ] Active mode: columns Status, Created, Source CE Job (short ID), States, PCA Dims, Actions (Cancel with X icon)
- [ ] Previous mode: checkbox, sortable headers (Status, Created, Source CE Job, States, Train Seqs, Log Likelihood), filter, pagination, bulk delete
- [ ] Per-row actions in previous mode: Review link to `/app/sequence-models/hmm-sequence/{jobId}`, Delete button (red text)
- [ ] `HMMSequenceJobTablePanel` wrapper matching the Continuous Embedding panel pattern
- [ ] Uses existing shared components

**Tests needed:**
- TypeScript type-check passes

---

### Task 5: Update page components and remove card components

**Files:**
- Modify: `frontend/src/components/sequence-models/ContinuousEmbeddingJobsPage.tsx`
- Modify: `frontend/src/components/sequence-models/HMMSequenceJobsPage.tsx`
- Delete: `frontend/src/components/sequence-models/ContinuousEmbeddingJobCard.tsx`
- Delete: `frontend/src/components/sequence-models/HMMSequenceJobCard.tsx`

**Acceptance criteria:**
- [ ] `ContinuousEmbeddingJobsPage` uses `ContinuousEmbeddingJobTablePanel` for Active and Previous sections (replaces card grid)
- [ ] `HMMSequenceJobsPage` uses `HMMSequenceJobTablePanel` for Active and Previous sections
- [ ] Create forms remain at top of each page
- [ ] Card components deleted; no remaining imports of deleted files
- [ ] Active panel hides when no active jobs

**Tests needed:**
- TypeScript type-check passes
- No broken imports (build succeeds)

---

### Task 6: Fix breadcrumbs

**Files:**
- Modify: `frontend/src/components/layout/Breadcrumbs.tsx`

**Acceptance criteria:**
- [ ] Static routes added for `/app/sequence-models/continuous-embedding` → Sequence Models > Continuous Embedding
- [ ] Static routes added for `/app/sequence-models/hmm-sequence` → Sequence Models > HMM Sequence
- [ ] Dynamic breadcrumb for detail pages: `/app/sequence-models/continuous-embedding/:id` → Sequence Models > Continuous Embedding > Job {short_id}
- [ ] Dynamic breadcrumb for detail pages: `/app/sequence-models/hmm-sequence/:id` → Sequence Models > HMM Sequence > Job {short_id}
- [ ] No more fallback to "Audio" on any Sequence Models route

**Tests needed:**
- TypeScript type-check passes
- Manual verification in browser that breadcrumbs render correctly on all four route types

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/services/continuous_embedding_service.py src/humpback/services/hmm_sequence_service.py src/humpback/api/routers/sequence_models.py`
2. `uv run ruff check src/humpback/services/continuous_embedding_service.py src/humpback/services/hmm_sequence_service.py src/humpback/api/routers/sequence_models.py`
3. `uv run pyright src/humpback/services/continuous_embedding_service.py src/humpback/services/hmm_sequence_service.py src/humpback/api/routers/sequence_models.py`
4. `uv run pytest tests/services/test_continuous_embedding_service.py tests/services/test_hmm_sequence_service.py tests/integration/test_sequence_models_api.py`
5. `cd frontend && npx tsc --noEmit`
6. `uv run pytest tests/`
