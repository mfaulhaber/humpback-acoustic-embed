# Training Data Source Type Filter Implementation Plan

**Goal:** Add source type filtering (detection vs embedding) to the Training Data Review page.
**Spec:** [docs/specs/2026-04-05-training-data-source-filter-design.md](../specs/2026-04-05-training-data-source-filter-design.md)

---

### Task 1: Add `source_type` query parameter to backend rows endpoint

**Files:**
- Modify: `src/humpback/api/routers/vocalization.py`

**Acceptance criteria:**
- [ ] `GET /vocalization/training-datasets/{dataset_id}/rows` accepts optional `source_type` query param
- [ ] When `source_type` is provided, only rows matching that source type are returned
- [ ] Filter is applied before type/group label filtering and before pagination
- [ ] Total count reflects the filtered result set
- [ ] Omitting the param returns all rows (existing behavior unchanged)

**Tests needed:**
- Test rows endpoint with `source_type=detection_job` returns only detection rows
- Test rows endpoint with `source_type=embedding_set` returns only embedding rows
- Test rows endpoint without `source_type` returns all rows (regression)
- Test composition: `source_type` + `type` + `group` filters work together

---

### Task 2: Wire `source_type` through frontend API client and query hook

**Files:**
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useVocalization.ts`

**Acceptance criteria:**
- [ ] `fetchTrainingDatasetRows` accepts optional `source_type` parameter
- [ ] `useTrainingDatasetRows` passes `source_type` through to the fetch function and query key
- [ ] Omitting `source_type` produces the same request as before (no breaking change)

**Tests needed:**
- TypeScript compilation verifies type correctness

---

### Task 3: Add source type filter UI to TrainingDataView

**Files:**
- Modify: `frontend/src/components/vocalization/TrainingDataView.tsx`

**Acceptance criteria:**
- [ ] Segmented button group (All | Detection | Embedding) appears in the filter bar, before the type filter buttons
- [ ] Styled consistently with the existing positive/negative toggle
- [ ] `sourceType` state defaults to `null` (All)
- [ ] Selecting a source type resets page to 0
- [ ] `sourceType` is passed to `useTrainingDatasetRows` as the `source_type` param
- [ ] Source filter composes with type and group filters

**Tests needed:**
- TypeScript compilation verifies component correctness

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/api/routers/vocalization.py`
2. `uv run ruff check src/humpback/api/routers/vocalization.py`
3. `uv run pyright src/humpback/api/routers/vocalization.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
