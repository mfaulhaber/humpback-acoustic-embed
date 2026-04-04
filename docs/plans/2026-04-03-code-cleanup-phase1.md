# Code Cleanup Phase 1: Dead Code Removal — Implementation Plan

**Goal:** Remove unused code and relocate orphaned test files to their correct locations.
**Spec:** [docs/specs/2026-04-03-code-cleanup-design.md](../specs/2026-04-03-code-cleanup-design.md)

---

### Task 1: Delete unused frontend components

**Files:**
- Delete: `frontend/src/components/vocalization/VocalizationResultsBrowser.tsx`
- Delete: `frontend/src/components/vocalization/VocalizationInferenceForm.tsx`
- Modify: `docs/reference/frontend.md` (remove these components from the component tree listing)

**Acceptance criteria:**
- [ ] Both files deleted
- [ ] No remaining imports, lazy-load references, or re-exports reference either component
- [ ] `docs/reference/frontend.md` component tree updated to remove both entries
- [ ] `cd frontend && npx tsc --noEmit` passes

**Tests needed:**
- No test changes — these components had no test coverage (they were unused)

---

### Task 2: Relocate orphaned search service tests

**Files:**
- Delete: `tests/test_search_service.py`
- Modify: `tests/unit/test_search_service.py` (append the 4 unique test classes)

**Acceptance criteria:**
- [ ] All 20 test functions from the root-level file are present in `tests/unit/test_search_service.py`
  - `TestScoreDistribution` (6 tests): basic_stats, percentiles, histogram_bins, histogram_bin_types, empty_scores, single_score
  - `TestPercentileRank` (5 tests): highest_score, lowest_score, middle_score, empty_scores, monotonic_ordering
  - `TestBruteForceSearch` projector tests (6 tests): no_projector, with_identity_projector, projector_transforms_vectors, empty_candidates, distribution_in_results, percentile_ranks_in_hits
  - `TestProjectedSearchModeRejection` (3 tests): projected_mode_rejected_similar, projected_mode_rejected_vector, invalid_search_mode_rejected
- [ ] Any imports needed by the moved tests are added to the unit file
- [ ] Root-level `tests/test_search_service.py` deleted
- [ ] `uv run pytest tests/unit/test_search_service.py` passes

**Tests needed:**
- No new tests — relocating existing tests. Verify all 20 moved tests pass in their new location.

---

### Task 3: Relocate orphaned clustering service tests

**Files:**
- Delete: `tests/test_clustering_service.py`
- Create: `tests/unit/test_clustering_service.py` (no existing file with this name)

**Acceptance criteria:**
- [ ] All 3 test functions relocated to `tests/unit/test_clustering_service.py`
  - `test_clustering_rejects_mixed_model_versions`
  - `test_clustering_accepts_same_model_version`
  - `test_clustering_rejects_mixed_vector_dims`
- [ ] Any imports and fixtures adjusted for the `tests/unit/` location
- [ ] Root-level `tests/test_clustering_service.py` deleted
- [ ] `uv run pytest tests/unit/test_clustering_service.py` passes

**Tests needed:**
- No new tests — relocating existing tests. Verify all 3 moved tests pass in their new location.

---

### Task 4: Delete POC script and update docs

**Files:**
- Delete: `scripts/noaa_gcs_poc.py`
- Modify: `scripts/README.md` (remove the noaa_gcs_poc.py usage section, lines ~117-127)

**Acceptance criteria:**
- [ ] `scripts/noaa_gcs_poc.py` deleted
- [ ] `scripts/README.md` no longer references the POC script
- [ ] No other files in the repo reference `noaa_gcs_poc`

**Tests needed:**
- No test changes — script had no tests

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check tests/unit/test_search_service.py tests/unit/test_clustering_service.py`
2. `uv run ruff check tests/unit/test_search_service.py tests/unit/test_clustering_service.py`
3. `uv run pyright tests/unit/test_search_service.py tests/unit/test_clustering_service.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
