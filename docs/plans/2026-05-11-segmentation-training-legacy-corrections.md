# Segmentation Training Legacy Corrections Implementation Plan

**Goal:** Include unambiguous legacy human boundary corrections when creating Segment Training datasets, and report actual contributing source jobs clearly.
**Spec:** `docs/specs/2026-05-11-segmentation-training-legacy-corrections-design.md`
**Primary domain:** call-parsing
**Neighbor domains:** frontend-shell, core-platform

---

### Task 1: Correction Resolution For Dataset Extraction

**Files:**
- Modify: `src/humpback/call_parsing/segmentation/extraction.py`

**Acceptance criteria:**
- [x] Dataset extraction uses segmentation-scoped boundary corrections when rows exist for the selected `event_segmentation_job_id`.
- [x] When no scoped rows exist, extraction can use legacy region-scoped rows where `event_segmentation_job_id IS NULL` only if the region detection job has exactly one segmentation job.
- [x] When legacy rows are ambiguous because multiple segmentation jobs share the region detection job, extraction returns no samples for that selected job and surfaces a skipped reason to the caller.
- [x] When both scoped and legacy rows exist for a region detection job, scoped rows are authoritative and legacy rows are not double-applied.
- [x] Existing crop subdivision, corrected event application, hydrophone-only validation, and missing-artifact behavior are preserved.
- [x] No correction rows, segmentation artifacts, or region artifacts are mutated.

**Tests needed:**
- Covered by the integration cases in Task 4, with direct assertions on source refs, skip counts, and modern-scoped precedence.

---

### Task 2: Dataset Creation Accounting And API Response

**Files:**
- Modify: `src/humpback/services/call_parsing.py`
- Modify: `src/humpback/schemas/call_parsing.py`
- Modify: `src/humpback/api/routers/call_parsing.py`

**Acceptance criteria:**
- [x] `create_dataset_from_corrections()` tracks selected job count, contributing source job count, skipped job count, and compact skipped-job details.
- [x] The public create response includes `sample_count`, `selected_job_count`, `source_job_count`, `skipped_job_count`, and `skipped_jobs`.
- [x] `source_job_count` reflects jobs that actually contributed saved samples, not merely selected jobs.
- [x] Jobs with no usable corrections are reported as skipped when at least one other selected job contributes samples.
- [x] The endpoint still returns 400 when no samples are collected across all selected jobs.
- [x] Existing dataset rows and sample rows keep the same persistence shape; `SegmentationTrainingSample.source_ref` remains the selected segmentation job id.
- [x] Existing list response semantics remain unchanged: dataset summaries derive source jobs from distinct saved sample source refs.

**Tests needed:**
- Add or update integration tests for the expanded create response shape.
- Verify existing list-dataset tests still pass and continue to report actual saved sample provenance.

---

### Task 3: Frontend Types And Segment Training Toast Copy

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/components/call-parsing/SegmentationJobPicker.tsx`
- Modify: `frontend/e2e/call-parsing-segment.spec.ts`

**Acceptance criteria:**
- [x] Frontend create-dataset response type includes `selected_job_count`, `source_job_count`, `skipped_job_count`, and `skipped_jobs`.
- [x] The success toast says how many jobs contributed samples rather than how many jobs were selected.
- [x] If the backend reports skipped jobs, the toast includes concise skipped-job context without implying those jobs contributed samples.
- [x] Existing selection reset, name reset, description reset, and error toast behavior are preserved.
- [x] Segment Training page E2E mocks match the expanded API response shape.

**Tests needed:**
- Update or add a Playwright case for creating a Segment Training dataset and seeing contributing-job copy.
- Run TypeScript after frontend type changes.

---

### Task 4: Backend Regression Tests

**Files:**
- Modify: `tests/integration/test_dataset_from_corrections.py`
- Modify: `tests/integration/test_call_parsing_router.py`

**Acceptance criteria:**
- [x] A mixed modern-plus-legacy dataset test creates samples from both a segmentation-scoped source job and an unambiguous legacy region-scoped source job.
- [x] The mixed test asserts distinct saved `source_ref` values include both selected contributing segmentation job ids.
- [x] An ambiguous legacy test creates two segmentation jobs for one region detection job, adds only legacy region-scoped corrections, and verifies the selected legacy job is skipped with an explanatory response.
- [x] A precedence test verifies scoped corrections win when scoped and legacy rows coexist.
- [x] Router response tests cover the new create-response fields and keep existing list-summary count expectations.

**Tests needed:**
- Run `uv run pytest tests/integration/test_dataset_from_corrections.py tests/integration/test_call_parsing_router.py -q`.

---

### Task 5: Local Production Sanity Check

**Files:**
- Modify: `docs/plans/2026-05-11-segmentation-training-legacy-corrections.md`

**Acceptance criteria:**
- [x] After implementation, use the configured `.env` database read-only to confirm `test-3-datasets` still shows the original 137-sample snapshot before any user deletes it.
- [x] Confirm the three relevant Orcasound jobs resolve as expected: two modern scoped contributors and one unambiguous legacy contributor.
- [x] Do not delete or recreate `test-3-datasets` during implementation; the user can recreate it after the fix lands.
- [x] Update this plan's checkboxes as tasks complete.

**Tests needed:**
- Read-only SQLite queries against the configured `HUMPBACK_DATABASE_URL` are sufficient for this sanity check.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/call_parsing/segmentation/extraction.py src/humpback/services/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/api/routers/call_parsing.py tests/integration/test_dataset_from_corrections.py tests/integration/test_call_parsing_router.py`
2. `uv run ruff check src/humpback/call_parsing/segmentation/extraction.py src/humpback/services/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/api/routers/call_parsing.py tests/integration/test_dataset_from_corrections.py tests/integration/test_call_parsing_router.py`
3. `uv run pyright src/humpback/call_parsing/segmentation/extraction.py src/humpback/services/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/api/routers/call_parsing.py`
4. `uv run pytest tests/integration/test_dataset_from_corrections.py tests/integration/test_call_parsing_router.py -q`
5. `cd frontend && npx playwright test e2e/call-parsing-segment.spec.ts`
6. `cd frontend && npx tsc --noEmit`
7. `uv run pytest tests/`
