# Segment Training Page Rework Implementation Plan

**Goal:** Make Call Parsing Segment Training match the Classifier Training page flow by showing models first, training directly from selected segmentation jobs, and surfacing queued/running training jobs.
**Spec:** `docs/specs/2026-05-15-segment-training-page-rework-design.md`
**Primary domain:** call-parsing
**Neighbor domains:** frontend-shell, core-platform

---

### Task 1: Extend Segmentation Training API Contract

**Files:**
- Modify: `src/humpback/schemas/call_parsing.py`
- Modify: `src/humpback/services/call_parsing.py`
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `tests/integration/test_dataset_from_corrections.py`

**Acceptance criteria:**
- [ ] `SegmentationTrainingConfig` supports nested feature extraction defaults while preserving existing flat config JSON compatibility.
- [ ] `CreateSegmentationTrainingJobRequest` accepts exactly one source mode: existing `training_dataset_id` or new `segmentation_job_ids`.
- [ ] `POST /call-parsing/segmentation-training-jobs` can create a dataset internally from selected segmentation jobs and queue a training job.
- [ ] `GET /call-parsing/segmentation-training-jobs` lists training jobs newest first.
- [ ] Existing dataset-based job creation still works.
- [ ] Invalid source-mode combinations return validation errors before service work.

**Tests needed:**
- Router coverage for direct job creation, dataset-based compatibility, list jobs, invalid source modes, and advanced config override serialization.

---

### Task 2: Use Feature Config Overrides in the Worker

**Files:**
- Modify: `src/humpback/workers/segmentation_training_worker.py`
- Modify: `tests/integration/test_dataset_from_corrections.py` or matching worker test if one exists

**Acceptance criteria:**
- [ ] The worker reads `training_config.feature_config` instead of reconstructing feature config from only `n_mels`.
- [ ] Model checkpoint/config payloads preserve the effective feature config used during training.
- [ ] Old training job config JSON remains loadable.
- [ ] Mismatched model/feature `n_mels` is rejected by schema validation.

**Tests needed:**
- Schema or worker-level coverage proving nested feature config is accepted and legacy config JSON is normalized.

---

### Task 3: Rework Segment Training Frontend

**Files:**
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/hooks/queries/useCallParsing.ts`
- Modify: `frontend/src/components/call-parsing/SegmentTrainingPage.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentModelTable.tsx`
- Create: `frontend/src/components/call-parsing/SegmentTrainingForm.tsx`
- Create: `frontend/src/components/call-parsing/SegmentTrainingJobTable.tsx`
- Delete or retire from route usage: `frontend/src/components/call-parsing/SegmentationJobPicker.tsx`
- Delete or retire from route usage: `frontend/src/components/call-parsing/TrainingDatasetTable.tsx`
- Modify: `frontend/e2e/call-parsing-segment.spec.ts`

**Acceptance criteria:**
- [ ] Segment Training page renders models first, the direct training form second, and previous jobs after that.
- [ ] Training form selects completed segmentation jobs and posts `segmentation_job_ids` directly to the training jobs endpoint.
- [ ] Training-dataset creation and management UI is no longer visible on the page.
- [ ] Advanced options expose feature, optimizer, split, and architecture defaults.
- [ ] Previous jobs panel displays queued, running, complete, and failed jobs with appropriate status badges.
- [ ] Query invalidation/polling makes newly queued jobs appear immediately and completed models refresh.

**Tests needed:**
- Update Playwright coverage for the new layout, direct training submit, advanced options, and previous jobs status.
- Run frontend TypeScript after implementation.

---

### Task 4: Documentation And Context Updates

**Files:**
- Modify: `docs/agent-context/domains/call-parsing/README.md`
- Modify: `docs/agent-context/domains/call-parsing/invariants.md` if the hidden dataset contract needs an explicit note
- Modify: `docs/agent-context/domains/call-parsing/tests.md` if targeted frontend tests change names or scope

**Acceptance criteria:**
- [ ] Domain-local context records that Segment Training now trains directly from selected segmentation jobs while persisting datasets internally.
- [ ] Existing correction and immutable artifact invariants remain intact.

**Tests needed:**
- Documentation-only changes do not require targeted tests beyond the task-level code verification already listed.

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/workers/segmentation_training_worker.py tests/integration/test_dataset_from_corrections.py`
2. `uv run ruff check src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/workers/segmentation_training_worker.py tests/integration/test_dataset_from_corrections.py`
3. `uv run pyright src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/workers/segmentation_training_worker.py`
4. `uv run pytest tests/integration/test_dataset_from_corrections.py tests/integration/test_call_parsing_router.py -q`
5. `uv run pytest tests/`
6. `cd frontend && npx tsc --noEmit`
7. `cd frontend && npx playwright test e2e/call-parsing-segment.spec.ts`
