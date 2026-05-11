# Delete Confirmation Dialogs Implementation Plan

**Goal:** Add shared delete confirmation dialogs and delete button styling across Call Parsing and Sequence Models, including Segment Training dataset deletion.
**Spec:** `docs/specs/2026-05-11-delete-confirmation-dialogs-design.md`
**Primary domain:** call-parsing
**Neighbor domains:** sequence-models, frontend-shell

---

### Task 1: Shared Delete Confirmation UI

**Files:**
- Create: `frontend/src/components/shared/DeleteConfirmationDialog.tsx`
- Create: `frontend/src/components/shared/DeleteConfirmationDialog.test.tsx`
- Modify: `frontend/src/components/ui/button.tsx`

**Acceptance criteria:**
- [ ] A reusable dialog supports single-resource and multi-resource delete confirmation copy.
- [ ] Dialog title follows `Delete [resource type]`.
- [ ] Dialog body identifies the selected resource or count and explains the consequence.
- [ ] Dialog footer uses `Cancel` and `Delete`; `Cancel` closes without side effects.
- [ ] Delete trigger and confirm buttons can use one shared rounded red background with white text style.
- [ ] The shared component has focused tests for rendering, cancel, and confirm behavior.

**Tests needed:**
- Add a Vitest component test for the shared dialog and delete button style.

---

### Task 2: Call Parsing Delete Button Dialogs

**Files:**
- Modify: `frontend/src/components/call-parsing/RegionJobTable.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentJobTable.tsx`
- Modify: `frontend/src/components/call-parsing/ClassifyJobTable.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentModelTable.tsx`
- Modify: `frontend/src/components/call-parsing/ClassifyModelTable.tsx`
- Modify: `frontend/src/components/call-parsing/ClassifyTrainingJobTable.tsx`
- Modify: `frontend/src/components/call-parsing/WindowClassifyJobTable.tsx`
- Modify: `frontend/src/components/call-parsing/EventDetailPanel.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`
- Modify: `frontend/src/components/call-parsing/WindowClassifyReviewWorkspace.tsx`
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.test.tsx`
- Modify: `frontend/src/components/call-parsing/WindowClassifyReviewWorkspace.test.tsx`

**Acceptance criteria:**
- [ ] Call Parsing bulk delete flows use the shared confirmation dialog instead of the classifier-specific bulk dialog.
- [ ] Call Parsing single-row Delete buttons open the shared dialog before mutations run.
- [ ] Model and training-job delete flows no longer use native browser `confirm()`.
- [ ] Review Delete Event buttons open a confirmation dialog before marking a pending delete correction.
- [ ] Existing error toasts, query invalidations, selection clearing, and disabled states are preserved.
- [ ] All in-scope Delete buttons use the shared delete button style.

**Tests needed:**
- Update or add focused frontend tests for delete confirmation behavior in Call Parsing tables and review workspaces.

---

### Task 3: Segment Training Dataset Deletion

**Files:**
- Modify: `src/humpback/services/call_parsing.py`
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useCallParsing.ts`
- Modify: `frontend/src/components/call-parsing/TrainingDatasetTable.tsx`
- Modify: `tests/integration/test_call_parsing_router.py`
- Modify: `tests/unit/test_call_parsing_service.py`

**Acceptance criteria:**
- [ ] `DELETE /call-parsing/segmentation-training-datasets/{dataset_id}` returns `204` after deleting an existing dataset.
- [ ] The delete service removes the dataset row and associated sample rows.
- [ ] The delete route returns `404` for a missing dataset.
- [ ] The delete route returns `409` when a queued or running segmentation training job references the dataset.
- [ ] Completed/failed training-job rows and produced models are not deleted as part of dataset deletion.
- [ ] The frontend exposes a delete client function and mutation hook that invalidates the segmentation training dataset query.
- [ ] Segment Training dataset rows include a Delete button using the shared style and confirmation dialog.

**Tests needed:**
- Add backend service and router tests for dataset delete success, not-found, sample cleanup, and in-flight job conflict.
- Add or update frontend component coverage for the Training Dataset row Delete flow.

---

### Task 4: Sequence Models Delete Button Dialogs

**Files:**
- Modify: `frontend/src/components/sequence-models/ContinuousEmbeddingJobTable.tsx`
- Modify: `frontend/src/components/sequence-models/EventEncoderJobTable.tsx`
- Create: `frontend/src/components/sequence-models/ContinuousEmbeddingJobTable.test.tsx`
- Create: `frontend/src/components/sequence-models/EventEncoderJobTable.test.tsx`

**Acceptance criteria:**
- [ ] Continuous Embedding bulk delete uses the shared confirmation dialog.
- [ ] Continuous Embedding row Delete opens the shared dialog before the mutation runs.
- [ ] Event Encoder row Delete opens the shared dialog before the mutation runs.
- [ ] Review/report links and active job Cancel behavior remain unchanged.
- [ ] All in-scope Delete buttons use the shared delete button style.

**Tests needed:**
- Add focused frontend component tests for Continuous Embedding and Event Encoder delete confirmation flows.

---

### Task 5: Verification And Cleanup

**Files:**
- Modify: `docs/plans/2026-05-11-delete-confirmation-dialogs.md`

**Acceptance criteria:**
- [ ] Plan checkboxes are updated as tasks complete.
- [ ] No unrelated delete behavior outside Call Parsing and Sequence Models is changed.
- [ ] No new native browser `confirm()` calls are introduced for delete actions.
- [ ] Documentation remains limited to the design and implementation plan unless implementation reveals a domain-local rule that belongs in a capsule.

**Tests needed:**
- Run targeted backend, frontend component, and TypeScript checks before the full gates.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py tests/integration/test_call_parsing_router.py tests/unit/test_call_parsing_service.py`
2. `uv run ruff check src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py tests/integration/test_call_parsing_router.py tests/unit/test_call_parsing_service.py`
3. `uv run pyright src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py`
4. `uv run pytest tests/unit/test_call_parsing_service.py tests/integration/test_call_parsing_router.py -q`
5. `cd frontend && npx vitest run src/components/shared/DeleteConfirmationDialog.test.tsx src/components/call-parsing/SegmentReviewWorkspace.test.tsx src/components/call-parsing/WindowClassifyReviewWorkspace.test.tsx src/components/sequence-models/ContinuousEmbeddingJobTable.test.tsx src/components/sequence-models/EventEncoderJobTable.test.tsx`
6. `cd frontend && npx tsc --noEmit`
7. `uv run pytest tests/`
