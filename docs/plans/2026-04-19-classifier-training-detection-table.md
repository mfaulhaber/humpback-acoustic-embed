# Classifier Training Detection Job Table — Implementation Plan

**Goal:** Add label counts, embedding status indicators, and an Embed button to the detection job picker table on the Classifier/Training page.
**Spec:** [docs/specs/2026-04-19-classifier-training-detection-table-design.md](../specs/2026-04-19-classifier-training-detection-table-design.md)

---

### Task 1: Backend — Label Counts Endpoint

**Files:**
- Modify: `src/humpback/schemas/classifier.py`
- Modify: `src/humpback/api/routers/classifier/training.py`

**Acceptance criteria:**
- [ ] New Pydantic schema `DetectionJobLabelCount` with fields `detection_job_id: str`, `positive: int`, `negative: int`
- [ ] New `GET /classifier/detection-jobs/label-counts` endpoint accepting `detection_job_ids` as repeated query params
- [ ] Endpoint reads each job's row store via `read_detection_row_store` and counts positive (humpback=1 or orca=1) and negative (ship=1 or background=1) rows
- [ ] Returns 200 with list of `DetectionJobLabelCount` for valid jobs; missing row stores return 0/0 counts (not an error)

**Tests needed:**
- Unit test: endpoint returns correct counts for detection jobs with known row store contents
- Unit test: endpoint handles missing row store gracefully (returns 0/0)
- Unit test: endpoint handles empty detection_job_ids list

---

### Task 2: Backend Test — Label Counts Endpoint

**Files:**
- Modify: `tests/integration/test_classifier_api.py`

**Acceptance criteria:**
- [ ] Integration test that creates a detection job with a row store, calls the label counts endpoint, and verifies positive/negative counts
- [ ] Test that missing row store returns 0/0
- [ ] Test that empty ID list returns empty list

---

### Task 3: Frontend — API Types, Client, and Hook for Label Counts

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useClassifier.ts`

**Acceptance criteria:**
- [ ] New `DetectionJobLabelCount` TypeScript interface with `detection_job_id`, `positive`, `negative`
- [ ] New `fetchDetectionJobLabelCounts(detectionJobIds: string[])` client function
- [ ] New `useDetectionJobLabelCounts(detectionJobIds: string[])` TanStack Query hook that fetches counts for all provided IDs
- [ ] Hook is enabled only when `detectionJobIds.length > 0`

---

### Task 4: Frontend — Upgrade DetectionSourcePicker to Table

**Files:**
- Modify: `frontend/src/components/classifier/DetectionSourcePicker.tsx`
- Delete: `frontend/src/components/classifier/ReembeddingStatusTable.tsx`

**Acceptance criteria:**
- [ ] Checkbox list replaced with a table: columns are Checkbox, Detection Job, Pos, Neg, Embedding, Action
- [ ] Label counts (Pos/Neg) always visible for all listed jobs; Pos in green, Neg in muted gray
- [ ] Embedding column shows dash ("—") when no model is selected
- [ ] Embedding column is blank when embeddings exist (status `complete`) for the selected model
- [ ] Embedding column shows "Missing" indicator when status is `not_started`
- [ ] Embedding column shows status badge (Queued/Running with progress %) when queued or running
- [ ] Embedding column shows "Failed" badge with error popover on failure
- [ ] Action column shows "Embed" button only when status is `not_started`
- [ ] Action column shows "Retry" button only when status is `failed`
- [ ] Action column is empty for complete/queued/running states
- [ ] Table is inside a scrollable container consistent with current `max-h-40` style
- [ ] `ReembeddingStatusTable.tsx` is deleted; no remaining imports reference it
- [ ] `isReady` logic unchanged — all selected jobs must have complete embeddings before training is enabled
- [ ] All existing embedding status query and polling behavior (useReembeddingStatus, 2s polling) preserved

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/api/routers/classifier/training.py src/humpback/schemas/classifier.py`
2. `uv run ruff check src/humpback/api/routers/classifier/training.py src/humpback/schemas/classifier.py`
3. `uv run pyright src/humpback/api/routers/classifier/training.py src/humpback/schemas/classifier.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
