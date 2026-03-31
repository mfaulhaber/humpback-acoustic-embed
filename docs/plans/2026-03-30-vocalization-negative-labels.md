# Vocalization Negative Labels Implementation Plan

**Goal:** Add explicit "(Negative)" labeling to the vocalization system, fix training to only use labeled windows, and fix model detail sample count display.
**Spec:** docs/specs/2026-03-30-vocalization-negative-labels-design.md

---

### Task 1: Fix training data assembly to skip unlabeled windows

**Files:**
- Modify: `src/humpback/workers/vocalization_worker.py`

**Acceptance criteria:**
- [ ] Detection job loop skips embedding rows with no entry in `labels_by_utc`
- [ ] Rows with "(Negative)" label have their label set converted to empty set `{}`
- [ ] Rows with type labels pass through unchanged
- [ ] Embedding set source (folder-based) is unaffected

**Tests needed:**
- Unit test: training assembly with mix of labeled, unlabeled, and "(Negative)" windows — verify only labeled rows are included and "(Negative)" becomes empty set

---

### Task 2: Add mutual exclusivity enforcement in label save API

**Files:**
- Modify: `src/humpback/api/routers/labeling.py`

**Acceptance criteria:**
- [ ] When saving label="(Negative)", existing type labels on the same (detection_job_id, start_utc, end_utc) are deleted
- [ ] When saving a type label, existing "(Negative)" labels on the same window are deleted
- [ ] Deletions happen within the same transaction as the new label insert

**Tests needed:**
- Integration test: save "(Negative)" on window with existing type label, verify type label deleted
- Integration test: save type label on window with existing "(Negative)", verify "(Negative)" deleted

---

### Task 3: Add vocabulary guard against "(Negative)" type name

**Files:**
- Modify: `src/humpback/api/routers/vocalization.py`

**Acceptance criteria:**
- [ ] `POST /vocalization/types` rejects name "(Negative)" (case-insensitive) with 400 status
- [ ] `PUT /vocalization/types/{id}` rejects rename to "(Negative)" with 400 status

**Tests needed:**
- Integration test: attempt to create type named "(Negative)", verify 400 response

---

### Task 4: Add "(Negative)" option to labeling UI with mutual exclusivity

**Files:**
- Modify: `frontend/src/components/vocalization/LabelingWorkspace.tsx`

**Acceptance criteria:**
- [ ] "(Negative)" appears at the bottom of the add-label popover, separated by a divider
- [ ] "(Negative)" badge uses red styling (`bg-red-100 text-red-800 border-red-200`)
- [ ] Adding "(Negative)" as pending clears any pending type adds and marks existing type labels for removal
- [ ] Adding a type label as pending clears any pending "(Negative)" add and marks existing "(Negative)" for removal
- [ ] "(Negative)" is excluded from `availableTypes` when already present (saved or pending)

**Tests needed:**
- Manual verification of label interaction states in the labeling workspace

---

### Task 5: Fix model detail sample count and add Negatives column

**Files:**
- Modify: `frontend/src/components/vocalization/VocalizationModelList.tsx`

**Acceptance criteria:**
- [ ] Samples column displays `n_positive + n_negative` (was `m.n_samples`)
- [ ] New "Negatives" column displays `n_negative` per type
- [ ] Column header order: Type, AP, F1, Precision, Recall, Samples, Negatives, Threshold

**Tests needed:**
- Manual verification that model detail shows correct counts after training

---

### Task 6: Tests for training assembly and API changes

**Files:**
- Modify: `tests/unit/test_vocalization_trainer.py`
- Modify: `tests/integration/test_vocalization_api.py`
- Modify: `tests/integration/test_labeling_api.py`

**Acceptance criteria:**
- [ ] Test: unlabeled windows excluded from training data assembly
- [ ] Test: "(Negative)" windows included as empty-set negatives
- [ ] Test: mutual exclusivity on label save
- [ ] Test: vocabulary guard rejects "(Negative)" type name
- [ ] All existing tests still pass

**Tests needed:**
- See acceptance criteria above

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/workers/vocalization_worker.py src/humpback/api/routers/labeling.py src/humpback/api/routers/vocalization.py`
2. `uv run ruff check src/humpback/workers/vocalization_worker.py src/humpback/api/routers/labeling.py src/humpback/api/routers/vocalization.py`
3. `uv run pyright src/humpback/workers/vocalization_worker.py src/humpback/api/routers/labeling.py src/humpback/api/routers/vocalization.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
