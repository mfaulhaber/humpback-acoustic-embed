# Corrections-Only Training Mode — Implementation Plan

**Goal:** Add a `corrections_only` config flag so event classifier feedback training can use only human-corrected labels, avoiding contamination from poor bootstrap pseudo-labels
**Spec:** [docs/specs/2026-04-15-corrections-only-training-design.md](../specs/2026-04-15-corrections-only-training-design.md)

---

### Task 1: Add `corrections_only` field to backend config

**Files:**
- Modify: `src/humpback/schemas/call_parsing.py`
- Modify: `src/humpback/call_parsing/event_classifier/trainer.py`

**Acceptance criteria:**
- [ ] `EventClassifierTrainingConfig` Pydantic model in `schemas/call_parsing.py` gains `corrections_only: bool = True`
- [ ] `EventClassifierTrainingConfig` dataclass in `trainer.py` gains `corrections_only: bool = True`
- [ ] Both default to `True`

**Tests needed:**
- Unit test: verify the Pydantic schema accepts `corrections_only` and defaults to `True`
- Unit test: verify the dataclass defaults to `True`

---

### Task 2: Wire `corrections_only` through the feedback worker

**Files:**
- Modify: `src/humpback/workers/event_classifier_feedback_worker.py`

**Acceptance criteria:**
- [ ] `_resolve_event_labels` accepts a `corrections_only: bool` parameter
- [ ] When `corrections_only=True`, uncorrected events get `labels[event_id] = None` (skip inference fallback)
- [ ] When `corrections_only=False`, existing behavior is preserved (inference fallback for uncorrected events)
- [ ] `_collect_samples` accepts and forwards `corrections_only` from the config
- [ ] `run_event_classifier_feedback_training` passes `config.corrections_only` through to `_collect_samples`

**Tests needed:**
- Unit test: `_resolve_event_labels` with `corrections_only=True` returns only corrected events
- Unit test: `_resolve_event_labels` with `corrections_only=False` includes inference labels

---

### Task 3: Add checkbox to ClassifyReviewWorkspace retrain

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] A "Corrected events only" checkbox appears near the Retrain button, default checked
- [ ] `handleRetrain` passes `{ corrections_only: correctionsOnly }` in the `config` field of the mutation payload
- [ ] Unchecking the box sends `corrections_only: false`

**Tests needed:**
- Playwright test: verify checkbox renders and is checked by default

---

### Task 4: Add checkbox to ClassificationJobPicker

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassificationJobPicker.tsx`

**Acceptance criteria:**
- [ ] A "Corrected events only" checkbox appears near the Train Model button, default checked
- [ ] `handleTrain` passes `{ corrections_only: correctionsOnly }` in the `config` field of the mutation payload

**Tests needed:**
- Playwright test: verify checkbox renders and is checked by default

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/schemas/call_parsing.py src/humpback/call_parsing/event_classifier/trainer.py src/humpback/workers/event_classifier_feedback_worker.py`
2. `uv run ruff check src/humpback/schemas/call_parsing.py src/humpback/call_parsing/event_classifier/trainer.py src/humpback/workers/event_classifier_feedback_worker.py`
3. `uv run pyright src/humpback/schemas/call_parsing.py src/humpback/call_parsing/event_classifier/trainer.py src/humpback/workers/event_classifier_feedback_worker.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
