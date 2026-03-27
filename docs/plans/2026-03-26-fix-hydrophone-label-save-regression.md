# Hydrophone Label Save Regression Implementation Plan

**Goal:** Restore persisted checkbox label saves for hydrophone detection jobs when the UI submits job-relative detection offsets.
**Spec:** Bug-fix workflow; no standalone design spec per `AGENTS.md`.

---

### Task 1: Translate hydrophone UI offsets during label save

**Files:**
- Modify: `src/humpback/api/routers/classifier.py`

**Acceptance criteria:**
- [ ] `PUT /classifier/detection-jobs/{job_id}/labels` still matches local detection rows by their stored offsets
- [ ] Hydrophone label saves accept the job-relative `start_sec` and `end_sec` values returned by `GET /content`
- [ ] Hydrophone saves update the intended row instead of silently leaving the row store unchanged
- [ ] Existing `has_positive_labels` behavior remains correct after the translation

**Tests needed:**
- Cover the hydrophone save path with a round-trip that loads row content, sends those returned offsets back to the save endpoint, and verifies the label persists

---

### Task 2: Add regression coverage for the persisted row state

**Files:**
- Modify: `tests/integration/test_hydrophone_api.py`

**Acceptance criteria:**
- [ ] Regression coverage reproduces the mismatch between stored file-relative offsets and UI job-relative offsets
- [ ] The test verifies a saved hydrophone label is visible through a subsequent `GET /content`
- [ ] The test would fail on the pre-fix behavior

**Tests needed:**
- Add an integration test for hydrophone `GET /content` plus `PUT /labels` round-trip persistence

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/api/routers/classifier.py tests/integration/test_hydrophone_api.py`
2. `uv run ruff check src/humpback/api/routers/classifier.py tests/integration/test_hydrophone_api.py`
3. `uv run pyright src/humpback/api/routers/classifier.py tests/integration/test_hydrophone_api.py`
4. `uv run pytest tests/integration/test_hydrophone_api.py`
