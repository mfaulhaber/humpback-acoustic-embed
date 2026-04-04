# Code Cleanup Phase 2: Utility Extraction — Implementation Plan

**Goal:** Extract shared parquet read helper and move schema converters to shared module, reducing duplication and prepping for Phase 3 file splitting.
**Spec:** [docs/specs/2026-04-03-code-cleanup-design.md](../specs/2026-04-03-code-cleanup-design.md)

**Scope revision:** Deep research found that job status helpers already exist in `queue.py` (task 2b skipped). Parquet consolidation is smaller than estimated (~6 sites, not 21). Schema converters are distinct per-model but extracting them reduces router sizes for Phase 3.

---

### Task 1: Add `read_embedding_vectors()` to `processing/embeddings.py`

**Files:**
- Modify: `src/humpback/processing/embeddings.py` (add new helper alongside existing `read_embeddings()`)

**Acceptance criteria:**
- [ ] `read_embedding_vectors(path) -> np.ndarray` reads just the embedding column as a float32 2D array
- [ ] Handles empty tables (returns empty array with correct shape)
- [ ] Existing `read_embeddings()` is unchanged

**Tests needed:**
- Unit test for `read_embedding_vectors()` — basic read, empty file, correct dtype/shape

---

### Task 2: Replace inline embedding reads with `read_embedding_vectors()`

**Files:**
- Modify: `src/humpback/classifier/trainer.py` (~line 190)
- Modify: `src/humpback/workers/vocalization_worker.py` (~lines 155-164)
- Modify: Other sites where only the embedding column is read into a numpy array with no additional columns

**Acceptance criteria:**
- [ ] Each replaced site produces identical output to the original inline code
- [ ] Sites that read additional columns (row_id, confidence, filename, etc.) are left unchanged
- [ ] No behavior changes

**Tests needed:**
- Existing tests cover these paths — no new tests needed beyond Task 1

---

### Task 3: Extract classifier router converters to `schemas/converters.py`

**Files:**
- Create: `src/humpback/schemas/converters.py`
- Modify: `src/humpback/api/routers/classifier.py` (remove converter functions, add import)

**Acceptance criteria:**
- [ ] All 6 classifier converter functions moved: `_training_job_to_out`, `_autoresearch_candidate_to_summary`, `_autoresearch_candidate_to_detail`, `_model_to_out`, `_detection_job_to_out`, `_retrain_workflow_to_out`
- [ ] Functions renamed to drop leading underscore (they're now public module functions)
- [ ] `classifier.py` imports from `schemas.converters`
- [ ] All existing tests pass unchanged

**Tests needed:**
- Existing integration tests cover classifier API endpoints — no new tests needed

---

### Task 4: Extract remaining router converters to `schemas/converters.py`

**Files:**
- Modify: `src/humpback/schemas/converters.py` (append)
- Modify: `src/humpback/api/routers/vocalization.py` (remove converters, add import)
- Modify: `src/humpback/api/routers/clustering.py` (remove converters, add import)
- Modify: `src/humpback/api/routers/processing.py` (remove converters, add import)
- Modify: `src/humpback/api/routers/label_processing.py` (remove converters, add import)
- Modify: `src/humpback/api/routers/audio.py` (remove converters, add import)

**Acceptance criteria:**
- [ ] All 9 remaining converter functions moved from their routers
- [ ] Each router imports its converters from `schemas.converters`
- [ ] No name collisions (use domain-prefixed names where needed, e.g. `clustering_job_to_out`, `processing_job_to_out`)
- [ ] All existing tests pass unchanged

**Tests needed:**
- Existing integration tests cover all router endpoints — no new tests needed

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/schemas/converters.py src/humpback/processing/embeddings.py`
2. `uv run ruff check src/humpback/schemas/converters.py src/humpback/processing/embeddings.py`
3. `uv run pyright src/humpback/schemas/converters.py src/humpback/processing/embeddings.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit` (no frontend changes expected, but verify)
