# Detection Window Confidence Implementation Plan

**Goal:** Store per-window classifier confidence in detection_embeddings.parquet and thread it through the vocalization inference pipeline to enable server-side confidence sorting in the labeling workspace.
**Spec:** docs/specs/2026-03-29-detection-window-confidence-design.md

---

### Task 1: Add confidence to detection embedding records

**Files:**
- Modify: `src/humpback/classifier/detector.py`

**Acceptance criteria:**
- [ ] Embedding record dict includes `"confidence": best_conf` when building per-detection embedding records in `run_detection()`
- [ ] `write_detection_embeddings()` parquet schema includes `("confidence", pa.float32())`
- [ ] Records missing `confidence` key write `None` for the column

**Tests needed:**
- `write_detection_embeddings` round-trip: write records with confidence, read back, verify schema and values
- `write_detection_embeddings` with missing confidence key: verify None written without error

---

### Task 2: Thread confidence through vocalization inference pipeline

**Files:**
- Modify: `src/humpback/workers/vocalization_worker.py`
- Modify: `src/humpback/classifier/vocalization_inference.py`

**Acceptance criteria:**
- [ ] `_load_source_embeddings` reads `confidence` column from detection_embeddings.parquet when present, returns `list[float] | None`
- [ ] `_load_source_embeddings` returns `None` for confidences when column absent (old jobs)
- [ ] `run_inference` accepts optional `confidences: list[float] | None` parameter
- [ ] `run_inference` writes `confidence` float32 column to predictions parquet when provided
- [ ] `read_predictions` reads `confidence` column from predictions parquet when present, includes it in returned row dicts

**Tests needed:**
- `run_inference` with confidences: verify predictions parquet includes confidence column with correct values
- `run_inference` without confidences: verify predictions parquet omits confidence column
- `read_predictions` with and without confidence column in parquet
- `_load_source_embeddings` backward compat: mock parquet without confidence column, verify None returned

---

### Task 3: API sort parameter and schema update

**Files:**
- Modify: `src/humpback/schemas/vocalization.py`
- Modify: `src/humpback/api/routers/vocalization.py`

**Acceptance criteria:**
- [ ] `VocalizationPredictionRow` has `confidence: float | None = None` field
- [ ] `GET /inference-jobs/{job_id}/results` accepts `sort: str | None = Query(None)` parameter
- [ ] When `sort=confidence_desc`, results are sorted by confidence descending (nulls last) before offset/limit pagination
- [ ] When sort is None or unrecognized, existing behavior unchanged

**Tests needed:**
- API endpoint with `sort=confidence_desc`: create inference job with known confidence values, verify paginated results arrive in confidence order
- API endpoint with no sort param: verify existing behavior unchanged
- Old-format predictions (no confidence): verify `confidence: null` in response, sort degrades gracefully

---

### Task 4: Frontend confidence sort mode

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/components/vocalization/LabelingWorkspace.tsx`

**Acceptance criteria:**
- [ ] `VocClassifierPredictionRow` type includes optional `confidence?: number`
- [ ] API client passes `sort` query parameter to inference results endpoint
- [ ] `LabelingWorkspace` has `confidence_desc` sort mode that passes `sort=confidence_desc` to API and skips client-side re-sorting
- [ ] Confidence sort option only shown when source is a detection job and confidence data is present
- [ ] Default sort for detection job sources is `confidence_desc`

**Tests needed:**
- Verify TypeScript types compile with new field (`npx tsc --noEmit`)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/classifier/detector.py src/humpback/workers/vocalization_worker.py src/humpback/classifier/vocalization_inference.py src/humpback/schemas/vocalization.py src/humpback/api/routers/vocalization.py`
2. `uv run ruff check src/humpback/classifier/detector.py src/humpback/workers/vocalization_worker.py src/humpback/classifier/vocalization_inference.py src/humpback/schemas/vocalization.py src/humpback/api/routers/vocalization.py`
3. `uv run pyright src/humpback/classifier/detector.py src/humpback/workers/vocalization_worker.py src/humpback/classifier/vocalization_inference.py src/humpback/schemas/vocalization.py src/humpback/api/routers/vocalization.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
