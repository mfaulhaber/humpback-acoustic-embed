# Search Score Calibration & Projected Search — Implementation Plan

**Goal:** Add score distribution context to search results and prepare a pluggable projector for future classifier-projected search.
**Spec:** [docs/specs/2026-03-28-search-calibration-projected-design.md](../specs/2026-03-28-search-calibration-projected-design.md)

---

### Task 1: Score distribution computation and pluggable projector

**Files:**
- Modify: `src/humpback/services/search_service.py`

**Acceptance criteria:**
- [ ] New `ScoreDistribution` dataclass with fields: mean, std, min, max, p25, p50, p75, histogram (list of bin dicts)
- [ ] `_brute_force_search` accumulates all scores into a flat numpy array during scan
- [ ] After scan, compute distribution stats via numpy (mean, std, min, max, `np.percentile` for p25/p50/p75, `np.histogram` with ~20 bins)
- [ ] `_brute_force_search` returns `ScoreDistribution` as a third tuple element
- [ ] Each hit in the returned list gains a `percentile_rank` float computed as `(all_scores < hit_score).sum() / total`
- [ ] `_brute_force_search` accepts an optional `projector: Callable[[np.ndarray], np.ndarray] | None` parameter
- [ ] When projector is provided, query vector is projected via `projector(query.reshape(1, -1))[0]` before scoring
- [ ] When projector is provided, each candidate batch is projected via `projector(embeddings)` before scoring
- [ ] Both `similarity_search` and `similarity_search_by_vector` pass through the new return values

**Tests needed:**
- Score distribution correctness with known synthetic embeddings (verify mean, percentiles, histogram bin counts)
- Percentile rank ordering matches score ordering
- Projector transforms are applied to both query and candidates
- Identity projector produces same results as no projector
- Empty candidate set returns sensible defaults (zero distribution)

---

### Task 2: Search request and response schema updates

**Files:**
- Modify: `src/humpback/schemas/search.py`
- Modify: `frontend/src/api/types.ts`

**Acceptance criteria:**
- [ ] New `ScoreHistogramBin` Pydantic model with `bin_start: float`, `bin_end: float`, `count: int`
- [ ] New `ScoreDistribution` Pydantic model with `mean`, `std`, `min`, `max`, `p25`, `p50`, `p75`, `histogram: list[ScoreHistogramBin]`
- [ ] `SimilaritySearchHit` gains `percentile_rank: float`
- [ ] `SimilaritySearchResponse` gains `score_distribution: ScoreDistribution`
- [ ] `SimilaritySearchRequest` gains `search_mode: str = "raw"` (pattern `^(raw|projected)$`) and `classifier_model_id: str | None = None`
- [ ] `VectorSearchRequest` gains same two fields
- [ ] `AudioSearchRequest` gains same two fields
- [ ] TypeScript types in `types.ts` mirror all new Pydantic models and fields

**Tests needed:**
- Schema validation: `search_mode` rejects invalid values
- Schema validation: `classifier_model_id` required when `search_mode` is `"projected"` (validated at API level, not schema)

---

### Task 3: API endpoint wiring

**Files:**
- Modify: `src/humpback/api/routers/search.py`
- Modify: `src/humpback/services/search_service.py` (pass search_mode through)

**Acceptance criteria:**
- [ ] `POST /search/similar` passes `search_mode` and `classifier_model_id` from request to service
- [ ] `POST /search/similar-by-vector` passes same fields
- [ ] `POST /search/similar-by-audio` stores `search_mode` and `classifier_model_id` on the SearchJob (may require adding columns or passing through job metadata)
- [ ] When `search_mode == "projected"`, all three endpoints return HTTP 400 with message "Projected search mode not yet implemented"
- [ ] When `search_mode == "raw"`, behavior is identical to current (no projector passed)
- [ ] Score distribution and percentile rank are populated in all search responses

**Tests needed:**
- Integration test: `POST /search/similar` returns score_distribution with correct structure
- Integration test: `search_mode=projected` returns 400
- Integration test: existing search behavior unchanged when search_mode omitted (defaults to raw)

---

### Task 4: Frontend score calibration UI

**Files:**
- Modify: `frontend/src/components/search/SearchTab.tsx`
- Modify: `frontend/src/api/types.ts` (done in Task 2)

**Acceptance criteria:**
- [ ] Primary score display shows percentile rank as "Top N%" (e.g., percentile_rank 0.99 displays as "Top 1%")
- [ ] Secondary score display shows raw cosine score in smaller muted text below percentile
- [ ] Score color thresholds based on percentile rank: green >= 0.95, yellow >= 0.75, muted below
- [ ] Score histogram bar chart rendered below results table using the score_distribution.histogram data
- [ ] Histogram marks the positions of returned hits (e.g., small triangles or dots above relevant bins)
- [ ] Search mode dropdown added: "Raw Embedding" / "Classifier Projected" (projected option disabled with tooltip "Requires trained vocalization classifier")
- [ ] Search mode state stored and passed through API calls

**Tests needed:**
- Percentile display formatting (0.99 -> "Top 1%", 0.75 -> "Top 25%", etc.)

---

### Task 5: Backend tests

**Files:**
- Create: `tests/test_search_service.py`

**Acceptance criteria:**
- [ ] Test score distribution with synthetic embeddings: known vectors produce expected mean, std, min, max
- [ ] Test percentile rank: verify ranks are monotonically ordered with scores
- [ ] Test histogram: verify bin counts sum to total candidates
- [ ] Test projector passthrough: mock projector called with correct shapes, results use projected vectors
- [ ] Test identity projector: same results as no projector
- [ ] Test empty candidates: returns zero-valued distribution without error
- [ ] Test search_mode=projected returns error at service level

**Tests needed:**
- All above are the tests

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/services/search_service.py src/humpback/schemas/search.py src/humpback/api/routers/search.py tests/test_search_service.py`
2. `uv run ruff check src/humpback/services/search_service.py src/humpback/schemas/search.py src/humpback/api/routers/search.py tests/test_search_service.py`
3. `uv run pyright src/humpback/services/search_service.py src/humpback/schemas/search.py src/humpback/api/routers/search.py tests/test_search_service.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
