# Search Score Calibration & Classifier-Projected Search

**Date:** 2026-03-28
**Status:** Approved

## Problem

Raw cosine similarity on SurfPerch embeddings produces misleadingly high scores
(50%+) between unrelated audio. Ship noise and whale vocalizations score similarly
because SurfPerch encodes generic acoustic features (harmonics, noise floor,
temporal patterns) that overlap across sound types. The 1536-dimensional embedding
space exhibits concentration of measure, where the useful discrimination range is
narrow (e.g., 0.40-0.95) rather than the full 0-100% the UI implies.

Users searching with a negative (non-whale) embedding against accepted whale
vocalizations see high scores that undermine trust in the search tool.

## Solution

Two-phase improvement:

1. **Score calibration** — add distribution context so users understand where a
   score falls relative to all candidates, not just its raw value.
2. **Classifier-projected search** — project embeddings through a trained
   vocalization classifier into a low-dimensional class-probability space where
   cosine similarity is semantically meaningful.

## Phase 1: Score Calibration

### Backend

Add score distribution statistics computed during the brute-force scan. Since the
scan already iterates all candidates, collecting stats adds negligible cost.

**New response fields on `SimilaritySearchResponse`:**

```
score_distribution:
  mean: float
  std: float
  min: float
  max: float
  p25: float
  p50: float
  p75: float
  histogram: [{bin_start: float, bin_end: float, count: int}, ...]
```

**New field on each `SimilaritySearchHit`:**

```
percentile_rank: float  # fraction of candidates scoring below this hit (0.0-1.0)
```

**Implementation:** During `_brute_force_search`, accumulate all scores into a
flat numpy array (alongside the heap for top-K). After scan, compute stats:
mean, std, min, max via numpy; percentiles via `np.percentile`; histogram via
`np.histogram` with ~20 bins spanning [min, max]. Percentile rank per hit:
`(scores < hit_score).sum() / total`. At current scale (< 50K candidates), the
flat array fits comfortably in memory. If scale grows beyond 1M candidates,
switch to streaming stats (sum/sum-of-squares/fixed-bin histogram).

### Pluggable Scoring Backend

Refactor `_brute_force_search` to accept an optional projector:

```python
def _brute_force_search(
    query: np.ndarray,
    candidate_sets: list[tuple[str, str]],
    top_k: int,
    metric: str,
    projector: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[list[tuple[float, str, int]], int, ScoreDistribution]:
```

When `projector` is provided:
- Query: `query = projector(query.reshape(1, -1))[0]`
- Each candidate batch: `embeddings = projector(embeddings)`
- Scoring proceeds as normal on projected vectors.

The projector is any callable mapping `(N, D_in) -> (N, D_out)`. This supports
predict_proba, PCA, UMAP, or any future transformation without changing the
search core.

### Frontend

- **Primary score display:** Percentile rank (e.g., "Top 1%") replaces raw
  cosine percentage as the prominent score.
- **Secondary display:** Raw cosine score shown in smaller/muted text.
- **Color thresholds:** Based on percentile, not raw score. Green >= 95th
  percentile, yellow >= 75th, muted below.
- **Score histogram:** Lightweight bar chart below the results table showing the
  full candidate score distribution with markers for returned hits.

### Search Request Schema Changes

Add to `SimilaritySearchRequest` and `VectorSearchRequest`:

```
search_mode: "raw" | "projected"  (default: "raw")
classifier_model_id: str | None   (required when search_mode is "projected")
```

Phase 1 implements the schema change but only supports `"raw"` mode. The
`"projected"` mode returns an error until Phase 2 wires it up.

## Phase 2: Classifier-Projected Search

### Prerequisite: Fix Labeling Classifier Bug

The multi-class vocalization training crashes at `np.vstack(all_embeddings)` when
source detection jobs use different embedding models with different vector
dimensions. Fix: validate that all source detection jobs share the same embedding
model version and vector dimension before training begins. Reject with a clear
error if they differ.

### Projection via predict_proba

A trained vocalization classifier maps `1536-d SurfPerch -> N-d class
probabilities` (N = 15-20 vocalization types). Cosine similarity in this
class-probability space is discriminative:

- Ship noise -> [0.01, 0.02, 0.90, ...] (high "ship" class)
- Whale moan -> [0.80, 0.05, 0.01, ...] (high "moan" class)
- Another moan -> [0.75, 0.08, 0.02, ...] (high "moan" class)
- cosine(ship, moan) = low (different class distributions)
- cosine(moan, moan) = high (similar class distributions)

**Why predict_proba, not hidden layer extraction:**
- Works with both LogisticRegression and MLP classifiers
- Class probabilities are interpretable (users can see *why* things match)
- 15-20 dimensions eliminates concentration-of-measure
- No architecture-specific code for extracting intermediate features

### Search Service Changes

When `search_mode == "projected"`:
1. Load the vocalization classifier model from joblib (LRU-cached by model ID)
2. Wrap `pipeline.predict_proba` as the projector callable
3. Pass projector into `_brute_force_search`

No pre-computation of projected embeddings. `predict_proba` on an sklearn
pipeline (StandardScaler + matrix multiply) is milliseconds for 50K candidates.

### Frontend Changes

- **Search mode dropdown:** "Raw Embedding" / "Classifier Projected"
- **Classifier model picker:** Shown when "Classifier Projected" is selected.
  Populated from vocalization classifier models (not binary classifiers).
- **Class probability tooltip:** Optional per-hit tooltip showing the top 3
  class probabilities as context for why the match scored high.

## Phasing

**Phase 1 (single feature branch):**
- ScoreDistribution computation in brute-force search
- score_distribution and percentile_rank in response schema
- Pluggable projector parameter on _brute_force_search
- search_mode / classifier_model_id on request schemas (projected returns error)
- Frontend: percentile display, histogram, search mode UI
- Tests: score distribution, percentile ranking, projector passthrough

**Phase 2 (separate feature branch, blocked on classifier fix):**
- Fix labeling classifier dimension-mismatch bug
- Load vocalization classifier as projector in search service
- Wire projected search mode end-to-end
- Frontend: classifier model picker, class probability tooltip
- Tests: integration test with small trained classifier

## Not In Scope

- Fixing other labeling classifier issues (UTC precision, label priority)
- Pre-computed projected embeddings or FAISS indexing
- Changes to binary classifier or detection pipeline
- Changes to embedding storage format

## Risks

- Phase 2 usefulness depends on having enough labeled data per vocalization type.
  Sparse classes (< 10 samples) produce poorly calibrated probabilities. Phase 1
  score calibration mitigates this — percentile ranks surface relative quality
  even with a mediocre classifier.
- predict_proba on very large candidate sets (100K+) may need batching. Current
  scale (< 50K) should be fine without pre-computation.
