# Masked Transformer Projection-Head Geometry Diagnostics - Design

**Date:** 2026-05-05
**Status:** Draft

## 1. Purpose

Extend the Masked Transformer retrieval-aware Phase 0 analysis endpoint with a
projection-head geometry diagnostic report before any further contrastive
lambda sweeps. Recent sweep output suggests the retrieval head may be producing
a saturated cone: random-pair retrieval cosines are high enough that nearest
neighbors can be dominated by a common direction rather than useful call-shape
structure.

This design adds first-class geometry metrics across contextual and retrieval
embedding spaces, then adds a frozen-transformer projection-head-only ablation
to isolate whether the projection head can learn a healthy retrieval geometry
from conservative labels before unfreezing or sweeping contrastive weights.

## 2. Current Anchors

- `src/humpback/sequence_models/retrieval_diagnostics.py`
  - Owns the Phase 0 nearest-neighbor diagnostic backend.
  - Already computes random-pair cosine percentiles per embedding variant.
  - Already supports contextual vs retrieval artifacts and variants including
    `remove_pc10` and `whiten_pca`.
- `POST /sequence-models/masked-transformers/{job_id}/nearest-neighbor-report`
  - Current API surface for Phase 0 analysis.
- `src/humpback/sequence_models/masked_transformer.py`
  - `MaskedTransformer` already has an optional retrieval projection head.
  - `extract_transformer_embeddings()` currently returns contextual hidden
    states and optional post-L2 retrieval embeddings.
- `src/humpback/workers/masked_transformer_worker.py`
  - Persists `contextual_embeddings.parquet` and, for retrieval-head jobs,
    `retrieval_embeddings.parquet`.
  - Current retrieval artifact stores post-head retrieval embeddings after the
    model's configured L2 normalization.
- `src/humpback/sequence_models/retrieval_sweeps.py`
  - Plans and compares lambda sweeps; this is the right place to enforce
    geometry stop rules before submitting more lambda runs.

## 3. Goals

1. Report geometry for exactly these six spaces by default:
   - contextual raw
   - contextual remove_pc10
   - contextual whiten_pca
   - retrieval raw
   - retrieval remove_pc10
   - retrieval whiten_pca
2. Compute, for every available space:
   - random-pair cosine percentiles: `p0`, `p1`, `p5`, `p25`, `p50`, `p75`,
     `p95`, `p99`, `p100`
   - normalized mean-vector norm
   - effective rank
   - PCA explained variance for PC1, PC1-5, and PC1-10
   - per-dimension standard deviation summary
   - pre-L2 norm distribution where available
3. Flag likely cone collapse or anisotropy directly in the report.
4. Add a projection-head-only ablation mode: freeze transformer weights and
   train only the retrieval head using conservative same-label/different-region
   positives and safe different-label-family negatives.
5. Block further retrieval lambda sweeps until retrieval raw geometry is no
   longer saturated.

## 4. Non-Goals

- Add frontend charts in this change. The API should return structured JSON
  suitable for a future panel, but the first consumer is the sweep CLI/report.
- Replace nearest-neighbor retrieval metrics. Geometry diagnostics explain
  whether the space is healthy enough for retrieval; they do not replace
  same-label neighbor overlap.
- Backfill pre-L2 retrieval-head outputs for historical jobs. Existing jobs can
  report `pre_l2_norms.available=false`.
- Decide the final contrastive lambda. This design adds a gate before sweeps,
  not a new sweep result.

## 5. Alternatives Considered

### Option A - Add geometry fields to each existing variant metric

Attach mean norm, effective rank, PCA variance, dimension std, and norm
distribution inside every `results[mode][variant]` metrics object.

Pros:
- Small API surface change.
- Keeps random-pair cosine next to neighbor quality metrics.

Cons:
- Geometry is independent of retrieval mode, so it would be duplicated under
  `unrestricted`, `exclude_same_event`, and `exclude_same_event_and_region`.
- Harder for the sweep CLI to compare contextual vs retrieval geometry in one
  table.

### Option B - Add a sibling `geometry_report` section to the existing endpoint

Extend `nearest-neighbor-report` with an optional `include_geometry_report`
flag and a response field keyed by semantic space names such as
`contextual.raw_l2` and `retrieval.whiten_pca`.

Pros:
- Keeps Phase 0 analysis behind the existing endpoint.
- Avoids retrieval-mode duplication.
- Lets callers request nearest-neighbor metrics, geometry metrics, or both in
  one deterministic payload over the same job/context.
- Backward-compatible because the new field is optional.

Cons:
- The endpoint name becomes slightly narrower than the payload.

### Option C - Create a separate geometry endpoint

Add `POST /sequence-models/masked-transformers/{job_id}/geometry-report`.

Pros:
- Clean endpoint semantics.
- Smaller response schemas per endpoint.

Cons:
- Splits Phase 0 analysis into two calls and makes sweep comparison code juggle
  multiple payloads.
- Easier for future users to run neighbor diagnostics without the required
  geometry gate.

## 6. Decision

Use Option B. Extend the existing Phase 0 analysis endpoint with a sibling
`geometry_report` section.

Request additions:

```text
include_geometry_report: bool = false
geometry_embedding_spaces: list[str] | null = null
geometry_random_pairs: int = 20000
geometry_pca_components: int = 20
```

When `geometry_embedding_spaces` is omitted and
`include_geometry_report=true`, report the six default spaces listed in
Section 3. Missing artifacts are represented per-space as unavailable rather
than failing the whole report unless no requested space can be loaded.

Response addition:

```text
geometry_report:
  spaces:
    contextual.raw_l2: GeometrySpaceReport
    contextual.remove_pc10: GeometrySpaceReport
    contextual.whiten_pca: GeometrySpaceReport
    retrieval.raw_l2: GeometrySpaceReport
    retrieval.remove_pc10: GeometrySpaceReport
    retrieval.whiten_pca: GeometrySpaceReport
  summary:
    retrieval_raw_saturated: bool
    lambda_sweeps_blocked: bool
    warnings: list[str]
```

## 7. Metric Definitions

### 7.1 Space construction

Each semantic space has a source artifact and variant:

```text
contextual raw        -> contextual_embeddings.parquet + raw_l2
contextual remove_pc10 -> contextual_embeddings.parquet + remove_pc10
contextual whiten_pca -> contextual_embeddings.parquet + whiten_pca
retrieval raw         -> retrieval_embeddings.parquet + raw_l2
retrieval remove_pc10 -> retrieval_embeddings.parquet + remove_pc10
retrieval whiten_pca  -> retrieval_embeddings.parquet + whiten_pca
```

`raw_l2`, `remove_pc10`, and `whiten_pca` reuse the existing variant builders
so geometry and neighbor metrics evaluate exactly the same matrices.

### 7.2 Random-pair cosine percentiles

Use the existing seeded pair sampler and report percentile keys with `p`
prefixes:

```text
p0, p1, p5, p25, p50, p75, p95, p99, p100
```

Warnings:

```text
median_gt_0p3 when p50 > 0.30
p75_gt_0p7 when p75 > 0.70
p95_gt_0p95 when p95 > 0.95
```

### 7.3 Normalized mean-vector norm

For the evaluated matrix, L2-normalize rows, average rows, then compute the
norm of the mean vector.

Interpretation bands:

```text
good: < 0.05
okay: 0.05 <= norm < 0.15
suspicious: 0.15 <= norm < 0.30
collapse_risk: >= 0.30
```

### 7.4 Per-dimension standard deviation

Report per-dimension std over the evaluated matrix before the final diagnostic
L2 normalization. For `raw_l2`, this means the loaded artifact's vectors before
the diagnostic normalization step. For `remove_pc10` and `whiten_pca`, this
means the transformed vectors before their final L2 normalization.

Fields:

```text
min, p1, p5, p25, p50, p75, p95, p99, max, mean
near_zero_fraction
dominance_ratio
```

`near_zero_fraction` uses `std < 1e-5`. `dominance_ratio` is
`max(std) / mean(std)` with safe handling for zero means.

### 7.5 Effective rank

Center the evaluated matrix, compute singular values, normalize singular values
to a probability distribution, and return `exp(entropy)`.

Interpretation bands for 128-dimensional retrieval spaces:

```text
severe_collapse: < 10
weak: 10 <= rank < 30
plausible: 30 <= rank < 80
broad: >= 80
```

For contextual spaces whose dimensionality may differ from 128, return both the
absolute rank and `effective_rank_fraction = effective_rank / vector_dim`.

### 7.6 PCA explained variance

Center the evaluated matrix and fit PCA with up to
`min(geometry_pca_components, n_rows, n_dims)` components.

Fields:

```text
pc1
pc1_5
pc1_10
components_available
```

Warnings:

```text
pc1_dominant when pc1 >= 0.30
pc5_dominant when pc1_5 >= 0.70
pc10_dominant when pc1_10 >= 0.85
```

### 7.7 Pre-L2 norm distribution

For contextual spaces, the pre-L2 norm distribution is available from the
loaded contextual vectors before diagnostic normalization.

For retrieval spaces, the true projection-head pre-L2 norm distribution is only
available for future jobs that persist a pre-normalization retrieval-head
artifact. Existing `retrieval_embeddings.parquet` rows are already post-L2 when
`retrieval_l2_normalize=true`, so the report must not pretend those norms are
pre-L2.

Add a future artifact for retrieval-head jobs:

```text
retrieval_head_outputs.parquet
```

Schema mirrors `retrieval_embeddings.parquet`, but `embedding` stores the raw
projection-head output before optional L2 normalization. The geometry report
uses this artifact for retrieval `pre_l2_norm_distribution` when present.

Distribution fields:

```text
available: bool
source: contextual_artifact | retrieval_head_outputs | unavailable
min, p1, p5, p25, p50, p75, p95, p99, max, mean
```

## 8. Saturation Verdict

The report should produce per-space warnings and a top-level retrieval raw
verdict. `retrieval.raw_l2` is saturated when any high-confidence cone-collapse
rule fires:

```text
p75 > 0.70
or p95 > 0.95
or mean_vector_norm >= 0.30
or effective_rank < 10
or pc1 >= 0.30
or pc1_5 >= 0.70
```

It is suspicious, but not hard-blocking, when:

```text
p50 > 0.30
or mean_vector_norm >= 0.15
or effective_rank < 30
or pc1_10 >= 0.85
```

The sweep CLI treats saturated retrieval raw geometry as a stop condition:

```text
lambda_sweeps_blocked = retrieval_raw_saturated
```

## 9. Projection-Head-Only Ablation

### 9.1 Purpose

Before changing lambda values, isolate whether the retrieval projection head can
learn useful geometry when the contextual transformer is fixed. This separates
two failure modes:

- The transformer hidden states do not contain enough label geometry.
- The projection head/training objective is collapsing otherwise usable hidden
  states into a cone.

### 9.2 Training mode

Add a training mode to masked-transformer jobs:

```text
training_freeze_mode:
  none
  transformer_frozen_projection_head_only
```

For `transformer_frozen_projection_head_only`:

- Load an existing completed masked-transformer model as the initialization
  source.
- Freeze `input_proj`, `encoder`, and `output_proj`.
- Train only `retrieval_head` parameters.
- Use full-region extraction for artifact alignment.
- Do not update contextual embeddings except by copying or reusing the source
  job's contextual artifact.
- Persist new retrieval artifacts and per-k bundles under a new job ID so
  idempotency and comparisons remain clean.

### 9.3 Label policy

Use human-correction labels only. Positives are deliberately conservative:

```text
positive(anchor, candidate)
  = same surviving human label
  and different region_id
```

Negatives are safe different-label-family pairs:

```text
negative(anchor, candidate)
  = both have exactly one surviving human label
  and labels are in different safe families
  and pair is not in related-label exclusions
```

Initial safe family grouping:

```text
moan_family: Moan, Ascending Moan, Descending Moan
creak_vibrate_family: Creak, Vibrate
growl_buzz_family: Growl, Buzz
whup_grunt_family: Whup, Grunt
other: every other label as its own family
```

Safe negatives require different family names. This avoids treating acoustically
adjacent labels as hard negatives while testing whether the projection head can
separate broad families.

### 9.4 Objective

Use supervised contrastive loss over event-level mean-pooled projection-head
embeddings. No masked reconstruction gradient flows because the transformer is
frozen and only the projection head is optimized.

Recommended first ablation defaults:

```text
contrastive_loss_weight: 1.0
contrastive_temperature: 0.10
batch_size: 16 for 250 ms chunks, 4 for 100 ms chunks
labels_per_batch: 4
events_per_label: 4
max_epochs: 10
early_stop_patience: 2
```

Track geometry at the end of every epoch in memory and persist the final
geometry report. If retrieval raw remains saturated after this ablation, lambda
sweeps remain blocked.

## 10. API And Schema Changes

### 10.1 Request schema

Extend `MaskedTransformerNearestNeighborReportRequest`:

```text
include_geometry_report: bool = false
geometry_embedding_spaces: list[GeometryEmbeddingSpace] | null = null
geometry_random_pairs: int = 20000
geometry_pca_components: int = 20
```

`GeometryEmbeddingSpace` values:

```text
contextual.raw_l2
contextual.remove_pc10
contextual.whiten_pca
retrieval.raw_l2
retrieval.remove_pc10
retrieval.whiten_pca
```

### 10.2 Response schema

Add:

```text
geometry_report: GeometryReport | null
```

Core models:

```text
GeometryReport
  spaces: dict[str, GeometrySpaceReport]
  summary: GeometrySummary

GeometrySpaceReport
  available: bool
  reason: str | null
  artifact_path: str | null
  source_space: contextual | retrieval
  variant: raw_l2 | remove_pc10 | whiten_pca
  row_count: int
  vector_dim: int
  random_pair_percentiles: dict[str, float]
  mean_vector_norm: float | null
  mean_vector_band: string | null
  effective_rank: float | null
  effective_rank_fraction: float | null
  effective_rank_band: string | null
  pca_explained_variance: dict[str, float | int]
  dimension_std: dict[str, float]
  pre_l2_norm_distribution: dict[str, float | bool | string]
  warnings: list[str]

GeometrySummary
  retrieval_raw_saturated: bool
  lambda_sweeps_blocked: bool
  warnings: list[str]
```

### 10.3 Job create schema

For the ablation, extend `MaskedTransformerJobCreate` with:

```text
training_freeze_mode: Literal["none", "transformer_frozen_projection_head_only"] = "none"
source_masked_transformer_job_id: str | null = null
negative_label_family_policy_json: str | null = null
```

Validation:

- `source_masked_transformer_job_id` is required for projection-head-only
  ablation.
- The source job must be completed and must share the same upstream continuous
  embedding job.
- `retrieval_head_enabled=true` is required.
- `contrastive_label_source="human_corrections"` is required.
- The ablation accepts `sequence_construction_mode="region"` because no masked
  reconstruction training is performed; event-level contrastive pooling still
  uses effective event labels.

## 11. Storage Changes

Add optional artifact:

```text
masked_transformer/<job_id>/retrieval_head_outputs.parquet
```

It is written only for retrieval-head jobs after extraction. It mirrors
`retrieval_embeddings.parquet` row order so diagnostics can compare raw
projection-head norms against post-L2 retrieval vectors without another model
forward pass.

For projection-head-only ablation jobs:

- `contextual_embeddings.parquet` can be copied from the source job.
- `retrieval_head_outputs.parquet` is newly written.
- `retrieval_embeddings.parquet` is newly written.
- Per-k bundles are newly fit from retrieval embeddings.

## 12. Sweep Gating

Update `scripts/masked_transformer_retrieval_sweep.py` and
`src/humpback/sequence_models/retrieval_sweeps.py`:

- Compare mode requests `include_geometry_report=true`.
- CSV/Markdown/JSON outputs include:
  - retrieval raw p50/p75/p95
  - retrieval raw mean norm
  - retrieval raw effective rank
  - retrieval raw PC1/PC1-5/PC1-10 explained variance
  - retrieval raw lambda blocked flag
- Submit mode refuses planned lambda-sweep submissions when the selected
  baseline or ablation report has `lambda_sweeps_blocked=true`.
- The initial preset inserts a required projection-head-only ablation before
  any further lambda runs.

This rule is intentionally strict: do not continue lambda sweeps until
`retrieval.raw_l2` geometry is no longer saturated.

## 13. Testing

Unit tests:

- Geometry metric helpers on synthetic isotropic, single-cone, and low-rank
  matrices.
- Percentile keys use `p0` through `p100` and remain deterministic under seed.
- Mean-vector norm warnings trigger at the configured thresholds.
- Effective rank bands classify collapsed and broad matrices correctly.
- PCA explained variance handles fewer than 10 available components.
- Pre-L2 norm distribution reports unavailable for historical retrieval jobs
  without `retrieval_head_outputs.parquet`.
- Geometry report marks missing retrieval artifacts unavailable while still
  reporting contextual spaces.
- Sweep comparison rows flatten retrieval raw geometry columns.
- Sweep submit blocks lambda runs when retrieval raw is saturated.
- Projection-head-only job validation rejects missing source job, incomplete
  source job, disabled retrieval head, and non-human contrastive labels.

Integration tests:

- `nearest-neighbor-report` returns `geometry_report` when requested and keeps
  existing responses valid when omitted.
- Completed contextual-only jobs return contextual geometry and unavailable
  retrieval spaces.
- Retrieval-head jobs with a synthetic `retrieval_head_outputs.parquet` return
  pre-L2 norm distributions.

Worker/model tests:

- Extraction can return both pre-L2 and post-L2 retrieval outputs.
- Projection-head-only ablation freezes transformer parameters and updates only
  retrieval-head parameters.
- Ablation artifacts preserve contextual row alignment and produce fresh
  retrieval/per-k artifacts.

## 14. Acceptance Criteria

- Phase 0 analysis endpoint can report the six default geometry spaces in one
  request.
- The report directly identifies retrieval raw cone collapse or anisotropy.
- Historical jobs remain analyzable without pre-L2 retrieval artifacts.
- New retrieval-head jobs can persist pre-L2 projection-head outputs for norm
  diagnostics.
- Sweep tooling refuses additional lambda runs while retrieval raw geometry is
  saturated.
- A projection-head-only ablation can be submitted and compared before lambda
  sweeps resume.
