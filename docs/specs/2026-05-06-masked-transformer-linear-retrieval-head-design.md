# Masked Transformer Linear Retrieval Head - Design

**Date:** 2026-05-06
**Status:** Draft

## 1. Purpose

Run the "Simplify the head" Masked Transformer experiment by comparing the
current retrieval MLP head against a linear projection head:

```text
MLP head:
  LayerNorm -> Linear(d_model -> hidden_dim) -> GELU -> Linear(hidden_dim -> retrieval_dim) -> L2 normalize

Linear head:
  LayerNorm -> Linear(d_model -> retrieval_dim) -> L2 normalize
```

For the default preset this is the requested:

```text
LayerNorm -> Linear(256 -> 128) -> L2 normalize
```

The experiment asks whether the current MLP projection head is too expressive,
poorly conditioned, over-smoothing, or contributing to retrieval-space collapse.
If the linear head improves retrieval geometry or same-label cross-region
nearest-neighbor behavior, the failure mode is likely in projection-head
architecture rather than only in contrastive labels, sampler construction, or
the transformer encoder.

## 2. Current Anchors

- `src/humpback/sequence_models/masked_transformer.py`
  - `MaskedTransformer` builds the retrieval head inline when
    `retrieval_head_enabled=true`.
  - Current head is `LayerNorm -> Linear -> GELU -> Linear`, followed by
    optional L2 normalization in `forward()`.
  - `_pool_event_retrieval_embeddings()` reuses the same head for event-level
    contrastive training.
  - `extract_transformer_embeddings_with_pre_l2()` already captures both
    normalized retrieval embeddings and pre-L2 head outputs.
- `src/humpback/services/masked_transformer_service.py`
  - Owns training-signature idempotency.
  - Current signature includes retrieval dimensions when the retrieval head is
    enabled.
- `src/humpback/models/sequence_models.py`
  - `MaskedTransformerJob` stores retrieval-head enablement, dimensions, and
    L2 normalization.
- `src/humpback/schemas/sequence_models.py`
  - `MaskedTransformerJobCreate` normalizes retrieval defaults to
    `retrieval_dim=128`, `retrieval_hidden_dim=512` when the retrieval head is
    enabled.
- `src/humpback/workers/masked_transformer_worker.py`
  - Persists `contextual_embeddings.parquet`, `retrieval_embeddings.parquet`,
    `retrieval_head_outputs.parquet`, model metadata, and per-k token bundles.
- `src/humpback/sequence_models/retrieval_diagnostics.py`
  - Already reports geometry fields needed to detect collapse, including
    pre-L2 norm and dimension-std diagnostics.
- `docs/specs/2026-05-05-retrieval-aware-transformer-training-design.md`
  - Defines the retrieval-aware training roadmap and current MLP default.
- `docs/specs/2026-05-06-freeze-transformer-projection-head-design.md`
  - Defines the projection-head-only freeze ablation; the linear head should be
    usable both in normal joint training and in that frozen-head mode.

## 3. Goals

1. Make the retrieval-head architecture an explicit Masked Transformer job
   configuration.
2. Preserve the current MLP head as the default and as the interpretation of
   all existing jobs.
3. Add a linear head option that uses only `LayerNorm` and one projection
   matrix.
4. Include head architecture in training identity so linear and MLP jobs do not
   collide.
5. Keep retrieval artifacts, k-means tokenization, motifs, diagnostics, and
   sweep comparison on existing rails.
6. Record enough metadata to compare MLP and linear runs after the fact.
7. Keep the first implementation small: model construction, schema/service
   normalization, migration, worker metadata, optional UI/sweep affordance, and
   tests.

## 4. Non-Goals

- Replace the transformer encoder.
- Change the reconstruction objective, contrastive objective, temperature, or
  sampler policy.
- Add a separate ProjectionHeadJob resource.
- Retokenize or migrate historical artifacts.
- Make the linear head the default before an experiment proves it should be.
- Introduce deeper projection-head variants such as residual MLPs, bottlenecks,
  dropout heads, or weight-normalized heads.
- Add a new frontend analysis page; existing detail and diagnostic views should
  remain sufficient.

## 5. Experiment Contract

Add a new retrieval-head architecture field:

```text
retrieval_head_arch = "mlp" | "linear"
```

Default:

```text
retrieval_head_arch = "mlp"
```

MLP behavior remains:

```text
LayerNorm(d_model)
Linear(d_model -> retrieval_hidden_dim)
GELU
Linear(retrieval_hidden_dim -> retrieval_dim)
optional L2 normalize
```

Linear behavior:

```text
LayerNorm(d_model)
Linear(d_model -> retrieval_dim)
optional L2 normalize
```

The linear head ignores `retrieval_hidden_dim` for model construction. The
service and schema should normalize `retrieval_hidden_dim` to `None` for linear
jobs so the stored row, job output, and training signature do not imply a
hidden layer that does not exist.

For the requested default-preset experiment:

```text
preset: default
d_model: 256
retrieval_dim: 128
retrieval_head_arch: linear
retrieval_l2_normalize: true
```

## 6. Alternatives Considered

### Option A - Add explicit `retrieval_head_arch`

Add a first-class enum-style job field with values `mlp` and `linear`.

Pros:

- Clear experiment identity.
- Backward-compatible default.
- Easy to expose in the create form and sweep manifests.
- Keeps future diagnostics and PR review legible.
- Avoids overloading dimensions to mean architecture.

Cons:

- Requires a small database migration.
- Adds another config field to service, schema, API, frontend types, and tests.

### Option B - Treat `retrieval_hidden_dim=None` as linear

Use the existing nullable hidden-dim field to switch architecture.

Pros:

- Avoids adding a new column.
- Minimal API shape change.

Cons:

- Ambiguous because `retrieval_hidden_dim=None` currently means "use default
  512" when the retrieval head is enabled.
- Risky for idempotency and backwards compatibility.
- Makes job rows harder to interpret.
- Couples architecture choice to a numeric parameter in a way that will age
  poorly.

### Option C - Add a one-off script or sweep-only flag

Keep API and job schema unchanged, but allow a research script to construct a
linear head directly.

Pros:

- Fastest one-off path.
- No migration.

Cons:

- Bypasses queueing, job identity, artifact metadata, and frontend visibility.
- Harder to reproduce.
- Diagnostics and sweep comparisons would need special handling.
- Easy for future runs to accidentally compare incompatible artifacts.

## 7. Decision

Use Option A: add explicit `retrieval_head_arch`.

This is small enough to implement cleanly and keeps the experiment reproducible
inside the existing Masked Transformer job system. The MLP remains the default;
linear is an opt-in ablation dimension.

## 8. Data Model And Migration

Add a new column to `masked_transformer_jobs`:

```text
retrieval_head_arch TEXT NOT NULL DEFAULT 'mlp'
```

Migration requirements:

- Use Alembic with `op.batch_alter_table()` for SQLite compatibility.
- Existing rows receive `mlp`.
- Downgrade removes the column.
- Add a migration unit test following the style of migration 067, 069, 070,
  071, and 072 tests.

Model update:

- `MaskedTransformerJob.retrieval_head_arch: Mapped[str] = mapped_column(Text, default="mlp")`

No artifact migration is required. Existing retrieval artifacts continue to be
interpreted as MLP-head outputs because existing rows default to `mlp`.

## 9. API, Schema, And Service

### 9.1 Schema

Add to `MaskedTransformerJobCreate`:

```text
retrieval_head_arch: Literal["mlp", "linear"] = "mlp"
```

Add to `MaskedTransformerJobOut`:

```text
retrieval_head_arch: str = "mlp"
```

Validation and normalization:

- If `retrieval_head_enabled=false`, set:
  - `retrieval_dim = None`
  - `retrieval_hidden_dim = None`
  - `retrieval_head_arch = "mlp"`
- If `retrieval_head_enabled=true` and `retrieval_head_arch="mlp"`:
  - default `retrieval_dim` to `128`;
  - default `retrieval_hidden_dim` to `512`;
  - require positive `retrieval_hidden_dim`.
- If `retrieval_head_enabled=true` and `retrieval_head_arch="linear"`:
  - default `retrieval_dim` to `128`;
  - normalize `retrieval_hidden_dim` to `None`;
  - allow callers to omit `retrieval_hidden_dim`;
  - if callers provide `retrieval_hidden_dim`, ignore it only after
    validation has made the behavior explicit in tests.

Projection-head-only freeze mode must accept either architecture. Source-job
compatibility for freeze mode should require:

```text
source_job.retrieval_head_arch == requested_retrieval_head_arch
source_job.retrieval_dim == requested_retrieval_dim
source_job.retrieval_hidden_dim == requested_retrieval_hidden_dim
source_job.retrieval_l2_normalize == requested_retrieval_l2_normalize
```

That keeps checkpoint loading strict and prevents trying to load MLP weights
into a linear head or vice versa.

### 9.2 Service

Update `normalize_retrieval_head_config()` to return:

```text
enabled
retrieval_dim
retrieval_hidden_dim
retrieval_l2_normalize
retrieval_head_arch
```

Update `create_masked_transformer_job()` to:

- accept `retrieval_head_arch`;
- store it on new jobs;
- pass it to `compute_training_signature()`;
- include it in source-job compatibility checks for freeze mode.

Training signature:

- Include `retrieval_head_arch` when `retrieval_head_enabled=true` and the
  architecture is not the historical default `mlp`.
- This preserves idempotency for existing and future default MLP submissions
  while guaranteeing linear jobs have a distinct signature.
- Continue excluding `k_values`.

## 10. Model And Trainer

Update `MaskedTransformerConfig` and `MaskedTransformer.__init__()` with:

```text
retrieval_head_arch: Literal["mlp", "linear"] = "mlp"
```

Construction:

```text
if retrieval_head_enabled and retrieval_head_arch == "mlp":
  retrieval_head = Sequential(
    LayerNorm(d_model),
    Linear(d_model, retrieval_hidden_dim or 512),
    GELU(),
    Linear(retrieval_hidden_dim or 512, retrieval_dim or 128),
  )

if retrieval_head_enabled and retrieval_head_arch == "linear":
  retrieval_head = Sequential(
    LayerNorm(d_model),
    Linear(d_model, retrieval_dim or 128),
  )
```

Forward behavior is unchanged:

- `retrieval_pre_l2 = retrieval_head(hidden)`
- `retrieval = retrieval_pre_l2`
- if `retrieval_l2_normalize`, apply `F.normalize(..., p=2, dim=-1, eps=1e-12)`

Event pooling remains unchanged:

```text
retrieval_event = retrieval_head(mean(hidden_t over event chunk range))
```

Projection-head-only freeze mode remains parameter-name based:

```text
requires_grad = name starts with "retrieval_head."
```

That rule works for both MLP and linear heads.

Checkpoint loading must stay strict. A linear job should not load an MLP
checkpoint unless the source job also records `retrieval_head_arch="linear"`.

## 11. Worker And Artifacts

Worker config construction should pass `job.retrieval_head_arch` into
`MaskedTransformerConfig`.

Model metadata saved with `transformer.pt` should include:

```text
retrieval_head_arch
retrieval_head_enabled
retrieval_dim
retrieval_hidden_dim
retrieval_l2_normalize
```

Artifact behavior is otherwise unchanged:

- `contextual_embeddings.parquet` always comes from transformer hidden states.
- `retrieval_embeddings.parquet` comes from the selected head when enabled.
- `retrieval_head_outputs.parquet` stores pre-L2 outputs for both MLP and
  linear heads.
- Per-k token bundles use retrieval embeddings when the retrieval head is
  enabled.
- Geometry diagnostics should work without special-casing because dimensions
  and pre-L2 outputs keep the same storage contract.

## 12. Frontend And Sweep Tooling

Frontend create form:

- Add a compact architecture selector shown only when retrieval head is enabled:
  - `MLP`
  - `Linear`
- Keep `MLP` selected by default.
- Hide or disable `retrieval_hidden_dim` when `Linear` is selected.
- Submit `retrieval_head_arch` in the create payload.
- Ensure job tables/detail payloads can display the architecture if they already
  show retrieval-head configuration; no new analysis surface is required.

Sweep tooling:

- Add a linear-head variant to retrieval-aware sweep manifests where the
  comparison is otherwise identical to the current MLP run.
- For projection-head-only ablation sweeps, allow both:
  - frozen transformer + MLP head;
  - frozen transformer + linear head.
- Metadata should include:
  - `retrieval_head_arch`;
  - `failure_mode_probe: "linear_projection_head"`;
  - the matched MLP baseline job ID when available.

## 13. Recommended First Comparison

Use matched pairs so architecture is the only intended variable.

### Joint retrieval-aware training

MLP baseline:

```text
preset: default
retrieval_head_enabled: true
retrieval_head_arch: mlp
retrieval_dim: 128
retrieval_hidden_dim: 512
retrieval_l2_normalize: true
sequence_construction_mode: mixed or event_centered
contrastive_label_source: human_corrections
contrastive_loss_weight: same as current best sweep
```

Linear variant:

```text
preset: default
retrieval_head_enabled: true
retrieval_head_arch: linear
retrieval_dim: 128
retrieval_hidden_dim: null
retrieval_l2_normalize: true
same remaining config as MLP baseline
```

### Frozen-transformer projection-head-only training

Run the same comparison after selecting a completed source job whose source
architecture matches the requested ablation architecture. If the first source
job is MLP-only, train a matched linear source job before running a frozen
linear-head continuation.

## 14. Success Criteria

The linear head is a promising replacement if, against a matched MLP run, it:

- improves same-human-label cross-region nearest-neighbor overlap;
- improves event-level mean-pooled retrieval metrics;
- reduces cone-collapse indicators such as high random-pair cosine;
- improves effective rank or per-dimension standard deviation distribution;
- keeps pre-L2 norm distribution healthy rather than saturating;
- preserves or improves motif quality at the same `k_values`.

The linear head is a useful negative result if:

- retrieval geometry remains collapsed with valid contrastive batches;
- MLP and linear behave similarly after whitening or dominant-PC removal;
- linear improves geometry but hurts motif quality or token stability;
- linear only helps when L2 normalization is disabled, which would make this a
  separate normalization experiment rather than a head-simplification result.

## 15. Testing Plan

Unit tests:

- `MaskedTransformer` builds an MLP head with `LayerNorm, Linear, GELU, Linear`
  by default.
- `MaskedTransformer` builds a linear head with `LayerNorm, Linear` when
  `retrieval_head_arch="linear"`.
- Linear head forward pass returns `(batch, T, retrieval_dim)` retrieval and
  pre-L2 tensors.
- L2-normalized linear outputs have unit norm within tolerance.
- `MaskedTransformerConfig` rejects unknown head architectures if validation is
  implemented in the trainer layer.
- Schema defaults old-style retrieval-head requests to `mlp`.
- Schema normalizes linear `retrieval_hidden_dim` to `None`.
- Service training signature differs between otherwise identical MLP and linear
  jobs.
- Service training signature for default MLP jobs preserves historical
  idempotency behavior.
- Freeze-mode source compatibility rejects MLP source to linear ablation and
  linear source to MLP ablation.

Migration tests:

- Upgrade adds `retrieval_head_arch` with default `'mlp'`.
- Existing rows read back as `mlp`.
- Downgrade removes the column.

Worker tests:

- Worker passes `retrieval_head_arch` into `MaskedTransformerConfig`.
- Completed linear-head jobs write retrieval embeddings, pre-L2 head outputs,
  and per-k token artifacts.
- Saved model metadata records `retrieval_head_arch`.

Frontend tests:

- Create form defaults to MLP when retrieval head is enabled.
- Selecting Linear hides or disables hidden-dim input.
- Submitted payload includes `retrieval_head_arch: "linear"` and omits or nulls
  `retrieval_hidden_dim`.

Verification:

```text
uv run ruff format --check src/humpback/sequence_models/masked_transformer.py src/humpback/services/masked_transformer_service.py src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/workers/masked_transformer_worker.py tests/sequence_models/test_masked_transformer.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/unit/test_sequence_models_schemas.py
uv run ruff check src/humpback/sequence_models/masked_transformer.py src/humpback/services/masked_transformer_service.py src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/workers/masked_transformer_worker.py tests/sequence_models/test_masked_transformer.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/unit/test_sequence_models_schemas.py
uv run pyright src/humpback/sequence_models/masked_transformer.py src/humpback/services/masked_transformer_service.py src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/workers/masked_transformer_worker.py
uv run pytest tests/sequence_models/test_masked_transformer.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/unit/test_sequence_models_schemas.py
cd frontend && npm run typecheck
cd frontend && npm test -- MaskedTransformerCreateForm
uv run pytest tests/
```

## 16. Open Questions

- Should `retrieval_head_arch="linear"` be available in the frontend initially,
  or only through API/sweep tooling until the first comparison is complete?
- Should `retrieval_hidden_dim` be stored as `NULL` for linear jobs even if a
  caller supplied a value, or should the API reject supplied hidden dims for
  stricter experiment hygiene? The recommended implementation normalizes to
  `NULL` and covers it with tests.
- Should model metadata include a derived `retrieval_head_param_count` to make
  comparisons easier in diagnostics? Helpful, but not required for the first
  implementation.
- If linear wins, should the MLP remain the default until a second run confirms
  the result on a different upstream continuous-embedding job?
