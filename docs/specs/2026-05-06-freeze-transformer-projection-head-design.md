# Freeze Transformer, Train Projection Head - Design

**Date:** 2026-05-06
**Status:** Draft

## 1. Purpose

Run a clean retrieval-head ablation for Masked Transformer jobs:

```text
contextual transformer embeddings frozen
projection head trainable
contrastive loss trains only projection head
```

The experiment asks whether the already-good contextual hidden states can be
mapped into a better retrieval metric space without changing the transformer.
If the projection head still collapses or underperforms, the likely failure is
in contrastive setup, batch construction, loss formulation, label policy, or
head architecture rather than transformer damage from joint training.

This is not a hard architectural constraint. The preferred implementation
minimizes code touch points, but it may decompose the Masked Transformer
training API where that avoids coupling the ablation to masked-reconstruction
training internals.

## 2. Current Anchors

- `src/humpback/sequence_models/masked_transformer.py`
  - `MaskedTransformer` already supports an optional retrieval projection head.
  - `MaskedTransformerConfig` carries retrieval-head and contrastive settings.
  - `train_masked_transformer()` is the current training entry point.
- `src/humpback/workers/masked_transformer_worker.py`
  - Loads upstream CRNN chunk embeddings.
  - Builds region, event-centered, or mixed training windows.
  - Persists `transformer.pt`, `contextual_embeddings.parquet`,
    `retrieval_embeddings.parquet`, optional pre-L2 head outputs, and per-k
    token bundles.
- `src/humpback/services/masked_transformer_service.py`
  - Owns idempotent job creation and training signatures.
  - Already treats `k_values` as excluded from training identity.
- `src/humpback/schemas/sequence_models.py`
  - Owns API validation for Masked Transformer create requests.
- `src/humpback/sequence_models/contrastive_loss.py`
  - Builds supervised contrastive masks from event metadata.
- `src/humpback/sequence_models/contrastive_labels.py`
  - Loads authoritative human-correction labels for effective events.
- `src/humpback/sequence_models/retrieval_sweeps.py`
  - Plans retrieval-aware experiments and can add this ablation to sweep
    manifests.

Current `main` already has early support for
`training_freeze_mode="transformer_frozen_projection_head_only"`. This spec
records the intended experiment contract so the implementation can be verified,
tightened, or refactored without broadening the feature.

## 3. Goals

1. Train a new retrieval projection head while keeping the source transformer
   hidden states fixed.
2. Persist the ablation as a new Masked Transformer job ID so downstream
   diagnostics, k-means tokenization, motif extraction, and sweep comparison
   stay on existing rails.
3. Use human-correction labels only for supervised contrastive positives.
4. Prefer same-label, different-region positives.
5. Keep contextual artifacts comparable to the source job and write new
   retrieval artifacts for the ablation job.
6. Record enough metadata to distinguish the ablation from joint transformer
   plus projection-head contrastive training.
7. Keep touch points concentrated in the Masked Transformer train, worker,
   service/schema validation, and sweep tooling layers.

## 4. Non-Goals

- Replace the current transformer architecture.
- Change historical Masked Transformer artifacts.
- Add a new frontend workflow in the first implementation.
- Add database tables for a separate projection-head job type.
- Use model-only Classify labels as contrastive supervision.
- Tune the final loss, sampler, or projection-head architecture beyond the
  first ablation defaults.

## 5. Experiment Contract

The ablation creates a new Masked Transformer job with:

```text
training_freeze_mode = transformer_frozen_projection_head_only
source_masked_transformer_job_id = <completed source job>
retrieval_head_enabled = true
contrastive_label_source = human_corrections
contrastive_loss_weight > 0
```

The worker loads the source job checkpoint, freezes all non-head parameters,
and trains only parameters whose names belong to `retrieval_head`.

Frozen modules:

```text
input_proj
encoder
output_proj
```

Trainable module:

```text
retrieval_head
```

The masked reconstruction objective must not contribute gradients in this mode.
The total train and validation losses should represent the weighted contrastive
objective, while the loss curve may still report masked-reconstruction loss as
diagnostic-only context.

## 6. Alternatives Considered

### Option A - Reuse Masked Transformer jobs with a freeze mode

Add or preserve a `training_freeze_mode` field on Masked Transformer jobs. The
worker loads a completed source job, freezes transformer parameters, trains the
existing retrieval head through the existing training function, then writes
normal Masked Transformer artifacts under the new job ID.

Pros:

- Smallest downstream surface area.
- Reuses job lifecycle, queueing, storage layout, per-k tokenization, and
  nearest-neighbor diagnostics.
- Keeps idempotency in `create_masked_transformer_job()`.
- Lets sweep tooling compare ablation jobs and normal jobs uniformly.

Cons:

- The training path still knows about masked reconstruction even when the
  ablation only needs contrastive head training.
- It may forward through the transformer instead of consuming cached
  contextual embeddings directly.
- Care is required so diagnostic masked loss does not accidentally train frozen
  modules or influence early stopping in misleading ways.

### Option B - Train projection head directly from contextual embeddings parquet

Create a lower-level trainer that reads the source job's
`contextual_embeddings.parquet`, pools event windows from fixed contextual
vectors, trains a projection-head module, and then writes retrieval artifacts.

Pros:

- Matches the experiment wording most literally: contextual embeddings are
  fixed inputs.
- Avoids repeated transformer forward passes.
- Removes masked-reconstruction coupling from the ablation trainer.

Cons:

- Requires new artifact-to-model plumbing because `transformer.pt` currently
  stores one model state dict containing both transformer and head.
- More bespoke code for mapping event intervals onto contextual rows.
- More chances to diverge from the worker's existing alignment and extraction
  semantics.

### Option C - Add a separate ProjectionHeadJob resource

Introduce a new job model, API route, worker path, and storage namespace for
projection-head-only experiments.

Pros:

- Clean domain boundary for head-only experiments.
- Avoids overloading Masked Transformer job semantics.

Cons:

- Highest implementation cost and broadest touch points.
- Duplicates queueing, idempotency, status, diagnostics, and artifact linking.
- Premature before this ablation proves useful.

## 7. Decision

Use Option A for the first experiment, with a small internal seam that keeps
Option B available if implementation friction appears.

The implementation should preserve the current Masked Transformer job resource
and add only the minimum extra mode-specific behavior:

- validation for `training_freeze_mode`;
- source-job compatibility checks;
- checkpoint loading from `source_masked_transformer_job_id`;
- parameter freezing in the trainer;
- contrastive-only gradient flow and early-stopping behavior;
- new retrieval artifacts and per-k token bundles written under the ablation
  job ID.

If the trainer becomes difficult to keep clean, extract a helper with this
shape rather than creating a new job type:

```text
train_projection_head_only(
  source_model,
  training_sequences_or_contextual_vectors,
  contrastive_events,
  config,
) -> trained_model_or_head
```

That helper can initially be called by `train_masked_transformer()` and later
switch from frozen forward passes to cached contextual embeddings without
changing the public job API.

## 8. Job Validation

Create request validation:

- `training_freeze_mode` must be one of:
  - `none`
  - `transformer_frozen_projection_head_only`
- `source_masked_transformer_job_id` is required for projection-head-only mode.
- Source job must exist.
- Source job must be complete.
- Source job must share the same `continuous_embedding_job_id` as the ablation
  job.
- Source job must have a retrieval head compatible with requested
  `retrieval_dim`, `retrieval_hidden_dim`, and `retrieval_l2_normalize`, or the
  service must reject the request with a clear error.
- `retrieval_head_enabled=true` is required.
- `contrastive_loss_weight > 0` is required.
- `contrastive_label_source="human_corrections"` is required.
- `sequence_construction_mode="region"` is accepted for this mode because
  masked reconstruction is not the training target; the worker may internally
  construct event-centered contrastive windows from effective events.

Training signature:

- Include `training_freeze_mode`.
- Include `source_masked_transformer_job_id`.
- Include contrastive label and sampler settings.
- Include retrieval-head dimensions and normalization.
- Continue excluding `k_values`.
- Ensure the ablation job does not collide with the source job signature.

## 9. Training Behavior

### 9.1 Initialization

The worker loads the source checkpoint from:

```text
masked_transformer/<source_job_id>/transformer.pt
```

The ablation model starts from the source model's full state dict, not from a
fresh transformer. This preserves the source contextual embedding function.

### 9.2 Freezing

Before optimizer construction:

```text
for each named parameter:
  requires_grad = name starts with "retrieval_head."
```

The optimizer receives only trainable projection-head parameters. If no
trainable parameters exist, training fails before the first epoch.

### 9.3 Loss

In projection-head-only mode:

```text
total_loss = contrastive_loss_weight * supervised_contrastive_loss
```

Masked reconstruction loss is computed only for reporting. Retrieval
consistency loss is disabled. No masked-loss or reconstruction-loss term may
backpropagate.

Validation loss should follow the same contrastive-only rule. If the validation
split has no valid contrastive anchors, the run should still surface skipped
validation contrastive batches explicitly rather than pretending the model
improved.

### 9.4 Event Pooling

Use the existing event-level mean pooling contract:

```text
hidden_event = mean(hidden_t over event chunk range)
retrieval_event = retrieval_head(hidden_event)
optional L2 normalization
```

Only events with surviving human labels participate in contrastive loss.

### 9.5 Positives And Negatives

Positive pair:

```text
same surviving human label
and different region_id when require_cross_region_positive=true
```

For the first ablation, keep negatives conservative. The default contrastive
mask can treat disjoint labels as negatives, but the experiment should support
an optional safe-family policy that excludes acoustically adjacent labels from
negative pressure.

Initial safe family grouping:

```text
moan_family: Moan, Ascending Moan, Descending Moan
creak_vibrate_family: Creak, Vibrate
growl_buzz_family: Growl, Buzz
whup_grunt_family: Whup, Grunt
other: every other label as its own family
```

Safe negatives require both events to have exactly one surviving human label
and different family names.

## 10. Artifact Behavior

For the ablation job:

- Save a new `transformer.pt` containing the frozen source transformer weights
  plus the trained retrieval head.
- Write `contextual_embeddings.parquet` under the ablation job ID.
- Write `retrieval_embeddings.parquet` under the ablation job ID.
- Write `retrieval_head_outputs.parquet` when pre-L2 outputs are available.
- Fit per-k bundles from retrieval embeddings when the retrieval head is
  enabled.

The contextual embeddings should be identical to the source job up to normal
floating-point determinism. A later optimization may copy the source
`contextual_embeddings.parquet` instead of recomputing it, but recomputation is
acceptable for the first implementation if it avoids new artifact-copy logic.

## 11. Sweep Defaults

Recommended first run for 250 ms chunks:

```text
contrastive_loss_weight: 1.0
contrastive_temperature: 0.10
batch_size: 16
contrastive_labels_per_batch: 4
contrastive_events_per_label: 4
contrastive_max_unlabeled_fraction: 0.25
contrastive_region_balance: true
require_cross_region_positive: true
max_epochs: 10
early_stop_patience: 2
```

Recommended first run for 100 ms chunks:

```text
contrastive_loss_weight: 1.0
contrastive_temperature: 0.10
batch_size: 4
contrastive_labels_per_batch: 2
contrastive_events_per_label: 2
contrastive_max_unlabeled_fraction: 0.25
contrastive_region_balance: true
require_cross_region_positive: true
max_epochs: 10
early_stop_patience: 2
```

The first comparison should include:

- source contextual raw;
- source contextual whitened;
- source retrieval raw if the source had a retrieval head;
- ablation retrieval raw;
- ablation retrieval remove-PC10;
- ablation retrieval whitened;
- event-level mean-pooled retrieval metrics where available;
- geometry report fields for retrieval cone collapse.

## 12. Success Criteria

The ablation is useful if:

- projection-head parameters change while frozen transformer parameters remain
  byte-for-byte or numerically unchanged;
- train loss records valid contrastive batches and nonzero positive pairs;
- ablation raw retrieval improves same-human-label cross-region overlap over
  source raw retrieval or moves materially toward source contextual whitened;
- retrieval geometry is not saturated by the existing cone-collapse checks;
- contextual raw and contextual whitened baselines remain comparable to the
  source job.

The ablation is a strong negative result if:

- contrastive batches are valid but retrieval raw still collapses;
- retrieval raw underperforms contextual raw and contextual whitened by the
  same margin as joint contrastive runs;
- geometry diagnostics show high random-pair cosine, low effective rank, or a
  dominant mean direction after head-only training.

## 13. Testing Plan

Unit tests:

- Schema accepts valid projection-head-only requests and rejects missing source
  job ID, missing retrieval head, non-human contrastive label source, and zero
  contrastive weight.
- Service rejects missing, incomplete, or mismatched source jobs.
- Training freezes all non-`retrieval_head` parameters and updates at least one
  head parameter on a small synthetic contrastive fixture.
- Training loss in freeze mode excludes masked reconstruction gradients.
- Contrastive masks require cross-region positives when configured.
- Negative-label family policy excludes same-family label pairs from negatives.

Worker tests:

- Projection-head-only job loads the source checkpoint as `initial_model`.
- The completed ablation writes contextual, retrieval, pre-L2 head output, and
  per-k artifacts.
- The ablation job status and error handling match normal Masked Transformer
  worker behavior.

Sweep tests:

- Sweep preset emits a projection-head-only ablation row before unblocked
  lambda sweeps.
- Dry-run manifests include `training_freeze_mode`,
  `source_masked_transformer_job_id`, source references, and label semantics.
- Comparison ranking keeps ablation rows comparable with normal retrieval-aware
  jobs.

Verification:

```text
uv run ruff format --check src/humpback/sequence_models/masked_transformer.py src/humpback/workers/masked_transformer_worker.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/sequence_models/retrieval_sweeps.py tests/sequence_models/test_masked_transformer.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/unit/test_sequence_models_schemas.py
uv run ruff check src/humpback/sequence_models/masked_transformer.py src/humpback/workers/masked_transformer_worker.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/sequence_models/retrieval_sweeps.py tests/sequence_models/test_masked_transformer.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/unit/test_sequence_models_schemas.py
uv run pyright src/humpback/sequence_models/masked_transformer.py src/humpback/workers/masked_transformer_worker.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/sequence_models/retrieval_sweeps.py
uv run pytest tests/sequence_models/test_masked_transformer.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/unit/test_sequence_models_schemas.py
uv run pytest tests/
```

## 14. Open Questions

- Should the first implementation recompute ablation contextual embeddings or
  copy them from the source artifact? Recompute has fewer new file operations;
  copy is faster and makes identity easier to assert.
- Should early stopping monitor validation contrastive loss only, or a geometry
  metric from a lightweight in-memory report? Use validation contrastive loss
  first unless collapse remains hard to detect.
- Should the projection head start from the source retrieval head weights or be
  reinitialized? Default to source weights for minimal disruption, but expose a
  future ablation option for fresh-head initialization if source head collapse
  appears sticky.
- Should safe-family negatives be required for the first run or optional? The
  safer default is optional with a documented preset so the baseline remains
  comparable to current contrastive loss behavior.
