# Freeze Transformer Projection Head Implementation Plan

**Goal:** Add and verify a Masked Transformer ablation mode that freezes a completed source transformer and trains only the retrieval projection head with human-correction contrastive loss.
**Spec:** [docs/specs/2026-05-06-freeze-transformer-projection-head-design.md](../specs/2026-05-06-freeze-transformer-projection-head-design.md)

---

### Task 1: Tighten Job Schema And Service Validation

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/services/masked_transformer_service.py`
- Modify: `src/humpback/models/sequence_models.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`
- Modify: `tests/services/test_masked_transformer_service.py`

**Acceptance criteria:**
- [ ] `MaskedTransformerJobCreate` accepts `training_freeze_mode="transformer_frozen_projection_head_only"` only with `source_masked_transformer_job_id`, `retrieval_head_enabled=true`, `contrastive_loss_weight > 0`, and `contrastive_label_source="human_corrections"`.
- [ ] Non-ablation requests normalize `source_masked_transformer_job_id` and `negative_label_family_policy_json` to null so they do not affect normal job identity.
- [ ] The service rejects missing, incomplete, or upstream-mismatched source jobs before enqueueing an ablation.
- [ ] The service rejects incompatible source and requested retrieval-head settings with a clear error message.
- [ ] The training signature includes freeze mode, source job ID, contrastive settings, retrieval-head settings, and negative-family policy while continuing to exclude `k_values`.
- [ ] A projection-head-only ablation job never collides with the source job or with normal joint-training jobs.

**Tests needed:**
- Schema tests for valid ablation payloads and each validation failure.
- Service tests for missing source, incomplete source, mismatched upstream CEJ, incompatible retrieval-head dimensions, signature separation, and idempotent reuse.

---

### Task 2: Isolate Projection-Head-Only Training Behavior

**Files:**
- Modify: `src/humpback/sequence_models/masked_transformer.py`
- Modify: `src/humpback/sequence_models/contrastive_loss.py`
- Modify: `tests/sequence_models/test_masked_transformer.py`
- Modify: `tests/sequence_models/test_contrastive_loss.py`

**Acceptance criteria:**
- [ ] Projection-head-only mode freezes every non-`retrieval_head` parameter before optimizer construction.
- [ ] The optimizer receives only trainable projection-head parameters and fails clearly if no trainable parameters exist.
- [ ] Train and validation total loss in projection-head-only mode is the weighted supervised contrastive loss, with masked reconstruction loss recorded as diagnostics only.
- [ ] Retrieval consistency loss is disabled in projection-head-only mode.
- [ ] Existing joint masked-transformer and retrieval-aware training behavior remains unchanged when `training_freeze_mode="none"`.
- [ ] Loss-curve and validation metric fields make skipped, valid-batch, anchor-count, positive-pair, masked-loss, contrastive-loss, and total-loss values auditable for the ablation.

**Tests needed:**
- Unit test proving frozen parameters are unchanged and at least one projection-head parameter changes.
- Unit test proving masked reconstruction loss does not backpropagate in projection-head-only mode.
- Unit test proving normal retrieval-aware training still updates transformer parameters.
- Unit tests for loss-curve fields and skipped-batch accounting.

---

### Task 3: Load Source Checkpoint And Write Ablation Artifacts

**Files:**
- Modify: `src/humpback/workers/masked_transformer_worker.py`
- Modify: `src/humpback/storage.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`

**Acceptance criteria:**
- [ ] The worker loads the completed source job checkpoint as the ablation `initial_model`.
- [ ] The worker uses authoritative human-correction event labels for projection-head-only contrastive windows.
- [ ] `sequence_construction_mode="region"` remains accepted for ablation requests while the worker internally builds trainable contrastive event windows.
- [ ] The ablation saves a new `transformer.pt` under the ablation job ID containing frozen source transformer weights plus the trained retrieval head.
- [ ] The ablation writes `contextual_embeddings.parquet`, `retrieval_embeddings.parquet`, and `retrieval_head_outputs.parquet` under the ablation job ID.
- [ ] Per-k token bundles are fit from ablation retrieval embeddings when the retrieval head is enabled.
- [ ] Completed-job k-extension behavior remains unchanged for normal and ablation Masked Transformer jobs.

**Tests needed:**
- Worker integration test proving the source checkpoint is passed as `initial_model`.
- Worker integration test proving all expected ablation artifacts and per-k outputs are written.
- Regression test proving normal completed-job k extension still reuses existing model and embeddings.

---

### Task 4: Add Conservative Negative-Family Policy Support

**Files:**
- Modify: `src/humpback/sequence_models/contrastive_loss.py`
- Modify: `src/humpback/sequence_models/masked_transformer.py`
- Modify: `src/humpback/services/masked_transformer_service.py`
- Modify: `tests/sequence_models/test_contrastive_loss.py`
- Modify: `tests/sequence_models/test_masked_transformer.py`
- Modify: `tests/services/test_masked_transformer_service.py`

**Acceptance criteria:**
- [ ] `negative_label_family_policy_json` is parsed and validated only for projection-head-only ablations.
- [ ] Safe-family negatives require both events to have exactly one surviving human label and different family names.
- [ ] Same-family and related-label pairs are excluded from negative pressure.
- [ ] Missing or empty negative-family policy preserves current contrastive negative behavior.
- [ ] Invalid family policy JSON fails with an actionable validation error.

**Tests needed:**
- Unit tests for default negative behavior, safe-family exclusions, same-family exclusion, multi-label exclusion, malformed JSON, and interaction with related-label policy.

---

### Task 5: Add Sweep Preset And Comparison Metadata

**Files:**
- Modify: `src/humpback/sequence_models/retrieval_sweeps.py`
- Modify: `scripts/masked_transformer_retrieval_sweep.py`
- Modify: `tests/sequence_models/test_retrieval_sweeps.py`
- Modify: `tests/scripts/test_masked_transformer_retrieval_sweep.py`

**Acceptance criteria:**
- [ ] The retrieval sweep preset emits a projection-head-only ablation row before lambda sweeps that depend on unsaturated retrieval geometry.
- [ ] Dry-run and submit manifests include freeze mode, source job ID, source job references, label semantics, and negative-family policy when configured.
- [ ] The ablation create payload satisfies `MaskedTransformerJobCreate` validation.
- [ ] Compare mode ranks ablation jobs with the same same-human-label cross-region retrieval metrics as normal retrieval-aware jobs.
- [ ] Stop-rule messaging distinguishes projection-head-only collapse from joint-training collapse.

**Tests needed:**
- Sweep expansion tests for ablation ordering, payload validation, source metadata, dry-run output, and comparison ranking with synthetic rows.

---

### Task 6: Update Documentation

**Files:**
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `docs/specs/2026-05-05-retrieval-aware-transformer-training-design.md`
- Modify: `scripts/README.md`

**Acceptance criteria:**
- [ ] Sequence Models API docs describe projection-head-only ablation create fields and validation rules.
- [ ] Storage docs describe ablation artifacts and the optional `retrieval_head_outputs.parquet` artifact.
- [ ] Behavioral constraints state that ablation contrastive supervision uses authoritative human corrections only.
- [ ] Retrieval-aware transformer design references this standalone ablation spec instead of burying the experiment only in geometry diagnostics.
- [ ] Script docs show how to dry-run and submit the ablation through the retrieval sweep helper.

**Tests needed:**
- Documentation review for consistency with implemented schema fields, artifact names, and CLI flags.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/contrastive_loss.py src/humpback/workers/masked_transformer_worker.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/models/sequence_models.py src/humpback/storage.py src/humpback/sequence_models/retrieval_sweeps.py scripts/masked_transformer_retrieval_sweep.py tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_contrastive_loss.py tests/workers/test_masked_transformer_worker.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/sequence_models/test_retrieval_sweeps.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
2. `uv run ruff check src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/contrastive_loss.py src/humpback/workers/masked_transformer_worker.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/models/sequence_models.py src/humpback/storage.py src/humpback/sequence_models/retrieval_sweeps.py scripts/masked_transformer_retrieval_sweep.py tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_contrastive_loss.py tests/workers/test_masked_transformer_worker.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/sequence_models/test_retrieval_sweeps.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
3. `uv run pyright src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/contrastive_loss.py src/humpback/workers/masked_transformer_worker.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/models/sequence_models.py src/humpback/storage.py src/humpback/sequence_models/retrieval_sweeps.py scripts/masked_transformer_retrieval_sweep.py`
4. `uv run pytest tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_contrastive_loss.py tests/workers/test_masked_transformer_worker.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/sequence_models/test_retrieval_sweeps.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
5. `uv run pytest tests/`
