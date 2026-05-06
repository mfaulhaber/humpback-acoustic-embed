# Masked Transformer Projection Geometry Diagnostics Implementation Plan

**Goal:** Extend the Masked Transformer Phase 0 analysis endpoint with geometry diagnostics, pre-L2 retrieval-head norm support, a projection-head-only ablation, and sweep gating that blocks further lambda sweeps while retrieval raw geometry is saturated.
**Spec:** [docs/specs/2026-05-05-masked-transformer-projection-geometry-diagnostics-design.md](../specs/2026-05-05-masked-transformer-projection-geometry-diagnostics-design.md)

---

### Task 1: Add Geometry Metric Primitives

**Files:**
- Modify: `src/humpback/sequence_models/retrieval_diagnostics.py`
- Modify: `tests/sequence_models/test_retrieval_diagnostics.py`

**Acceptance criteria:**
- [x] Geometry helper functions compute random-pair cosine percentiles with `p0`, `p1`, `p5`, `p25`, `p50`, `p75`, `p95`, `p99`, and `p100` keys.
- [x] Geometry helper functions compute normalized mean-vector norm, effective rank, effective-rank fraction, PCA explained variance for PC1/PC1-5/PC1-10, per-dimension std summary, and pre-L2 norm distribution.
- [x] Helper output includes warning codes for cosine saturation, high mean-vector norm, low effective rank, and dominant PCA components.
- [x] Existing nearest-neighbor metrics continue to use the current variant builder semantics for `raw_l2`, `remove_pc10`, and `whiten_pca`.

**Tests needed:**
- Synthetic isotropic, single-cone, and low-rank matrices.
- Deterministic percentile sampling under a fixed seed.
- PCA handling when fewer than 10 components are available.
- Threshold warning coverage for mean norm, effective rank, cosine percentiles, and PCA variance.

---

### Task 2: Extend Diagnostics API Schemas And Response Payload

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `src/humpback/sequence_models/retrieval_diagnostics.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`
- Modify: `tests/integration/test_masked_transformer_retrieval_diagnostics_api.py`

**Acceptance criteria:**
- [x] `MaskedTransformerNearestNeighborReportRequest` accepts `include_geometry_report`, `geometry_embedding_spaces`, `geometry_random_pairs`, and `geometry_pca_components`.
- [x] The response schema includes nullable `geometry_report` with per-space reports and a top-level geometry summary.
- [x] Omitting `include_geometry_report` preserves the existing response shape and behavior for current callers.
- [x] Requesting geometry with no explicit spaces reports contextual raw, contextual remove_pc10, contextual whiten_pca, retrieval raw, retrieval remove_pc10, and retrieval whiten_pca.
- [x] Missing retrieval artifacts mark retrieval spaces unavailable while still returning contextual geometry when available.
- [x] Retrieval raw saturation sets `retrieval_raw_saturated=true` and `lambda_sweeps_blocked=true`.

**Tests needed:**
- Schema validation for default and explicit geometry space requests.
- API response validation for geometry omitted, contextual-only historical jobs, and retrieval-head jobs.
- API error behavior remains 404 for missing job or k, 409 for incomplete jobs or missing required base artifacts, and 422 for invalid options.

---

### Task 3: Persist Pre-L2 Retrieval-Head Outputs

**Files:**
- Modify: `src/humpback/storage.py`
- Modify: `src/humpback/sequence_models/masked_transformer.py`
- Modify: `src/humpback/workers/masked_transformer_worker.py`
- Modify: `tests/sequence_models/test_masked_transformer.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`

**Acceptance criteria:**
- [x] Retrieval-head extraction can return both raw projection-head outputs before L2 normalization and final retrieval embeddings after configured L2 normalization.
- [x] New retrieval-head jobs persist `retrieval_head_outputs.parquet` with row order matching `retrieval_embeddings.parquet`.
- [x] Historical jobs without `retrieval_head_outputs.parquet` report retrieval pre-L2 norms as unavailable rather than using post-L2 retrieval norms.
- [x] Existing contextual-only jobs and extend-k-sweep jobs continue to work without requiring the new artifact.

**Tests needed:**
- Model forward/extraction tests proving pre-L2 vectors differ from post-L2 vectors when normalization is enabled.
- Worker artifact tests for `retrieval_head_outputs.parquet` row alignment and absence on contextual-only jobs.
- Diagnostics tests for available and unavailable pre-L2 norm distributions.

---

### Task 4: Add Projection-Head-Only Ablation Job Metadata

**Files:**
- Create: `alembic/versions/072_masked_transformer_projection_head_ablation.py`
- Modify: `src/humpback/models/sequence_models.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/services/masked_transformer_service.py`
- Create: `tests/unit/test_migration_072_masked_transformer_projection_head_ablation.py`
- Modify: `tests/services/test_masked_transformer_service.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`

**Acceptance criteria:**
- [x] Before running the Alembic migration on any persistent database, read the SQLite path with `DB_PATH=$(grep '^HUMPBACK_DATABASE_URL=' .env | sed 's#^HUMPBACK_DATABASE_URL=sqlite:///##')`, set `BACKUP_PATH="${DB_PATH}.$(date -u +%Y%m%dT%H%M%SZ).bak"`, copy it with `cp "$DB_PATH" "$BACKUP_PATH"`, and verify it with `test -s "$BACKUP_PATH"`; do not run `alembic upgrade head` until the backup exists and has non-zero size.
- [x] Migration adds nullable or defaulted columns for `training_freeze_mode`, `source_masked_transformer_job_id`, and `negative_label_family_policy_json` using `op.batch_alter_table()` for SQLite compatibility.
- [x] Model and schema expose the new fields with defaults preserving existing job behavior.
- [x] Service validation requires a completed source masked-transformer job for `transformer_frozen_projection_head_only`.
- [x] Service validation requires matching upstream continuous embedding job, `retrieval_head_enabled=true`, and `contrastive_label_source="human_corrections"` for the ablation.
- [x] Training signatures include ablation-affecting fields and continue excluding `k_values`.

**Tests needed:**
- Migration upgrade and downgrade tests.
- Service and schema validation for missing source job, incomplete source job, upstream mismatch, disabled retrieval head, non-human label source, and signature changes.

---

### Task 5: Implement Projection-Head-Only Ablation Training Path

**Files:**
- Modify: `src/humpback/sequence_models/contrastive_loss.py`
- Modify: `src/humpback/sequence_models/contrastive_labels.py`
- Modify: `src/humpback/sequence_models/masked_transformer.py`
- Modify: `src/humpback/workers/masked_transformer_worker.py`
- Modify: `tests/sequence_models/test_contrastive_loss.py`
- Modify: `tests/sequence_models/test_contrastive_labels.py`
- Modify: `tests/sequence_models/test_masked_transformer.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`

**Acceptance criteria:**
- [x] Ablation jobs load the completed source model and freeze transformer/input/output projection parameters.
- [x] Only retrieval-head parameters receive optimizer updates in projection-head-only mode.
- [x] Conservative positives require same surviving human label and different `region_id`.
- [x] Safe negatives require exactly one surviving human label per event, different safe label families, and no related-label exclusion.
- [x] Ablation training can run with `sequence_construction_mode="region"` while pooling effective-event embeddings for supervised contrastive loss.
- [x] Ablation jobs write fresh `retrieval_head_outputs.parquet`, `retrieval_embeddings.parquet`, and per-k bundles under the new job ID.
- [x] Contextual embeddings are reused or copied from the source job without changing their row alignment.

**Tests needed:**
- Parameter-freezing test that detects no source transformer parameter updates.
- Positive and negative mask tests for cross-region positives, same-family negative exclusion, related-label exclusion, unlabeled events, and multi-label events.
- Worker artifact tests for contextual alignment and fresh retrieval/per-k outputs.

---

### Task 6: Gate Retrieval Sweeps On Geometry Verdicts

**Files:**
- Modify: `src/humpback/sequence_models/retrieval_sweeps.py`
- Modify: `scripts/masked_transformer_retrieval_sweep.py`
- Modify: `tests/sequence_models/test_retrieval_sweeps.py`
- Modify: `tests/sequence_models/test_retrieval_sweep_outputs.py`
- Modify: `tests/scripts/test_masked_transformer_retrieval_sweep.py`

**Acceptance criteria:**
- [x] Compare mode requests `include_geometry_report=true` by default.
- [x] Comparison rows flatten retrieval raw p50/p75/p95, mean-vector norm, effective rank, PC1/PC1-5/PC1-10 explained variance, and lambda-blocked status.
- [x] Submit mode refuses lambda-sweep submissions when the selected baseline or required ablation report has `lambda_sweeps_blocked=true`.
- [x] Initial preset inserts a required projection-head-only ablation before any new lambda run.
- [x] Failure rows preserve geometry-gate errors in CSV, Markdown, and JSON outputs.

**Tests needed:**
- Comparison flattening tests with saturated and unsaturated geometry reports.
- CLI submit tests proving blocked lambda runs are not submitted.
- Initial preset tests proving lambda runs depend on the ablation gate.

---

### Task 7: Documentation And API Reference Updates

**Files:**
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/specs/2026-05-05-retrieval-aware-transformer-training-design.md`
- Modify: `docs/backlog.md`

**Acceptance criteria:**
- [x] API reference documents geometry request fields and response sections.
- [x] Retrieval-aware transformer roadmap names the geometry gate before further lambda sweeps.
- [x] Backlog notes any deferred frontend visualization work for geometry diagnostics.

**Tests needed:**
- Documentation-only review for consistency with implemented schema names and sweep behavior.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/storage.py src/humpback/sequence_models/masked_transformer.py src/humpback/workers/masked_transformer_worker.py src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/sequence_models/contrastive_loss.py src/humpback/sequence_models/contrastive_labels.py src/humpback/sequence_models/retrieval_sweeps.py scripts/masked_transformer_retrieval_sweep.py tests/sequence_models/test_retrieval_diagnostics.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_retrieval_diagnostics_api.py tests/sequence_models/test_masked_transformer.py tests/workers/test_masked_transformer_worker.py tests/unit/test_migration_072_masked_transformer_projection_head_ablation.py tests/services/test_masked_transformer_service.py tests/sequence_models/test_contrastive_loss.py tests/sequence_models/test_contrastive_labels.py tests/sequence_models/test_retrieval_sweeps.py tests/sequence_models/test_retrieval_sweep_outputs.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
2. `uv run ruff check src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/storage.py src/humpback/sequence_models/masked_transformer.py src/humpback/workers/masked_transformer_worker.py src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/sequence_models/contrastive_loss.py src/humpback/sequence_models/contrastive_labels.py src/humpback/sequence_models/retrieval_sweeps.py scripts/masked_transformer_retrieval_sweep.py tests/sequence_models/test_retrieval_diagnostics.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_retrieval_diagnostics_api.py tests/sequence_models/test_masked_transformer.py tests/workers/test_masked_transformer_worker.py tests/unit/test_migration_072_masked_transformer_projection_head_ablation.py tests/services/test_masked_transformer_service.py tests/sequence_models/test_contrastive_loss.py tests/sequence_models/test_contrastive_labels.py tests/sequence_models/test_retrieval_sweeps.py tests/sequence_models/test_retrieval_sweep_outputs.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
3. `uv run pyright src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/storage.py src/humpback/sequence_models/masked_transformer.py src/humpback/workers/masked_transformer_worker.py src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/sequence_models/contrastive_loss.py src/humpback/sequence_models/contrastive_labels.py src/humpback/sequence_models/retrieval_sweeps.py scripts/masked_transformer_retrieval_sweep.py`
4. `uv run pytest tests/sequence_models/test_retrieval_diagnostics.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_retrieval_diagnostics_api.py tests/sequence_models/test_masked_transformer.py tests/workers/test_masked_transformer_worker.py tests/unit/test_migration_072_masked_transformer_projection_head_ablation.py tests/services/test_masked_transformer_service.py tests/sequence_models/test_contrastive_loss.py tests/sequence_models/test_contrastive_labels.py tests/sequence_models/test_retrieval_sweeps.py tests/sequence_models/test_retrieval_sweep_outputs.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
5. `uv run pytest tests/`
