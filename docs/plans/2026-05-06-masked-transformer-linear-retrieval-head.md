# Masked Transformer Linear Retrieval Head Implementation Plan

**Goal:** Add a first-class linear retrieval-head option for Masked Transformer jobs so matched MLP-vs-linear retrieval experiments are reproducible through the normal job, artifact, diagnostics, frontend, and sweep paths.
**Spec:** `docs/specs/2026-05-06-masked-transformer-linear-retrieval-head-design.md`

---

### Task 1: Add Retrieval Head Architecture Storage

**Files:**
- Create: `alembic/versions/073_masked_transformer_retrieval_head_arch.py`
- Create: `tests/unit/test_migration_073_masked_transformer_retrieval_head_arch.py`
- Modify: `src/humpback/models/sequence_models.py`

**Acceptance criteria:**
- [ ] Before running or testing the migration against any existing SQLite database, perform the CLAUDE.md §3.5 backup step: read the SQLite path from `.env` with `grep -E '^HUMPBACK_DATABASE_URL=' .env`, derive the file path from the `sqlite` URL, copy it to a UTC-stamped backup path with `cp "$DB_PATH" "$DB_PATH.backup.$(date -u +%Y%m%dT%H%M%SZ)"`, and verify the backup is non-zero with `test -s "$BACKUP_PATH"`.
- [ ] Alembic revision 073 adds `masked_transformer_jobs.retrieval_head_arch` as non-null text with default `mlp`.
- [ ] Migration uses `op.batch_alter_table()` for SQLite compatibility.
- [ ] Existing masked-transformer job rows read back with `retrieval_head_arch='mlp'` after upgrade.
- [ ] Downgrade removes `retrieval_head_arch`.
- [ ] SQLAlchemy `MaskedTransformerJob` exposes `retrieval_head_arch` with default `mlp`.

**Tests needed:**
- Unit migration test covering upgrade defaults, existing-row backfill behavior, and downgrade column removal.
- Existing model round-trip tests updated where explicit masked-transformer rows are constructed.

---

### Task 2: Thread Architecture Through API Schema And Service Identity

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/services/masked_transformer_service.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`
- Modify: `tests/services/test_masked_transformer_service.py`
- Modify: `tests/integration/test_masked_transformer_api.py`

**Acceptance criteria:**
- [ ] `MaskedTransformerJobCreate` accepts `retrieval_head_arch` as `mlp` or `linear`, defaulting to `mlp`.
- [ ] `MaskedTransformerJobOut` returns `retrieval_head_arch`.
- [ ] Disabled retrieval heads normalize architecture to `mlp` and dimensions to `None`.
- [ ] MLP retrieval heads keep current defaults: `retrieval_dim=128`, `retrieval_hidden_dim=512`.
- [ ] Linear retrieval heads default `retrieval_dim=128` and normalize `retrieval_hidden_dim` to `None`.
- [ ] `create_masked_transformer_job()` accepts, normalizes, stores, and returns `retrieval_head_arch`.
- [ ] Training signatures differ between otherwise identical MLP and linear retrieval-head jobs.
- [ ] Historical MLP idempotency behavior is preserved for callers that omit `retrieval_head_arch`.
- [ ] Projection-head-only freeze mode rejects a source job whose architecture does not match the requested architecture.
- [ ] API create route passes `retrieval_head_arch` through to the service and returns it in responses.

**Tests needed:**
- Schema tests for defaults, linear normalization, invalid architecture rejection, and disabled-head normalization.
- Service tests for signature uniqueness, historical MLP idempotency, stored job fields, and freeze-mode source compatibility.
- Integration API test asserting create payload and response include the architecture field.

---

### Task 3: Implement Linear Head In The Masked Transformer Trainer

**Files:**
- Modify: `src/humpback/sequence_models/masked_transformer.py`
- Modify: `tests/sequence_models/test_masked_transformer.py`

**Acceptance criteria:**
- [ ] `MaskedTransformerConfig` carries `retrieval_head_arch` with default `mlp`.
- [ ] `MaskedTransformer` validates supported head architectures when a retrieval head is enabled.
- [ ] MLP construction remains `LayerNorm`, `Linear`, `GELU`, `Linear`.
- [ ] Linear construction is exactly `LayerNorm`, `Linear(d_model -> retrieval_dim)`.
- [ ] Forward output shape and pre-L2 output shape remain `(batch, time, retrieval_dim)` for both architectures.
- [ ] L2 normalization behavior is unchanged for both architectures.
- [ ] Event-level pooling reuses the selected head without architecture-specific branches.
- [ ] Projection-head-only freeze mode continues to train only parameters whose names start with `retrieval_head.`.

**Tests needed:**
- Unit tests checking module layout for MLP and linear heads.
- Forward-pass tests for linear retrieval output and pre-L2 output shape.
- Norm test for L2-normalized linear outputs.
- Freeze-mode parameter test confirming linear head parameters are trainable and non-head parameters are frozen.

---

### Task 4: Persist Linear Head Artifacts Through The Worker

**Files:**
- Modify: `src/humpback/workers/masked_transformer_worker.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`

**Acceptance criteria:**
- [ ] Worker passes `job.retrieval_head_arch` into `MaskedTransformerConfig`.
- [ ] Worker checkpoint metadata records `retrieval_head_arch` beside retrieval dimensions and L2 normalization.
- [ ] Completed linear-head jobs write `contextual_embeddings.parquet`.
- [ ] Completed linear-head jobs write `retrieval_embeddings.parquet`.
- [ ] Completed linear-head jobs write `retrieval_head_outputs.parquet`.
- [ ] Completed linear-head jobs produce per-k tokenization artifacts from retrieval embeddings.
- [ ] Strict checkpoint loading prevents MLP and linear head state dicts from being used interchangeably in freeze mode.

**Tests needed:**
- Worker test for completed linear-head job artifact creation and metadata.
- Worker test for projection-head-only source loading with matching linear architecture.
- Worker or service-level regression test covering architecture mismatch failure before incompatible checkpoint loading.

---

### Task 5: Expose Head Architecture In Frontend Create Flow

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/MaskedTransformerCreateForm.tsx`
- Create: `frontend/src/components/sequence-models/MaskedTransformerCreateForm.test.tsx`

**Acceptance criteria:**
- [ ] API type for masked-transformer create payload includes `retrieval_head_arch`.
- [ ] Job response type includes `retrieval_head_arch`.
- [ ] Create form shows an MLP/Linear architecture selector only when retrieval head is enabled.
- [ ] MLP remains the default selected architecture.
- [ ] Linear selection hides or disables the hidden-dimension input.
- [ ] Linear submission sends `retrieval_head_arch='linear'` and `retrieval_hidden_dim=null`.
- [ ] MLP submission preserves current hidden-dimension behavior.

**Tests needed:**
- Component test for default MLP selection after enabling retrieval head.
- Component test for hidden-dimension control behavior when Linear is selected.
- Component test for submitted linear payload.
- Typecheck for updated API types.

---

### Task 6: Add Linear Variant To Retrieval Sweep Tooling

**Files:**
- Modify: `src/humpback/sequence_models/retrieval_sweeps.py`
- Modify: `scripts/masked_transformer_retrieval_sweep.py`
- Modify: `tests/sequence_models/test_retrieval_sweeps.py`
- Modify: `tests/scripts/test_masked_transformer_retrieval_sweep.py`

**Acceptance criteria:**
- [ ] Sweep manifests can emit a matched linear-head variant for retrieval-aware runs.
- [ ] Projection-head-only ablation sweep entries carry `retrieval_head_arch` and require matching source architecture.
- [ ] Linear-head sweep metadata includes `failure_mode_probe='linear_projection_head'`.
- [ ] Sweep comparison keeps linear and MLP rows comparable without changing existing diagnostic metrics.
- [ ] Existing sweep defaults remain unchanged unless the user opts into the linear-head comparison.

**Tests needed:**
- Sweep unit test asserting linear create payload includes `retrieval_head_arch='linear'` and `retrieval_hidden_dim=null`.
- Sweep test asserting existing MLP payloads preserve current defaults.
- Script dry-run test asserting linear-head metadata appears in the manifest.

---

### Task 7: Final Integration And Documentation Polish

**Files:**
- Modify: `docs/specs/2026-05-06-masked-transformer-linear-retrieval-head-design.md`
- Modify: `docs/plans/2026-05-06-masked-transformer-linear-retrieval-head.md`
- Modify: `DECISIONS.md` only if implementation discovers a durable architectural decision not already captured by existing ADRs

**Acceptance criteria:**
- [ ] Spec is updated only for implementation discoveries or clarified decisions.
- [ ] Plan checkboxes are updated as tasks complete.
- [ ] No unrelated docs are changed.
- [ ] If no new ADR-worthy decision appears, `DECISIONS.md` remains unchanged.
- [ ] Final implementation summary names whether linear-head support is API-only, frontend-exposed, or both.

**Tests needed:**
- Documentation review for consistency with implemented field names, defaults, and verification results.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/masked_transformer.py src/humpback/services/masked_transformer_service.py src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/workers/masked_transformer_worker.py src/humpback/sequence_models/retrieval_sweeps.py scripts/masked_transformer_retrieval_sweep.py tests/sequence_models/test_masked_transformer.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/unit/test_sequence_models_schemas.py tests/unit/test_migration_073_masked_transformer_retrieval_head_arch.py tests/sequence_models/test_retrieval_sweeps.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
2. `uv run ruff check src/humpback/sequence_models/masked_transformer.py src/humpback/services/masked_transformer_service.py src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/workers/masked_transformer_worker.py src/humpback/sequence_models/retrieval_sweeps.py scripts/masked_transformer_retrieval_sweep.py tests/sequence_models/test_masked_transformer.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/unit/test_sequence_models_schemas.py tests/unit/test_migration_073_masked_transformer_retrieval_head_arch.py tests/sequence_models/test_retrieval_sweeps.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
3. `uv run pyright src/humpback/sequence_models/masked_transformer.py src/humpback/services/masked_transformer_service.py src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/workers/masked_transformer_worker.py src/humpback/sequence_models/retrieval_sweeps.py scripts/masked_transformer_retrieval_sweep.py`
4. `uv run pytest tests/unit/test_migration_073_masked_transformer_retrieval_head_arch.py tests/sequence_models/test_masked_transformer.py tests/services/test_masked_transformer_service.py tests/workers/test_masked_transformer_worker.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_api.py tests/sequence_models/test_retrieval_sweeps.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npm test -- MaskedTransformerCreateForm`
7. `uv run pytest tests/`
