# Retrieval-Aware Transformer Phase 1 Implementation Plan

**Goal:** Add an optional retrieval projection head to masked-transformer jobs, persist retrieval embeddings, and make new retrieval-head jobs tokenize from the retrieval embedding space while preserving existing contextual jobs.

**Spec:** [docs/specs/2026-05-05-retrieval-aware-transformer-training-design.md](../specs/2026-05-05-retrieval-aware-transformer-training-design.md)

---

### Task 1: Add retrieval-head job configuration and signature fields

**Files:**
- Create: `alembic/versions/067_masked_transformer_retrieval_head.py`
- Create: `tests/unit/test_migration_067_masked_transformer_retrieval_head.py`
- Modify: `src/humpback/models/sequence_models.py`
- Modify: `src/humpback/services/masked_transformer_service.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `tests/services/test_masked_transformer_service.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`

**Acceptance criteria:**
- [x] **Production DB backup taken FIRST per CLAUDE.md §3.5:** read `HUMPBACK_DATABASE_URL` from `.env` with `DB_URL=$(grep '^HUMPBACK_DATABASE_URL=' .env | cut -d= -f2-)`, derive the SQLite path with `DB_PATH=${DB_URL#sqlite+aiosqlite:///}`, create a UTC backup name with `BACKUP="$DB_PATH.$(date -u +%Y-%m-%d-%H:%M).bak"`, copy with `cp "$DB_PATH" "$BACKUP"`, and verify non-zero size with `test -s "$BACKUP"` before running any migration command. If any backup command fails or is skipped, stop and do not apply the migration.
- [x] Migration `067_masked_transformer_retrieval_head.py` adds `retrieval_head_enabled`, `retrieval_dim`, `retrieval_hidden_dim`, and `retrieval_l2_normalize` to `masked_transformer_jobs` using `op.batch_alter_table()` for SQLite compatibility.
- [x] Existing rows backfill to `retrieval_head_enabled=false`, `retrieval_dim=NULL`, `retrieval_hidden_dim=NULL`, and `retrieval_l2_normalize=true`.
- [x] `MaskedTransformerJob` exposes the new fields with defaults matching the migration and preserves existing rows without manual data repair.
- [x] `MaskedTransformerJobCreate` accepts the retrieval-head fields with defaults that keep current behavior unchanged.
- [x] `MaskedTransformerJobOut` returns the retrieval-head fields so jobs and detail pages can show which embedding space owns tokenization.
- [x] `create_masked_transformer_job()` validates retrieval dimensions only when the retrieval head is enabled; omitted enabled-job dimensions resolve to the Phase 1 defaults `retrieval_dim=128` and `retrieval_hidden_dim=512`.
- [x] `compute_training_signature()` includes the normalized retrieval-head config fields and continues to exclude `k_values`.
- [x] Re-submitting an existing non-retrieval config returns the same job as before, while changing any retrieval-head config field changes the signature.

**Tests needed:**
- Migration upgrade/downgrade test against SQLite showing defaults on pre-existing `masked_transformer_jobs` rows.
- Service idempotency tests proving retrieval-head config participates in `training_signature` and `k_values` still does not.
- Schema validation tests for defaults, invalid dimensions, disabled-head null dimensions, and enabled-head default dimensions.

---

### Task 2: Extend the masked-transformer model with a retrieval projection head

**Files:**
- Modify: `src/humpback/sequence_models/masked_transformer.py`
- Modify: `tests/sequence_models/test_masked_transformer.py`

**Acceptance criteria:**
- [x] `MaskedTransformerConfig` carries `retrieval_head_enabled`, `retrieval_dim`, `retrieval_hidden_dim`, and `retrieval_l2_normalize`.
- [x] `MaskedTransformer` can be constructed with the retrieval head disabled and remains backward-compatible for existing non-retrieval training and extraction behavior.
- [x] When enabled, the model adds `LayerNorm -> Linear -> GELU -> Linear` after the transformer hidden state and returns retrieval embeddings alongside reconstructed outputs and hidden states.
- [x] Retrieval outputs have shape `(batch, T, retrieval_dim)` and are L2-normalized when `retrieval_l2_normalize=true`.
- [x] The forward result has an explicit named contract, such as a dataclass, so callers do not rely on ambiguous tuple positions once a third output exists.
- [x] Masked reconstruction remains based on transformer hidden states, preserving the current reconstruction objective.
- [x] Phase 1 resolves the spec tension that `total_loss=masked_loss` does not naturally train an otherwise-unused projection head: implementation either documents and tests the chosen gradient path or updates the design before code lands.
- [x] Tests assert that retrieval-head parameters receive gradients during Phase 1 training when `retrieval_head_enabled=true`.

**Tests needed:**
- Forward-shape test for disabled and enabled retrieval-head modes.
- L2-normalization test with tolerance on non-padding retrieval vectors.
- Backward-compatibility test for existing contextual extraction callers.
- Gradient-flow regression test proving retrieval-head parameters are trainable under the Phase 1 objective.

---

### Task 3: Persist retrieval embeddings from the worker

**Files:**
- Modify: `src/humpback/workers/masked_transformer_worker.py`
- Modify: `src/humpback/storage.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`

**Acceptance criteria:**
- [x] First-pass retrieval-head jobs write both `contextual_embeddings.parquet` and `retrieval_embeddings.parquet`.
- [x] `retrieval_embeddings.parquet` mirrors the contextual embedding schema: `region_id`, `chunk_index_in_region`, `audio_file_id`, `start_timestamp`, `end_timestamp`, `tier`, and `embedding`.
- [x] Retrieval embeddings are written atomically through a temporary file and `atomic_rename()`.
- [x] Existing non-retrieval jobs continue to write only `contextual_embeddings.parquet`.
- [x] Saved `transformer.pt` metadata includes enough retrieval-head config to reload or inspect the trained model contract later.
- [x] Extend-k-sweep follow-up passes can read retrieval embeddings for retrieval-head jobs without retraining or re-extracting the transformer.
- [x] Missing retrieval artifacts on a retrieval-head follow-up fail clearly instead of silently falling back to contextual tokenization.

**Tests needed:**
- Worker test for a retrieval-head job asserting both parquet artifacts exist and have matching row alignment.
- Worker test that retrieval artifact vectors have dimension `retrieval_dim` and unit norm when normalization is enabled.
- Worker test that a non-retrieval job remains artifact-compatible with existing expectations.
- Extend-k-sweep test proving retrieval embeddings are reused and transformer/contextual artifact modification times are unchanged.

---

### Task 4: Tokenize retrieval-head jobs from retrieval embeddings

**Files:**
- Modify: `src/humpback/workers/masked_transformer_worker.py`
- Modify: `src/humpback/sequence_models/tokenization.py`
- Modify: `src/humpback/sequence_models/retrieval_diagnostics.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`
- Modify: `tests/sequence_models/test_retrieval_diagnostics.py`

**Acceptance criteria:**
- [x] `_write_per_k_bundle()` receives the embedding space selected by the job rather than assuming contextual embeddings.
- [x] New retrieval-head jobs fit k-means and write `k<N>/decoded.parquet` from `retrieval_embeddings.parquet`.
- [x] Existing jobs and retrieval-head-disabled jobs continue to fit k-means from `contextual_embeddings.parquet`.
- [x] Per-k bundle shape remains unchanged, including `decoded.parquet`, `kmeans.joblib`, `run_lengths.json`, `overlay.parquet`, `exemplars.json`, and `label_distribution.json`.
- [x] Overlay, exemplar, run-length, label-distribution, and motif-extraction consumers continue to read the per-k bundle without requiring a schema change.
- [x] The nearest-neighbor report can compare `embedding_space="contextual"` and `embedding_space="retrieval"` for the same completed retrieval-head job.
- [x] Requesting retrieval-space diagnostics for an old job without `retrieval_embeddings.parquet` still returns the Phase 0 409 behavior.

**Tests needed:**
- Worker regression test using distinguishable contextual and retrieval fixture embeddings to prove k-means labels are fit from retrieval embeddings when the head is enabled.
- Worker regression test proving retrieval-head-disabled tokenization remains contextual.
- Diagnostics unit test covering successful retrieval-space loading from the new artifact.
- Diagnostics regression test covering the existing 409 path for missing retrieval artifacts.

---

### Task 5: Add minimal frontend support for creating retrieval-head jobs

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/MaskedTransformerCreateForm.tsx`
- Modify: `frontend/e2e/sequence-models/masked-transformer.spec.ts`

**Acceptance criteria:**
- [x] Frontend API types include the retrieval-head config fields on create and job responses.
- [x] The create form can submit `retrieval_head_enabled=true` with optional retrieval dimensions and L2 normalization.
- [x] The create form defaults preserve current non-retrieval job creation.
- [x] Retrieval dimension controls are disabled or omitted unless the retrieval head is enabled.
- [x] Frontend validation prevents non-positive retrieval dimensions before submit.
- [x] Existing masked-transformer create E2E coverage is updated so the submitted payload includes the new fields when the user enables retrieval mode.

**Tests needed:**
- E2E test or update showing the retrieval-head toggle changes the create request payload.
- TypeScript compile check for the updated API types and form state.

---

### Task 6: Update reference documentation and compatibility notes

**Files:**
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `docs/reference/frontend.md`

**Acceptance criteria:**
- [x] Data-model reference documents the new `MaskedTransformerJob` retrieval-head fields and their default/backward-compatibility behavior.
- [x] Storage-layout reference documents `retrieval_embeddings.parquet` and states that retrieval-head per-k bundles are tokenized from retrieval embeddings.
- [x] API reference documents create/job response retrieval-head fields and the unchanged nearest-neighbor embedding-space behavior from Phase 0.
- [x] Behavioral constraints keep the `training_signature` rule explicit: retrieval-head training config is included and `k_values` is excluded.
- [x] Frontend reference records the create-form retrieval-head controls.

**Tests needed:**
- Documentation-only task; covered by review plus the verification commands below.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check alembic/versions/067_masked_transformer_retrieval_head.py src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/tokenization.py src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/workers/masked_transformer_worker.py tests/unit/test_migration_067_masked_transformer_retrieval_head.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_retrieval_diagnostics.py tests/workers/test_masked_transformer_worker.py`
2. `uv run ruff check alembic/versions/067_masked_transformer_retrieval_head.py src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/tokenization.py src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/workers/masked_transformer_worker.py tests/unit/test_migration_067_masked_transformer_retrieval_head.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_retrieval_diagnostics.py tests/workers/test_masked_transformer_worker.py`
3. `uv run pyright src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/tokenization.py src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/workers/masked_transformer_worker.py`
4. `uv run pytest tests/unit/test_migration_067_masked_transformer_retrieval_head.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_retrieval_diagnostics.py tests/workers/test_masked_transformer_worker.py`
5. `uv run pytest tests/`
6. `cd frontend && npm test -- --run src/components/sequence-models`
7. `cd frontend && npx tsc --noEmit`
8. `cd frontend && npx playwright test e2e/sequence-models/masked-transformer.spec.ts`
