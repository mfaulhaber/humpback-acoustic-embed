# Retrieval-Aware Transformer Phase 0 Implementation Plan

**Goal:** Move masked-transformer nearest-neighbor diagnostics into the backend source tree and expose them through a structured API endpoint before changing transformer training.

**Spec:** [docs/specs/2026-05-05-retrieval-aware-transformer-training-design.md](../specs/2026-05-05-retrieval-aware-transformer-training-design.md)

---

### Task 1: Extract nearest-neighbor diagnostics into a source module

**Files:**

- Create: `src/humpback/sequence_models/retrieval_diagnostics.py`
- Delete: `scripts/masked_transformer_nn_report.py`
- Delete or modify: `tests/scripts/test_masked_transformer_nn_report.py`

**Acceptance criteria:**

- [ ] Reusable report logic lives in `src/humpback/sequence_models/retrieval_diagnostics.py`; the script no longer owns diagnostic business logic.
- [ ] The source module accepts an existing `AsyncSession`, `Settings` or explicit `storage_root`, `job_id`, and options object rather than reading `.env` or parsing CLI args internally.
- [ ] The source module loads completed `MaskedTransformerJob` rows, validates the requested `k`, and resolves all artifact paths through `humpback.storage`.
- [ ] The source module supports contextual embeddings now and retrieval embeddings when the Phase 1 artifact exists.
- [ ] `scripts/masked_transformer_nn_report.py` is removed; diagnostics are invoked through the source module and API endpoint.
- [ ] Existing script tests are removed or rewritten as source/API tests with equivalent coverage.

**Tests needed:**

- Unit tests that import `retrieval_diagnostics.py` directly and verify report construction does not depend on CLI parsing or `.env`.
- Regression coverage showing old contextual-embedding jobs still work when no retrieval artifact exists.

---

### Task 2: Add human-correction-only label extraction

**Files:**

- Modify: `src/humpback/sequence_models/retrieval_diagnostics.py`
- Modify: `tests/sequence_models/test_retrieval_diagnostics.py`

**Acceptance criteria:**

- [ ] Diagnostic labels are derived only from `VocalizationCorrection` add/remove rows overlapped against effective event boundaries.
- [ ] Classify model `TypedEvent` labels are not used for positive, negative, or same-label retrieval metrics.
- [ ] Effective event boundaries are loaded with `load_effective_events()` so boundary corrections are respected.
- [ ] Event-relative seconds are bridged to absolute UTC using the upstream `RegionDetectionJob.start_timestamp`, matching ADR-062 and ADR-063 timestamp semantics.
- [ ] Human label sets support multiple labels per event and empty sets for unlabeled events.
- [ ] A remove correction without a surviving add leaves the event unlabeled for human-correction metrics.

**Tests needed:**

- Unit test with one event and two overlapping add corrections; the event receives both labels.
- Unit test with add then remove for the same type; the type is absent.
- Unit test with only model Classify labels and no human corrections; the event is treated as unlabeled for retrieval metrics.
- Unit test with a boundary-adjusted event; row membership follows the corrected interval.

---

### Task 3: Implement retrieval modes and embedding variants

**Files:**

- Modify: `src/humpback/sequence_models/retrieval_diagnostics.py`
- Modify: `tests/sequence_models/test_retrieval_diagnostics.py`

**Acceptance criteria:**

- [ ] Supported retrieval modes are `unrestricted`, `exclude_same_event`, and `exclude_same_event_and_region`.
- [ ] Each retrieval mode reuses the same sampled query indices across embedding variants for deterministic comparisons.
- [ ] Supported embedding variants include `raw_l2`, `centered_l2`, `remove_pc1`, `remove_pc3`, `remove_pc5`, `remove_pc10`, and `whiten_pca`.
- [ ] Variant builders handle small fixture datasets by clamping PCA component counts safely.
- [ ] Metrics include same human label overlap, exact human label-set match, same event, same region, similar duration, same token, average cosine, random-pair cosine percentiles, verdict counts, and label-specific same-label overlap.
- [ ] Optional event-level mean-pooled evaluation is available from the same module and clearly marked as event-level in the response.

**Tests needed:**

- Unit tests for each retrieval mode showing excluded candidates cannot appear in top neighbors.
- Unit tests for all embedding variants on a small synthetic matrix.
- Unit tests showing sampled query indices are stable under a fixed seed and shared across variants.
- Unit tests for label-specific metric aggregation.

---

### Task 4: Add request and response schemas

**Files:**

- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`

**Acceptance criteria:**

- [ ] Add a request schema for the nearest-neighbor report endpoint with fields for `k`, `embedding_space`, `samples`, `topn`, `seed`, `retrieval_modes`, `embedding_variants`, `include_query_rows`, and `include_neighbor_rows`.
- [ ] Request validation rejects invalid embedding spaces, retrieval modes, embedding variants, non-positive `samples`, and non-positive `topn`.
- [ ] Add response schemas for job metadata, label coverage, aggregate metrics by mode and variant, verdict counts, representative query summaries, and optional detail rows.
- [ ] Response schemas use JSON-friendly primitives only; no NumPy scalar or Path objects escape the backend.
- [ ] Defaults match the shared design spec and existing Stage 0 report behavior where applicable.

**Tests needed:**

- Schema validation tests for defaults.
- Schema validation tests for invalid options.
- Response serialization test with representative aggregate, query, and neighbor rows.

---

### Task 5: Expose the diagnostics endpoint

**Files:**

- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `tests/integration/test_masked_transformer_api.py` or create `tests/integration/test_masked_transformer_retrieval_diagnostics_api.py`
- Modify: `docs/reference/sequence-models-api.md`

**Acceptance criteria:**

- [ ] Add `POST /sequence-models/masked-transformers/{job_id}/nearest-neighbor-report`.
- [ ] Endpoint calls the source diagnostic module directly and does not shell out to `scripts/masked_transformer_nn_report.py`.
- [ ] Endpoint returns structured JSON matching the response schema.
- [ ] Endpoint returns 404 for missing jobs and requested k values not configured on the job.
- [ ] Endpoint returns 409 for incomplete jobs, missing embeddings, missing decoded artifacts, or other missing required artifacts.
- [ ] Endpoint returns 422 for invalid request options through schema validation.
- [ ] API reference documents the endpoint, request fields, response shape, and status codes.

**Tests needed:**

- Integration test for a completed masked-transformer job returning aggregate metrics.
- Integration test for unavailable k returning 404.
- Integration test for incomplete job returning 409.
- Integration test that monkeypatches the diagnostic module and verifies the router path invokes it rather than the script.

---

### Task 6: Preserve artifact and backward compatibility behavior

**Files:**

- Modify: `src/humpback/sequence_models/retrieval_diagnostics.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`
- Modify: `tests/integration/test_masked_transformer_api.py` or the new diagnostics API test file

**Acceptance criteria:**

- [ ] Contextual embedding diagnostics work for existing jobs that have `contextual_embeddings.parquet` and no `retrieval_embeddings.parquet`.
- [ ] Requesting `embedding_space="retrieval"` before Phase 1 artifacts exist returns a clear 409 instead of silently falling back to contextual embeddings.
- [ ] Requesting `embedding_space="contextual"` continues to use `contextual_embeddings.parquet`.
- [ ] The response records the embedding space actually used.
- [ ] Per-k decoded tokens are read from the existing `k<N>/decoded.parquet` bundle and are not recomputed by the diagnostics endpoint.

**Tests needed:**

- Integration or unit test for contextual fallback on an old-style fixture job.
- Integration or unit test for retrieval-space request with missing retrieval artifact returning 409.
- Unit test confirming decoded artifacts are read but not written.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/api/routers/sequence_models.py src/humpback/schemas/sequence_models.py tests/sequence_models/test_retrieval_diagnostics.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_retrieval_diagnostics_api.py`
2. `uv run ruff check src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/api/routers/sequence_models.py src/humpback/schemas/sequence_models.py tests/sequence_models/test_retrieval_diagnostics.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_retrieval_diagnostics_api.py`
3. `uv run pyright src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/api/routers/sequence_models.py src/humpback/schemas/sequence_models.py`
4. `uv run pytest tests/sequence_models/test_retrieval_diagnostics.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_retrieval_diagnostics_api.py`
5. `uv run pytest tests/`
