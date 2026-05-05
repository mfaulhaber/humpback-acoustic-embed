# Retrieval-Aware Transformer Phase 2 Implementation Plan

**Goal:** Add event-centered masked-transformer training windows while preserving full-region extraction and keeping contrastive loss disabled.
**Spec:** [docs/specs/2026-05-05-retrieval-aware-transformer-training-design.md](../specs/2026-05-05-retrieval-aware-transformer-training-design.md)

---

### Task 1: Add sequence-construction job configuration and signature fields

**Files:**
- Create: `alembic/versions/068_masked_transformer_event_centered_sequences.py`
- Create: `tests/unit/test_migration_068_masked_transformer_event_centered_sequences.py`
- Modify: `src/humpback/models/sequence_models.py`
- Modify: `src/humpback/services/masked_transformer_service.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `tests/services/test_masked_transformer_service.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`
- Modify: `tests/integration/test_masked_transformer_api.py`

**Acceptance criteria:**
- [ ] **Production DB backup taken FIRST per CLAUDE.md §3.5:** read `HUMPBACK_DATABASE_URL` from `.env` with `DB_URL=$(grep '^HUMPBACK_DATABASE_URL=' .env | cut -d= -f2-)`, derive the SQLite path with `DB_PATH=${DB_URL#sqlite+aiosqlite:///}`, create a UTC backup name with `BACKUP="$DB_PATH.$(date -u +%Y-%m-%d-%H:%M).bak"`, copy with `cp "$DB_PATH" "$BACKUP"`, and verify non-zero size with `test -s "$BACKUP"` before running any migration command. If any backup command fails or is skipped, stop and do not apply the migration.
- [ ] Migration `068_masked_transformer_event_centered_sequences.py` adds `sequence_construction_mode`, `event_centered_fraction`, `pre_event_context_sec`, and `post_event_context_sec` to `masked_transformer_jobs` using `op.batch_alter_table()` for SQLite compatibility.
- [ ] Existing rows backfill to region-only behavior: `sequence_construction_mode="region"`, `event_centered_fraction=0.0`, and null pre/post context fields.
- [ ] `MaskedTransformerJob`, `MaskedTransformerJobCreate`, and `MaskedTransformerJobOut` expose the new fields with defaults that preserve existing job creation behavior.
- [ ] Accepted modes are `region`, `event_centered`, and `mixed`.
- [ ] Event-centered and mixed modes normalize omitted pre/post context to the Phase 2 defaults `2.0` seconds each.
- [ ] Region mode normalizes `event_centered_fraction` to `0.0` and clears pre/post context fields.
- [ ] Event-centered mode normalizes `event_centered_fraction` to `1.0`.
- [ ] Mixed mode requires `0.0 < event_centered_fraction < 1.0`.
- [ ] `compute_training_signature()` includes the normalized sequence-construction fields and continues to exclude `k_values`.
- [ ] Re-submitting an existing region-only config returns the same pre-068 job as before, while changing any sequence-construction field changes the signature.

**Tests needed:**
- Migration upgrade/downgrade test against SQLite showing defaults on pre-existing `masked_transformer_jobs` rows.
- Service idempotency tests proving sequence-construction config participates in `training_signature` and `k_values` still does not.
- Schema validation tests for defaults, invalid modes, invalid mixed fractions, region-mode normalization, event-centered defaults, and explicit context values.
- API create/list/detail tests showing the new fields round-trip in request and response payloads.

---

### Task 2: Add a pure event-centered training window builder

**Files:**
- Create: `src/humpback/sequence_models/masked_transformer_sequences.py`
- Create: `tests/sequence_models/test_masked_transformer_sequences.py`
- Modify: `src/humpback/sequence_models/masked_transformer.py`
- Modify: `tests/sequence_models/test_masked_transformer.py`

**Acceptance criteria:**
- [ ] A pure builder accepts full-region CRNN embedding sequences plus aligned tier/start/end metadata and effective event intervals, and returns training sequences with aligned tier lists.
- [ ] Region mode returns the original full-region training sequences unchanged.
- [ ] Event-centered mode returns one training sequence per effective event that overlaps available CRNN chunks.
- [ ] Each event-centered training window includes configured pre-event and post-event context, clamped to the available region chunk range.
- [ ] Short events, long events, and events near the beginning or end of a region produce non-empty, correctly bounded windows.
- [ ] Events with no overlapping upstream CRNN chunks are skipped without affecting extraction-time full-region metadata.
- [ ] Mixed mode combines region and event-centered candidates according to `event_centered_fraction`.
- [ ] Mixed mode selection is deterministic under `seed`.
- [ ] Mixed mode always returns at least one trainable sequence when either source has candidates.
- [ ] The builder does not read files, query the database, mutate input arrays, or apply contrastive labels.
- [ ] `MaskedTransformerConfig` carries the sequence-construction fields so trainer-facing configuration has one source of truth.

**Tests needed:**
- Unit tests for short, long, and edge-near events with explicit expected chunk index ranges.
- Unit test proving region mode returns the original arrays and tiers unchanged.
- Unit tests for event-centered-only mode skipping no-overlap events and preserving tier alignment.
- Unit tests for mixed-mode deterministic sampling with a fixed seed and a changed selection under a changed seed.
- Regression test proving `train_masked_transformer()` remains backward-compatible when callers omit sequence-construction fields.

---

### Task 3: Integrate event-centered training into the masked-transformer worker

**Files:**
- Modify: `src/humpback/workers/masked_transformer_worker.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`
- Modify: `tests/fixtures/sequence_models/classify_binding.py`

**Acceptance criteria:**
- [ ] First-pass workers load effective events through the existing segmentation-correction path for event-centered and mixed modes.
- [ ] Event seconds are bridged to the same timestamp domain as CRNN chunk `start_timestamp` and `end_timestamp` using the upstream `RegionDetectionJob.start_timestamp`, consistent with ADR-062 and ADR-063 semantics.
- [ ] Training uses event-centered or mixed training sequences only when configured.
- [ ] Extraction still runs over the original full-region sequences for contextual embeddings, retrieval embeddings, reconstruction error, and per-k tokenization.
- [ ] Retrieval-head jobs still write one retrieval embedding row per upstream CRNN chunk, not per event-centered training window.
- [ ] Non-retrieval jobs still write one contextual embedding row per upstream CRNN chunk.
- [ ] `job.total_sequences` and `job.total_chunks` continue to report full-region extraction counts so existing UI and storage contracts do not change.
- [ ] Saved `transformer.pt` metadata records sequence-construction mode, event-centered fraction, and pre/post context values.
- [ ] If an event-centered or mixed job has no usable event-centered windows, the worker fails clearly rather than silently training as region-only.
- [ ] Extend-k-sweep follow-up passes do not rebuild event-centered windows or retrain the transformer.

**Tests needed:**
- Worker test for event-centered mode with effective events asserting full-region artifact row counts still match the upstream CRNN parquet.
- Worker test for retrieval-head event-centered mode asserting `retrieval_embeddings.parquet` row count still matches the upstream CRNN chunk count.
- Worker test for mixed mode showing training receives both region and event-centered windows under a deterministic seed.
- Worker failure test for event-centered mode when the upstream segmentation has no effective events overlapping CRNN chunks.
- Extend-k-sweep regression test proving event-centered configuration does not touch `transformer.pt`, `contextual_embeddings.parquet`, or `retrieval_embeddings.parquet`.

---

### Task 4: Add minimal frontend controls for sequence-construction mode

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/MaskedTransformerCreateForm.tsx`
- Modify: `frontend/e2e/sequence-models/masked-transformer.spec.ts`

**Acceptance criteria:**
- [ ] Frontend API types include `sequence_construction_mode`, `event_centered_fraction`, `pre_event_context_sec`, and `post_event_context_sec` on create and job response types.
- [ ] The create form exposes a compact mode control for region, event-centered, and mixed training construction.
- [ ] Region mode remains the default and submits the same effective payload shape as existing contextual jobs plus explicit Phase 2 defaults.
- [ ] Event-centered and mixed modes enable numeric pre/post context controls.
- [ ] Mixed mode enables an `event_centered_fraction` numeric control and prevents values outside `0.0 < fraction < 1.0`.
- [ ] Event-centered mode submits `event_centered_fraction=1.0`; region mode submits `event_centered_fraction=0.0`.
- [ ] Existing retrieval-head controls continue to work independently of sequence-construction mode.

**Tests needed:**
- E2E update showing the default create request remains region mode.
- E2E update showing event-centered mode submits pre/post context and fraction defaults.
- E2E update showing mixed mode submits the chosen `event_centered_fraction`.
- TypeScript compile check for updated API types and form state.

---

### Task 5: Update reference documentation and compatibility notes

**Files:**
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `docs/reference/frontend.md`

**Acceptance criteria:**
- [ ] Data-model reference documents the new `MaskedTransformerJob` sequence-construction fields and their default/backward-compatibility behavior.
- [ ] API reference documents create/job response sequence-construction fields, accepted modes, and normalization rules.
- [ ] Behavioral constraints make the invariant explicit: event-centered windows affect training only; extraction artifacts stay full-region aligned.
- [ ] Behavioral constraints keep the `training_signature` rule explicit: sequence-construction config is included and `k_values` is excluded.
- [ ] Frontend reference records the create-form sequence-construction controls.

**Tests needed:**
- Documentation-only task; covered by review plus the verification commands below.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check alembic/versions/068_masked_transformer_event_centered_sequences.py src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/masked_transformer_sequences.py src/humpback/workers/masked_transformer_worker.py tests/unit/test_migration_068_masked_transformer_event_centered_sequences.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_api.py tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_masked_transformer_sequences.py tests/workers/test_masked_transformer_worker.py`
2. `uv run ruff check alembic/versions/068_masked_transformer_event_centered_sequences.py src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/masked_transformer_sequences.py src/humpback/workers/masked_transformer_worker.py tests/unit/test_migration_068_masked_transformer_event_centered_sequences.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_api.py tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_masked_transformer_sequences.py tests/workers/test_masked_transformer_worker.py`
3. `uv run pyright src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/masked_transformer_sequences.py src/humpback/workers/masked_transformer_worker.py`
4. `uv run pytest tests/unit/test_migration_068_masked_transformer_event_centered_sequences.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_api.py tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_masked_transformer_sequences.py tests/workers/test_masked_transformer_worker.py`
5. `uv run pytest tests/`
6. `cd frontend && npm test -- --run src/components/sequence-models`
7. `cd frontend && npx tsc --noEmit`
8. `cd frontend && npx playwright test e2e/sequence-models/masked-transformer.spec.ts`
