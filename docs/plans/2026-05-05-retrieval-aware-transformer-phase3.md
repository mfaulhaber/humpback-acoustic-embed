# Retrieval-Aware Transformer Phase 3 Implementation Plan

**Goal:** Add human-correction supervised contrastive loss to retrieval-head masked-transformer training while preserving masked modeling, full-region extraction, and existing artifact contracts.
**Spec:** [docs/specs/2026-05-05-retrieval-aware-transformer-training-design.md](../specs/2026-05-05-retrieval-aware-transformer-training-design.md)

---

### Task 1: Add contrastive-training job configuration and signature fields

**Files:**
- Create: `alembic/versions/069_masked_transformer_contrastive_training.py`
- Create: `tests/unit/test_migration_069_masked_transformer_contrastive_training.py`
- Modify: `src/humpback/models/sequence_models.py`
- Modify: `src/humpback/services/masked_transformer_service.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `tests/services/test_masked_transformer_service.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`
- Modify: `tests/integration/test_masked_transformer_api.py`

**Acceptance criteria:**
- [x] **Production DB backup taken FIRST per CLAUDE.md §3.5:** read `HUMPBACK_DATABASE_URL` from `.env` with `DB_URL=$(grep '^HUMPBACK_DATABASE_URL=' .env | cut -d= -f2-)`, derive the SQLite path with `DB_PATH=${DB_URL#sqlite+aiosqlite:///}`, create a UTC backup name with `BACKUP="$DB_PATH.$(date -u +%Y-%m-%d-%H:%M).bak"`, copy with `cp "$DB_PATH" "$BACKUP"`, and verify non-zero size with `test -s "$BACKUP"` before running any migration command. If any backup command fails or is skipped, stop and do not apply the migration.
- [x] Migration `069_masked_transformer_contrastive_training.py` adds `contrastive_loss_weight`, `contrastive_temperature`, `contrastive_label_source`, `contrastive_min_events_per_label`, `contrastive_min_regions_per_label`, `require_cross_region_positive`, and `related_label_policy_json` to `masked_transformer_jobs` using `op.batch_alter_table()` for SQLite compatibility.
- [x] Existing rows backfill to contrastive-disabled behavior: `contrastive_loss_weight=0.0`, `contrastive_temperature=0.07`, `contrastive_label_source="none"`, `contrastive_min_events_per_label=4`, `contrastive_min_regions_per_label=2`, `require_cross_region_positive=true`, and `related_label_policy_json=NULL`.
- [x] `MaskedTransformerJob`, `MaskedTransformerJobCreate`, and `MaskedTransformerJobOut` expose the new fields with defaults that preserve existing job creation behavior.
- [x] Accepted `contrastive_label_source` values are `none` and `human_corrections`.
- [x] Any positive `contrastive_loss_weight` requires `retrieval_head_enabled=true`, `contrastive_label_source="human_corrections"`, and an `event_classification_job_id` bound to the selected upstream workflow.
- [x] `contrastive_temperature`, `contrastive_min_events_per_label`, and `contrastive_min_regions_per_label` reject non-positive values.
- [x] Omitted related-label policy resolves to the Phase 3 default exclusions from the design spec and is stored or serialized consistently for signature computation.
- [x] `compute_training_signature()` includes normalized contrastive fields, continues to exclude `k_values`, and includes `event_classification_job_id` when human-correction contrastive training is enabled.
- [x] Re-submitting a contrastive-disabled config remains idempotent with equivalent pre-069 jobs, while changing any enabled contrastive field changes the signature.

**Tests needed:**
- Migration upgrade/downgrade test against SQLite showing defaults on pre-existing `masked_transformer_jobs` rows.
- Service idempotency tests proving contrastive config participates in `training_signature`, `event_classification_job_id` participates only when contrastive labels consume it, and `k_values` still does not.
- Schema validation tests for defaults, invalid label source, invalid positive weight without retrieval head, invalid positive weight without bound Classify workflow, invalid temperature/support thresholds, and default related-label policy normalization.
- API create/list/detail tests showing the new fields round-trip in request and response payloads.

---

### Task 2: Add event-label and contrastive batch metadata structures

**Files:**
- Create: `src/humpback/sequence_models/contrastive_labels.py`
- Create: `tests/sequence_models/test_contrastive_labels.py`
- Modify: `src/humpback/sequence_models/masked_transformer_sequences.py`
- Modify: `tests/sequence_models/test_masked_transformer_sequences.py`

**Acceptance criteria:**
- [x] A reusable loader returns effective events annotated only with human correction label sets, reusing the Phase 0 human-correction semantics and never using Classify model labels as positives.
- [x] Effective event boundaries are loaded through `load_effective_events()` so boundary corrections are respected.
- [x] Event-relative seconds are bridged to absolute UTC with the upstream `RegionDetectionJob.start_timestamp`, matching ADR-062 and ADR-063 timestamp semantics.
- [x] Events with no surviving human correction labels are represented as unlabeled and remain eligible for masked-modeling sequence construction.
- [x] Add/remove correction overlays support multi-label events, add-then-remove behavior, and remove-only behavior.
- [x] Event-centered training candidates carry stable event metadata, absolute event interval, region id, candidate chunk start/end indexes, and human label sets when available.
- [x] Region-context candidates remain valid masked-modeling examples and carry no contrastive event label unless explicitly event-aligned later.
- [x] The structures are pure data contracts with no tensor, model, or optimizer dependency.

**Tests needed:**
- Unit tests with one event and two overlapping add corrections showing a multi-label set.
- Unit tests with add then remove, remove-only, and model-label-only inputs showing the expected human label set or unlabeled event.
- Unit test with a boundary-adjusted event showing membership follows the corrected interval.
- Unit tests proving event-centered sequence candidates retain event metadata and region-mode candidates stay backward-compatible.

---

### Task 3: Implement supervised-contrastive mask and loss helpers

**Files:**
- Create: `src/humpback/sequence_models/contrastive_loss.py`
- Create: `tests/sequence_models/test_contrastive_loss.py`
- Modify: `src/humpback/sequence_models/masked_transformer.py`
- Modify: `tests/sequence_models/test_masked_transformer.py`

**Acceptance criteria:**
- [x] Event embeddings are mean-pooled from transformer hidden states over each event's unpadded chunk span and then passed through the existing retrieval projection head.
- [x] Positive masks use human label-set intersection and include only events whose labels satisfy the configured minimum event and region support thresholds.
- [x] Negative masks require both events to have eligible human labels, disjoint label sets, and no pair blocked by the related-label exclusion policy.
- [x] When `require_cross_region_positive=true`, same-label different-region positives are preferred and same-region-only positives are excluded when a cross-region positive exists for the anchor.
- [x] Batches with no valid positives return a zero contrastive loss that preserves autograd compatibility and lets masked loss train normally.
- [x] Supervised-contrastive loss is finite for multi-label events, singleton labels, empty labels, rare labels, and all-related-label batches.
- [x] `MaskedTransformerConfig` carries contrastive loss weight, temperature, label source, support thresholds, cross-region-positive behavior, and related-label policy.
- [x] `train_masked_transformer()` combines losses as masked loss plus `contrastive_loss_weight * contrastive_loss` only when enabled.
- [x] Retrieval-head consistency loss from Phase 1 continues to operate when contrastive loss is disabled and does not double-count as the supervised contrastive term.

**Tests needed:**
- Unit tests for positive masks with overlapping multi-label sets and disjoint sets.
- Unit tests for related-label exclusions removing pairs from the negative mask.
- Unit tests for rare-label filtering under event-count and region-count thresholds.
- Unit tests for cross-region positive preference with both same-region and different-region same-label candidates.
- Loss tests proving finite values for valid multi-label batches and zero autograd-compatible loss for no-positive batches.
- Training regression tests proving contrastive-enabled training updates retrieval-head parameters and contrastive-disabled training preserves existing loss behavior.

---

### Task 4: Add region-aware contrastive sampling to training batches

**Files:**
- Modify: `src/humpback/sequence_models/masked_transformer.py`
- Modify: `src/humpback/sequence_models/contrastive_loss.py`
- Modify: `tests/sequence_models/test_masked_transformer.py`
- Modify: `tests/sequence_models/test_contrastive_loss.py`

**Acceptance criteria:**
- [x] Training batches receive optional per-sequence contrastive event metadata aligned 1:1 with the constructed training sequence list.
- [x] Contrastive-enabled training prefers batches with multiple eligible labels, multiple regions, and same-label different-region examples when available.
- [x] Labels without the configured minimum support remain in masked modeling batches but are excluded from contrastive masks.
- [x] Batch construction remains deterministic under `seed`.
- [x] If the available training set cannot form a contrastive batch, training logs or records that contrastive was skipped for that batch and continues with masked loss only.
- [x] Validation computes masked, contrastive, and total losses using the same mask rules without mutating training sampler state.
- [x] Existing callers that omit contrastive metadata continue to train exactly as contrastive-disabled masked-transformer jobs.

**Tests needed:**
- Unit test showing a deterministic batch order and same-label cross-region co-occurrence under a fixed seed.
- Unit test showing rare labels stay present in masked batches but are absent from contrastive masks.
- Unit test showing no-eligible-positive training batches safely fall back to masked loss.
- Regression test proving old `train_masked_transformer()` callers without contrastive metadata remain backward-compatible.

---

### Task 5: Integrate human-correction contrastive training into the worker

**Files:**
- Modify: `src/humpback/workers/masked_transformer_worker.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`
- Modify: `tests/fixtures/sequence_models/classify_binding.py`

**Acceptance criteria:**
- [x] First-pass contrastive-enabled workers load human-correction event labels through the shared contrastive label loader and the bound Classify workflow context.
- [x] Contrastive label loading uses only human `VocalizationCorrection` rows; model Classify labels never create positives or negatives.
- [x] Event-centered and mixed training candidates receive aligned human label metadata before calling `train_masked_transformer()`.
- [x] Events without human labels remain in the constructed training sequences for masked modeling and are excluded only from contrastive masks.
- [x] If `contrastive_loss_weight>0` and no eligible human-labeled positives exist, the worker fails clearly before spending epochs on a run that cannot exercise contrastive loss.
- [x] Full-region extraction still writes one contextual row and, for retrieval-head jobs, one retrieval row per upstream CRNN chunk.
- [x] Per-k tokenization still uses retrieval embeddings for retrieval-head jobs and contextual embeddings for non-retrieval jobs.
- [x] Extend-k-sweep follow-up passes do not reload human labels, rebuild contrastive batches, retrain the transformer, or rewrite transformer/contextual/retrieval artifacts.
- [x] Saved `transformer.pt` metadata records contrastive loss config and related-label policy.

**Tests needed:**
- Worker test for contrastive-enabled event-centered mode with human-labeled events asserting training receives labels and artifact row counts remain full-region aligned.
- Worker test proving model-only Classify labels do not produce contrastive positives.
- Worker test proving unlabeled effective events remain available for masked modeling.
- Worker failure test for positive contrastive weight with no eligible human-labeled positives.
- Extend-k-sweep regression test proving contrastive config does not touch model or embedding artifacts on follow-up passes.

---

### Task 6: Persist separated loss curves and summary metrics

**Files:**
- Modify: `src/humpback/sequence_models/masked_transformer.py`
- Modify: `src/humpback/workers/masked_transformer_worker.py`
- Modify: `tests/sequence_models/test_masked_transformer.py`
- Modify: `tests/workers/test_masked_transformer_worker.py`
- Modify: `docs/reference/storage-layout.md`

**Acceptance criteria:**
- [x] `TrainResult.loss_curve` records train and validation series for masked loss, contrastive loss, total loss, and contrastive-skipped batch counts.
- [x] Existing `train_loss` and `val_loss` in `loss_curve.json` continue to mean total loss for backward-compatible consumers.
- [x] `loss_curve.json` adds explicit `train_masked_loss`, `train_contrastive_loss`, `train_total_loss`, `val_masked_loss`, `val_contrastive_loss`, and `val_total_loss` arrays.
- [x] `val_metrics` includes final masked, contrastive, total, and skipped-batch values using JSON-friendly primitives.
- [x] Non-contrastive jobs write contrastive series as zeros or omit only fields documented as optional; frontend and existing tests must not break.
- [x] Storage-layout reference documents the expanded loss-curve artifact.

**Tests needed:**
- Unit test showing contrastive-enabled training returns all loss series with aligned epoch counts.
- Unit test showing contrastive-disabled training preserves existing `train` and `val` curve keys.
- Worker artifact test showing `loss_curve.json` contains total and separated loss series for a contrastive-enabled job.
- Worker regression test showing existing loss-curve consumers still find `epochs`, `train_loss`, and `val_loss`.

---

### Task 7: Add minimal frontend controls for contrastive training

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/MaskedTransformerCreateForm.tsx`
- Modify: `frontend/e2e/sequence-models/masked-transformer.spec.ts`
- Modify: `frontend/src/components/sequence-models/__tests__/MaskedTransformerCreateForm.test.tsx`

**Acceptance criteria:**
- [x] Frontend API types include contrastive training fields on create and job response types.
- [x] The create form keeps contrastive training disabled by default and submits backward-compatible defaults.
- [x] Contrastive controls are enabled only when retrieval head is enabled.
- [x] Enabling contrastive training sets `contrastive_label_source="human_corrections"` and exposes loss weight, temperature, support thresholds, and cross-region-positive controls.
- [x] Frontend validation prevents positive contrastive weight without retrieval head and prevents non-positive temperature/support thresholds.
- [x] Existing retrieval-head and sequence-construction controls continue to work independently when contrastive training remains disabled.

**Tests needed:**
- E2E update showing the default create request remains contrastive-disabled.
- E2E update showing contrastive-enabled creation submits human-correction label source, weight, temperature, support thresholds, and cross-region flag.
- Component or E2E test showing contrastive controls are disabled until retrieval head is enabled.
- TypeScript compile check for updated API types and form state.

---

### Task 8: Update reference documentation and behavioral constraints

**Files:**
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `docs/reference/frontend.md`

**Acceptance criteria:**
- [x] Data-model reference documents the new `MaskedTransformerJob` contrastive fields and their default/backward-compatibility behavior.
- [x] API reference documents create/job response contrastive fields, accepted label sources, normalization rules, and validation constraints.
- [x] Storage-layout reference documents separated loss-curve fields and confirms contextual/retrieval embedding artifact schemas remain unchanged.
- [x] Behavioral constraints make the invariant explicit: human corrections are the only contrastive supervision source; model Classify labels cannot create contrastive positives.
- [x] Behavioral constraints record that unlabeled events remain valid masked-modeling examples but do not contribute to contrastive loss.
- [x] Behavioral constraints keep the `training_signature` rule explicit: contrastive config is included, contrastive-enabled human correction training includes the bound Classify workflow id, and `k_values` is excluded.
- [x] Frontend reference records the contrastive create-form controls and defaults.

**Tests needed:**
- Documentation-only task; covered by review plus the verification commands below.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check alembic/versions/069_masked_transformer_contrastive_training.py src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/contrastive_labels.py src/humpback/sequence_models/contrastive_loss.py src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/masked_transformer_sequences.py src/humpback/workers/masked_transformer_worker.py tests/unit/test_migration_069_masked_transformer_contrastive_training.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_api.py tests/sequence_models/test_contrastive_labels.py tests/sequence_models/test_contrastive_loss.py tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_masked_transformer_sequences.py tests/workers/test_masked_transformer_worker.py`
2. `uv run ruff check alembic/versions/069_masked_transformer_contrastive_training.py src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/contrastive_labels.py src/humpback/sequence_models/contrastive_loss.py src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/masked_transformer_sequences.py src/humpback/workers/masked_transformer_worker.py tests/unit/test_migration_069_masked_transformer_contrastive_training.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_api.py tests/sequence_models/test_contrastive_labels.py tests/sequence_models/test_contrastive_loss.py tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_masked_transformer_sequences.py tests/workers/test_masked_transformer_worker.py`
3. `uv run pyright src/humpback/models/sequence_models.py src/humpback/services/masked_transformer_service.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/sequence_models/contrastive_labels.py src/humpback/sequence_models/contrastive_loss.py src/humpback/sequence_models/masked_transformer.py src/humpback/sequence_models/masked_transformer_sequences.py src/humpback/workers/masked_transformer_worker.py`
4. `uv run pytest tests/unit/test_migration_069_masked_transformer_contrastive_training.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_api.py tests/sequence_models/test_contrastive_labels.py tests/sequence_models/test_contrastive_loss.py tests/sequence_models/test_masked_transformer.py tests/sequence_models/test_masked_transformer_sequences.py tests/workers/test_masked_transformer_worker.py`
5. `uv run pytest tests/`
6. `cd frontend && npm test -- --run src/components/sequence-models`
7. `cd frontend && npx tsc --noEmit`
8. `cd frontend && npx playwright test e2e/sequence-models/masked-transformer.spec.ts`
