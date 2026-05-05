# Retrieval-Aware Transformer Contrastive Batching Improvements

**Date:** 2026-05-05
**Branch:** `feature/retrieval-aware-transformer-phase3`
**Motivation:** Phase 0 diagnostics on the `batch_size=16` event-centered contrastive run proved that contrastive loss is now active, but training still skipped 14 contrastive batches per epoch and raw retrieval embeddings underperformed contextual embeddings. The current sampler sorts labeled examples before unlabeled examples, but it does not actively build batches with same-label cross-region positives.

## Goal

Reduce skipped contrastive batches and make retrieval-head training receive useful same-label cross-region positives consistently, while keeping masked modeling examples, full-region extraction artifacts, and existing default behavior intact.

## Current Root Cause

- `src/humpback/sequence_models/masked_transformer.py` uses `_contrastive_epoch_order()` to sort labeled events by label, region, and event id, then slices the ordered list into fixed-size batches.
- That ordering is not a batch sampler. It can still split same-label positives across batch boundaries, overpack one region, and leave many batches with no valid positives.
- Support thresholds are evaluated inside each batch by `build_contrastive_masks()`, so a globally useful label can still be excluded if the local batch does not contain enough events or regions.
- The UI exposes raw advanced fields in one large panel, making it easy to set batch size and contrastive controls without understanding which settings belong together.

## Implementation Strategy

Add a deterministic, label-balanced contrastive batch sampler that constructs batches from eligible human-labeled events first, then fills remaining space with masked-modeling-only examples. Persist its useful parameters on `MaskedTransformerJob`, expose them in a consolidated create UI, and record richer batch diagnostics in `loss_curve.json`.

## Task 1: Add Contrastive Sampler Configuration

Files:
- `alembic/versions/071_masked_transformer_contrastive_sampler.py`
- `src/humpback/models/sequence_models.py`
- `src/humpback/schemas/sequence_models.py`
- `src/humpback/services/masked_transformer_service.py`
- `src/humpback/api/routers/sequence_models.py`
- `src/humpback/workers/masked_transformer_worker.py`
- `frontend/src/api/sequenceModels.ts`

New persisted fields:
- `contrastive_sampler_enabled: bool = true`
- `contrastive_labels_per_batch: int = 4`
- `contrastive_events_per_label: int = 4`
- `contrastive_max_unlabeled_fraction: float = 0.25`
- `contrastive_region_balance: bool = true`

Acceptance criteria:
- Migration uses `op.batch_alter_table()` and backfills existing rows to sampler-enabled defaults.
- `MaskedTransformerJobCreate` validates positive integer sampler counts and `0.0 <= contrastive_max_unlabeled_fraction < 1.0`.
- Sampler fields only affect `training_signature` when contrastive loss is enabled.
- Contrastive-disabled job creation remains idempotent with existing disabled jobs.
- Worker `_config_from_job()` passes sampler settings into `MaskedTransformerConfig`.

Tests:
- Migration upgrade/downgrade test for defaults.
- Schema validation tests for invalid sampler values.
- Service signature tests proving sampler fields participate only when `contrastive_loss_weight > 0`.
- API create/list/detail round-trip test for sampler fields.

## Task 2: Implement Global Label Eligibility and Batch Planning

Files:
- `src/humpback/sequence_models/contrastive_loss.py`
- `src/humpback/sequence_models/masked_transformer.py`
- `tests/sequence_models/test_contrastive_loss.py`
- `tests/sequence_models/test_masked_transformer.py`

Implementation notes:
- Add a pure helper that computes globally eligible labels from the full train split using `contrastive_min_events_per_label` and `contrastive_min_regions_per_label`.
- Add a deterministic `build_contrastive_epoch_batches()` helper that returns `list[list[int]]` rather than a flat permutation.
- For each batch, choose up to `contrastive_labels_per_batch` eligible labels.
- For each selected label, choose up to `contrastive_events_per_label` labeled events, preferring distinct regions when `contrastive_region_balance=true`.
- Ensure each chosen label has at least two same-label events in the batch when possible, and at least two regions when cross-region positives exist.
- Fill remaining batch capacity with unlabeled or rare-label examples up to `contrastive_max_unlabeled_fraction`; preserve the rest for masked-only batches after labeled batches.
- Keep deterministic behavior under `seed`.

Acceptance criteria:
- Eligible labels are computed globally, not re-derived solely from local batch support.
- A label with enough global support can form positives even if some individual batches contain fewer than the old threshold count.
- Batches with eligible labels include same-label positives by construction whenever the train split contains them.
- Rare-label and unlabeled examples remain in training for masked modeling.
- Existing non-contrastive training still uses ordinary random permutation batching.

Tests:
- Pure sampler test: given multi-region labels, every contrastive batch contains at least one valid anchor.
- Pure sampler test: region balancing prefers cross-region same-label examples.
- Pure sampler test: unlabeled fill never exceeds `contrastive_max_unlabeled_fraction`.
- Training regression: skipped contrastive batches drop to zero for a synthetic dataset with enough eligible labels.
- Backward compatibility: contrastive-disabled callers preserve current loss behavior.

## Task 3: Make Contrastive Mask Thresholds Compatible With Planned Batches

Files:
- `src/humpback/sequence_models/contrastive_loss.py`
- `src/humpback/sequence_models/masked_transformer.py`
- `tests/sequence_models/test_contrastive_loss.py`

Implementation notes:
- Keep support thresholds as global eligibility gates.
- Add an optional `eligible_labels` argument to `build_contrastive_masks()` and `supervised_contrastive_loss()`.
- When `eligible_labels` is provided, masks should filter labels against that set rather than recomputing support from the current batch.
- Keep the old behavior when `eligible_labels` is omitted so direct unit callers remain compatible.

Acceptance criteria:
- Batch-level masks can produce positives for globally eligible labels even when the batch itself has fewer than `contrastive_min_events_per_label` total events for that label.
- Same-label intersection, related-label negative exclusions, and cross-region-positive preference remain unchanged.
- Zero-positive batches still return zero contrastive loss safely.

Tests:
- Unit test showing a globally eligible label with two in-batch examples yields positives even when `min_events_per_label=4`.
- Unit test showing labels not in `eligible_labels` are excluded even if they appear in the batch.
- Existing contrastive-loss tests continue to pass unchanged.

## Task 4: Expand Loss-Curve Diagnostics

Files:
- `src/humpback/sequence_models/masked_transformer.py`
- `src/humpback/workers/masked_transformer_worker.py`
- `tests/sequence_models/test_masked_transformer.py`
- `tests/workers/test_masked_transformer_worker.py`

New loss-curve fields:
- `train_contrastive_valid_batches`
- `train_contrastive_valid_anchor_count`
- `train_contrastive_positive_pair_count`
- `train_contrastive_eligible_label_count`
- `train_contrastive_labeled_event_count`
- `train_contrastive_unlabeled_fill_count`

Acceptance criteria:
- Existing `train`, `val`, `train_masked`, `train_contrastive`, and skipped-batch fields remain.
- Diagnostics are JSON-friendly numeric arrays aligned with `epochs`.
- For contrastive-disabled jobs, new contrastive diagnostics are zero arrays.
- Worker artifact tests assert the new fields are present for contrastive-enabled runs.

Tests:
- Unit test checking all arrays align to epoch count.
- Training regression test showing valid batch count plus skipped batch count equals train batch count.
- Worker artifact test for the new JSON keys.

## Task 5: Consolidate Create UI Into Meaningful Sections

Files:
- `frontend/src/components/sequence-models/MaskedTransformerCreateForm.tsx`
- `frontend/e2e/sequence-models/masked-transformer.spec.ts`
- `docs/reference/frontend.md`

Recommended sections:
- Source: upstream embedding job, Classify binding, `k_values`.
- Model: preset, dropout, mask fraction, span lengths, mask-weight bias.
- Retrieval Head: enable projection head, dimensions, L2 normalization.
- Training Windows: region/event-centered/mixed mode, event fraction, pre/post context.
- Contrastive Learning: enable human-correction contrastive loss, loss weight, temperature, support thresholds, cross-region-positive flag, sampler settings.
- Run Controls: `batch_size`, `max_epochs`, early stop patience, validation split, seed.

UI behavior:
- Keep contrastive disabled by default.
- Enabling contrastive still requires retrieval head and switches region mode to mixed `0.7`.
- Show sampler settings only when contrastive is active.
- Default sampler values should match backend defaults.
- Keep all numeric fields as inputs with validation states.

Acceptance criteria:
- Submit payload includes sampler settings when contrastive is active.
- Default create payload remains backward-compatible and contrastive-disabled.
- The form no longer presents one undifferentiated advanced grid.
- Controls remain compact and scan-friendly; no explanatory marketing text inside the app.

Tests:
- Playwright test for default submission.
- Playwright test for contrastive submission with sampler overrides.
- Playwright test that contrastive enablement switches region mode to mixed `0.7`.
- Playwright test for disabled/invalid sampler values blocking submit.

## Task 6: Add Docs and Behavioral Notes

Files:
- `docs/reference/data-model.md`
- `docs/reference/sequence-models-api.md`
- `docs/reference/frontend.md`

Acceptance criteria:
- Data model documents new sampler fields and defaults.
- API reference explains that support thresholds are global eligibility gates and sampler fields control per-batch composition.
- Frontend reference documents the sectioned create form and contrastive defaults.
- Notes explicitly say full-region extraction artifacts remain unchanged.

## Task 7: Verification

Required checks:
- `uv run ruff format --check src tests alembic`
- `uv run ruff check src tests alembic`
- `uv run pyright`
- `uv run pytest tests/sequence_models/test_contrastive_loss.py tests/sequence_models/test_masked_transformer.py tests/services/test_masked_transformer_service.py tests/unit/test_sequence_models_schemas.py tests/integration/test_masked_transformer_api.py tests/workers/test_masked_transformer_worker.py`
- `uv run pytest tests/`
- `cd frontend && npm run build`
- `cd frontend && npx playwright test e2e/sequence-models/masked-transformer.spec.ts`

Manual validation:
- Create a contrastive job from the UI with defaults.
- Confirm submitted payload uses mixed windows, `batch_size=16` if selected, and sampler defaults.
- After completion, inspect `loss_curve.json` and require fewer skipped contrastive batches than the current batch-size-16 baseline of 14 skipped batches per epoch.
- Run Phase 0 diagnostics and compare `exclude_same_event_and_region` retrieval raw/whitened metrics against the baseline report for job `63b72897-fb98-44d3-ac2d-2354a4d3f515`.

## Non-Goals

- Do not change full-region artifact alignment or per-k tokenization output contracts.
- Do not use Classify model labels as contrastive positives.
- Do not add hard-negative mining in this pass; region-balanced positives should land first.
- Do not remove the ability to train masked-only or retrieval-head-only jobs.
