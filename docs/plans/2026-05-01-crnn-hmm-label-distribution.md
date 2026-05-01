# CRNN-Source HMM Label Distribution Implementation Plan

**Goal:** Enable CRNN-source HMM jobs to render the state-to-label distribution chart on the detail page by extending the Phase 1 source-agnostic loader Protocol with a label-distribution method, emitting a unified tier-aware JSON shape on disk, and removing the CRNN skip in the regenerate endpoint.

**Spec:** [docs/specs/2026-05-01-crnn-hmm-label-distribution-design.md](../specs/2026-05-01-crnn-hmm-label-distribution-design.md)

---

### Task 1: Extend loader Protocol with `LabelDistributionInputs` and `load_label_distribution_inputs()`

**Files:**
- Modify: `src/humpback/sequence_models/loaders/__init__.py`

**Acceptance criteria:**
- [ ] New `LabelDistributionInputs` dataclass with fields `hydrophone_id: int | None`, `state_rows: list[dict[str, Any]]`, `tier_per_row: list[str] | None`.
- [ ] `SequenceArtifactLoader` Protocol gains `async def load_label_distribution_inputs(self, session, storage_root, hmm_job, cej) -> LabelDistributionInputs`.
- [ ] Existing `OverlayInputs`, `_LOADERS` registry, and `get_loader()` are unchanged.
- [ ] Module exports both `LabelDistributionInputs` and the new Protocol method without breaking existing imports.

**Tests needed:**
- Verify the Protocol is satisfied by both loaders after Tasks 2 and 3 land (caught by pyright; no new test file required).

---

### Task 2: Implement `load_label_distribution_inputs()` on `SurfPerchLoader`

**Files:**
- Modify: `src/humpback/sequence_models/loaders/surfperch.py`

**Acceptance criteria:**
- [ ] `SurfPerchLoader.load_label_distribution_inputs()` traverses `cej.event_segmentation_job_id → EventSegmentationJob → region_detection_job_id → RegionDetectionJob → hydrophone_id`.
- [ ] Reads `states.parquet` (via `hmm_sequence_states_path`); builds `state_rows` with `start_timestamp` (float), `end_timestamp` (float), `viterbi_state` (int) keys.
- [ ] Returns `tier_per_row=None`.
- [ ] Raises `ValueError` if the EventSegmentationJob or RegionDetectionJob is missing (mirrors current service behavior).

**Tests needed:**
- Unit test in `tests/sequence_models/` covering the happy path (returns expected `hydrophone_id` and aligned `state_rows`).
- Unit test for missing ESJ/RDJ raising `ValueError`.

---

### Task 3: Implement `load_label_distribution_inputs()` on `CrnnRegionLoader`

**Files:**
- Modify: `src/humpback/sequence_models/loaders/crnn_region.py`

**Acceptance criteria:**
- [ ] `CrnnRegionLoader.load_label_distribution_inputs()` traverses `cej.region_detection_job_id → RegionDetectionJob → hydrophone_id` directly (no EventSegmentationJob).
- [ ] Reads `states.parquet` including the `tier` column; builds `state_rows` aligned with `tier_per_row` in the same parquet row order.
- [ ] `state_rows` carries `start_timestamp` / `end_timestamp` / `viterbi_state` (matching SurfPerch's shape).
- [ ] `tier_per_row` is a `list[str]` with the same length as `state_rows`; values are the per-chunk tier strings (`event_core` / `near_event` / `background`).
- [ ] Raises `ValueError` if `cej.region_detection_job_id` is `None` or the RDJ is missing.

**Tests needed:**
- Unit test for the happy path with a synthetic RDJ + hydrophone.
- Unit test verifying `tier_per_row` length and ordering match `state_rows`.
- Unit test for the missing-RDJ failure mode.

---

### Task 4: Extend `compute_label_distribution()` with `tier_per_row` parameter

**Files:**
- Modify: `src/humpback/sequence_models/label_distribution.py`

**Acceptance criteria:**
- [ ] Signature gains `tier_per_row: list[str] | None = None` as the final parameter.
- [ ] Output shape is always `{state: {tier: {label: count}}}` (nested), never flat.
- [ ] When `tier_per_row` is `None`, every row buckets to a synthetic `"all"` tier key.
- [ ] When `tier_per_row` is provided, each row buckets to its own tier value.
- [ ] `"unlabeled"` is incremented per (state, tier) when no detection-window match exists for that row.
- [ ] When `tier_per_row` is provided, raises `ValueError` if `len(tier_per_row) != len(states)`.
- [ ] `total_windows: int` and `n_states: int` are preserved in the output dict.

**Tests needed:**
- Update `tests/sequence_models/test_label_distribution.py`:
  - Existing flat-output tests rewritten to assert the new nested shape under the `"all"` tier.
  - New test: passing `tier_per_row` produces correctly bucketed nested output.
  - New test: a state with rows in multiple tiers produces multiple inner tier keys, each with independent label counts.
  - New test: `len(tier_per_row) != len(states)` raises `ValueError`.

---

### Task 5: Refactor `generate_label_distribution()` in service to be source-agnostic

**Files:**
- Modify: `src/humpback/services/hmm_sequence_service.py`

**Acceptance criteria:**
- [ ] `generate_label_distribution()` resolves `cej`, derives `source_kind = source_kind_for(cej.model_version)`, and calls `get_loader(source_kind)`.
- [ ] Calls `await loader.load_label_distribution_inputs(...)` to obtain `LabelDistributionInputs`.
- [ ] Inline ESJ/RDJ traversal removed; inline `states.parquet` read removed.
- [ ] DetectionJob + VocalizationLabel SQL fetch retained in the service (uses `inputs.hydrophone_id`).
- [ ] Calls `compute_label_distribution(inputs.state_rows, det_windows, labels, job.n_states, tier_per_row=inputs.tier_per_row)`.
- [ ] Atomic write to `hmm_sequence_label_distribution_path()` is preserved (same write contract).
- [ ] Function returns the unified-shape dict.

**Tests needed:**
- Update `tests/services/test_hmm_sequence_service.py` to assert nested-shape output for both SurfPerch and CRNN paths.
- New test: CRNN-source HMM job produces nested output with real tier keys.
- New test: SurfPerch-source HMM job produces nested output with `"all"` tier keys.

---

### Task 6: Update `LabelDistributionResponse` Pydantic schema

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`

**Acceptance criteria:**
- [ ] `LabelDistributionResponse.states` becomes `dict[str, dict[str, dict[str, int]]]`.
- [ ] `n_states` and `total_windows` fields are unchanged.
- [ ] No other schema changes in this file.

**Tests needed:**
- Pydantic validation exercised via the integration tests in Task 8.

---

### Task 7: Add read-time legacy adapter on `GET /label-distribution`

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`

**Acceptance criteria:**
- [ ] Reads `label_distribution.json`; before returning, projects any state whose value is a flat `dict[str, int]` to `{"all": that_dict}`.
- [ ] Detection rule: a state's value is "legacy flat" if its first inner value is `int` (not `dict`); unified shape's first inner value is `dict`.
- [ ] Adapter is a small private helper near the existing `_project_legacy_overlay_row` / `_project_legacy_exemplar_row` helpers.
- [ ] Comment names the adapter as transitional.
- [ ] On-disk file is never rewritten by this code path.

**Tests needed:**
- Integration test in `tests/integration/test_sequence_models_api.py`: GET on a job with a synthetic flat-shape JSON on disk returns the unified shape; the file's bytes are unchanged after the GET.

---

### Task 8: Remove CRNN skip in regenerate endpoint

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`

**Acceptance criteria:**
- [ ] `POST /hmm-sequences/{job_id}/generate-interpretations` always calls `generate_label_distribution()` regardless of source kind.
- [ ] The `if source_kind_for(cej.model_version) != SOURCE_KIND_REGION_CRNN:` branch and the related multi-line comment (lines ~497–504) are removed.
- [ ] `label_distribution_generated` is either removed from the response or kept as a constant `True` for response-shape stability — pick one and apply consistently.
- [ ] If kept, frontend code reading `label_distribution_generated` is unaffected; if removed, all frontend readers updated in Task 11.

**Tests needed:**
- Integration test: POST regenerate on a CRNN-source HMM job returns 200 and writes a valid unified-shape JSON to disk.
- Integration test: POST regenerate on a SurfPerch-source HMM job continues to return 200 and writes unified shape (rewriting any legacy flat file in unified format).

---

### Task 9: Update frontend `LabelDistribution` TS type and chart `useMemo` collapse

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`

**Acceptance criteria:**
- [ ] `LabelDistribution.states: Record<string, Record<string, Record<string, number>>>`.
- [ ] `LabelDistributionChart`'s `useMemo` collapses the inner tier dimension into a flat `Record<string, number>` per state before building Plotly traces.
- [ ] Collapse logic: for each state, sum counts across all tier keys per label; pass the resulting `{label: count}` dict into the existing trace-building loop.
- [ ] No visual change to the chart for SurfPerch jobs.
- [ ] CRNN jobs render the same single-stack chart with bars summed across tiers.
- [ ] `data-testid="hmm-label-distribution"` and Plotly layout are preserved.

**Tests needed:**
- Frontend component test (Playwright or component-level): renders both a SurfPerch-shape payload (single `"all"` tier) and a CRNN-shape payload (multiple tier keys); asserts identical bar totals when CRNN tier counts sum to the same labels.

---

### Task 10: Verify `frontend/src/api/types.ts` references

**Files:**
- Modify (if needed): `frontend/src/api/types.ts`

**Acceptance criteria:**
- [ ] Confirm whether `label_distribution: Record<string, number>` references at `types.ts:570` / `types.ts:576` are HMM-related or labeling-related.
- [ ] If HMM-related, update to the unified nested type and align with Task 9's TS type.
- [ ] If labeling-related (per the spec's note), no change.

**Tests needed:**
- TypeScript compilation (`npx tsc --noEmit`) covers this; no new test file.

---

### Task 11: Documentation and ADR-060

**Files:**
- Modify: `CLAUDE.md` (§9.1 Sequence Models bullet)
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] `CLAUDE.md` §9.1: strike "label distribution remains SurfPerch-only pending Phase 2"; add a one-line note that label distribution is unified across sources under ADR-060 with tier-aware on-disk storage.
- [ ] `docs/reference/sequence-models-api.md`: document the unified nested `LabelDistribution` shape, the `"all"` synthetic tier key, the read-time legacy adapter as transitional, and the unconditional regenerate behavior.
- [ ] `DECISIONS.md`: append `ADR-060: Source-agnostic HMM label distribution with tier-aware storage` covering decisions 1–4 from spec §9.

**Tests needed:**
- None (documentation only).

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/ src/humpback/services/hmm_sequence_service.py src/humpback/api/routers/sequence_models.py src/humpback/schemas/sequence_models.py`
2. `uv run ruff check src/humpback/sequence_models/ src/humpback/services/hmm_sequence_service.py src/humpback/api/routers/sequence_models.py src/humpback/schemas/sequence_models.py`
3. `uv run pyright src/humpback/sequence_models/ src/humpback/services/hmm_sequence_service.py src/humpback/api/routers/sequence_models.py src/humpback/schemas/sequence_models.py tests/sequence_models/ tests/services/test_hmm_sequence_service.py tests/integration/test_sequence_models_api.py`
4. `uv run pytest tests/sequence_models/ tests/services/test_hmm_sequence_service.py tests/integration/test_sequence_models_api.py`
5. `uv run pytest tests/`
6. `cd frontend && npx tsc --noEmit`
7. `cd frontend && npx playwright test` (full or scoped to the HMM detail page)
