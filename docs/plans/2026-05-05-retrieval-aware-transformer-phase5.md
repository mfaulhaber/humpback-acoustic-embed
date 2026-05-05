# Retrieval-Aware Transformer Phase 5 Implementation Plan

**Goal:** Add repeatable retrieval-aware sweep and comparison tooling that submits or compares masked-transformer research runs and ranks them by cross-region same-authoritative-human-label retrieval metrics.
**Spec:** [docs/specs/2026-05-05-retrieval-aware-transformer-training-design.md](../specs/2026-05-05-retrieval-aware-transformer-training-design.md)

---

## Current Label Correction

The current research Classify job has human-corrected authoritative single labels per effective event. Phase 5 should therefore rank by same authoritative human label, not by broad model Classify label sets and not by multi-label set intersection as the expected case. Existing set-capable diagnostic plumbing can remain for compatibility, but sweep summaries must surface whether any event has zero or more than one human label so research comparisons do not silently mix label semantics.

No database migration is expected for this phase.

---

## Initial Sweep Experiments

Recent session analysis produced three anchor observations that should become the first supported Phase 5 sweep set.

- Stage 0 250 ms contextual baseline, job `9fd95e63-9f06-4cfb-8242-63a03dbbedd0`, `k=150`: cross-region `exclude_same_event_and_region` metrics were raw `24.8%`, remove-PC10 `31.8%`, and whitened `40.2%` same-human-label overlap. This remains the historical baseline for the retrieval objective.
- Pre-sampler 250 ms contrastive run, job `63b72897-fb98-44d3-ac2d-2354a4d3f515`: contrastive loss was active, but the run skipped 14 contrastive batches per epoch and raw retrieval embeddings underperformed contextual embeddings. This is the failure-mode baseline for sampler and lambda changes.
- First completed 100 ms CRNN chunk run, Masked Transformer job `5e160936-2f5a-4a10-9311-452d818d8ac9`, upstream CEJ `42900b68-d830-40e0-af4b-e8f0a20456e7`: completed with `40,099` chunks, retrieval head enabled, event-centered human-correction contrastive training, `batch_size=4`, train loss `0.181`, val loss `0.299`, and 15 epochs. Cross-region `exclude_same_event_and_region` diagnostics showed retrieval raw `18.4%`, retrieval whitened `34.2%`, contextual raw `24.8%`, and contextual whitened `50.2%` same-human-label overlap. This proves the 100 ms path is viable but also shows raw retrieval geometry is not yet beating contextual + whitening.

The first sweep matrix should be intentionally small, ordered, and stop-aware:

1. **Baseline re-report:** Re-run comparison reports for the historical 250 ms baseline, the pre-sampler contrastive baseline, and the completed 100 ms job with the same diagnostic options, including chunk-level and event-level mean-pooled results. Purpose: make every later run comparable in one output artifact.
2. **Sampler confirmation at 250 ms:** Submit one post-sampler 250 ms contrastive job with the current UI/backend defaults: retrieval head enabled, mixed training windows with `event_centered_fraction=0.7`, human-correction contrastive loss `0.10`, temperature `0.07`, `batch_size=16`, labels per batch `4`, events per label `4`, max unlabeled fill `0.25`, region balance on, support thresholds `4` events and `2` regions. Purpose: confirm skipped contrastive batches fall below the old 14-per-epoch baseline and see whether raw retrieval improves.
3. **Lambda sweep at 250 ms:** If the sampler confirmation has valid contrastive batches, run loss weights `0.05`, `0.10`, `0.25`, and `0.50` with all other sampler settings fixed. Purpose: find whether stronger or weaker contrastive pressure improves raw retrieval without damaging contextual/whitened baselines.
4. **Context-window sweep at 250 ms:** For the best lambda from the previous step, compare mixed event context `2s/2s` versus `4s/4s`, keeping `event_centered_fraction=0.7`. Purpose: test whether more bout context helps or reintroduces region leakage.
5. **100 ms memory-safe confirmation:** Submit one 100 ms job using the same upstream geometry as CEJ `42900b68-d830-40e0-af4b-e8f0a20456e7`, but keep extraction memory safe: `batch_size=4`, labels per batch `2`, events per label `2`, and require the implementation or operator notes to avoid full-region MPS batches larger than the safe path discovered during job `56d5700a-0f1b-45c2-9c30-a8151757c6fa`. Purpose: confirm 100 ms signal is repeatable without the MPS OOM path.
6. **100 ms lambda mini-sweep:** Only after the memory-safe confirmation completes, run `0.05`, `0.10`, and `0.25` at 100 ms. Skip `0.50` until one lower-weight run improves raw retrieval or event-level retrieval, because the first completed 100 ms run already lagged contextual + whitening.
7. **Policy ablation:** On the best 250 ms and best 100 ms lambda runs, compare `require_cross_region_positive=true` versus `false` and default related-label exclusions versus empty exclusions. Purpose: measure whether the related-label policy is protecting motif-adjacent labels or hiding useful negatives.

First-sweep ranking should use `exclude_same_event_and_region` and prefer raw retrieval same-human-label overlap, but every report must include contextual raw and contextual whitened columns. A run is not considered an improvement unless raw retrieval moves closer to or past contextual raw while not losing more than five absolute percentage points against contextual whitened.

---

### Task 1: Add a Reusable Sweep Planning Module

**Files:**
- Create: `src/humpback/sequence_models/retrieval_sweeps.py`
- Create: `tests/sequence_models/test_retrieval_sweeps.py`

**Acceptance criteria:**
- [ ] The module defines typed sweep configuration objects for baseline job settings, contrastive loss weight values, supported negative or related-label policy variants, k values, diagnostic options, and output paths.
- [ ] Sweep expansion is deterministic and produces stable run names from normalized config values.
- [ ] Lambda sweep values default to the spec values `0.05`, `0.10`, `0.25`, and `0.50`, while allowing caller-provided values.
- [ ] Policy sweep support is limited to currently accepted masked-transformer create fields, such as related-label policy, require-cross-region-positive behavior, and sampler settings; true hard-negative policies that do not yet exist in the job schema are rejected with a clear message.
- [ ] Expanded create payloads preserve existing service invariants: retrieval head enabled for contrastive runs, human-correction label source for positive contrastive weight, event-centered or mixed sequence construction, and `k_values` excluded from job identity.
- [ ] The module records the corrected label assumption in run metadata as `label_semantics="authoritative_single_human_label"` unless diagnostics later observe otherwise.
- [ ] The module can emit the initial sweep matrix above as a named preset without requiring the caller to hand-enter every run.
- [ ] The preset records source job references, baseline metrics, and the stop rules used to decide whether later 100 ms or high-lambda runs should proceed.

**Tests needed:**
- Unit tests for deterministic sweep expansion and run naming.
- Unit tests for lambda defaults and caller overrides.
- Unit tests proving unsupported hard-negative policy fields are rejected instead of silently ignored.
- Unit tests proving generated contrastive payloads satisfy `MaskedTransformerJobCreate` validation.
- Unit tests proving the initial sweep preset expands in the documented order and carries baseline job ids.

---

### Task 2: Add a Sweep Submission CLI

**Files:**
- Create: `scripts/masked_transformer_retrieval_sweep.py`
- Modify: `scripts/README.md`
- Create or modify: `tests/scripts/test_masked_transformer_retrieval_sweep.py`

**Acceptance criteria:**
- [ ] The script supports a submit mode that loads settings, opens an async session, and calls `create_masked_transformer_job()` rather than inserting rows directly.
- [ ] Submit mode accepts upstream continuous embedding job id, optional Classify binding, k values, preset/training defaults, lambda values, related-label policy variants, sampler overrides, seed values, and dry-run mode.
- [ ] Dry-run mode prints deterministic planned run rows without creating jobs.
- [ ] Non-dry-run mode creates or reuses jobs idempotently and writes a JSON manifest containing run name, requested config, resolved job id, created or reused status, and label semantics.
- [ ] The script can optionally extend k sweeps on completed reused jobs through the existing `extend_k_sweep_job()` service path.
- [ ] The CLI never starts worker loops itself; it reports queued or existing job ids for the existing worker system to process.
- [ ] The script supports `--preset initial-retrieval-aware-sweep` for dry-run and submit modes, and marks runs that are blocked by unmet stop rules as planned-but-not-submitted.

**Tests needed:**
- CLI parser tests for submit arguments and defaults.
- Dry-run test using a temporary output path and no database mutation.
- Service-monkeypatched submit test proving create and extend-k-sweep service functions are called with normalized payloads.
- Error test for positive contrastive lambda without a Classify binding when the service cannot resolve one.
- Dry-run test for the initial sweep preset showing baseline comparison rows plus the first runnable submit rows.

---

### Task 3: Add Deterministic Comparison and Ranking

**Files:**
- Modify: `src/humpback/sequence_models/retrieval_sweeps.py`
- Modify: `scripts/masked_transformer_retrieval_sweep.py`
- Modify: `tests/sequence_models/test_retrieval_sweeps.py`
- Modify: `tests/scripts/test_masked_transformer_retrieval_sweep.py`

**Acceptance criteria:**
- [ ] The module provides a compare workflow that accepts explicit masked-transformer job ids or a submit manifest.
- [ ] Compare mode calls `build_nearest_neighbor_report()` for completed jobs and does not duplicate nearest-neighbor math.
- [ ] Default diagnostic options use retrieval embedding space, the required retrieval modes, the required embedding variants, a fixed seed, and a configured k.
- [ ] Baseline contextual jobs can be included by explicitly setting their embedding space to contextual in the comparison manifest.
- [ ] Ranking primary key is cross-region same authoritative human label overlap from `exclude_same_event_and_region` and `raw_l2`.
- [ ] Secondary columns include retrieval `remove_pc1`, `remove_pc3`, `remove_pc5`, `remove_pc10`, `whiten_pca`, same event rate, same region rate, similar duration rate, random-pair cosine percentiles, and good/mixed/bad verdict counts.
- [ ] Jobs with zero human-labeled query coverage, missing retrieval artifacts, incomplete status, or unavailable k values are kept in the output with a failure status instead of aborting the whole comparison.
- [ ] The comparison manifest records label coverage counts, including single-label event count, unlabeled event count, and multi-label event count.
- [ ] Compare mode can ingest known baseline metric rows for historical jobs when the operator wants the first report to include already-observed session metrics before rerunning diagnostics.
- [ ] Compare mode highlights the first-sweep go/no-go checks: skipped contrastive batches, raw retrieval versus contextual raw, raw retrieval versus contextual whitened, and event-level mean-pooled retrieval.

**Tests needed:**
- Unit tests for ranking order with synthetic diagnostic responses.
- Unit tests proving failed jobs remain represented with error status.
- Unit tests proving same-authoritative-label ranking uses the cross-region mode and raw retrieval variant.
- Regression test showing a response with multi-label rows is flagged in coverage rather than treated as the expected sweep label shape.
- Unit test for first-sweep stop-rule evaluation from synthetic comparison rows.

---

### Task 4: Write Markdown and CSV Research Outputs

**Files:**
- Modify: `src/humpback/sequence_models/retrieval_sweeps.py`
- Modify: `scripts/masked_transformer_retrieval_sweep.py`
- Create: `tests/sequence_models/test_retrieval_sweep_outputs.py`

**Acceptance criteria:**
- [ ] Compare mode writes `comparison.csv`, `comparison.md`, and `comparison.json` to a caller-specified output directory.
- [ ] CSV output is stable under seed and uses one row per job/config/k/embedding-space comparison.
- [ ] Markdown output includes a compact ranked table, label coverage summary, metric definitions, diagnostic options, and any failure rows.
- [ ] JSON output preserves the full normalized comparison payload for later notebook or script use.
- [ ] Output filenames are deterministic unless the caller requests timestamped output.
- [ ] Output rendering does not include large neighbor detail rows unless explicitly requested.

**Tests needed:**
- Snapshot-style tests for CSV and Markdown from small synthetic comparison rows.
- JSON serialization test proving no Path, NumPy, or datetime objects leak into the artifact.
- Determinism test for repeated compare runs with the same input order and seed.

---

### Task 5: Align Diagnostic Label Semantics for Sweep Reporting

**Files:**
- Modify: `src/humpback/sequence_models/retrieval_diagnostics.py`
- Modify: `src/humpback/sequence_models/contrastive_labels.py`
- Modify: `tests/sequence_models/test_retrieval_diagnostics.py`
- Modify: `tests/sequence_models/test_contrastive_loss.py`

**Acceptance criteria:**
- [ ] Diagnostic metadata clearly distinguishes authoritative human labels from model Classify labels in field names or documentation comments.
- [ ] Label coverage includes counts for unlabeled, single-label, and multi-label human-corrected effective events.
- [ ] Current single-label human-corrected events produce the same same-label metric as set intersection, but the report explicitly marks the observed label cardinality.
- [ ] Model-only Classify labels remain excluded from retrieval sweep positives and contrastive positives.
- [ ] Existing multi-label compatibility tests remain valid, but are framed as defensive compatibility rather than the expected Phase 5 research case.
- [ ] The comparison layer uses the new label-cardinality fields to warn when a run is not evaluating the intended single-label corrected dataset.

**Tests needed:**
- Unit test for all single-label corrected events reporting zero multi-label events.
- Unit test for a mixed single-label and multi-label fixture reporting the multi-label count.
- Regression test proving model-only labels still do not create human-label coverage.
- Regression test proving same-label metrics are unchanged for single-label corrected fixtures.

---

### Task 6: Update Documentation

**Files:**
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `docs/specs/2026-05-05-retrieval-aware-transformer-training-design.md`
- Modify: `scripts/README.md`

**Acceptance criteria:**
- [ ] Sequence Models API docs describe the sweep helper workflow and clarify that nearest-neighbor diagnostics remain the source of comparison metrics.
- [ ] Behavioral constraints state that Phase 5 sweep ranking uses authoritative human-corrected labels and rejects model-only Classify positives.
- [ ] Storage or script docs identify where sweep manifests and comparison outputs are written.
- [ ] The retrieval-aware transformer design keeps the corrected single-label research assumption visible near Phase 5.
- [ ] Documentation says true hard-negative training policies are not added by Phase 5 unless Phase 4 schema support lands first.

**Tests needed:**
- Documentation-only review for consistency with implemented CLI flags, output filenames, and label semantics.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/retrieval_sweeps.py src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/sequence_models/contrastive_labels.py scripts/masked_transformer_retrieval_sweep.py tests/sequence_models/test_retrieval_sweeps.py tests/sequence_models/test_retrieval_sweep_outputs.py tests/sequence_models/test_retrieval_diagnostics.py tests/sequence_models/test_contrastive_loss.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
2. `uv run ruff check src/humpback/sequence_models/retrieval_sweeps.py src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/sequence_models/contrastive_labels.py scripts/masked_transformer_retrieval_sweep.py tests/sequence_models/test_retrieval_sweeps.py tests/sequence_models/test_retrieval_sweep_outputs.py tests/sequence_models/test_retrieval_diagnostics.py tests/sequence_models/test_contrastive_loss.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
3. `uv run pyright src/humpback/sequence_models/retrieval_sweeps.py src/humpback/sequence_models/retrieval_diagnostics.py src/humpback/sequence_models/contrastive_labels.py scripts/masked_transformer_retrieval_sweep.py`
4. `uv run pytest tests/sequence_models/test_retrieval_sweeps.py tests/sequence_models/test_retrieval_sweep_outputs.py tests/sequence_models/test_retrieval_diagnostics.py tests/sequence_models/test_contrastive_loss.py tests/scripts/test_masked_transformer_retrieval_sweep.py`
5. `uv run pytest tests/`
