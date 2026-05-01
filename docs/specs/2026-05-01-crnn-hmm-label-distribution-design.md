# CRNN-Source HMM Label Distribution — Design

**Status:** Approved (brainstorming complete 2026-05-01)
**Track:** Sequence Models (extends ADR-057, ADR-059)
**Phase:** 2 of 2 (label distribution; Phase 1 — overlay + exemplars — shipped under ADR-059)

---

## 1. Goal

CRNN-source HMM jobs produce a state-to-label distribution artifact and render it on the detail page using the existing `LabelDistributionChart`. The artifact is **tier-aware on disk** (each row buckets to its own CRNN tier); the chart sums tiers in `useMemo`, preserving the current visual for both sources. The richer tier-stratified data is preserved on disk for a future tier-aware UI without re-running jobs.

The interpretation pipeline's source-agnostic abstraction (introduced in Phase 1) extends to label distribution: each source's loader knows how to resolve its hydrophone and (optionally) attach a per-row tier; the join math is shared.

## 2. Background

ADR-059 made overlay + exemplars source-agnostic via a `SequenceArtifactLoader` Protocol with one impl per embedding source kind. The same ADR's §12 explicitly deferred CRNN label distribution to Phase 2 and noted: "Phase 2 will likely extend the loader Protocol with a parallel method or introduce a sibling `LabelDistributionLoader` Protocol — TBD when scoped."

Concrete deferred surfaces today:

- `generate_label_distribution()` in [`src/humpback/services/hmm_sequence_service.py`](src/humpback/services/hmm_sequence_service.py) traverses `HMM → CEJ → EventSegmentationJob → RegionDetectionJob → hydrophone_id`. The ESJ link does not exist on CRNN-source CEJs (CRNN CEJs carry `region_detection_job_id` directly).
- The regenerate endpoint at [`src/humpback/api/routers/sequence_models.py`](src/humpback/api/routers/sequence_models.py:497) carries an explicit `if source_kind != REGION_CRNN` skip with a comment naming the deferral.
- `LabelDistributionResponse.states: dict[str, dict[str, int]]` ([`schemas/sequence_models.py:388`](src/humpback/schemas/sequence_models.py:388)) and the matching frontend `LabelDistribution` type ([`frontend/src/api/sequenceModels.ts:769`](frontend/src/api/sequenceModels.ts:769)) are flat — no tier dimension.
- `compute_label_distribution()` in [`src/humpback/sequence_models/label_distribution.py`](src/humpback/sequence_models/label_distribution.py) takes `(states, detection_windows, labels, n_states)` only; no tier awareness.

CRNN `states.parquet` already carries per-chunk `tier`, `audio_file_id`, and start/end timestamps — every input the join needs is on disk. The CRNN loader path also already resolves `region_id → audio_file_id` (used by overlay + exemplars in Phase 1).

## 3. Scope

### In scope (Phase 2)

- Extend `SequenceArtifactLoader` Protocol with a second method `load_label_distribution_inputs()`.
- Add `LabelDistributionInputs` dataclass: `hydrophone_id`, `state_rows`, `tier_per_row` (optional list parallel to `state_rows`; `None` for sources without a tier dimension).
- Implement `load_label_distribution_inputs()` on `SurfPerchLoader` (lifts current ESJ→RDJ traversal from the service; returns `tier_per_row=None`) and `CrnnRegionLoader` (RDJ→hydrophone traversal; reads `tier` column from `states.parquet`).
- Extend pure `compute_label_distribution()` with an optional `tier_per_row` parameter; when `None`, every row buckets to a synthetic `"all"` tier key. Output shape is always nested: `{state: {tier: {label: count}}}`.
- Refactor `generate_label_distribution()` to be source-agnostic: resolve loader, fetch inputs, run shared `DetectionJob`/`VocalizationLabel` SQL, call the pure compute function, write JSON.
- Update `LabelDistributionResponse` Pydantic schema and the frontend `LabelDistribution` TS type to the unified nested shape.
- Update `LabelDistributionChart` to collapse the tier dimension in `useMemo` before building Plotly traces. **No visual change** in this PR.
- Remove the CRNN skip branch in `POST /generate-interpretations/{id}` and the related comment; `label_distribution_generated` is always `True`.
- Read-time legacy adapter at the API layer projects pre-PR flat-shaped SurfPerch JSON files to the unified nested shape on `GET /label-distribution`.
- ADR-060 in `DECISIONS.md`.

### Explicit non-goals (deferred)

- Tier filter / tier toggle UI on the chart. The data is preserved on disk for a future PR.
- Forced one-shot regeneration of pre-PR SurfPerch label distribution files. The legacy read-time adapter handles them lazily; the existing Refresh button rewrites in unified format on demand.
- Backfill of CRNN HMM jobs that completed before this PR. New jobs compute label distribution on first GET; old jobs re-render after Refresh, identical to the current SurfPerch behavior.
- Worker-time eager generation. Label distribution remains lazy via GET / eager via Refresh, matching today's SurfPerch behavior for both sources. (Overlay + exemplars are worker-time eager because they depend on the live `pca.joblib` from training; label distribution is purely a downstream join and is fine on demand.)
- Discriminated-union typing for tier keys. The synthetic `"all"` bucket and the CRNN tier strings (`event_core` / `near_event` / `background`) live as string keys; the validator (Pydantic + pyright) treats them as opaque dict keys.
- Click-through navigation from chart bars into the timeline.
- Performance optimization for the join itself. Same complexity as today's SurfPerch path; one extra dict lookup per row for tier bucketing.

## 4. Architecture

```
generate_label_distribution(session, storage_root, hmm_job)
   │
   ├─ resolve cej, source_kind = source_kind_for(cej.model_version)
   ├─ loader = get_loader(source_kind)
   ├─ inputs = await loader.load_label_distribution_inputs(session, storage_root, hmm_job, cej)
   │       ├─ SurfPerchLoader  → ESJ → RDJ → hydrophone_id; tier_per_row = None
   │       └─ CrnnRegionLoader → RDJ → hydrophone_id; tier_per_row = [...] from states.parquet
   │
   ├─ fetch DetectionJobs for inputs.hydrophone_id; build (DetectionWindow, LabelRecord) lists
   │       (kept in service — identical for both sources)
   │
   └─ dist = compute_label_distribution(
                 inputs.state_rows, det_windows, labels, n_states,
                 tier_per_row=inputs.tier_per_row,
             )
       └─ output shape: {n_states, total_windows, states: {state: {tier: {label: count}}}}
   ↓
   atomic write to label_distribution.json
```

**Hard rule (preserved from Phase 1):** Loader code is the only place that knows a source's parquet column names and entity-graph traversal. Downstream pure functions consume generic dataclasses and never branch on source kind.

## 5. New and Modified Modules

### Modified — `src/humpback/sequence_models/loaders/__init__.py`

Add to the existing module:

- `LabelDistributionInputs` dataclass with fields:
  - `hydrophone_id: int | None`
  - `state_rows: list[dict[str, Any]]` — each row carrying `start_timestamp` (float), `end_timestamp` (float), `viterbi_state` (int)
  - `tier_per_row: list[str] | None` — parallel to `state_rows`; `None` for SurfPerch
- Extend `SequenceArtifactLoader` Protocol with:
  - `async def load_label_distribution_inputs(self, session, storage_root, hmm_job, cej) -> LabelDistributionInputs`

The existing `_LOADERS` registry, `get_loader()`, and `OverlayInputs` dataclass are unchanged.

### Modified — `src/humpback/sequence_models/loaders/surfperch.py`

Add `load_label_distribution_inputs()`:

- Traverses `cej.event_segmentation_job_id → EventSegmentationJob → region_detection_job_id → RegionDetectionJob → hydrophone_id` (lifted verbatim from the current service code).
- Reads `states.parquet` for the HMM job; builds `state_rows` from `start_timestamp` / `end_timestamp` / `viterbi_state` columns.
- Returns `tier_per_row=None`.

### Modified — `src/humpback/sequence_models/loaders/crnn_region.py`

Add `load_label_distribution_inputs()`:

- Traverses `cej.region_detection_job_id → RegionDetectionJob → hydrophone_id` directly.
- Reads `states.parquet` including the `tier` column. CRNN `states.parquet` contains per-chunk rows already; build `state_rows` from `start_timestamp` / `end_timestamp` / `viterbi_state`, and `tier_per_row` from the `tier` column (parallel order).
- Returns `tier_per_row=[...]`.

Defensive: if `cej.region_detection_job_id is None`, raise `ValueError` (mirrors SurfPerch's missing-ESJ failure mode; service surfaces the same way).

### Modified — `src/humpback/sequence_models/label_distribution.py`

Extend `compute_label_distribution()` signature:

```
def compute_label_distribution(
    states: list[dict],
    detection_windows: list[DetectionWindow],
    labels: list[LabelRecord],
    n_states: int,
    tier_per_row: list[str] | None = None,
) -> dict
```

Behavior:

- `per_state: dict[str, dict[str, dict[str, int]]]` — outer = state, middle = tier key, inner = label.
- For each state row, bucket key is `tier_per_row[i]` if provided, else `"all"`.
- Increment is per-(state, tier, label) — including the `"unlabeled"` label when no detection-window match.
- Return shape: `{"n_states": int, "total_windows": int, "states": per_state}`.

If `tier_per_row` is provided, its length must equal `len(states)`; otherwise raise `ValueError`.

### Modified — `src/humpback/services/hmm_sequence_service.py`

Refactor `generate_label_distribution()`:

- Resolve `cej` and `source_kind = source_kind_for(cej.model_version)`.
- `loader = get_loader(source_kind)`.
- `inputs = await loader.load_label_distribution_inputs(session, storage_root, job, cej)`.
- Run the existing DetectionJob + VocalizationLabel SQL using `inputs.hydrophone_id` (kept here, not in the loader, since it's identical for both sources once `hydrophone_id` is known).
- Call `compute_label_distribution(inputs.state_rows, det_windows, labels, job.n_states, tier_per_row=inputs.tier_per_row)`.
- Atomic write to `hmm_sequence_label_distribution_path()` (unchanged write contract).

Remove the inline ESJ/RDJ traversal and the inline `states.parquet` read.

### Modified — `src/humpback/schemas/sequence_models.py`

`LabelDistributionResponse.states: dict[str, dict[str, dict[str, int]]]` (state → tier → label → count). `n_states` and `total_windows` unchanged.

### Modified — `src/humpback/api/routers/sequence_models.py`

- `GET /hmm-sequences/{job_id}/label-distribution`: read JSON; if a state's value type is `dict[str, int]` (legacy flat), project to `{"all": that_dict}` per state before validation. Adapter is ~10 lines, transitional, mirrors the Phase 1 overlay/exemplars adapter pattern. Disk file is never rewritten by this path.
- `POST /hmm-sequences/{job_id}/generate-interpretations`: drop the `if source_kind != SOURCE_KIND_REGION_CRNN` skip and the related comment block (lines ~497–504). Always call `generate_label_distribution()`. `label_distribution_generated` becomes a constant `True` in the response and can be removed entirely (or kept as `True` for response-shape stability — see Task 8 for the call).

### Modified — `frontend/src/api/sequenceModels.ts`

`LabelDistribution.states` type becomes `Record<string, Record<string, Record<string, number>>>`.

### Modified — `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`

`LabelDistributionChart`'s `useMemo` collapses the tier dimension before building traces. The collapse is a small reduction over each state's tier dict; no other rendering logic changes. The `data-testid="hmm-label-distribution"` attribute and Plotly layout are unchanged.

### Modified — `frontend/src/api/types.ts`

If `label_distribution: Record<string, number>` appears as a sub-type of any sequence-models payload, update accordingly. (The lines at `types.ts:570` / `types.ts:576` refer to `labeling.py` schemas, not HMM — verify and skip if unrelated.)

## 6. Data Model

No SQL schema changes. No Alembic migration. No `encoding_signature` changes. Label distribution is a downstream artifact of HMM training and does not participate in idempotency.

### `label_distribution.json` schema (unified, written by both source paths)

```json
{
  "n_states": 12,
  "total_windows": 8421,
  "states": {
    "0": {"all": {"song_unit": 17, "unlabeled": 102}},
    "5": {
      "event_core": {"song_unit": 14},
      "near_event": {"song_unit": 3},
      "background": {"unlabeled": 412}
    }
  }
}
```

- SurfPerch jobs: every state has exactly one inner-tier key, `"all"`.
- CRNN jobs: each state has one or more tier keys drawn from `{"event_core", "near_event", "background"}` (only tiers that actually appear in the chunks contributing to that state).
- The `"all"` synthetic key is reserved; future sources without a tier dimension reuse it. New sources with a real tier dimension pick distinct keys.

Pre-PR SurfPerch flat-shape JSON (with `states[state] = {label: count}` directly) remains on disk untouched and is projected to the unified shape by the API read-time adapter on each GET. The existing Refresh button rewrites the file in unified shape on demand.

## 7. Workflow

### New CRNN HMM job (post-PR)

1. Worker runs `_run_region_crnn_hmm()` end-to-end (overlay + exemplars are eager, label distribution is lazy — same model as SurfPerch today).
2. User opens the detail page; `useHMMLabelDistribution` triggers `GET /label-distribution`.
3. Endpoint sees `label_distribution.json` does not exist; calls `generate_label_distribution()`.
4. Service resolves the CRNN loader → `LabelDistributionInputs` with `hydrophone_id` and `tier_per_row`.
5. SQL fetches DetectionJobs/labels for that hydrophone; pure compute function emits unified-shape JSON.
6. Atomic write; response returns the unified payload.

### New SurfPerch HMM job (post-PR)

Same flow; `tier_per_row=None`; output uses `"all"` bucket; chart renders identically to today.

### Existing pre-PR SurfPerch HMM job (flat JSON on disk)

`GET /label-distribution` reads the flat file; adapter projects each state to `{"all": flat_dict}`; response is unified shape. Disk file unchanged. Refresh button rewrites in unified format on demand.

### Existing pre-PR CRNN HMM job (no JSON on disk; CRNN was previously skipped)

First GET computes via the new CRNN loader path; written in unified shape from the start.

### Refresh button

`POST /generate-interpretations/{id}` always runs label distribution regardless of source; the no-op-for-CRNN branch is removed. Output is unified shape.

## 8. Failure Modes (must be tested)

- CRNN job, every chunk in a state has `tier="background"` → that state has only `{"background": {...}}` in its inner dict; no `event_core` / `near_event` keys present.
- CRNN job, no detection labels exist on the hydrophone → every state has `{tier: {"unlabeled": N}}` entries (one inner key per tier that appears in that state).
- CRNN job, no detection jobs at all on the hydrophone → service's existing `if rdj.hydrophone_id:` guard short-circuits with empty `det_windows` / `labels` lists; output has every row bucketed under its tier with `"unlabeled"` as the only label.
- SurfPerch job (post-PR) → every state has exactly one inner key, `"all"`. Chart renders identically to today.
- Pre-PR SurfPerch flat JSON on disk → GET adapter projects to `{"all": ...}`; on-disk file unchanged.
- New SurfPerch job (post-PR) → on-disk file is unified shape; legacy adapter is a no-op for that file.
- CRNN HMM job whose `cej.region_detection_job_id is None` (defensive) → loader raises `ValueError`; service propagates; route returns 500. Symmetric to SurfPerch's missing-ESJ failure mode.
- `tier_per_row` length mismatch with `state_rows` → `compute_label_distribution()` raises `ValueError` (defensive, exercised in unit tests by passing a deliberately mismatched list).
- Frontend chart with mixed-tier CRNN data → `useMemo` collapse produces the same per-(state, label) totals as the equivalent flat shape would. Snapshot test validates equivalence.

## 9. Key Decisions

1. **Extend `SequenceArtifactLoader` Protocol with `load_label_distribution_inputs()`** rather than introducing a sibling Protocol. Same Protocol resolves a source's parquet schema (Phase 1) and its hydrophone-traversal path (Phase 2). One file per source still owns both. *(Q2 → A)*
2. **Always-tiered JSON shape on disk.** SurfPerch uses synthetic `"all"` bucket; CRNN uses real tier keys. Single reader, single Pydantic schema, single TS type. Pre-PR flat files handled by API-layer adapter. *(Q3 → i)*
3. **Frontend collapses tier dimension in `useMemo`.** No visual change in this PR; the data is preserved on disk for a future tier-aware chart PR. *(Q4 → a)*
4. **Chunk's own CRNN tier is the bucket key** for CRNN. Preserves the "what does this state look like under each tier?" question. The labeled detection window's identity does not influence the tier bucket. *(Q5 → a)*
5. **Service keeps the SQL fetch.** Only the source-specific traversal lives in the loader; the shared DetectionJob + VocalizationLabel query stays in the service.
6. **No worker change.** Label distribution remains lazy via GET / eager via Refresh, for both sources, matching today's SurfPerch behavior.
7. **Read-time legacy adapter at the API layer** for pre-PR flat SurfPerch JSON. No on-disk migration; Refresh button rewrites. Mirrors Phase 1.
8. **`compute_label_distribution()` raises on length mismatch.** Defensive — protects against future loaders accidentally returning misaligned `tier_per_row`. The cost (one length check) is negligible.

## 10. Risks and Mitigations

1. **Schema rename touches every test that constructs the legacy `dict[str, int]` payload.** *Mitigation:* mechanical rename caught by pyright; existing tests at `tests/sequence_models/test_label_distribution.py` and `tests/services/test_hmm_sequence_service.py` get updated in lock-step.
2. **Legacy adapter rot.** Same risk Phase 1 carries. *Mitigation:* a comment naming the adapter as transitional; revisit removal in a future cleanup PR after pre-PR SurfPerch jobs have been refreshed or deleted.
3. **Tier key collision.** A future source kind that uses `"all"` as a real tier key would collide with the synthetic SurfPerch bucket. *Mitigation:* `"all"` is documented as reserved.
4. **PCA staleness vs. training set.** Not relevant — label distribution does not project through PCA.
5. **Performance.** A single CRNN HMM job can have ~200k state rows. Same complexity class as SurfPerch; one extra dict lookup per row for tier bucketing. No new mitigation.

## 11. Documentation Updates (CLAUDE.md §10.2)

- `CLAUDE.md` §9.1 — strike "label distribution remains SurfPerch-only pending Phase 2" from the Sequence Models bullet; add a one-liner noting Phase 2 shipped under ADR-060.
- `docs/reference/sequence-models-api.md` — document the unified nested `LabelDistribution` shape, the `"all"` synthetic tier key, the legacy read-time adapter as transitional, and the unconditional regenerate behavior.
- `DECISIONS.md` — append `ADR-060: Source-agnostic HMM label distribution with tier-aware storage` covering decisions 1–4 from §9.
- No README change — purely internal abstraction + on-disk shape change.
- No Alembic migration — no SQL schema change.

## 12. Future Work (out of scope here)

- Tier filter / toggle UI on `LabelDistributionChart` (the data is already on disk).
- Per-state per-tier facet plots (one row per tier instead of one row per state).
- Click-through from a chart bar to a filtered state-timeline view.
- Removal of the read-time legacy flat-JSON adapter once all pre-PR SurfPerch jobs have been refreshed or deleted.
