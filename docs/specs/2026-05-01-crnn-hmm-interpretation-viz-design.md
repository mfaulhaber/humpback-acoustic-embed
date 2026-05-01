# CRNN-Source HMM Interpretation Visualizations — Design

**Status:** Approved (brainstorming complete 2026-05-01)
**Track:** Sequence Models (extends ADR-057)
**Phase:** 1 of 2 (overlay + exemplars; label distribution deferred to Phase 2)

---

## 1. Goal

CRNN-source HMM jobs render the same **PCA / UMAP Overlay** and **State Exemplars** panels on the detail page as SurfPerch-source jobs. The interpretation pipeline becomes source-agnostic via a `SequenceArtifactLoader` Protocol so future embedding sources slot in by adding one loader file.

Phase 2 (CRNN label distribution) lands separately and reuses the same abstraction.

## 2. Background

ADR-057 (CRNN region-based HMM sequence embedding source) explicitly deferred PCA/UMAP overlay, exemplars, and label distribution for CRNN-source HMM jobs as Phase 2 of that source's spec. The deferred path leaves three concrete artifacts unbuilt:

- `_run_region_crnn_hmm()` in `src/humpback/workers/hmm_sequence_worker.py` does **not** call `generate_interpretations()` (a placeholder `_ = source` line and a comment note the deferral).
- `_load_overlay_inputs()` in `src/humpback/services/hmm_sequence_service.py` is hardcoded to the SurfPerch parquet schema (`merged_span_id`, `window_index_in_span`).
- `OverlayPoint` and `ExemplarRecord` Pydantic schemas require `merged_span_id: int` and `window_index_in_span: int` — incompatible with CRNN region UUIDs and chunk indices.

`pca.joblib` is already persisted by both worker paths, so the inputs to overlay computation already exist on disk for CRNN-source jobs.

## 3. Scope

### In scope (Phase 1)

- New `SequenceArtifactLoader` Protocol with one impl per embedding source kind.
- Generic identifier model `(sequence_id: str, position_in_sequence: int)` replacing the SurfPerch-specific `(merged_span_id: int, window_index_in_span: int)` and CRNN-specific `(region_id: str, chunk_index_in_region: int)` field pairs throughout the abstraction layer.
- Source-specific metadata channel via `extras: dict[str, str | int | float | None]` on `WindowMeta` and `ExemplarRecord`. CRNN populates `extras["tier"]`; SurfPerch leaves it empty.
- Source-agnostic rewrite of `OverlayMetadata`, `WindowMeta`, `compute_overlay()`, `select_exemplars()`, and `generate_interpretations()`.
- Two concrete loaders: `SurfPerchLoader` (replaces existing inline logic) and `CrnnRegionLoader` (new).
- Worker change: `_run_region_crnn_hmm()` calls `generate_interpretations()` after summary write, with the same try/except + `logger.warning` swallow pattern the SurfPerch path uses.
- Read-time legacy adapter on `GET /hmm-sequences/{id}/overlay` and `/exemplars` so pre-PR SurfPerch artifacts (with the old column names) continue rendering. New jobs and refreshed old jobs produce the unified format.
- Frontend tier badge on exemplar cards when `record.extras?.tier` is present.
- ADR-059 in `DECISIONS.md`.

### Explicit non-goals (deferred to Phase 2 or later)

- CRNN-source label distribution (`generate_label_distribution()` extension). Tracked for Phase 2.
- Forced one-shot regeneration of pre-PR SurfPerch overlay/exemplar artifacts. The legacy read-time adapter handles them lazily.
- Backfill of CRNN HMM jobs that completed before this PR. The user has deleted them; new jobs run interpretations via the worker.
- Discriminated-union typing for `extras` (per-source-kind dataclasses). The `dict[str, str | int | float | None]` shape is explicit YAGNI — extensibility without type tax.
- Click-to-navigate behavior from overlay points or exemplar cards into the timeline.
- Performance optimization for large-region UMAP fitting (deferred until measured).

## 4. Architecture

```
                       ┌─────────────────────────────┐
                       │  generate_interpretations() │
                       │  (source-agnostic)          │
                       └────────────┬────────────────┘
                                    │
                       ┌────────────▼────────────────┐
                       │  get_loader(source_kind)    │
                       │  registry: { surfperch,     │
                       │              region_crnn }  │
                       └────────┬───────────┬────────┘
                                │           │
                ┌───────────────▼─┐       ┌─▼────────────────┐
                │ SurfPerchLoader │       │ CrnnRegionLoader │
                │  reads SurfPerch│       │  reads CRNN      │
                │  parquet schema │       │  parquet schema  │
                └─────────┬───────┘       └─────────┬────────┘
                          │                         │
                          └────────────┬────────────┘
                                       │
                              ┌────────▼─────────┐
                              │  OverlayInputs   │
                              │  (generic shape) │
                              └────────┬─────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
        ┌───────▼─────────┐                          ┌────────▼─────────┐
        │ compute_overlay │                          │ select_exemplars │
        │ → overlay.parquet                          │ → exemplars.json │
        └─────────────────┘                          └──────────────────┘
```

**Hard rule:** Loader code is the only place that knows a source's parquet column names. Downstream pure functions (`compute_overlay`, `select_exemplars`) consume the generic `OverlayInputs` and never branch on source kind.

## 5. New and Modified Modules

### New modules

- `src/humpback/sequence_models/loaders/__init__.py` — defines:
  - `SequenceArtifactLoader` Protocol with single method `load(storage_root, hmm_job, cej) -> OverlayInputs`.
  - `OverlayInputs` dataclass: `pca_model`, `raw_sequences: list[np.ndarray]`, `viterbi_states: list[np.ndarray]`, `max_probs: list[np.ndarray]`, `metadata: OverlayMetadata`, `window_metas: list[WindowMeta]`.
  - Internal `_LOADERS` registry keyed on `source_kind_for()` constants.
  - `get_loader(source_kind: str) -> SequenceArtifactLoader` raising `ValueError` on unknown source.
- `src/humpback/sequence_models/loaders/surfperch.py` — `SurfPerchLoader` impl. Body is the current `_load_overlay_inputs()` logic with column reads renamed to populate the generic identifier pair (`sequence_id = str(merged_span_id)`, `position_in_sequence = window_index_in_span`) and `extras = {}` on every `WindowMeta`.
- `src/humpback/sequence_models/loaders/crnn_region.py` — `CrnnRegionLoader` impl. Reads `region_id` (string), `chunk_index_in_region`, `tier`, `audio_file_id`, timestamps, embeddings from CRNN parquets. Groups sequences by `region_id` sorted by `min(start_timestamp)` per region (matching frontend nav order). Populates `extras["tier"]` on every `WindowMeta`.

### Modified modules

- `src/humpback/sequence_models/overlay.py` — `OverlayMetadata` becomes `sequence_ids: list[str]`, `positions_in_sequence: list[int]`, plus existing timestamps. `compute_overlay()` writes `sequence_id` (string) and `position_in_sequence` (int32) columns to the overlay parquet. PCA/UMAP math unchanged.
- `src/humpback/sequence_models/exemplars.py` — `WindowMeta` becomes `sequence_id: str`, `position_in_sequence: int`, `audio_file_id: int | None`, timestamps, `max_state_probability: float`, `extras: dict[str, str | int | float | None]`. `select_exemplars()` carries `extras` through unchanged.
- `src/humpback/services/hmm_sequence_service.py` — delete `_load_overlay_inputs()`. `generate_interpretations()` resolves `source_kind = source_kind_for(cej.model_version)`, fetches the loader, calls it, then runs `compute_overlay()` and `select_exemplars()` on the returned `OverlayInputs`.
- `src/humpback/workers/hmm_sequence_worker.py` — `_run_region_crnn_hmm()` calls `generate_interpretations(settings.storage_root, job, source)` after summary write, wrapped in the same `try/except` + `logger.warning(...)` pattern as the SurfPerch path. Remove the placeholder `_ = source` line.
- `src/humpback/api/routers/sequence_models.py` — read-time legacy adapter on `GET /overlay` and `/exemplars`: detect legacy column / field names (`merged_span_id`, `window_index_in_span`) and translate to `sequence_id` / `position_in_sequence` before serializing the response. ~20 lines confined to the API layer.
- `src/humpback/schemas/sequence_models.py` — `OverlayPoint` becomes `sequence_id: str`, `position_in_sequence: int`, plus existing fields. `ExemplarRecord` becomes `sequence_id: str`, `position_in_sequence: int`, `audio_file_id: int | None`, timestamps, `max_state_probability: float`, `exemplar_type: str`, `extras: dict[str, str | int | float | None] = {}`.

### Frontend changes

- `frontend/src/api/sequenceModels.ts` — `OverlayPoint` and `ExemplarRecord` TS interfaces match the unified Pydantic shape; `extras: Record<string, string | number | null>` added to `ExemplarRecord`.
- `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx` — exemplar gallery row component renders a small tier badge when `record.extras?.tier` is present (CRNN). SurfPerch records have no `tier` key and render no badge. `PcaUmapScatter` is unchanged — it never read identifier columns.

## 6. Data Model

No SQL schema changes. No Alembic migration. No `encoding_signature` change (interpretation artifacts are downstream of HMM training and don't participate in idempotency).

### `overlay.parquet` schema (unified, written by all loaders)

| column | type | notes |
|---|---|---|
| `sequence_id` | string | SurfPerch stringifies `merged_span_id`; CRNN passes `region_id` |
| `position_in_sequence` | int32 | window index (SurfPerch) or chunk index (CRNN) |
| `start_timestamp` | float64 | UTC epoch seconds |
| `end_timestamp` | float64 | UTC epoch seconds |
| `pca_x` / `pca_y` | float32 | PCA first two components |
| `umap_x` / `umap_y` | float32 | UMAP 2-D projection (NaN if N < 3) |
| `viterbi_state` | int16 | |
| `max_state_probability` | float32 | |

Pre-PR SurfPerch overlay parquets (legacy schema with `merged_span_id` / `window_index_in_span`) remain on disk and are translated by the API read-time adapter.

### `exemplars.json` schema (unified)

```json
{
  "n_states": 12,
  "states": {
    "0": [
      {
        "sequence_id": "<region UUID or stringified span id>",
        "position_in_sequence": 17,
        "audio_file_id": 42,
        "start_timestamp": 1709123456.0,
        "end_timestamp": 1709123456.25,
        "max_state_probability": 0.93,
        "exemplar_type": "high_confidence",
        "extras": {"tier": "event_core"}
      }
    ]
  }
}
```

Pre-PR SurfPerch exemplars JSON files remain on disk with the legacy keys (`merged_span_id`, `window_index_in_span`) and are translated by the API read-time adapter.

## 7. Workflow

### New CRNN HMM job

1. Worker runs `_run_region_crnn_hmm()` end-to-end (PCA fit → HMM fit → decode → states.parquet write → summary.json write).
2. After summary write, worker calls `generate_interpretations(settings.storage_root, job, source)` inside try/except.
3. `generate_interpretations()` calls `get_loader(SOURCE_KIND_REGION_CRNN)`, loader reads CRNN parquets, builds `OverlayInputs`.
4. `compute_overlay()` writes `overlay.parquet` (unified schema). `select_exemplars()` writes `exemplars.json` (unified schema).
5. On any failure inside the try block, worker logs a warning and continues; HMM job status remains `complete`.

### New SurfPerch HMM job

Same as today, but `generate_interpretations()` now goes through `get_loader(SOURCE_KIND_SURFPERCH)` → `SurfPerchLoader.load()`. Output `overlay.parquet` and `exemplars.json` use the unified schema.

### Existing SurfPerch HMM job (pre-PR artifacts on disk)

- `GET /overlay`: API endpoint reads the parquet, detects missing `sequence_id` column, projects `merged_span_id.cast(str) → sequence_id` and `window_index_in_span → position_in_sequence`, returns unified response. Disk file unchanged.
- `GET /exemplars`: API endpoint reads JSON, detects legacy keys per record, translates in-memory, returns unified response. Disk file unchanged.
- User clicks the existing "Refresh" button next to Label Distribution → `POST /generate-interpretations/{id}` rewrites overlay.parquet and exemplars.json in unified format. Subsequent GETs skip the adapter (no-op since columns already match).

## 8. Failure Modes (must be tested)

- CRNN HMM job with all chunks `tier="background"` → exemplars carry `extras["tier"]="background"`; no special handling required.
- Single-region CRNN job (only one sequence) → loader returns one `raw_sequences` entry; UMAP falls back to NaN if total points < 3 (existing `compute_overlay` behavior).
- Pre-PR SurfPerch overlay parquet served via GET `/overlay` → response contains `sequence_id` (string) and `position_in_sequence` (int) populated from legacy columns. No on-disk write.
- New SurfPerch HMM job → on-disk overlay.parquet has unified columns; legacy adapter is a no-op (columns already match).
- Unknown `source_kind` (defensive, shouldn't happen) → `get_loader()` raises `ValueError`; `generate_interpretations()` is wrapped in worker try/except and the warning is logged.
- Missing `pca.joblib` (e.g., HMM job failed mid-flight) → loader raises; `generate_interpretations()` propagates; worker swallows.

## 9. Key Decisions

1. **Generalized abstraction (Approach 3) over schema-dispatched single function.** Future embedding sources are anticipated; a Protocol with one impl per source is the foundation. *(Q: approach choice)*
2. **Unified `(sequence_id: str, position_in_sequence: int)` identifier pair.** SurfPerch stringifies its int span id. Lossless and trivially reversible if needed; eliminates the alternative-types switch. *(Approach 3 implementation)*
3. **`extras: dict[str, str | int | float | None]` for source-specific metadata.** Avoids the discriminated-union type tax for two source kinds. CRNN populates `extras["tier"]`; future sources add their own keys. *(Q3 follow-on)*
4. **Tier badge on exemplar cards.** Free at the producer (already on every CRNN row), answers the most obvious "is this state actually call-related?" question per state. *(Q3)*
5. **Legacy read-time adapter, not forced regeneration.** Pre-PR SurfPerch artifacts remain readable without on-disk migration. The existing Refresh button rewrites in unified format on demand. *(Q4 / Q5 follow-on)*
6. **No backfill for old CRNN HMM jobs.** User deleted them via the existing UI delete flow. New jobs run interpretations via the worker; deletion + recreation is the documented recovery path for any old job. *(Q4)*
7. **`PcaUmapScatter` frontend component unchanged.** Already schema-agnostic — reads only `pca_x/pca_y/umap_x/umap_y/viterbi_state/max_state_probability`. *(Q5 confirmed)*

## 10. Risks and Mitigations

1. **Schema breakage in tests.** The generic-dataclass rename touches `OverlayMetadata` / `WindowMeta` field names everywhere they're constructed. *Mitigation:* mechanical rename caught by pyright; the PR is large but every touch is local.
2. **Legacy adapter rot.** The read-time adapter must stay until all pre-PR SurfPerch HMM jobs are refreshed or deleted. *Mitigation:* clear comment in code naming the adapter as transitional; revisit removal in a future cleanup PR.
3. **PCA staleness on overlay vs. training set.** `compute_overlay()` projects all chunks/windows even when training used a sub-sequence subset. Consistent with current SurfPerch behavior; no mitigation.
4. **L2-normalize parity.** `compute_overlay()` must use the HMM job's `l2_normalize` setting. Already plumbed; preserved unchanged.
5. **Large-region UMAP cost.** A single CRNN HMM job can have ~200k chunks; UMAP fit is multi-minute at that size. *Mitigation:* deferred. Existing SurfPerch path runs UMAP unconditionally and has not been a problem in practice. If measured to be a problem, add deterministic stratified row sampling in a follow-up PR.

## 11. Documentation Updates (CLAUDE.md §10.2)

- `CLAUDE.md` §9.1 — strike "(SurfPerch source only for Phase 1)" from the overlay/exemplars line; add a note that label distribution is still SurfPerch-only pending Phase 2.
- `docs/reference/sequence-models-api.md` — document the unified `OverlayPoint` / `ExemplarRecord` Pydantic shape, the `extras` field convention, and the legacy read-time adapter as transitional.
- `DECISIONS.md` — append `ADR-059: Source-agnostic HMM interpretation loader Protocol` covering the abstraction (decisions 1, 2, 3 from §9).
- No README change — purely internal abstraction.
- No Alembic migration — no SQL schema change.

## 12. Phase 2 Sketch (out of scope here)

CRNN label distribution extends `generate_label_distribution()` to traverse `RegionDetectionJob → hydrophone_id → DetectionJobs` and join chunk center-times against `vocalization_labels`. Structurally identical to the SurfPerch path (which traverses `EventSegmentationJob → RegionDetectionJob → hydrophone_id → DetectionJobs`). The tier metadata enables a richer per-state breakdown (per-tier label counts) on the frontend chart. Phase 2 will likely extend the loader Protocol with a parallel method or introduce a sibling `LabelDistributionLoader` Protocol — TBD when scoped.
