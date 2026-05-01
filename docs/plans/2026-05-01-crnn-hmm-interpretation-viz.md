# CRNN-Source HMM Interpretation Visualizations Implementation Plan

**Goal:** Make CRNN-source HMM jobs render PCA / UMAP overlay and State Exemplars panels by introducing a source-agnostic `SequenceArtifactLoader` Protocol, plus a tier badge on CRNN exemplar cards.
**Spec:** [docs/specs/2026-05-01-crnn-hmm-interpretation-viz-design.md](../specs/2026-05-01-crnn-hmm-interpretation-viz-design.md)

---

### Task 1: Generalize `OverlayMetadata` and `compute_overlay()` to the unified identifier pair

**Files:**
- Modify: `src/humpback/sequence_models/overlay.py`

**Acceptance criteria:**
- [ ] `OverlayMetadata` dataclass has `sequence_ids: list[str]`, `positions_in_sequence: list[int]`, `start_timestamps: list[float]`, `end_timestamps: list[float]`. The previous int-typed `merged_span_ids` and `window_indices` fields are removed.
- [ ] `compute_overlay()` writes the parquet table with `sequence_id` (pa.string()) and `position_in_sequence` (pa.int32()) columns instead of `merged_span_id` / `window_index_in_span`.
- [ ] PCA / UMAP math (concatenation, optional L2-normalize, PCA transform, UMAP fit_transform, NaN fallback when N < 3) is byte-identical to the current implementation.
- [ ] Function signature otherwise unchanged: same `pca_model`, `raw_sequences`, `viterbi_states`, `max_state_probs`, `l2_normalize`, UMAP knobs, returns `(pa.Table, np.ndarray)`.

**Tests needed:**
- Unit test asserting `compute_overlay()` produces a table with the new column names and types, given a synthetic input.
- Unit test asserting PCA/UMAP numeric outputs are stable for a fixed-seed synthetic input (regression guard against accidental math changes).

---

### Task 2: Generalize `WindowMeta` and `select_exemplars()` to the unified identifier pair plus `extras`

**Files:**
- Modify: `src/humpback/sequence_models/exemplars.py`

**Acceptance criteria:**
- [ ] `WindowMeta` dataclass fields: `sequence_id: str`, `position_in_sequence: int`, `audio_file_id: int | None`, `start_timestamp: float`, `end_timestamp: float`, `max_state_probability: float`, `extras: dict[str, str | int | float | None]` (default empty dict via `field(default_factory=dict)`).
- [ ] `select_exemplars()` carries `extras` from `WindowMeta` into each per-state record dict under the `extras` key.
- [ ] Per-state record dicts use the new identifier names (`sequence_id`, `position_in_sequence`) and add `extras`.
- [ ] Selection math (top-k by probability ascending/descending, nearest-to-centroid via L2) is unchanged.

**Tests needed:**
- Unit test asserting `select_exemplars()` returns records with the new field names and propagates `extras` exactly.
- Unit test asserting deduplication semantics (a window picked as `high_confidence` is not re-emitted as `mean_nearest`) are preserved.

---

### Task 3: Define the `SequenceArtifactLoader` Protocol, `OverlayInputs` dataclass, and registry

**Files:**
- Create: `src/humpback/sequence_models/loaders/__init__.py`

**Acceptance criteria:**
- [ ] `SequenceArtifactLoader` Protocol exposes a single method: `load(self, storage_root: Path, hmm_job: HMMSequenceJob, cej: ContinuousEmbeddingJob) -> OverlayInputs`.
- [ ] `OverlayInputs` dataclass has fields: `pca_model: Any`, `raw_sequences: list[np.ndarray]`, `viterbi_states: list[np.ndarray]`, `max_probs: list[np.ndarray]`, `metadata: OverlayMetadata`, `window_metas: list[WindowMeta]`.
- [ ] Internal `_LOADERS: dict[str, SequenceArtifactLoader]` registry keyed on the existing `SOURCE_KIND_*` constants from `humpback.services.continuous_embedding_service` (or wherever they currently live).
- [ ] `get_loader(source_kind: str) -> SequenceArtifactLoader` raises `ValueError` with a descriptive message on unknown source.
- [ ] Module imports the two concrete loaders (Task 4 + Task 5) and registers them.
- [ ] Pyright passes with `--strict` on this file (the Protocol must type-check).

**Tests needed:**
- Unit test asserting `get_loader(SOURCE_KIND_SURFPERCH)` and `get_loader(SOURCE_KIND_REGION_CRNN)` return loader instances of the expected concrete types.
- Unit test asserting `get_loader("unknown")` raises `ValueError`.

---

### Task 4: Implement `SurfPerchLoader`

**Files:**
- Create: `src/humpback/sequence_models/loaders/surfperch.py`

**Acceptance criteria:**
- [ ] `SurfPerchLoader` class implements `SequenceArtifactLoader.load()`.
- [ ] `load()` reads `pca.joblib`, `embeddings.parquet`, `states.parquet` from the resolved paths.
- [ ] Sequences are grouped by `merged_span_id` and sorted ascending (matching current behavior).
- [ ] For each row in the flat `OverlayMetadata`, `sequence_id = str(merged_span_id)` and `position_in_sequence = window_index_in_span`.
- [ ] Each `WindowMeta` carries `sequence_id`, `position_in_sequence`, `audio_file_id`, timestamps, `max_state_probability`, and `extras = {}`.
- [ ] Behavior is byte-identical to the current `_load_overlay_inputs()` (modulo the column-name remapping at the boundary).

**Tests needed:**
- Unit test using a fixture SurfPerch states.parquet + embeddings.parquet + saved PCA: asserts the returned `OverlayInputs` has the expected sequence count, identifier stringification, and empty `extras` on every `WindowMeta`.
- Regression test asserting the loader produces identical `raw_sequences` / `viterbi_states` / `max_probs` arrays as the pre-rename `_load_overlay_inputs()` for a fixed fixture.

---

### Task 5: Implement `CrnnRegionLoader`

**Files:**
- Create: `src/humpback/sequence_models/loaders/crnn_region.py`

**Acceptance criteria:**
- [ ] `CrnnRegionLoader` class implements `SequenceArtifactLoader.load()`.
- [ ] `load()` reads `pca.joblib`, `embeddings.parquet`, `states.parquet` for a CRNN-source job.
- [ ] Sequences are grouped by `region_id` (string), sorted by the per-region minimum `start_timestamp` ascending. This matches the timeline navigation order the frontend uses for CRNN jobs.
- [ ] For each row in the flat `OverlayMetadata`, `sequence_id = region_id`, `position_in_sequence = chunk_index_in_region`.
- [ ] Each `WindowMeta` carries `sequence_id`, `position_in_sequence`, `audio_file_id`, timestamps, `max_state_probability`, and `extras = {"tier": tier}` where `tier` is read from the states.parquet `tier` column.

**Tests needed:**
- Unit test using a fixture CRNN states.parquet + embeddings.parquet + saved PCA: asserts the returned `OverlayInputs` has the expected region count, sequence ordering by min start_timestamp, and `extras["tier"]` populated on every `WindowMeta`.
- Unit test with a single-region fixture: asserts loader returns one sequence and the downstream UMAP NaN fallback is reachable when N < 3.

---

### Task 6: Refactor `generate_interpretations()` to use the loader registry

**Files:**
- Modify: `src/humpback/services/hmm_sequence_service.py`

**Acceptance criteria:**
- [ ] `_load_overlay_inputs()` is deleted from `hmm_sequence_service.py`.
- [ ] `generate_interpretations()` resolves `source_kind = source_kind_for(cej.model_version)`, calls `get_loader(source_kind).load(...)`, then runs `compute_overlay()` and `select_exemplars()` on the returned `OverlayInputs`.
- [ ] Atomic write semantics for `overlay.parquet` and `exemplars.json` are preserved.
- [ ] No direct imports of pyarrow-on-CRNN-or-SurfPerch parquet column names remain in this file. The service layer is source-agnostic.

**Tests needed:**
- Integration test asserting `generate_interpretations()` writes `overlay.parquet` with unified columns and `exemplars.json` with unified keys for a SurfPerch fixture.
- Integration test asserting the same for a CRNN fixture, with `extras["tier"]` set on each exemplar record.

---

### Task 7: Wire `_run_region_crnn_hmm()` to call `generate_interpretations()`

**Files:**
- Modify: `src/humpback/workers/hmm_sequence_worker.py`

**Acceptance criteria:**
- [ ] `_run_region_crnn_hmm()` calls `generate_interpretations(settings.storage_root, job, source)` after `summary.json` is written, wrapped in `try/except Exception` that logs a `logger.warning(..., exc_info=True)` and continues. This matches the SurfPerch path pattern at the existing call site.
- [ ] The placeholder `_ = source` line and its preceding comment are removed.
- [ ] HMM job status remains `complete` even if interpretation generation fails (interpretation failure is non-fatal).

**Tests needed:**
- Worker-level integration test (or extension of an existing CRNN HMM end-to-end test) asserting `overlay.parquet` and `exemplars.json` exist on disk with the unified schema after a CRNN HMM job runs to completion.
- Test asserting that injecting a failure in `generate_interpretations()` does not change the job status from `complete` to `failed`.

---

### Task 8: Update Pydantic schemas for `OverlayPoint` and `ExemplarRecord`

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`

**Acceptance criteria:**
- [ ] `OverlayPoint` fields: `sequence_id: str`, `position_in_sequence: int`, `start_timestamp: float`, `end_timestamp: float`, `pca_x: float`, `pca_y: float`, `umap_x: float`, `umap_y: float`, `viterbi_state: int`, `max_state_probability: float`. The old `merged_span_id` and `window_index_in_span` fields are removed.
- [ ] `ExemplarRecord` fields: `sequence_id: str`, `position_in_sequence: int`, `audio_file_id: int | None`, `start_timestamp: float`, `end_timestamp: float`, `max_state_probability: float`, `exemplar_type: str`, `extras: dict[str, str | int | float | None] = {}`. The old `merged_span_id` and `window_index_in_span` fields are removed.

**Tests needed:**
- Unit test asserting Pydantic validates a unified-shape payload and rejects a payload missing `sequence_id`.

---

### Task 9: Add the read-time legacy adapter to the overlay and exemplars GET endpoints

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`

**Acceptance criteria:**
- [ ] `GET /hmm-sequences/{job_id}/overlay` reads the parquet, detects whether `sequence_id` column is present; if not, projects `merged_span_id` (cast to string) → `sequence_id` and `window_index_in_span` → `position_in_sequence` before constructing `OverlayPoint` instances. Disk file is not modified.
- [ ] `GET /hmm-sequences/{job_id}/exemplars` reads the JSON, detects legacy keys per record (`merged_span_id`, `window_index_in_span`); if present, translates to `sequence_id` (stringified) and `position_in_sequence` per record before serializing the response. Empty `extras: {}` is added when missing. Disk file is not modified.
- [ ] When the parquet/JSON is already in unified format, the adapter is a structural no-op (no column rewrites, no field renames).
- [ ] The adapter logic lives in small private helpers in this file and is named with a comment marking it as transitional.

**Tests needed:**
- API test using a legacy-format overlay parquet fixture (`merged_span_id` / `window_index_in_span` columns): asserts the GET response contains `sequence_id` (string) and `position_in_sequence` (int).
- API test using a unified-format overlay parquet fixture: asserts the adapter is a no-op (response matches the on-disk values exactly).
- Same two tests for the exemplars endpoint.

---

### Task 10: Update frontend TypeScript interfaces and exemplar tier badge

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`

**Acceptance criteria:**
- [ ] `OverlayPoint` interface in `sequenceModels.ts`: `sequence_id: string`, `position_in_sequence: number`, plus existing fields. Old `merged_span_id` / `window_index_in_span` removed.
- [ ] `ExemplarRecord` interface in `sequenceModels.ts`: `sequence_id: string`, `position_in_sequence: number`, `audio_file_id: number | null`, timestamps, `max_state_probability: number`, `exemplar_type: string`, `extras: Record<string, string | number | null>`. Old `merged_span_id` / `window_index_in_span` removed.
- [ ] The `ExemplarGallery` row component (or its sub-component) renders a small tier badge (existing badge styling, e.g., shadcn Badge) when `record.extras?.tier` is a non-null string. The badge text is the tier string verbatim (`event_core` / `near_event` / `background`). When `extras.tier` is undefined or null, no badge renders.
- [ ] `PcaUmapScatter` component is unchanged.
- [ ] No other references to `merged_span_id` or `window_index_in_span` remain in frontend code that consumes the overlay or exemplars APIs (a grep across `frontend/src/components/sequence-models/` confirms zero hits for either name in those code paths). Note: the timeline / states API uses these field names independently and is out of scope for this task.

**Tests needed:**
- Playwright test loading a CRNN HMM detail-page fixture asserts `data-testid="hmm-pca-umap-scatter"` is visible and exemplar cards display a tier badge with one of the three documented tier values.
- Playwright test loading a SurfPerch HMM detail-page fixture asserts the overlay panel renders and exemplar cards display no tier badge.

---

### Task 11: Add ADR-059 and update reference docs

**Files:**
- Modify: `DECISIONS.md`
- Modify: `CLAUDE.md` (§9.1)
- Modify: `docs/reference/sequence-models-api.md`

**Acceptance criteria:**
- [ ] `DECISIONS.md` appends `## ADR-059: Source-agnostic HMM interpretation loader Protocol` with: context (ADR-057 deferral), decision (Protocol + registry + generic identifier pair + extras dict), and consequences (new sources slot in by writing one loader file; legacy adapter is transitional).
- [ ] `CLAUDE.md` §9.1 — strike "(SurfPerch source only for Phase 1)" from the overlay/exemplars description in the Sequence Models bullet. Add a clarifying note that label distribution remains SurfPerch-only pending Phase 2.
- [ ] `docs/reference/sequence-models-api.md` documents: the unified `OverlayPoint` and `ExemplarRecord` shape, the `extras` field convention with the documented `tier` key for CRNN sources, and the legacy read-time adapter as transitional.

**Tests needed:**
- N/A (documentation task). Verified by inspection during code review.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/loaders src/humpback/sequence_models/overlay.py src/humpback/sequence_models/exemplars.py src/humpback/services/hmm_sequence_service.py src/humpback/workers/hmm_sequence_worker.py src/humpback/api/routers/sequence_models.py src/humpback/schemas/sequence_models.py tests/sequence_models tests/api`
2. `uv run ruff check src/humpback/sequence_models/loaders src/humpback/sequence_models/overlay.py src/humpback/sequence_models/exemplars.py src/humpback/services/hmm_sequence_service.py src/humpback/workers/hmm_sequence_worker.py src/humpback/api/routers/sequence_models.py src/humpback/schemas/sequence_models.py tests/sequence_models tests/api`
3. `uv run pyright src/humpback/sequence_models/loaders src/humpback/sequence_models/overlay.py src/humpback/sequence_models/exemplars.py src/humpback/services/hmm_sequence_service.py src/humpback/workers/hmm_sequence_worker.py src/humpback/api/routers/sequence_models.py src/humpback/schemas/sequence_models.py tests/sequence_models tests/api`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test` (only the sequence-models specs touched by Task 10)
