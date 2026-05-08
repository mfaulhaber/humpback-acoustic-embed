# Event Encoder Cluster Projections Implementation Plan

**Goal:** Add a selector-controlled UMAP/PCA cluster projection panel to the Event Encoder detail timeline viewer.
**Spec:** `docs/specs/2026-05-08-event-encoder-cluster-projections-design.md`
**Primary domain:** `sequence-models`
**Neighbor domains:** `signal-timeline`, `frontend-shell`, `vocalization-clustering`

---

### Task 1: Add Artifact-Backed Projection API

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `src/humpback/clustering/reducer.py`
- Modify: `tests/integration/test_sequence_models_api.py`
- Modify: `tests/unit/test_clustering_pipeline.py`

**Acceptance criteria:**
- [ ] Completed Event Encoder jobs expose `GET /sequence-models/event-encoders/{job_id}/projection` with `method=umap|pca` and optional `k`.
- [ ] The endpoint joins selected token rows to persisted `event_vector` rows without reading current Pass 2 event state.
- [ ] The endpoint reuses existing PCA/UMAP reduction helpers through a stable two-dimensional projection wrapper.
- [ ] Invalid `k`, incomplete jobs, missing token artifacts, and missing vector artifacts produce explicit HTTP errors.
- [ ] PCA and UMAP responses include plot-ready points with token metadata and deterministic coordinates for tiny datasets.
- [ ] Integration tests cover PCA, UMAP/tiny fallback, invalid `k`, and missing vectors.

**Tests needed:**
- `uv run pytest tests/unit/test_clustering_pipeline.py -q`
- `uv run pytest tests/integration/test_sequence_models_api.py -q`

---

### Task 2: Add Frontend Projection API Types And Query Hook

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`

**Acceptance criteria:**
- [ ] Frontend types model projection methods, response metadata, and projection points.
- [ ] A fetcher and TanStack query hook request projection data by job id, selected `k`, and projection method.
- [ ] The projection query stays disabled until the Event Encoder job is complete and a job id is available.

**Tests needed:**
- TypeScript verification via `cd frontend && npx tsc --noEmit`.

---

### Task 3: Extract Shared Cluster Plot Component

**Files:**
- Create: `frontend/src/components/shared/ClusterProjectionPlot.tsx`
- Modify: `frontend/src/components/vocalization/VocalizationUmapPlot.tsx`

**Acceptance criteria:**
- [ ] Shared Plotly scatter rendering handles grouped points, legends, axis titles, responsive sizing, optional selected-point highlighting, and optional point click handlers.
- [ ] `VocalizationUmapPlot` adapts existing vocalization visualization data into the shared component.
- [ ] Vocalization cluster colors, noise styling, and audio-on-click behavior are preserved.

**Tests needed:**
- TypeScript verification via `cd frontend && npx tsc --noEmit`.

---

### Task 4: Render Selector-Controlled Event Encoder Cluster Plot Panel

**Files:**
- Create: `frontend/src/components/sequence-models/EventEncoderClusterProjectionPanel.tsx`
- Modify: `frontend/src/components/sequence-models/EventEncoderTimelinePanel.tsx`
- Modify: `frontend/e2e/sequence-models/event-encoder.spec.ts`

**Acceptance criteria:**
- [ ] The new panel appears directly beneath Selected Event Features in the Event Encoder timeline viewer.
- [ ] The selector switches between UMAP and PCA projections.
- [ ] Plotly traces are grouped by token and use `labelColor(token_id, selected_k)`.
- [ ] The currently selected event is visually distinguished in the plot.
- [ ] Loading, empty, and error states match the existing detail page tone.
- [ ] Playwright coverage verifies panel placement and selector-driven PCA requests.

**Tests needed:**
- `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
- `cd frontend && npx tsc --noEmit`

---

### Task 5: Update Domain Context

**Files:**
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/vocalization-clustering/README.md`

**Acceptance criteria:**
- [ ] Sequence Models context mentions the Event Encoder projection endpoint and detail-page cluster projection panel.
- [ ] Vocalization Clustering context mentions that its UMAP plot uses the shared cluster projection plot component.
- [ ] The context preserves the artifact-authoritative completed-job rule.

**Tests needed:**
- Documentation-only task; no targeted automated test required.

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/clustering/reducer.py tests/integration/test_sequence_models_api.py tests/unit/test_clustering_pipeline.py`
2. `uv run ruff check src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/clustering/reducer.py tests/integration/test_sequence_models_api.py tests/unit/test_clustering_pipeline.py`
3. `uv run pyright src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/clustering/reducer.py tests/integration/test_sequence_models_api.py tests/unit/test_clustering_pipeline.py`
4. `uv run pytest tests/unit/test_clustering_pipeline.py tests/integration/test_sequence_models_api.py -q`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
7. `uv run pytest tests/`
