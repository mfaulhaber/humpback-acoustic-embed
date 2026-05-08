# Event Encoder Cluster Projections Design

## Goal

Add a read-only cluster projection panel beneath the Event Encoder timeline
viewer's Selected Event Features panel. The panel lets users switch between
UMAP and PCA scatter plots for the currently selected Event Encoder `k`, with
points colored by the same job-local token palette used by the timeline token
overlay.

## Domains

- Primary domain: `sequence-models`
- Neighbor domains: `signal-timeline`, `frontend-shell`

## Context

Completed Event Encoder jobs already persist `event_vectors.parquet` and
`event_tokens.parquet`. The timeline viewer reads those immutable artifacts and
renders token assignments for a selected `k`. Its Selected Event Features panel
uses descriptor metadata from the same timeline response, and the token overlay
uses `labelColor(token_id, selected_k)` from
`frontend/src/components/sequence-models/constants.ts`.

The new projection panel should preserve those contracts:

- It must be artifact-authoritative for completed jobs.
- It must treat `Txx` token labels and token ids as job-local and `k`-local.
- It must not mutate Event Encoder artifacts or introduce a new worker step.
- It should reuse the existing frontend Plotly dependency.

The Vocalization / Clustering detail page already has two useful pieces:

- Backend dimensionality reduction helpers in `humpback.clustering.reducer`
  (`reduce_umap`, `reduce_pca`), used by the clustering worker.
- A `VocalizationUmapPlot` Plotly wrapper that groups points by cluster label
  and adds vocalization-specific audio playback on point click.

The existing vocalization plot component is not directly reusable as-is because
it owns its data fetch, palette, cluster/noise naming, hover payload, and audio
click behavior. The Plotly trace/layout mechanics are reusable.

## Options Considered

### Option 1: Compute UMAP/PCA on demand in a new API endpoint

The API joins selected-`k` token rows with `event_vector` rows from
`event_vectors.parquet`, computes either a two-dimensional UMAP or PCA
projection, and returns plot-ready points with token assignment metadata.
It reuses the existing backend reduction helpers where their output shape is
appropriate, with a thin wrapper that pads tiny datasets to a stable 2D result.

Pros:

- Existing completed Event Encoder jobs gain the panel without reruns.
- The calculation remains artifact-authoritative and backend-owned.
- Frontend code stays focused on plot presentation and token coloring.

Cons:

- UMAP has some per-request cost.
- A future high-volume job may need caching if this endpoint becomes hot.

### Option 2: Precompute projections in the Event Encoder worker

The worker writes projection sidecars when tokenization completes.

Pros:

- Fast reads for the detail page.
- Projection parameters are captured at job-completion time.

Cons:

- Existing completed jobs would not have projections.
- Worker/artifact scope expands for a UI-only diagnostic.
- Changing projection defaults later would require backfills or versioned
  sidecars.

### Option 3: Compute projections entirely in the browser

The frontend downloads timeline/vector data and computes PCA/UMAP locally.

Pros:

- Avoids a new backend route.

Cons:

- The frontend does not have a UMAP dependency today.
- Large vector payloads and projection logic would move into the browser.
- Browser-side results could drift from backend scientific dependencies.

### Option 4: Refactor the existing vocalization UMAP plot into a shared plot component

Extract a presentational Plotly scatter component that accepts normalized
points, group/color/label functions, axis titles, optional selected point
highlighting, and an optional click handler. Keep domain-specific fetching and
click behavior in `VocalizationUmapPlot` and the new Event Encoder panel.

Pros:

- Both visualizations share Plotly layout, grouping, marker, empty-state, and
  responsive behavior.
- Vocalization remains free to use its existing cluster palette and audio
  click behavior.
- Event Encoder can use the timeline token color palette without coupling to
  vocalization semantics.

Cons:

- Adds one small shared component and requires a focused vocalization adapter
  update.

## Decision

Use Option 1 plus the frontend portion of Option 4.

Add `GET /sequence-models/event-encoders/{job_id}/projection` with query
parameters:

- `k`: optional positive integer, defaulting to the lowest available `k`.
- `method`: `umap` or `pca`, defaulting to `umap`.

The response includes:

- `job_id`, `selected_k`, `valid_k_values`, and `method`.
- Axis labels.
- One point per tokenized event that has a valid `event_vector`.
- Each point carries `event_id`, `token_id`, `token_label`,
  `token_confidence`, `distance_to_centroid`, timestamps, and `x`/`y`.

Projection behavior:

- PCA uses the persisted `event_vector` values and returns two components when
  possible, reusing `reduce_pca` with deterministic zero padding for
  one-dimensional or single-point cases.
- UMAP uses the persisted `event_vector` values with a deterministic random
  seed from the job, reusing `reduce_umap` when it can produce meaningful
  coordinates. For tiny datasets where UMAP is not meaningful, the route returns
  a deterministic two-dimensional fallback so the UI can still render the
  selected events.
- Invalid `k`, incomplete jobs, missing token artifacts, or missing vector
  artifacts return the same style of HTTP errors as the timeline endpoint.

## Frontend Design

Add a shared presentational component, tentatively
`frontend/src/components/shared/ClusterProjectionPlot.tsx`, that handles
Plotly scatter rendering from normalized point data. Refactor
`VocalizationUmapPlot` to adapt existing vocalization visualization data into
that component while preserving its current palette and audio-on-click behavior.

Add an `EventEncoderClusterProjectionPanel` below
`SelectedEventFeatureTable` inside the existing timeline viewer frame. The
panel:

- Shows a compact selector for `UMAP` versus `PCA`.
- Fetches projection data for `job.id`, `timeline.selected_k`, and the selected
  projection method.
- Groups Plotly scatter traces by token id and colors each trace with
  `labelColor(token_id, timeline.selected_k)`.
- Highlights the currently selected event with a larger marker outline.
- Emits empty/loading/error states consistent with the timeline detail panels.

The existing timeline `k` selector remains the source of truth for the selected
tokenization. Changing `k` changes both the token overlay and the projection
query. The projection selector controls only the dimensionality-reduction
method.

## Tests

- Backend integration tests for the new projection endpoint:
  - default `k` and PCA response shape;
  - UMAP/fallback response shape;
  - invalid `k`;
  - missing vector artifact.
- Frontend Event Encoder Playwright coverage:
  - completed detail page renders the projection panel below selected features;
  - selector switches from UMAP to PCA and requests the PCA projection;
  - point coloring uses the shared token palette through the same `labelColor`
    helper used by the timeline overlay.
- Frontend TypeScript verifies both the refactored vocalization plot adapter and
  the new Event Encoder plot panel.
- Final verification includes Sequence Models backend tests, Event Encoder
  frontend smoke, TypeScript, and the full backend test suite.
