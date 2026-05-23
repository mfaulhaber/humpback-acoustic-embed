# Frontend Stack & Development

> Read this when working on frontend components, routes, styling, or build configuration.

The web UI is a React SPA in the `frontend/` directory, built with:

| Layer | Technology |
|-------|-----------|
| Build | Vite + TypeScript |
| UI Framework | React 18 |
| Styling | Tailwind CSS |
| Component Library | shadcn/ui (Radix primitives, copy-paste model in `frontend/src/components/ui/`) |
| Server State | TanStack Query (polling, caching, mutations) |
| Charts | react-plotly.js / Plotly.js |
| Icons | lucide-react |
| API Client | Hand-rolled typed fetch wrapper (`frontend/src/api/client.ts`) |

**Navigation**: Side nav + top nav layout with react-router-dom. The default route is `/app/call-parsing/detection`. Classifier has sub-routes (`/app/classifier/training`, `/app/classifier/hydrophone`, `/app/classifier/embeddings`, `/app/classifier/tuning`, `/app/classifier/labeling`); Vocalization has sub-routes (`/app/vocalization/training`, `/app/vocalization/labeling`, `/app/vocalization/training-data`, `/app/vocalization/clustering`); Call Parsing has sub-routes (`/app/call-parsing/detection`, `/app/call-parsing/segment`, `/app/call-parsing/segment-training`, `/app/call-parsing/classify`, `/app/call-parsing/classify-training`, `/app/call-parsing/window-classify`); Sequence Models has retained routes for Continuous Embedding (`/app/sequence-models/continuous-embedding`) and Event Encoder (`/app/sequence-models/event-encoder`), and `/app/sequence-models` redirects to Continuous Embedding. The classifier timeline viewer is at `/app/classifier/timeline/:jobId`; the region detection timeline viewer is at `/app/call-parsing/region-timeline/:jobId`. Top-level Audio, Processing, Search, Label Processing, and standalone Clustering pages have been removed.

**Sequence Models retained components:**
- `ContinuousEmbeddingJobsPage`, `ContinuousEmbeddingCreateForm`, `ContinuousEmbeddingJobTable`, and `ContinuousEmbeddingDetailPage` are the active Continuous Embedding UI surface.
- `EventEncoderJobsPage`, `EventEncoderCreateForm`, `EventEncoderJobTable`, and `EventEncoderDetailPage` are the active Event Encoder UI surface. `EventEncoderDetailPage` includes an `EventEncoderTimelinePanel` as its second panel for read-only tokenized-event review with region-backed spectrogram tiles, audio playback, zoom controls, k selection, event navigation, and token badges. The selected token badge can be toggled to make previous/next navigation jump between events with the same job-local, selected-k token id without hiding other events. `EventEncoderJobTable` shows a `Notes` column with a `PianoRollNotesStatusPill` that links to the piano roll route for the row. `EventEncoderPianoRollPage` includes a controlled, collapsible bottom spectrogram strip that reuses region timeline tiles while preserving the piano roll's smooth viewport state. Its preferred v3 display is Ridge mode: one piano-roll rectangle per event, placed at the ridge median frequency and vertically spanning the trimmed ridge low/high descriptor bounds when coverage and ridge energy are trustworthy, with conservative spectral-envelope top expansion for broad harmonic events and F0/spectral peak fallbacks for older artifacts. V3 Ridge views default to a 6000 Hz vertical range so high whistles are visible without manually changing the Hz selector. The toolbar additionally renders a `NotesStatusControls` group with the same status pill, a `Generate notes` button when notes are absent or failed, an inline progress label when queued or running, and a `Re-run` button on failure that toggles the captured error message.
- When the Notes view is active and the resolved notes job is `v3` (ADR-069), `EventEncoderPianoRollPage` switches to a curved-ribbon renderer on a MIDI 12–120 log-frequency Y axis (C0…C9, with G9 anchored just below the top). The 88-key band (MIDI 21–108) gets normal black-key shading; the extended bands render with a desaturated tint to mark them as outside piano range. Per-note pitch contours are fetched in batches of up to 2000 `note_uid`s via `usePianoRollNoteContours` (POST `/notes/contours`) and accumulated in a per-page `contourCacheRef` map keyed by `note_uid` so panning does not refetch already-loaded notes. Notes without a fetched contour render as flat bars at `midi_pitch` and hydrate into ribbons when their contour arrives; a fetch failure shows a single non-blocking "Contours unavailable" toast and leaves notes in flat-bar state for the session. Hit-testing uses ≤ 6 px polyline distance (ribbons) with a row-height bounding-box fallback (flat bars). Legacy `v1`/`v2` sidecars still render as flat bars because their rows have no `note_uid`.
- `PianoRollNotesStatusPill` surfaces a "v3 available" badge when the encoder has a completed v2 (or older) notes sidecar but no v3 job yet; clicking it POSTs a v3 notes job and catches a 409 (already enqueued) with a non-blocking toast rather than the global error path. The MIDI download button's tooltip names MPE / DAW compatibility for v3 exports, and the export status panel shows `Format: MPE v3` under the file size when the latest export is v3.
- `frontend/src/api/sequenceModels.ts` exports Continuous Embedding, Event Encoder, and Piano Roll Notes types, fetchers, and TanStack Query hooks. Piano Roll Notes hooks (`usePianoRollNotesStatus`, `usePianoRollNotes`, `usePianoRollNoteContours`, `useCreatePianoRollNotesJob`) keep refetching `notes-status` while queued or running, and invalidate the status and timeline queries on mutation success. `usePianoRollNoteContours` keys the batch query on a 32-bit FNV-1a content hash of the sorted-uid join and populates per-uid entries under `["piano-roll-note-contour", jobId, version, uid]` so single-note reads skip the network. `ApiError` (re-exported from this module) lets callers distinguish HTTP status codes (used by the v3-available pill to handle 409 explicitly).
- `DiscreteSequenceBar` remains as a generic future-use state/token timeline strip. Props include `mode: "rows" | "single-row"`, `numLabels`, `colorPalette`, and `tooltipFormatter?`.
- `RegionNavBar` and `SpanNavBar` remain as reusable future-use navigation controls for region/span visualizations.
- `LABEL_COLORS` + `labelColor(idx, total)` in `constants.ts` provide deterministic colors for state or motif visualizations.
- `MotifHighlightOverlay`, `MotifTimelineLegend`, and `colorForMotifKey()` remain as generic visualization primitives. They use local timeline/motif types and do not import retired Sequence Models API types.
- `CollapsiblePanelCard` remains available as a generic future-use collapsible card helper.
- `frontend/src/components/timeline/index.ts` is retained as a generic timeline barrel for future shared timeline imports.
- These generic primitives are intentionally retained despite no active route importing all of them; their focused tests are the reuse contract.

**Timeline tiles**: Frontend consumers keep the existing URL contracts:
classifier timelines request `timelineTileUrl`, and region-backed timelines
request `regionTileUrl`. The backend stores and reuses PNG tiles through a
shared hydrophone-span repository keyed by source span, renderer id/version,
zoom, frequency range, and pixel geometry, so classifier, region, and review
views can share the same disk tiles without frontend
prop changes.

## Frontend Package Management
*   Use `npm` for all frontend package operations. Run commands from the `frontend/` directory.
*   `npm install` — install dependencies
*   `npm run dev` — start Vite dev server on `:5173` (proxies API calls to `:8000`)
*   `npm run build` — production build to `src/humpback/static/dist/`
*   `npx tsc --noEmit` — type-check without emitting

## Frontend File Structure
```
frontend/
├── package.json, vite.config.ts, tsconfig.json, tailwind.config.ts
├── playwright.config.ts         (Playwright test config)
├── components.json              (shadcn/ui config)
├── index.html
├── e2e/                         (Playwright test specs)
└── src/
    ├── main.tsx                 (QueryClientProvider + App mount)
    ├── App.tsx                  (routes + AppShell wrapper)
    ├── index.css                (Tailwind directives + shadcn CSS vars)
    ├── lib/utils.ts             (cn() helper)
    ├── api/
    │   ├── client.ts            (typed fetch wrapper, all endpoints)
    │   └── types.ts             (TS interfaces mirroring Pydantic schemas)
    ├── hooks/queries/           (TanStack Query hooks per domain)
    ├── components/
    │   ├── ui/                  (shadcn primitives)
    │   ├── layout/              (AppShell, TopNav, SideNav, Breadcrumbs)
    │   ├── call-parsing/        (DetectionPage, RegionJobForm, RegionJobTable, RegionDetectionTimeline, SegmentPage, SegmentJobForm, SegmentJobTable, SegmentJobDetail, SegmentReviewWorkspace, RegionTable, ClassifyReviewWorkspace, ClassifyTrainingPage, ClassifyTrainingJobTable, WindowClassifyPage, WindowClassifyReviewWorkspace, WindowClassifyJobForm, WindowClassifyJobTable, EventDetailPanel, SegmentTrainingPage, TrainingDatasetTable, SegmentModelTable, ClassifyModelTable)
    │   ├── classifier/          (TrainingTab, AutoresearchCandidatesSection, HydrophoneTab, LabelingTab, EmbeddingsPage, DetectionSourcePicker, ActiveEmbeddingBanner, BulkDeleteDialog)
    │   ├── vocalization/        (VocalizationTrainingTab, VocabularyManager, VocalizationTrainForm, VocalizationModelList, VocalizationLabelingTab, SourceSelector, InferencePanel, LabelingWorkspace, RetrainFooter, TrainingDataView, VocalizationClusteringPage, VocalizationClusteringDetail)
    │   ├── sequence-models/     (ContinuousEmbeddingJobsPage, ContinuousEmbeddingCreateForm, ContinuousEmbeddingJobTable, ContinuousEmbeddingDetailPage, EventEncoderJobsPage, EventEncoderCreateForm, EventEncoderJobTable, EventEncoderDetailPage, EventEncoderTimelinePanel, EventEncoderTokenOverlay, eventEncoderTimelineNavigation, plus retained generic primitives: DiscreteSequenceBar, RegionNavBar, SpanNavBar, MotifTimelineLegend, CollapsiblePanelCard)
    │   ├── timeline/            (ClassifierTimeline, Spectrogram, TileCanvas, TimelineProvider, PlaybackControls, ZoomSelector, OverlayToggles, EditToggle, EditToolbar, EventNav, DetectionOverlay, VocalizationOverlay, RegionOverlay, RegionEditOverlay, EventBarOverlay, RegionBandOverlay, MotifHighlightOverlay, LabelEditor, LabelToolbar, VocLabelEditor, VocLabelPopover, VocLabelToolbar, etc.)
    │   ├── admin/               (AdminTab, ModelRegistry, ModelScanner, DatabaseAdmin)
    │   └── shared/              (ComputeDeviceBadge, DatabaseErrorBanner, FolderBrowser, StatusBadge, MessageToast, DateRangePickerUtc)
    └── utils/                   (format.ts)
```

## Dev Workflow
```bash
# Terminal 1: Backend
uv run humpback-api          # API on :8000
uv run humpback-worker       # Worker process

# Terminal 2: Frontend dev server
cd frontend && npm run dev   # Vite on :5173, proxies to :8000
```

## Production Build & Serving
```bash
cd frontend && npm run build  # outputs to src/humpback/static/dist/
uv run humpback-api           # serves SPA at / and API on :8000
```

The FastAPI backend detects `static/dist/index.html` at startup. When present, it serves the built SPA at `/` and mounts `/assets` for JS/CSS bundles. When absent, it falls back to the legacy `static/index.html`.
Deployment/runtime configuration should come from a repo-root `.env` plus
`HUMPBACK_` env vars. The API and worker entrypoints explicitly load the
repo-root `.env`; direct `Settings()` construction should stay hermetic.
Production host allowlisting belongs in FastAPI via `HUMPBACK_ALLOWED_HOSTS`;
do not use Vite `allowedHosts` for deployed host validation.
