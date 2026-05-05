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
| Charts | react-plotly.js (wraps Plotly.js basic dist) |
| Icons | lucide-react |
| API Client | Hand-rolled typed fetch wrapper (`frontend/src/api/client.ts`) |

**Navigation**: Side nav + top nav layout with react-router-dom. The default route is `/app/call-parsing/detection`. Classifier has sub-routes (`/app/classifier/training`, `/app/classifier/hydrophone`, `/app/classifier/embeddings`, `/app/classifier/tuning`, `/app/classifier/labeling`); Vocalization has sub-routes (`/app/vocalization/training`, `/app/vocalization/labeling`, `/app/vocalization/training-data`, `/app/vocalization/clustering`); Call Parsing has sub-routes (`/app/call-parsing/detection`, `/app/call-parsing/segment`, `/app/call-parsing/segment-training`, `/app/call-parsing/classify`, `/app/call-parsing/classify-training`, `/app/call-parsing/window-classify`); Sequence Models has sub-routes (`/app/sequence-models/continuous-embedding`, `/app/sequence-models/hmm-sequence`, `/app/sequence-models/masked-transformer`). HMM Sequence and Masked Transformer detail pages include a Motifs panel for creating motif extraction jobs (parent-kind discriminator), reviewing ranked motifs, and aligning motif examples around event midpoints. Masked Transformer detail page reads the active k from the URL via `?k=` (`useSearchParams`); per-k panels key TanStack Query cache on `(jobId, k)` so KPicker switches don't remount the page. The classifier timeline viewer is at `/app/classifier/timeline/:jobId`; the region detection timeline viewer is at `/app/call-parsing/region-timeline/:jobId`. Top-level Audio, Processing, Search, Label Processing, and standalone Clustering pages have been removed.

**Sequence Models shared components (ADR-061):**
- `DiscreteSequenceBar` — generic timeline strip refactored from the old `HMMStateBar`. Props add `mode: "rows" | "single-row"`, `numLabels`, `colorPalette`, `tooltipFormatter?`. `mode="rows"` reproduces the original HMM bar pixel-equivalently; `mode="single-row"` is a 60px-tall canvas used for masked-transformer token strips. Both HMM and Masked Transformer detail pages render through this single component.
- `RegionNavBar` — extracted from the inline CRNN region nav on the HMM detail page (A/D keyboard shortcuts, region-count display, current-region highlight). Reused on the Masked Transformer detail page.
- `LABEL_COLORS` (renamed from `STATE_COLORS`) + `labelColor(idx, total)` helper in `constants.ts`: returns `palette[idx % palette.length]` for `total ≤ 30`, generated HSL ramp for larger `total` (covers the wide k-sweep range: k=50, 100, 200, ...).
- `KPicker` — top-of-page tab/select for the Masked Transformer detail page only; URL-synced via `useSearchParams("k")`, defaults to first entry of `k_values`.
- `CollapsiblePanelCard` — wraps a shadcn `<Card>` with a chevron toggle and persists open/closed state in `localStorage[`seq-models:panel:${storageKey}`]`. Closed panels **unmount** their children (not just hide), so heavy charts don't keep running off-screen. Props: `title`, `storageKey`, `defaultOpen?`, `headerExtra?` (clicks inside the slot do not toggle the panel), `testId?`. Used on both Sequence Model detail pages for every panel except the top metadata Card and the timeline-viewer Card.
- `MotifTimelineLegend` — selected-motif legend rendered inside the timeline-viewer Card on each Sequence Model detail page (inside `TimelineProvider` on Masked Transformer; between `SpanNavBar` and `TimelineProvider` on HMM). Renders state swatches with arrow separators, an `idx+1 / total` counter, prev/next buttons, an optional `▶ play` button, and an optional `tokenSelector` slot. Both `tokenSelector` and `onPlay` are opt-in — HMM consumers omit them and see no change. The page lifts `MotifExtractionPanel`'s active-occurrence selection via the panel's `onSelectionChange` / `activeOccurrenceIndex` / `onActiveOccurrenceChange` props so prev/next can drive both the timeline (`seekTo` to occurrence midpoint) and the per-row highlight in `MotifExampleAlignment`.
- `MotifTokenCountSelector` — `Token Count: 2 | 3 | 4` toggle group rendered to the right of the legend prev/next/Play controls on the Masked Transformer detail page. Selecting a value enters byLength mode: every visible occurrence of every length-N motif is highlighted with a deterministic per-motif color (`@/lib/motifColor.colorForMotifKey`, ~12-hue palette, hash-of-`motif_key` indexing), and prev/next/Play in the legend operate on the visible occurrence set sorted by `start_timestamp`. Clicking the active value returns to single-motif mode; clicking a row in the Motif panel also exits byLength mode (mutually exclusive). Buttons are disabled with a tooltip when no motif of that length exists in the active extraction job. Page-local state lives on `MaskedTransformerDetailPage`; the `useMotifsByLength(motifJobId, motifs, length)` hook (`@/api/sequenceModels`) fans the per-motif occurrence fetches via `useQueries` and shares React Query cache keys with `useMotifOccurrences`.

**Masked Transformer detail page panel order** (top → bottom, after the metadata Card and KPicker): Token Timeline (plain Card with the legend inside), Motifs, Loss Curve, Run-Length Histograms, Overlay, Exemplars, Label Distribution. Motifs sits directly under the timeline so the legend, prev/next, and the alignment-strip are all visible without scrolling.

**Masked Transformer create form**: Advanced options are grouped into Model, Retrieval Head, Training Windows, Contrastive Learning, and Run Controls sections. Retrieval head controls default to 128 / 512 dimensions with L2 normalization on; the unchecked default submits `retrieval_head_enabled=false` and null dimensions. Region windows remain the default and submit `event_centered_fraction=0.0` with null context fields. Event-centered and mixed modes enable non-negative pre/post context controls defaulting to 2.0 seconds; mixed mode additionally requires `0.0 < event_centered_fraction < 1.0`. Human-correction contrastive controls remain disabled until the retrieval head is enabled; enabling contrastive switches region-only training to mixed mode with `event_centered_fraction=0.7` and reveals sampler controls. The sampler defaults match the API (`enabled=true`, 4 labels per batch, 4 events per label, max unlabeled fill 0.25, region balancing on). The default submit remains contrastive-disabled with `contrastive_loss_weight=0.0` and `contrastive_label_source="none"`.

**Timeline tiles**: Frontend consumers keep the existing URL contracts:
classifier timelines request `timelineTileUrl`, and region-backed timelines
request `regionTileUrl`. The backend stores and reuses PNG tiles through a
shared hydrophone-span repository keyed by source span, renderer id/version,
zoom, frequency range, and pixel geometry, so classifier, region, HMM, masked
transformer, and review views can share the same disk tiles without frontend
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
    │   ├── call-parsing/        (DetectionPage, RegionJobForm, RegionJobTable, RegionJobSummary, RegionDetectionTimeline, SegmentPage, SegmentJobForm, SegmentJobTable, SegmentJobDetail, SegmentReviewWorkspace, RegionTable, ClassifyReviewWorkspace, WindowClassifyReviewWorkspace, EventDetailPanel, RegionTable, SegmentTrainingPage, FeedbackTrainingJobTable, SegmentModelTable)
    │   ├── classifier/          (TrainingTab, AutoresearchCandidatesSection, HydrophoneTab, LabelingTab, EmbeddingsPage, DetectionTab, BulkDeleteDialog)
    │   ├── vocalization/        (VocalizationTrainingTab, VocabularyManager, VocalizationTrainForm, VocalizationModelList, VocalizationLabelingTab, SourceSelector, InferencePanel, LabelingWorkspace, RetrainFooter, TrainingDataView, VocalizationClusteringPage, VocalizationClusteringDetail)
    │   ├── sequence-models/     (ContinuousEmbeddingJobsPage, ContinuousEmbeddingDetailPage, HMMSequenceJobsPage, HMMSequenceDetailPage, MaskedTransformerJobsPage, MaskedTransformerCreateForm, MaskedTransformerDetailPage, KPicker, LossCurveChart, TokenRunLengthHistograms, DiscreteSequenceBar, RegionNavBar, MotifExtractionPanel, MotifExampleAlignment, MotifTimelineLegend, MotifTokenCountSelector, CollapsiblePanelCard)
    │   ├── timeline/            (ClassifierTimeline, Spectrogram, TileCanvas, TimelineProvider, PlaybackControls, ZoomSelector, OverlayToggles, EditToggle, EditToolbar, EventNav, DetectionOverlay, VocalizationOverlay, RegionOverlay, RegionEditOverlay, EventBarOverlay, RegionBandOverlay, MotifHighlightOverlay, LabelEditor, LabelToolbar, VocLabelEditor, VocLabelPopover, VocLabelToolbar, etc.)
    │   ├── admin/               (AdminTab, ModelRegistry, ModelScanner, DatabaseAdmin)
    │   └── shared/              (FolderTree, FolderBrowser, StatusBadge, MessageToast, DateRangePickerUtc)
    └── utils/                   (format.ts, audio.ts)
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

## Sequence Models UI Notes

- `MaskedTransformerCreateForm` keeps human-correction contrastive loss disabled by default. Enabling it requires the retrieval projection head and automatically switches region-only training to mixed mode with `event_centered_fraction=0.7`, because contrastive labels attach to event-centered windows rather than full-region sequences. Sampler controls are visible only while contrastive learning is active and submit the backend defaults unless overridden.
