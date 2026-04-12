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

**Navigation**: Side nav + top nav layout with react-router-dom. Classifier has sub-routes (`/app/classifier/training`, `/app/classifier/hydrophone`, `/app/classifier/embeddings`, `/app/classifier/tuning`, `/app/classifier/labeling`); Vocalization has sub-routes (`/app/vocalization/training`, `/app/vocalization/labeling`, `/app/vocalization/training-data`); Call Parsing has sub-routes (`/app/call-parsing/detection`); the timeline viewer is at `/app/classifier/timeline/:jobId`; other sections are single-route pages. The Tuning page (`TuningTab.tsx`) has three collapsible sections: Manifests (create/list/detail/delete data manifests), Searches (create/list/detail/delete hyperparameter search jobs with configurable search space), and Candidates (relocated `AutoresearchCandidatesSection` for artifact import, review, and candidate-backed promotion).

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
    │   ├── audio/               (AudioTab, AudioUpload, AudioList, AudioDetail, AudioPlayerBar, SpectrogramPlot, SimilarityMatrix)
    │   ├── processing/          (ProcessingTab, QueueJobForm, ProcessingJobsList, EmbeddingSetsList)
    │   ├── clustering/          (ClusteringTab, EmbeddingSetSelector, ClusteringParamsForm, ClusteringJobCard, ClusterTable, UmapPlot, EvaluationPanel, ExportReport)
    │   ├── call-parsing/        (DetectionPage, RegionJobForm, RegionJobTable, RegionJobSummary)
    │   ├── classifier/          (TrainingTab, AutoresearchCandidatesSection, HydrophoneTab, LabelingTab, EmbeddingsPage, DetectionTab, BulkDeleteDialog)
    │   ├── vocalization/        (VocalizationTrainingTab, VocabularyManager, VocalizationTrainForm, VocalizationModelList, VocalizationLabelingTab, SourceSelector, EmbeddingStatusPanel, InferencePanel, LabelingWorkspace, RetrainFooter, TrainingDataView)
    │   ├── timeline/            (TimelineViewer, SpectrogramViewport, TileCanvas, LabelEditor, LabelToolbar, VocalizationOverlay, VocLabelEditor, VocLabelPopover, VocLabelToolbar, etc.)
    │   ├── search/              (SearchTab — standalone + detection-sourced similarity search)
    │   ├── label-processing/    (LabelProcessingTab, LabelProcessingJobCard, LabelProcessingPreview)
    │   ├── admin/               (AdminTab, ModelRegistry, ModelScanner, DatabaseAdmin)
    │   └── shared/              (FolderTree, FolderBrowser, StatusBadge, MessageToast, DateRangePickerUtc, EmbeddingSetPanel)
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
