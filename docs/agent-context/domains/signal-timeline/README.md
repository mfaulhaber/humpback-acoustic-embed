# Signal Timeline Domain

Load this domain for audio IO, DSP, windowing, spectrograms, PCEN rendering,
timeline cache, playback audio, timeline API, or shared timeline UI primitives.

## Primary Paths

- `src/humpback/processing/`
- `src/humpback/services/timeline_export.py`
- `src/humpback/services/timeline_tile_service.py`
- `src/humpback/api/routers/timeline.py`
- `frontend/src/components/timeline/`
- `frontend/src/components/shared/DateRangePickerUtc.tsx`
- `frontend/e2e/timeline.spec.ts`
- `frontend/e2e/timeline-labeling.spec.ts`

## Artifact Roots

- Timeline tile caches are managed by processing services under the configured
  storage root.
- Timeline export output is domain-specific and should follow storage helper
  conventions.

## Likely Neighbors

- `ingest-detection` for detection timelines, hydrophone playback, and row
  stores.
- `call-parsing` for region/event overlays.
- `sequence-models` for continuous embedding audio source semantics.
- `frontend-shell` for shared UI, hooks, and navigation.

## Before Editing

1. Identify whether the change touches visualization, listening, or ML feature
   extraction. These paths deliberately have different normalization semantics.
2. Load consuming domain context when changing overlay contracts or playback
   behavior used by a workspace.
