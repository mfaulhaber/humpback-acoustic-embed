# Timeline Vocalization Overlay — Design Spec

**Date:** 2026-04-01
**Status:** Approved

## Overview

Add a "Vocalizations" toggle button to the timeline viewer that replaces the
detection label bars with vocalization type bars — showing both inference
suggestions and human-applied labels from the vocalization labeling workflow.
Read-only; editing stays in the vocalization labeling workspace.

## Data Source

**New API endpoint:** `GET /labeling/vocalization-labels/{detection_job_id}/all`
returns all vocalization labels for the entire detection job without requiring a
time-range filter. The existing `fetchVocalizationLabels` requires `start_utc`
and `end_utc` parameters, which doesn't suit the timeline's full-job viewport.

**Inference job discovery:** The frontend uses the existing
`fetchVocClassifierInferenceJobs()` endpoint, filtering client-side for
`source_type === "detection_job"` and `source_id === jobId` with
`status === "complete"` to determine whether vocalization data exists for the
current detection job.

## New Component: VocalizationOverlay.tsx

Mirrors `DetectionOverlay.tsx` structure:

- **Props:** `labels: VocalizationLabel[]`, `centerTimestamp`, `zoomLevel`,
  `width`, `height`, `visible`
- **Bar rendering:** Single uniform color for all vocalization bars (a distinct
  muted purple that contrasts with the detection label palette). One bar per
  `(start_utc, end_utc)` window that has at least one vocalization label.
- **Badges:** Small colored chips rendered inside each bar, one per vocalization
  type on that window. Badge colors assigned dynamically from a palette based on
  vocalization type vocabulary order. Inference-sourced labels get an outlined
  badge; manual labels get a filled badge.
- **Tooltip on hover:** Shows vocalization type(s), source (inference/manual),
  confidence score if present, and time range.
- **No click/edit interaction** — read-only overlay.

## State Management in TimelineViewer.tsx

New state: `overlayMode: "detection" | "vocalization"` (default `"detection"`).

- The existing `showLabels` toggle controls visibility of whichever overlay mode
  is active.
- The new "Vocalizations" button switches `overlayMode` to `"vocalization"`
  (toggling back to `"detection"` when pressed again).
- When `overlayMode === "vocalization"`, `SpectrogramViewport` renders
  `VocalizationOverlay` instead of `DetectionOverlay`.
- When `overlayMode === "detection"`, current behavior is unchanged.
- Switching to vocalization mode while in label edit mode exits label mode first
  (since vocalization overlay is read-only).

## Button Placement in PlaybackControls.tsx

A new "Vocalizations" button in the right-side group, next to the existing
"Labels: ON/OFF" button:

- **Enabled:** Shown when a completed vocalization inference job exists for this
  detection job.
- **Disabled:** Visible but grayed out with tooltip "No vocalization inference
  for this job".
- **Active:** Highlighted when `overlayMode === "vocalization"`.
- Toggling switches between detection and vocalization overlay modes.

## New Hook: useVocalizationOverlay

A TanStack Query hook that:

1. Fetches all vocalization labels for the detection job (the new `/all`
   endpoint).
2. Fetches inference jobs to determine if vocalization data exists for this
   detection job.
3. Returns `{ labels, hasVocalizationData, isLoading }`.

## Color Scheme

- **Bar fill:** `rgba(168, 130, 220, 0.40)` (muted purple), hover
  `rgba(168, 130, 220, 0.60)` — distinct from all detection label colors.
- **Badge colors:** Dynamically assigned from a 6–8 color palette based on
  vocalization type index. Filled badge = manual source, outlined badge =
  inference source.

## Scope Exclusions

- No editing vocalization labels from the timeline.
- No inference score heatmap or separate confidence strip.
- No simultaneous display of detection and vocalization overlays.
