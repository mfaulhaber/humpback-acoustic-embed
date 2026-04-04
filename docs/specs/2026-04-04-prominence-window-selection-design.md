# Prominence-Based Window Selection for Detection

**Date:** 2026-04-04
**Status:** Approved

## Problem

In windowed detection mode, NMS (non-maximum suppression) selects non-overlapping
5-second peak windows from merged events. This creates 2–4 second gaps between
detection items, even in regions where the classifier scores every window above 0.9.
Distinct vocalizations that fall in these gaps are missed — both for labeling and
for training data.

Example from job `d52e03cc`: at 02:16:53–02:16:56, the classifier scores 0.95–0.997,
but no detection item is emitted because NMS suppresses all windows within 5 seconds
of the adjacent selected peaks.

## Design

### Window Selection Toggle

A new `window_selection` parameter on detection jobs:

- **`"nms"`** (default) — current behavior. Greedy NMS with `window_size_seconds`
  suppression zone. Non-overlapping windows.
- **`"prominence"`** — peak prominence detection. Finds distinct vocalizations
  via score dips. Allows overlapping windows.

Persisted as a nullable column on `detection_jobs`. `NULL` treated as `"nms"` for
backward compatibility.

### Prominence Algorithm

Implemented as a new code path in `select_peak_windows_from_events()`:

1. **Smooth scores** — 3-window moving average (same as NMS) for peak finding.
2. **Find local peaks** — a window is a peak if its smoothed score >= both neighbors
   (or >= its one neighbor at event edges).
3. **Compute prominence using raw scores** — for each peak, prominence = peak's raw
   score minus the highest valley (minimum raw score) between it and the nearest
   higher peak on each side. Raw scores are used (not smoothed) to preserve the
   true dip depth.
4. **Filter by `min_prominence`** — discard peaks below threshold. Default **0.03**.
5. **Filter by `min_score`** — same gate as NMS; peaks below `high_threshold` are
   discarded.
6. **Emit windows** — each surviving peak gets a 5-second window at `peak_offset`.
   Overlapping windows are allowed.

### New Parameters

| Parameter | Type | Default | Stored | Notes |
|-----------|------|---------|--------|-------|
| `window_selection` | `"nms"` \| `"prominence"` | `"nms"` | DB column | Toggle between methods |
| `min_prominence` | float | 0.03 | DB column | Only used when `window_selection="prominence"` |

### Database Change

Alembic migration 038: add nullable `window_selection` (String) and `min_prominence`
(Float) columns to `detection_jobs`.

### API Changes

- `DetectionJobCreate` and `HydrophoneDetectionJobCreate`: add optional
  `window_selection` and `min_prominence` fields.
- `DetectionJobOut`: add `window_selection` and `min_prominence` fields.

### Frontend Changes

- Hydrophone detection UI: add window selection toggle (NMS / Prominence) and
  `min_prominence` input (visible when prominence is selected).
- Local detection creation: same toggle if a creation form exists.

### Downstream Impact

No changes needed. Overlapping windows are compatible with all downstream systems
(row store, embeddings, labeling, extraction, training datasets, vocalization
inference) because they are keyed by `row_id`, not time-range uniqueness.

Unit tests will be added to verify overlapping windows flow correctly through
downstream stages.
