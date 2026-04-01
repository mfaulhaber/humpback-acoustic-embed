# Timeline UX Improvements — Design Spec

**Date**: 2026-04-01
**Status**: Approved

## Overview

Four targeted improvements to the detection timeline page: fix viewport overflow, reposition overlay controls, enable label type changes in select mode, and improve tooltip positioning.

## 1. Viewport Overflow Fix

The root container uses `flex flex-col h-screen overflow-hidden` but a vertical scrollbar still appears, hiding the top portion of the view. The spectrogram viewport wrapper (`flex-1 min-h-0`) or its children likely have an intrinsic minimum height that prevents proper shrinking.

**Fix**: Audit the flex layout chain from root through header, viewport wrapper, and footer. Ensure every intermediate element in the chain uses `min-h-0` (or equivalent) so the spectrogram viewport absorbs remaining space. The `ResizeObserver` inside `SpectrogramViewport` already adapts to measured container size, so correcting the CSS constraint is sufficient.

**Files**: `TimelineViewer.tsx`, possibly `SpectrogramViewport.tsx`

## 2. Move Labels/Freq Buttons to Lower-Right Overlay

Remove the Labels toggle and Freq badge from `TimelineHeader`. Render them as an absolutely-positioned overlay at the **bottom-right** of the spectrogram viewport container.

- Same pill styling (small rounded, 10px text)
- Semi-transparent background to avoid fully obscuring the spectrogram
- z-index above spectrogram tiles but below tooltips
- Visible in both view mode and label mode

`TimelineHeader` simplifies to: back button, hydrophone name, and time range.

**Files**: `TimelineHeader.tsx`, `TimelineViewer.tsx` (move controls into the viewport wrapper)

## 3. Label Type Change in Select Mode

Currently label mode's select sub-mode only allows selecting labeled detections for drag/delete. Unlabeled detections are not selectable.

### Changes

- **Make unlabeled detections selectable** in select sub-mode. Clicking an unlabeled detection sets it as the active selection (white border highlight).
- **Label type buttons become dual-purpose**: When a detection is selected in select sub-mode, clicking a label type button dispatches `change_type` via the existing `useLabelEdits` reducer. Works for both unlabeled-to-labeled and relabeling.
- **Visual feedback**: Bar immediately updates to the new label color via the existing pending-edit rendering in merged rows.
- **No new components**: Wires existing toolbar buttons to `change_type` dispatch when there's an active selection.

The `change_type` action already exists in the reducer. The batch save endpoint already handles type changes.

**Files**: `LabelEditor.tsx` (allow unlabeled selection), `LabelToolbar.tsx` or `TimelineViewer.tsx` (wire type buttons to `change_type` dispatch)

## 4. Tooltip Positioning Near Mouse

Currently the tooltip anchors to bar top-center with `translate(-50%, -100%)`, which clips when bars are near the viewport top.

### Changes

- Capture mouse coordinates (relative to overlay container) on `mouseEnter`
- Position tooltip at fixed offset: **12px right, 12px below** the entry point
- Clamp within container bounds: flip left if overflowing right, flip above if overflowing bottom
- Tooltip stays at entry-point position for the hover duration (no `mouseMove` tracking)
- Use a ref on the tooltip element to measure dimensions and apply clamping after initial render

**Files**: `DetectionOverlay.tsx`

## Non-Goals

- No changes to the backend API or data model
- No changes to label mode's "add" sub-mode behavior
- No new components or external dependencies
- No frequency range editing UI (Freq badge remains read-only)
