# Timeline UX Improvements Implementation Plan

**Goal:** Fix viewport overflow, reposition overlay controls, enable label type changes in select mode, and improve tooltip positioning on the detection timeline page.
**Spec:** [docs/specs/2026-04-01-timeline-ux-improvements-design.md](../specs/2026-04-01-timeline-ux-improvements-design.md)

---

### Task 1: Fix viewport vertical overflow

**Files:**
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`
- Modify: `frontend/src/components/timeline/SpectrogramViewport.tsx` (if needed)

**Acceptance criteria:**
- [ ] Timeline page fits within the browser viewport with no vertical scrollbar at standard window sizes
- [ ] The flex layout chain from root `h-screen` through header, viewport, and footer all properly constrain height (every flex child in the column uses `min-h-0` or equivalent)
- [ ] `SpectrogramViewport` ResizeObserver continues to measure correct dimensions after the fix
- [ ] Footer controls (zoom selector, playback, label toolbar) remain pinned at the bottom

**Tests needed:**
- Visual verification: page loads without vertical scrollbar
- Resize browser window — spectrogram viewport shrinks/grows, no scrollbar appears

---

### Task 2: Move Labels/Freq buttons to spectrogram overlay

**Files:**
- Modify: `frontend/src/components/timeline/TimelineHeader.tsx`
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`

**Acceptance criteria:**
- [ ] Labels toggle and Freq badge removed from `TimelineHeader`
- [ ] `TimelineHeader` renders only: back button, hydrophone name, time range
- [ ] Labels toggle and Freq badge rendered as absolutely-positioned overlay at bottom-right of the spectrogram viewport container
- [ ] Overlay uses semi-transparent background so spectrogram beneath is partially visible
- [ ] z-index is above spectrogram tiles but below tooltips (e.g., z-index 8, between tiles at 5 and tooltips at 20)
- [ ] Controls visible in both view mode and label mode
- [ ] `TimelineHeader` props simplified (remove `showLabels`, `onToggleLabels`, `freqRange`, `onFreqRangeChange`)

**Tests needed:**
- Visual verification: buttons appear bottom-right of spectrogram, header is simplified
- Toggle Labels works from new position
- Freq badge displays correct range

---

### Task 3: Enable label type change on selected detections

**Files:**
- Modify: `frontend/src/components/timeline/LabelEditor.tsx`
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`

**Acceptance criteria:**
- [ ] Unlabeled detections are selectable in label mode's select sub-mode (click sets active selection with white border highlight)
- [ ] Unlabeled detection bars show `pointer` cursor in select mode (not `default`)
- [ ] When a detection (labeled or unlabeled) is selected in select sub-mode, clicking a label type radio button in `LabelToolbar` dispatches a `change_type` action
- [ ] After type change, the bar immediately shows the new label color via pending edit rendering
- [ ] The `change_type` edit is included in the batch save payload
- [ ] Selecting a different detection or switching modes clears the previous selection normally

**Tests needed:**
- Click unlabeled detection in select mode — verify it becomes selected (white border)
- With selection active, click a label type — verify bar changes color and edit appears in dirty state
- Save — verify the type change persists after reload
- Change type of an already-labeled detection — verify it updates correctly

---

### Task 4: Reposition tooltip near mouse entry point

**Files:**
- Modify: `frontend/src/components/timeline/DetectionOverlay.tsx`

**Acceptance criteria:**
- [ ] Tooltip positioned at mouse entry coordinates (relative to overlay container) plus 12px right and 12px below
- [ ] Tooltip stays at entry-point position for duration of hover (no mouseMove tracking)
- [ ] Tooltip clamped within container bounds: flips left of cursor if overflowing right, flips above if overflowing bottom
- [ ] Tooltip ref used to measure dimensions for clamping after initial render
- [ ] Tooltip no longer clips/hides when hovering bars near the top of the viewport

**Tests needed:**
- Hover a detection bar near the top — tooltip appears below-right of mouse, fully visible
- Hover a bar near the right edge — tooltip flips to left of mouse
- Hover a bar near the bottom-right corner — tooltip flips both left and above
- Tooltip content (label, times, confidence) unchanged

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test`
