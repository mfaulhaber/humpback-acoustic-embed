# Review UI Enhancements Implementation Plan

**Goal:** Unify time display, correction indicators, and drag-scroll across all Call Parsing review pages
**Spec:** docs/specs/2026-04-24-review-ui-enhancements-design.md

---

### Task 1: Add `formatRecordingTime` utility and update `EventBarOverlay` type union

**Files:**
- Modify: `frontend/src/utils/format.ts`
- Modify: `frontend/src/components/timeline/overlays/EventBarOverlay.tsx`

**Acceptance criteria:**
- [ ] `formatRecordingTime(offsetSec, jobStartEpoch)` returns `HH:MM:SS.d` (zero-padded, one decimal, UTC)
- [ ] `EffectiveEvent.typeSource` union includes `"approved"`
- [ ] `EventTypeBadge` renders lime ring for `"approved"`, green ring for `"correction"`, no ring for `"inference"`
- [ ] Visual constants `APPROVED_RING_COLOR` and `CORRECTED_RING_COLOR` defined

**Tests needed:**
- Unit test for `formatRecordingTime` covering zero-padding, boundary values, decimal precision

---

### Task 2: Fix drag-to-pan in `EventBarOverlay`

**Files:**
- Modify: `frontend/src/components/timeline/overlays/EventBarOverlay.tsx`

**Acceptance criteria:**
- [ ] `handleMouseMove` only calls `stopPropagation` when `dragRef.current` is set or `addMode` is true
- [ ] `handleMouseUp` only calls `stopPropagation` when `dragRef.current` is set
- [ ] `handleMouseLeave` only stops propagation during active drag
- [ ] `handleContainerClick` keeps unconditional `stopPropagation`
- [ ] Drag-to-pan works on empty spectrogram area in review timelines
- [ ] Edge-drag on selected event boundaries still works

**Tests needed:**
- Manual verification of drag-to-pan and edge-drag interaction (Playwright test for drag behavior)

---

### Task 3: Update `EventDetailPanel` with recording-based time

**Files:**
- Modify: `frontend/src/components/call-parsing/EventDetailPanel.tsx`

**Acceptance criteria:**
- [ ] New `jobStartEpoch` prop
- [ ] Start/end times displayed via `formatRecordingTime`
- [ ] "Was" annotations for adjusted boundaries use `formatRecordingTime`
- [ ] Duration remains offset-based (`Ns`)

**Tests needed:**
- Verify time format in Segment Review detail panel

---

### Task 4: Update `ReviewToolbar` and `RegionTable` with recording-based time

**Files:**
- Modify: `frontend/src/components/call-parsing/ReviewToolbar.tsx`
- Modify: `frontend/src/components/call-parsing/RegionTable.tsx`

**Acceptance criteria:**
- [ ] Both components accept `jobStartEpoch` prop
- [ ] Region start/end times use `formatRecordingTime`

**Tests needed:**
- Verify region time format in toolbar and table

---

### Task 5: Thread `jobStartEpoch` through `SegmentReviewWorkspace`

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] `jobStartEpoch` from timeline context passed to `EventDetailPanel`, `ReviewToolbar`, and `RegionTable`

**Tests needed:**
- TypeScript compilation confirms prop threading

---

### Task 6: Update `resolveEventType` and `ClassifyDetailPanel`

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`
- Modify: `frontend/src/components/call-parsing/ClassifyDetailPanel.tsx`

**Acceptance criteria:**
- [ ] `resolveEventType` returns `"approved"` when `correctedType === predictedType`
- [ ] `ClassifyDetailPanel` accepts `jobStartEpoch` prop and uses `formatRecordingTime`
- [ ] `ClassifyDetailPanel` accepts boundary correction info (`correctionType`, `originalStartSec`, `originalEndSec`) and renders boundary badges ("adjusted"/"deleted"/"added")
- [ ] `ClassifyDetailPanel` renders "was" annotations for adjusted boundaries
- [ ] Type badge gets lime ring for approved, green ring for corrected
- [ ] Text label shows "approved" (lime) or "corrected" (green) next to type badge
- [ ] `ClassifyReviewWorkspace` threads `jobStartEpoch` and boundary correction fields to `ClassifyDetailPanel`

**Tests needed:**
- Unit test for `resolveEventType` with approved case

---

### Task 7: Update `TypePalette` with three-state ring

**Files:**
- Modify: `frontend/src/components/call-parsing/TypePalette.tsx`
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] `TypePalette` accepts `typeSource` prop
- [ ] Active badge ring: type-colored for inference/null, lime for approved, green for corrected
- [ ] `ClassifyReviewWorkspace` passes `typeSource` to `TypePalette`

**Tests needed:**
- Verify ring color changes with type source

---

### Task 8: Update Window Classify local `EventDetailPanel`

**Files:**
- Modify: `frontend/src/components/call-parsing/WindowClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Local `EventDetailPanel` accepts `jobStartEpoch` and uses `formatRecordingTime`
- [ ] Boundary correction badges added (using the selected event's correction state from effective events)
- [ ] Label badges get three-state ring: lime for approved (correction "add" matching inferred above-threshold label), green for corrected (correction "add" for type not in model's above-threshold set), no ring for uncorrected
- [ ] `jobStartEpoch` threaded from workspace timeline context

**Tests needed:**
- Verify time format and ring treatment in Window Classify review

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test`
