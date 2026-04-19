# Timeline Label UX Enhancements Implementation Plan

**Goal:** Reduce friction in the detection label editing workflow with three enhancements: clickable unlabeled option, keyboard shortcuts, and click-to-enter label mode.
**Spec:** [docs/specs/2026-04-19-timeline-label-ux-enhancements-design.md](../specs/2026-04-19-timeline-label-ux-enhancements-design.md)

---

### Task 1: Add `clear_label` action to the label edits reducer

**Files:**
- Modify: `frontend/src/hooks/queries/useLabelEdits.ts`

**Acceptance criteria:**
- [ ] `Action` union includes `{ type: "clear_label"; row_id: string }`
- [ ] Reducer handles `clear_label` by setting `humpback`, `orca`, `ship`, `background` all to `null` on the target row in `mergedRows`
- [ ] Collapsing logic: if a prior `change_type` or `add` edit exists for the same `row_id`, update it in place (set `label` to `undefined`/remove the label field); otherwise create a new `clear_label` edit
- [ ] `LabelEdit` interface supports `action: "clear_label"` with `row_id`
- [ ] `mergedRows` applies `clear_label` edits: sets all four label fields to `null`

**Tests needed:**
- Unit test: dispatch `clear_label` on a labeled row, verify all label fields become `null`
- Unit test: dispatch `clear_label` after a `change_type` on the same row, verify edits collapse

---

### Task 2: Make "unlabeled" clickable in LabelToolbar and wire into TimelineViewer

**Files:**
- Modify: `frontend/src/components/timeline/LabelToolbar.tsx`
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`
- Modify: `frontend/src/components/timeline/LabelEditor.tsx`

**Acceptance criteria:**
- [ ] `LabelToolbarProps.onLabelChange` signature is `(label: LabelType | null) => void`
- [ ] "unlabeled" indicator is a clickable element (cursor pointer) that calls `onLabelChange(null)`
- [ ] In `TimelineViewer`, when `onLabelChange` receives `null` and a row is selected in select mode, dispatches `clear_label` for that row
- [ ] In `LabelEditor`, when `selectedLabel` is `null` and mode is `"add"`, ghost preview is suppressed and clicks do not create regions

**Tests needed:**
- Playwright: enter label mode, select a labeled bar, click "unlabeled", verify bar loses its colored label
- Playwright: switch to add mode, click "unlabeled", verify ghost cursor does not appear

---

### Task 3: Add keyboard shortcuts for label selection

**Files:**
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`

**Acceptance criteria:**
- [ ] `u`, `h`, `o`, `s`, `b` keys are handled in the existing `keydown` handler
- [ ] Shortcuts only fire when `labelMode && labelEditMode === "detection"`
- [ ] Shortcuts are ignored when `e.target` is an `INPUT`, `SELECT`, or `TEXTAREA`
- [ ] Each key updates `selectedLabel` state and, if in select mode with a selected row, dispatches `change_type` or `clear_label` accordingly
- [ ] `e.preventDefault()` is called for handled keys to avoid browser default behavior

**Tests needed:**
- Playwright: enter label mode, press `h`, verify humpback radio is selected
- Playwright: select a detection, press `o`, verify label changes to orca
- Playwright: select a labeled detection, press `u`, verify label clears

---

### Task 4: Click detection bar to enter label mode

**Files:**
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`

**Acceptance criteria:**
- [ ] A new callback is defined that, when a detection bar is clicked while preconditions are met (not playing, zoom 5m/1m, detection overlay visible, not in label mode), enters label mode with the clicked bar selected
- [ ] The callback sets `labelSubMode` to `"select"`, dispatches `select` for the clicked row, and sets `selectedLabel` to the row's current label (or `null`)
- [ ] The callback is passed through `SpectrogramViewport` to `DetectionOverlay` via the existing `onDetectionClick` prop — either replacing or wrapping the current handler depending on label-mode state
- [ ] When already in label mode or when playing, existing click behavior is preserved (popover in non-label mode, no-op in label mode)
- [ ] Zoom precondition is enforced: at coarser zoom levels the click does not enter label mode

**Tests needed:**
- Playwright: click a detection bar while stopped and zoomed to 5m, verify label toolbar appears and bar is selected
- Playwright: click a detection bar while playing, verify label mode is NOT entered
- Playwright: click a detection bar at 15m zoom, verify label mode is NOT entered

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test`
