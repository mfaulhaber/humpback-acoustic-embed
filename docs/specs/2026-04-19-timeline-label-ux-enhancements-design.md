# Timeline Label UX Enhancements

**Date:** 2026-04-19
**Status:** Approved

## Problem

The detection timeline label editing workflow has three friction points:

1. **No way to clear a label.** The "unlabeled" radio button is a passive indicator — clicking it does nothing. Once a detection is labeled, the only way to remove the label is to delete the detection entirely and re-add it.

2. **Mouse-only label selection.** Switching between humpback/orca/ship/background requires clicking the radio buttons in the toolbar. When reviewing dozens of detections, keyboard shortcuts would significantly speed up the workflow.

3. **Extra step to enter label mode.** Users must click the "Label" button in the playback controls before they can interact with detection bars. Clicking a detection bar directly should enter label mode when preconditions are met.

## Design

### 1. Mark as Unlabeled

Make the "unlabeled" indicator in `LabelToolbar` clickable. When clicked:

- Sets `selectedLabel` to `null`
- If in select mode with a selected row, dispatches a `clear_label` action that sets all four label fields (`humpback`, `orca`, `ship`, `background`) to `null` on that row
- If in add mode, selecting unlabeled disables the ghost placement cursor (adding an unlabeled region is meaningless since unlabeled windows come from the detection job)

Changes required:

- **`LabelToolbar.tsx`**: Convert the unlabeled `<span>` to a clickable element. Widen `onLabelChange` signature from `(label: LabelType)` to `(label: LabelType | null)`.
- **`useLabelEdits.ts`**: Add a `clear_label` action to the reducer that sets all four label fields to `null`. Add the action to the `Action` union type.
- **`TimelineViewer.tsx`**: Update the `onLabelChange` handler to dispatch `clear_label` when `label` is `null` and a row is selected.
- **`LabelEditor.tsx`**: When `selectedLabel` is `null` in add mode, suppress ghost preview and creation.

### 2. Keyboard Shortcuts

Extend the existing `keydown` handler in `TimelineViewer.tsx` with label shortcuts, active only when `labelMode && labelEditMode === "detection"`:

| Key | Action |
|-----|--------|
| `u` | Select unlabeled (null) |
| `h` | Select humpback |
| `o` | Select orca |
| `s` | Select ship |
| `b` | Select background |

Behavior is identical to clicking the corresponding radio button:

- Updates `selectedLabel` state
- If in select mode with a selected row, dispatches `change_type` (or `clear_label` for `u`)
- Guarded: ignored if `e.target` is an `INPUT`, `SELECT`, or `TEXTAREA`

### 3. Click Detection Bar to Enter Label Mode

When a detection bar in `DetectionOverlay` is clicked and all preconditions are met:

- **Preconditions:** not playing, zoom is 5m or 1m, detection overlay visible, not already in label mode
- **Effect:**
  1. Enters label mode (same as `toggleLabelMode()`)
  2. Sets `labelSubMode` to `"select"`
  3. Selects the clicked bar via `labelDispatch({ type: "select", id: row.row_id })`
  4. Sets `selectedLabel` to the bar's current label (or `null` if unlabeled)

Changes required:

- **`DetectionOverlay.tsx`**: Accept an `onBarClick` callback prop. On click, invoke it with the clicked row.
- **`SpectrogramViewport.tsx`**: Thread the `onBarClick` callback from `TimelineViewer` to `DetectionOverlay`.
- **`TimelineViewer.tsx`**: Define the callback that checks preconditions and enters label mode with the clicked bar selected.

When already in label mode or when playing, clicks behave as today (no change). The zoom precondition is preserved — at coarser zoom levels clicking does nothing extra.

## Files Modified

| File | Change |
|------|--------|
| `frontend/src/components/timeline/LabelToolbar.tsx` | Clickable unlabeled option, widen `onLabelChange` signature |
| `frontend/src/hooks/queries/useLabelEdits.ts` | Add `clear_label` reducer action |
| `frontend/src/components/timeline/TimelineViewer.tsx` | Keyboard shortcuts, `onLabelChange` null handling, click-to-enter callback |
| `frontend/src/components/timeline/LabelEditor.tsx` | Suppress ghost when `selectedLabel` is null |
| `frontend/src/components/timeline/DetectionOverlay.tsx` | Add `onBarClick` callback prop |
| `frontend/src/components/timeline/SpectrogramViewport.tsx` | Thread `onBarClick` prop |

## Testing

- Playwright test: enter label mode by clicking a detection bar, verify toolbar appears and bar is selected
- Playwright test: use keyboard shortcuts to change label, verify radio button state updates
- Playwright test: select a labeled detection, press `u`, verify label clears
- Playwright test: in add mode with unlabeled selected, verify ghost cursor is suppressed
