# Timeline Vocalization Label Editing — Design Spec

**Date**: 2026-04-08
**Status**: Draft

## 1. Goal

Add vocalization label editing to the detection timeline viewer. When in Vocalization overlay mode, the Label button enters a vocalization-focused edit mode where users can click detection windows to add/remove vocalization type labels via a compact popover. Changes accumulate locally and are saved atomically via a new batch API endpoint.

## 2. Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Control surface | Popup/popover per window | Multi-label editing needs more than radio buttons; popover keeps viewport clear |
| Window scope | Edit existing detection windows only | No new window creation; detection windows define the temporal grid |
| Backend persistence | New batch API endpoint | Atomic saves, matches frontend accumulate-then-commit pattern |
| Popup content | Minimal — labels only, no spectrogram/audio | Timeline viewport already provides spectrogram + playback context |
| Clickable windows | All detection windows (labeled + unlabeled) | Any detection window is a valid vocalization labeling target |
| Save semantics | Accumulate across windows, explicit Save/Cancel | Matches detection label editing pattern; enables review before commit |
| Visual treatment | Uniform neutral bars + vocalization badges | Focuses attention on vocalization context, not species detection labels |

## 3. Activation & Mode

- User must be in **Vocalization overlay mode** (existing "Vocalizations" toggle in PlaybackControls).
- "Label" button enabled at 1m/5m zoom when paused (same constraints as detection label mode).
- Clicking "Label" enters **vocalization label edit mode** — mutually exclusive with detection label edit mode.
- The existing `overlayMode` state (`"detection" | "vocalization"`) determines which label editor activates when "Label" is clicked.

## 4. Visual Treatment (Edit Mode Active)

- All detection windows render as **uniform neutral/dim bars** (no species detection colors).
- Windows with existing vocalization labels show **colored type badges** (same badge style as VocalizationOverlay).
- Windows with pending (unsaved) changes show a **visual indicator** (ring or dot) to distinguish dirty state from saved state.
- The selected window (popup open) gets a **highlighted border** (white, consistent with detection select mode).
- Existing VocalizationOverlay and DetectionOverlay are hidden while vocalization label edit mode is active — VocLabelEditor replaces them.

## 5. Popover UI

Compact popover anchored near the clicked detection window bar.

**Layout:**
- **Header**: time range in `HH:MM:SS – HH:MM:SS UTC` format.
- **Existing labels**: colored badges matching the vocalization type color palette. Clickable to toggle removal — pending removal shown as dimmed with strikethrough.
- **"+" button**: opens a dropdown listing available vocalization types from the vocabulary. Click a type to add it.
- **"(Negative)" toggle**: mutually exclusive with type labels. Adding "(Negative)" queues removal of all type labels; adding any type label queues removal of "(Negative)".
- **Dismiss**: click outside the popover or press Escape. Pending changes are retained (not discarded on dismiss).

No spectrogram thumbnail or audio play button in the popover.

## 6. State Management

### 6.1 New Hook: `useVocLabelEdits`

Reducer-based hook paralleling the existing `useLabelEdits` for detection labels.

**State shape:**
```typescript
type VocLabelEditState = {
  selectedRowId: string | null;
  edits: Map<string, { adds: Set<string>; removes: Set<string> }>;
};
```

**Actions:**
- `toggle_add(row_id, label)` — add a label; if already in `adds`, remove from `adds`; enforce "(Negative)" mutual exclusivity.
- `toggle_remove(row_id, label)` — mark an existing saved label for removal; if already in `removes`, un-mark it.
- `select(row_id)` — set the selected window (opens popover).
- `deselect` — clear selection (closes popover).
- `clear` — reset all edits and selection.

**Derived state:** merge pending edits with fetched vocalization labels to compute display state per window. For each window, the effective label set is: `(saved labels - removes) + adds`.

**Mutual exclusivity enforcement:** handled in the reducer. Adding `"(Negative)"` queues removal of all existing type labels and clears any pending type adds. Adding a type label queues removal of `"(Negative)"` and clears any pending `"(Negative)"` add.

### 6.2 Dirty State

The toolbar shows a dirty indicator and unsaved count. Dirty = any entry in the edits map with non-empty `adds` or `removes`.

## 7. Toolbar: VocLabelToolbar

Simpler than detection LabelToolbar — no Select/Add mode toggle, no label radio buttons, no drag support.

**Controls:**
- **Save** button — yellow dot indicator when dirty. Serializes reducer state to batch edit list, calls batch endpoint, clears reducer on success.
- **Cancel** button — discards all pending changes and exits label mode. Warns with confirmation dialog if dirty.
- **Unsaved change count** — text showing number of pending edits.

## 8. Backend: Batch Vocalization Label Endpoint

### 8.1 Endpoint

**Route:** `PATCH /labeling/vocalization-labels/{detection_job_id}/batch`

**Request schema (`VocalizationLabelBatchRequest`):**
```python
class VocalizationLabelBatchEditItem(BaseModel):
    action: Literal["add", "delete"]
    row_id: str
    label: str
    source: str = "manual"

class VocalizationLabelBatchRequest(BaseModel):
    edits: list[VocalizationLabelBatchEditItem]
```

**Behavior:**
- All edits applied in a single database transaction.
- `action: "add"` — creates a `VocalizationLabel` row. Server-side enforcement of `"(Negative)"` mutual exclusivity: adding `"(Negative)"` deletes all type labels for that `(detection_job_id, row_id)`; adding a type label deletes any `"(Negative)"` label for that row.
- `action: "delete"` — removes the `VocalizationLabel` matching `(detection_job_id, row_id, label)`. No-op if not found.
- Duplicate `add` for same `(row_id, label)` is idempotent (skip if already exists).

**Response:** `list[TimelineVocalizationLabel]` — the full updated label set for the detection job (same shape as the existing `/all` endpoint). This lets the frontend refresh its cache in one round-trip.

### 8.2 Schemas

New schemas added to `src/humpback/schemas/labeling.py`:
- `VocalizationLabelBatchEditItem`
- `VocalizationLabelBatchRequest`

## 9. Frontend Component Structure

```
TimelineViewer.tsx
├── SpectrogramViewport.tsx
│   ├── VocLabelEditor.tsx        (new — renders neutral bars + badges, click → popover)
│   │   └── VocLabelPopover.tsx   (new — anchored popover with badge toggles + type dropdown)
│   └── ... (existing overlays hidden during voc label mode)
├── VocLabelToolbar.tsx            (new — Save / Cancel / dirty count)
└── PlaybackControls.tsx           (Label button behavior extended for vocalization mode)
```

### 9.1 VocLabelEditor

- Renders inside SpectrogramViewport, positioned the same way as LabelEditor.
- Maps all detection windows (from detection content query) to neutral-colored bars.
- Overlays vocalization type badges on windows that have labels (saved or pending).
- Click handler on bars: dispatches `select(row_id)` to open popover.
- Receives merged label state (saved + pending) from `useVocLabelEdits`.

### 9.2 VocLabelPopover

- Anchored to the selected bar's position in the viewport.
- Fetches vocalization types from the existing `useVocalizationTypes` hook for the "+" dropdown.
- Displays current effective labels (saved - pending removes + pending adds) as colored badges.
- Badge click toggles add/remove via reducer dispatch.
- Closes on outside click or Escape; does not discard pending changes.

### 9.3 VocLabelToolbar

- Conditionally rendered when vocalization label edit mode is active.
- Save handler: converts reducer edits map to `VocalizationLabelBatchEditItem[]`, calls `patchVocalizationLabels()`, invalidates relevant queries, dispatches `clear`.
- Cancel handler: if dirty, shows confirmation dialog; dispatches `clear` and exits label mode.

## 10. Data Flow

1. **Enter mode**: user is in Vocalization overlay → clicks Label → `labelEditMode` set to `"vocalization"`. Detection content and vocalization labels are already fetched (or fetched now).
2. **Click window**: `select(row_id)` → popover opens showing effective labels for that window.
3. **Edit labels**: toggle badges / add from dropdown → reducer accumulates `adds`/`removes`.
4. **Close popover**: click outside or Escape → `deselect` → popover closes, edits retained.
5. **Click another window**: repeat steps 2–4.
6. **Save**: serialize edits → `PATCH .../batch` → on success, invalidate vocalization label queries → `clear` reducer.
7. **Cancel**: confirm if dirty → `clear` reducer → exit label mode.

## 11. Changes to Existing Code

| File | Change |
|------|--------|
| `TimelineViewer.tsx` | New `labelEditMode: "detection" \| "vocalization" \| null` state. Label button dispatches to correct editor based on `overlayMode`. |
| `PlaybackControls.tsx` | "Label" button works in both overlay modes with identical enable conditions. |
| `VocalizationOverlay.tsx` | Hidden when vocalization label edit mode is active. |
| `DetectionOverlay.tsx` | Hidden when vocalization label edit mode is active. |
| `src/humpback/api/routers/labeling.py` | New batch endpoint. |
| `src/humpback/schemas/labeling.py` | New batch edit schemas. |
| `frontend/src/api/client.ts` | New `patchVocalizationLabels()` function. |
| `frontend/src/api/types.ts` | New batch edit request/response types. |

## 12. What Doesn't Change

- Detection label editing mode (completely separate, untouched).
- Vocalization Labeling workspace page (independent workflow).
- Individual vocalization label CRUD endpoints (still available for other consumers).
- VocalizationOverlay read-only rendering (still used outside edit mode).
- Vocalization type management.

## 13. Testing

- **Backend**: unit tests for batch endpoint — add, delete, idempotent add, "(Negative)" mutual exclusivity, atomic transaction rollback on error.
- **Frontend**: Playwright tests for entering vocalization label mode, clicking a window, toggling labels in popover, save/cancel flow, dirty state warning.
