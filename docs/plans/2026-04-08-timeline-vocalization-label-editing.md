# Timeline Vocalization Label Editing Implementation Plan

**Goal:** Add vocalization label editing to the timeline viewer via popover-based multi-label editing on detection windows, with batch save through a new atomic API endpoint.
**Spec:** [docs/specs/2026-04-08-timeline-vocalization-label-editing-design.md](../specs/2026-04-08-timeline-vocalization-label-editing-design.md)

---

### Task 1: Backend batch vocalization label endpoint

**Files:**
- Modify: `src/humpback/schemas/labeling.py`
- Modify: `src/humpback/api/routers/labeling.py`

**Acceptance criteria:**
- [ ] `VocalizationLabelBatchEditItem` and `VocalizationLabelBatchRequest` schemas added
- [ ] `PATCH /labeling/vocalization-labels/{detection_job_id}/batch` endpoint implemented
- [ ] `action: "add"` creates a VocalizationLabel row; idempotent if already exists
- [ ] `action: "delete"` removes matching VocalizationLabel; no-op if not found
- [ ] "(Negative)" mutual exclusivity enforced server-side on add
- [ ] All edits applied in a single transaction
- [ ] Response returns full updated `list[TimelineVocalizationLabel]` for the job

**Tests needed:**
- Add, delete, idempotent add, "(Negative)" exclusivity, mixed add+delete in one batch, empty edits list

---

### Task 2: Frontend API client and types for batch endpoint

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`

**Acceptance criteria:**
- [ ] `VocalizationLabelBatchEditItem` and `VocalizationLabelBatchRequest` types defined
- [ ] `patchVocalizationLabels(detectionJobId, edits)` client function added calling `PATCH .../batch`
- [ ] Response type matches `TimelineVocalizationLabel[]`

**Tests needed:**
- None (covered by integration/E2E tests)

---

### Task 3: `useVocLabelEdits` reducer hook

**Files:**
- Create: `frontend/src/hooks/queries/useVocLabelEdits.ts`

**Acceptance criteria:**
- [ ] State shape: `{ selectedRowId, edits: Map<row_id, { adds, removes }> }`
- [ ] `toggle_add` action adds/un-adds a label for a row
- [ ] `toggle_remove` action marks/un-marks a saved label for removal
- [ ] `select` / `deselect` actions manage selected window
- [ ] `clear` resets all state
- [ ] "(Negative)" mutual exclusivity: adding it clears type adds and queues type removes; adding a type clears "(Negative)" add and queues "(Negative)" remove
- [ ] `isDirty` derived boolean and `editCount` derived number exported
- [ ] Helper to serialize edits map into `VocalizationLabelBatchEditItem[]` for the API call

**Tests needed:**
- Reducer logic: toggle_add, toggle_remove, mutual exclusivity, clear, serialization to batch edit items

---

### Task 4: `VocLabelPopover` component

**Files:**
- Create: `frontend/src/components/timeline/VocLabelPopover.tsx`

**Acceptance criteria:**
- [ ] Compact popover anchored to a position prop (bar location in viewport)
- [ ] Header shows time range in `HH:MM:SS – HH:MM:SS UTC`
- [ ] Displays effective labels as colored badges (saved - removes + adds)
- [ ] Clicking a badge toggles removal (dimmed + strikethrough for pending removal)
- [ ] "+" button opens dropdown of available vocalization types from `useVocalizationTypes`
- [ ] Clicking a type in dropdown dispatches `toggle_add`
- [ ] "(Negative)" toggle enforces mutual exclusivity via reducer
- [ ] Closes on outside click or Escape via `deselect` dispatch
- [ ] Pending changes are not discarded on close

**Tests needed:**
- Playwright: open popover, add/remove labels, close and verify pending state retained

---

### Task 5: `VocLabelEditor` component

**Files:**
- Create: `frontend/src/components/timeline/VocLabelEditor.tsx`

**Acceptance criteria:**
- [ ] Renders inside SpectrogramViewport, positioned like LabelEditor
- [ ] Maps all detection windows to uniform neutral/dim bars
- [ ] Overlays vocalization type badges on windows that have labels (saved or pending)
- [ ] Pending-dirty windows show a visual indicator (ring or dot)
- [ ] Selected window (popover open) shows highlighted white border
- [ ] Click handler dispatches `select(row_id)` to open VocLabelPopover
- [ ] Receives merged label state (saved + pending) from useVocLabelEdits

**Tests needed:**
- Playwright: enter voc label mode, verify bars render, click a bar opens popover

---

### Task 6: `VocLabelToolbar` component

**Files:**
- Create: `frontend/src/components/timeline/VocLabelToolbar.tsx`

**Acceptance criteria:**
- [ ] Save button with yellow dot when dirty; calls `patchVocalizationLabels()` with serialized edits, invalidates vocalization label queries, dispatches `clear`
- [ ] Cancel button with confirmation dialog when dirty; dispatches `clear` and exits label mode
- [ ] Unsaved change count displayed
- [ ] Save button disabled when not dirty or save in progress

**Tests needed:**
- Playwright: save flow (verify API call and state reset), cancel with dirty warning

---

### Task 7: Wire into TimelineViewer and PlaybackControls

**Files:**
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`
- Modify: `frontend/src/components/timeline/PlaybackControls.tsx`
- Modify: `frontend/src/components/timeline/SpectrogramViewport.tsx`

**Acceptance criteria:**
- [ ] `labelEditMode` state added: `"detection" | "vocalization" | null`
- [ ] Label button in PlaybackControls dispatches to correct editor based on `overlayMode`
- [ ] Label button enable conditions work in both overlay modes (paused + 1m/5m zoom)
- [ ] VocLabelEditor and VocLabelToolbar rendered when `labelEditMode === "vocalization"`
- [ ] DetectionOverlay and VocalizationOverlay hidden when `labelEditMode === "vocalization"`
- [ ] Existing detection label editing continues to work unchanged when `overlayMode === "detection"`
- [ ] Entering vocalization label mode exits detection label mode and vice versa
- [ ] Exiting label mode warns about unsaved changes

**Tests needed:**
- Playwright: toggle between detection and vocalization overlay modes, verify correct editor activates, verify mutual exclusivity of label modes

---

### Task 8: Backend unit tests for batch endpoint

**Files:**
- Modify: `tests/test_labeling_api.py` (or create `tests/test_vocalization_label_batch.py` if cleaner)

**Acceptance criteria:**
- [ ] Test add single label
- [ ] Test delete single label
- [ ] Test idempotent add (same row_id + label twice)
- [ ] Test delete non-existent label (no-op, no error)
- [ ] Test "(Negative)" mutual exclusivity on add
- [ ] Test mixed add + delete in one batch
- [ ] Test empty edits list returns current labels
- [ ] Test response matches TimelineVocalizationLabel shape

**Tests needed:**
- All of the above are the tests

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/schemas/labeling.py src/humpback/api/routers/labeling.py`
2. `uv run ruff check src/humpback/schemas/labeling.py src/humpback/api/routers/labeling.py`
3. `uv run pyright src/humpback/schemas/labeling.py src/humpback/api/routers/labeling.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
