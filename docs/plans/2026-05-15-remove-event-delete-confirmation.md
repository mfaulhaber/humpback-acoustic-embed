# Remove Event Delete Confirmation Implementation Plan

**Goal:** Let single-event deletes in the Call Parsing segmentation detail timeline apply immediately as pending corrections without opening a confirmation dialog.
**Spec:** Bug fix; no separate design spec.
**Primary domain:** call-parsing
**Neighbor domains:** none

---

### Task 1: Remove Pending Event Delete Dialog

**Files:**
- Modify: `frontend/src/components/call-parsing/EventDetailPanel.tsx`
- Modify: `frontend/src/components/call-parsing/EventDetailPanel.test.tsx`

**Acceptance criteria:**
- [x] The segmentation detail event panel Delete Event action calls the pending delete handler directly.
- [x] Undo Delete remains available for pending deleted events.
- [x] No shared delete confirmation dialog is shown for this pending correction flow.
- [x] The focused component test documents immediate pending-delete behavior.

**Tests needed:**
- Update the focused EventDetailPanel component test for immediate delete behavior.
- Run the focused EventDetailPanel test and TypeScript.

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx vitest run src/components/call-parsing/EventDetailPanel.test.tsx`
2. `cd frontend && npx tsc --noEmit`
