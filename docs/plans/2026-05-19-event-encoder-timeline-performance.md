# Event Encoder Timeline Performance Implementation Plan

**Goal:** Reduce slow Event Encoder job detail timeline rendering by avoiding unnecessary initial tile requests and limiting token overlay DOM work to visible events.
**Spec:** N/A — bug fix from session-debug root-cause investigation
**Primary domain:** sequence-models
**Neighbor domains:** signal-timeline, frontend-shell

---

### Task 1: Start the Event Encoder Timeline at the Selected Event

**Files:**
- Modify: `frontend/src/components/timeline/provider/types.ts`
- Modify: `frontend/src/components/timeline/provider/TimelineProvider.tsx`
- Modify: `frontend/src/components/sequence-models/EventEncoderTimelinePanel.tsx`

**Acceptance criteria:**
- [x] `TimelineProvider` supports an optional initial center timestamp without changing existing callers.
- [x] Event Encoder timeline passes the initially selected event center so initial tile requests target useful context instead of the region job midpoint.
- [x] Navigation and explicit selection still recenter the timeline as before.

**Tests needed:**
- Frontend coverage showing the Event Encoder detail timeline does not request midpoint tiles before centering on the first selected event.
- TypeScript verification for the provider prop change.

---

### Task 2: Render Only Visible Event Encoder Token Bars

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderTokenOverlay.tsx`
- Modify: `frontend/src/components/sequence-models/EventEncoderTokenOverlay.test.tsx`

**Acceptance criteria:**
- [x] Token overlay renders event bars only when their event interval intersects the current viewport.
- [x] Off-screen events remain selectable through existing previous/next controls because navigation still uses the full timeline event list outside the overlay.
- [x] Background click selection clearing still works.

**Tests needed:**
- Component test covering visible, edge-overlapping, and off-screen token events.
- Existing token overlay click and badge tests remain green.

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx vitest run src/components/sequence-models/EventEncoderTokenOverlay.test.tsx src/components/timeline/provider/TimelineProvider.test.tsx`
2. `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
3. `cd frontend && npx tsc --noEmit`
4. `uv run pytest tests/`
