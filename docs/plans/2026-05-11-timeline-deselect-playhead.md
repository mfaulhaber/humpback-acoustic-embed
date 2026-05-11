# Timeline Deselect Playhead Implementation Plan

**Goal:** Let review/detail timelines clear event or token selection on blank timeline clicks and make playback start from the current playhead when nothing is selected.
**Spec:** Bug fix; no separate design spec.
**Primary domain:** signal-timeline
**Neighbor domains:** call-parsing, sequence-models, frontend-shell

---

### Task 1: Preserve Cleared Selection In Call Parsing Review Timelines

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Clicking blank spectrogram space clears the selected event in Segment review without immediately reselecting from the navigation index.
- [ ] Clicking blank spectrogram space clears the selected event in Classify review instead of ignoring the null selection.
- [ ] Keyboard, previous/next buttons, and event-bar clicks still select and navigate events.
- [ ] When no event is selected, toolbar or keyboard playback starts from the visible playhead position rather than the old event or the left edge of the view.

**Tests needed:**
- Add focused component coverage for blank-click deselection or playback prop behavior where feasible.
- Run targeted Call Parsing frontend tests and TypeScript.

---

### Task 2: Add Blank-Click Deselect To Event Encoder Timeline

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderTokenOverlay.tsx`
- Modify: `frontend/src/components/sequence-models/EventEncoderTimelinePanel.tsx`
- Modify: `frontend/src/components/sequence-models/EventEncoderTokenOverlay.test.tsx`

**Acceptance criteria:**
- [ ] Clicking blank token timeline space clears the selected token/event.
- [ ] Cleared selection persists until the user selects a token/event or navigates to one.
- [ ] Token-scoped navigation turns off when there is no selected token.
- [ ] When no token/event is selected, Event Encoder playback starts from the current playhead position.

**Tests needed:**
- Extend token overlay tests for blank background deselection.
- Run targeted Event Encoder component tests and TypeScript.

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx vitest run src/components/sequence-models/EventEncoderTokenOverlay.test.tsx src/components/sequence-models/eventEncoderTimelineNavigation.test.ts src/components/call-parsing/SegmentReviewWorkspace.test.tsx src/components/call-parsing/ClassifyReviewWorkspace.epoch.test.tsx`
2. `cd frontend && npx tsc --noEmit`
