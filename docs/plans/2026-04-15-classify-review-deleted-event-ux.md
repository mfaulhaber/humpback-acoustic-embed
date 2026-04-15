# Classify Review Deleted Event UX — Implementation Plan

**Goal:** Improve classify review UX by suppressing overlapping deleted-event ghosts and filtering deleted events from navigation
**Spec:** [docs/specs/2026-04-15-classify-review-deleted-event-ux-design.md](../specs/2026-04-15-classify-review-deleted-event-ux-design.md)

---

### Task 1: Filter deleted events from navigation

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] Derive `navigableEvents` by filtering `events` to exclude events with saved boundary-deletion corrections (check `savedBoundaryCorrections` for matching event_id with `correction_type === "delete"`)
- [ ] Replace `events` with `navigableEvents` for: `currentEventIndex` bounds, `goPrev`/`goNext` callbacks, the "Event N of M" counter text, and `currentEvent` derivation
- [ ] Keep the unfiltered `events` list available for `regionEffectiveEvents` so ghost bars still render
- [ ] Reset `currentEventIndex` to 0 when `navigableEvents` changes identity (job switch already does this)

**Tests needed:**
- Playwright test: load classify review with a job that has boundary-deletion corrections, verify counter shows reduced count, verify arrow navigation skips deleted events

---

### Task 2: Suppress overlapping deleted-event ghosts in EventBarOverlay

**Files:**
- Modify: `frontend/src/components/call-parsing/EventBarOverlay.tsx`

**Acceptance criteria:**
- [ ] When computing `deletedEvents` for rendering, filter out any deleted event whose `[startSec, endSec]` range is fully covered by at least one active event in `sortedEvents` (active.startSec <= deleted.startSec AND active.endSec >= deleted.endSec)
- [ ] Deleted events NOT fully covered by an active event continue to render as ghost bars (unchanged)
- [ ] No changes to the active event rendering loop

**Tests needed:**
- Playwright test: verify that a deleted event overlapped by an adjusted event does not render a ghost bar; verify a deleted event NOT overlapped still renders

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test`
