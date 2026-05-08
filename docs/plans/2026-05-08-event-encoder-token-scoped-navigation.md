# Event Encoder Token-Scoped Navigation Implementation Plan

**Goal:** Let users toggle the selected Event Encoder token badge so previous and next navigation jumps between events with the same token type.
**Spec:** [docs/specs/2026-05-08-event-encoder-token-scoped-navigation-design.md](../specs/2026-05-08-event-encoder-token-scoped-navigation-design.md)
**Primary domain:** sequence-models
**Neighbor domains:** signal-timeline, frontend-shell

---

### Task 1: Add Pure Timeline Navigation Helper

**Files:**
- Create: `frontend/src/components/sequence-models/eventEncoderTimelineNavigation.ts`
- Create: `frontend/src/components/sequence-models/eventEncoderTimelineNavigation.test.ts`

**Acceptance criteria:**
- [x] The helper derives all-event navigation and same-token navigation from the loaded selected-k timeline events without mutating or resorting the array.
- [x] Same-token navigation scopes to the currently selected event's `token_id` and returns the selected event's same-token position and same-token total.
- [x] Previous and next targets are bounded to the active navigation list and report disabled states at active-list boundaries.
- [x] Missing selected events, empty event arrays, and token-scoped mode without a selected event fall back to normal safe disabled behavior.
- [x] The helper remains local to the Event Encoder timeline surface and does not introduce a shared token vocabulary abstraction.

**Tests needed:**
- Vitest coverage for normal event navigation, same-token skipping, boundary disabled states, missing selected event fallback, single-token occurrence behavior, and preservation of original event ordering.

---

### Task 2: Wire Token-Scoped Navigation Into The Timeline Panel

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderTimelinePanel.tsx`

**Acceptance criteria:**
- [x] `EventEncoderTimelineBody` owns `tokenScopedNavigation` local state initialized to off.
- [x] Toolbar previous and next buttons select targets through the navigation helper rather than raw full-list index arithmetic.
- [x] `A` and `D` keyboard shortcuts use the same active navigation target as the toolbar buttons.
- [x] Clicking a timeline event or cluster projection point still selects and centers that event; when token-scoped mode is active, the scope naturally follows the newly selected event's token.
- [x] Playback, zoom, pan, selected feature table, k selection, and cluster projection behavior remain otherwise unchanged.
- [x] Token-scoped navigation resets to off when the Event Encoder job changes or the selected k changes.

**Tests needed:**
- Playwright coverage for toolbar and keyboard navigation behavior.
- TypeScript coverage for the new helper integration and state wiring.

---

### Task 3: Make The Selected Token Badge An Accessible Toggle

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderTimelinePanel.tsx`

**Acceptance criteria:**
- [x] The selected token display becomes a badge-styled button with `aria-pressed` and `data-testid="eej-token-nav-toggle"`.
- [x] The toggle uses the selected token color for border and text, and uses a subtle active background while preserving label readability.
- [x] The toggle title communicates the current action, such as navigating by the selected token when inactive and returning to all-event navigation when active.
- [x] Previous and next button titles reflect normal mode or token-scoped mode with the selected token label.
- [x] In token-scoped mode, a compact occurrence counter shows the selected event's position within the same-token list.
- [x] Previous and next controls disable at same-token boundaries; if the selected token occurs once, both are disabled while the mode is active.
- [x] The existing selected token label text and confidence display remain visible and compact in the toolbar.

**Tests needed:**
- Playwright assertions for `aria-pressed`, active/inactive toggle behavior, same-token occurrence counter, and button disabled states at same-token boundaries.

---

### Task 4: Expand Event Encoder Frontend Coverage

**Files:**
- Modify: `frontend/e2e/sequence-models/event-encoder.spec.ts`

**Acceptance criteria:**
- [x] Complete-job timeline mock data includes at least three events where the first and third share a token and the second has a different token.
- [x] Existing timeline assertions for panel placement, selected features, cluster projection, k switching, and playback-triggered audio requests still pass.
- [x] Normal `D` navigation moves from event 1 to event 2 before the token toggle is active.
- [x] Toggling the selected token badge on at event 1 makes `D` and the next button jump to the next event with the same token.
- [x] Toggling the selected token badge off restores normal event-by-event navigation.
- [x] Changing selected k turns token-scoped navigation off.

**Tests needed:**
- `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`

---

### Task 5: Update Domain Context And References

**Files:**
- Modify: `docs/reference/frontend.md`
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/tests.md`

**Acceptance criteria:**
- [x] Frontend reference describes token-scoped navigation as a read-only Event Encoder detail timeline affordance.
- [x] Sequence Models domain context mentions that Event Encoder token navigation is selected-k and job-local.
- [x] Sequence Models targeted frontend tests include the navigation helper test and Event Encoder Playwright coverage.
- [x] Documentation does not imply token labels are stable across jobs or k values.

**Tests needed:**
- Documentation diff review and `git diff --check`.

---

### Verification

Run in order after all tasks:

1. `git diff --check`
2. `cd frontend && npx tsc --noEmit`
3. `cd frontend && npx vitest run src/components/sequence-models/eventEncoderTimelineNavigation.test.ts src/components/sequence-models/EventEncoderTokenOverlay.test.tsx`
4. `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
5. `uv run pytest tests/`
