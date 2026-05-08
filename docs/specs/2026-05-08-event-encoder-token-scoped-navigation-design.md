# Event Encoder Token-Scoped Navigation - Design

**Date:** 2026-05-08
**Status:** Draft
**Primary domain:** Sequence Models
**Neighbor domains:** Signal Timeline, Frontend Shell

## 1. Goal

Add a token-scoped navigation mode to the Event Encoder job detail timeline.
When the user toggles the selected event's token badge on, the existing
previous and next controls navigate to the previous or next event with the same
token id as the currently selected event. When toggled off, previous and next
return to normal event-by-event navigation.

The feature is an exploration aid for reading local repetitions of a token
type in audio context. It must stay read-only and must not imply that token ids
are stable outside the current Event Encoder job and selected k.

## 2. Scope

### In scope

- Add local UI state to `EventEncoderTimelinePanel` for token-scoped navigation.
- Make the selected token badge in the timeline toolbar an accessible toggle.
- Route toolbar previous and next buttons through either all events or the
  selected event's same-token subset depending on toggle state.
- Route `A` and `D` keyboard navigation through the same navigation mode.
- Keep event click selection, playback, zoom, selected feature table, and
  cluster projection selection behavior unchanged.
- Preserve token-scoped mode across event selection within the same job and k.
- Reset token-scoped mode when the Event Encoder job changes or selected k
  changes.
- Update focused frontend tests and Event Encoder Playwright coverage.

### Non-goals

- No backend API changes.
- No changes to Event Encoder artifacts or tokenization output.
- No global token filter, hidden events, or overlay-only subset view.
- No cross-job or cross-k token identity. Token ids remain job-local and
  k-local.
- No editing of event tokens or labels.

## 3. Existing Context

- `EventEncoderTimelinePanel` already fetches one selected-k timeline response
  through `useEventEncoderTimeline(job.id, selectedK, isComplete)`.
- The timeline response already contains all fields needed for this feature:
  `event_id`, `token_id`, `token_label`, `start_timestamp`, and
  `end_timestamp`.
- The existing panel stores `selectedEventId` and `selectedListIndex`, derives
  `selectedEvent`, and uses `selectIndex(index)` for toolbar and keyboard
  navigation.
- The selected token badge in the toolbar currently displays
  `selectedEvent.token_label` as a passive `Badge`.
- `EventEncoderTokenOverlay` already renders compact token badges on bars, but
  the full bar is the click target for event selection.
- Sequence Models invariants state that Event Encoder token ids and `Txx`
  labels are job-local and k-local, so token-scoped navigation must derive from
  the currently loaded timeline rows only.

## 4. Approaches Considered

### Approach A: Frontend-Only Navigation Scope

Keep all events visible, add a boolean `tokenScopedNavigation` state, and derive
the active navigation list in `EventEncoderTimelineBody`.

When the mode is off, the navigation list is `events`. When the mode is on and
an event is selected, the navigation list is all events whose `token_id` equals
`selectedEvent.token_id`, preserving the existing event sort order from the
timeline endpoint.

Pros:

- No API or artifact changes.
- Uses already-loaded, selected-k artifact data.
- Keeps all surrounding events visible for context.
- Gives toolbar buttons and keyboard shortcuts one shared navigation path.
- Handles large-enough current jobs without extra network round trips because
  the timeline endpoint already returns the rows being displayed.

Cons:

- The browser still holds all event rows even when the user only wants one token
  type.
- If a future timeline endpoint adds pagination, this behavior would need to
  become page-aware or server-assisted.

Verdict: recommended.

### Approach B: Backend Token Filter Parameter

Extend `GET /sequence-models/event-encoders/{job_id}/timeline` with a
`token_id` query parameter and refetch same-token rows when the mode is enabled.

Pros:

- Could reduce payload size for very large Event Encoder jobs.
- Gives the backend ownership of token filtering if timeline pagination appears.

Cons:

- Adds backend contract surface for a local navigation state.
- Requires preserving the full unfiltered timeline or refetching to return to
  normal navigation.
- Adds latency to a mode that should feel instantaneous.
- Makes the timeline response less directly tied to the visible overlay.

Verdict: defer unless timeline payload size becomes a real problem.

### Approach C: Token Filter Overlay Mode

Turn the selected token badge into a full token filter: hide or dim all events
with other token ids and make previous/next navigate only among visible events.

Pros:

- Strong visual emphasis on repeated token occurrences.
- Could be useful for cluster exploration.

Cons:

- Bigger behavior change than requested.
- Hiding events removes temporal context while listening.
- Dimming unrelated events introduces a second visual state that can be confused
  with selection or confidence.
- It overlaps with possible future token filter tools in cluster analysis.

Verdict: not recommended for this change.

## 5. Recommended Frontend Design

Implement Approach A entirely in
`frontend/src/components/sequence-models/EventEncoderTimelinePanel.tsx`.

Add local state in `EventEncoderTimelineBody`:

- `tokenScopedNavigation: boolean`

Derive:

- `selectedEvent`
- `selectedTokenId = selectedEvent?.token_id ?? null`
- `sameTokenEvents = selectedEvent ? events.filter((event) => event.token_id === selectedEvent.token_id) : []`
- `navigationEvents = tokenScopedNavigation && selectedEvent ? sameTokenEvents : events`
- `navigationIndex = navigationEvents.findIndex((event) => event.event_id === selectedEventId)`
- `effectiveNavigationIndex = navigationIndex >= 0 ? navigationIndex : 0`

Use a helper such as `selectNavigationOffset(delta)` for both buttons and
keyboard shortcuts:

- Bound `effectiveNavigationIndex + delta` to `navigationEvents`.
- Select the target event by `event_id`.
- Find its index in the full `events` list before calling the existing
  `onSelectIndex`.
- Center the selected event with the existing `centerEvent` helper.

Keep the existing global event counter as `Event X / N`, where `X` is the
selected event's position in the full event list. In token-scoped mode, add a
compact secondary counter next to the token badge, for example `2 / 7`, meaning
the selected event is the second visible occurrence of that token in the loaded
timeline.

## 6. Toggle UI

Replace the passive selected-token `Badge` in the timeline toolbar with a
badge-styled `button` or `Button` variant that displays the same token label.

Required behavior:

- Only render the toggle when `selectedEvent` exists.
- Use `aria-pressed={tokenScopedNavigation}`.
- Use `data-testid="eej-token-nav-toggle"`.
- Use the selected token color for border and text in both states.
- When active, use a subtle filled background derived from the token color and
  keep the label legible.
- Use a `title` such as `Navigate by T17 token` when inactive and
  `Return to all-event navigation` when active.
- Keep the displayed label as the token badge itself, for example `T17`.

Button titles should reflect mode:

- Normal mode: `Previous event`, `Next event`.
- Token-scoped mode: `Previous T17 event`, `Next T17 event`.

Disabled behavior:

- Normal mode disables previous at the first event and next at the last event.
- Token-scoped mode disables previous or next at the ends of the same-token
  list.
- If the selected token occurs only once, both previous and next are disabled
  while token-scoped mode is active.

The overlay bars stay visible and selectable. Clicking another event while
token-scoped mode is active selects that event and naturally changes the scoped
token to the newly selected event's `token_id`. This keeps the rule simple:
the mode follows the currently selected event's token.

## 7. State Rules

- Initial load: token-scoped navigation is off.
- Toggle on with a selected event: scope to that event's `token_id`.
- Toggle off: return to all-event navigation without changing selection.
- Click another event while on: keep the mode on and scope to the newly selected
  event's token.
- Change selected k: reset token-scoped navigation to off because token ids are
  k-local.
- Change Event Encoder job: reset token-scoped navigation to off.
- Empty timeline or no selected event: render no toggle and use normal disabled
  navigation behavior.
- Projection plot selection should behave like timeline bar selection: it may
  change the selected event, and if token-scoped mode is on the scope follows
  the selected event's token.

## 8. Keyboard Behavior

Keep the existing local shortcut handler and route only the event navigation
targets through the active navigation scope.

- `A`: previous event in the active navigation scope.
- `D`: next event in the active navigation scope.
- `Space`: unchanged, play or pause selected event slice.
- `+` / `=` and `-`: unchanged zoom behavior.
- `ArrowLeft` / `ArrowRight`: unchanged pan behavior.

Shortcut guards stay unchanged:

- Ignore shortcuts when focus is inside `input`, `textarea`, `select`, or a
  contenteditable element.
- Ignore modified keypresses using `Meta`, `Ctrl`, or `Alt`.

No new keyboard shortcut is required for toggling the token scope. The selected
token badge is the explicit control.

## 9. Implementation Notes

- Keep the current `events` array as the only source of truth. Do not mutate or
  resort it.
- Avoid storing a separate locked token id. Derive the active token id from the
  current selected event so clicks and projection selections remain intuitive.
- Keep `selectedListIndex` as the full-list fallback used across k changes.
- Prefer extracting small pure helpers if the component body becomes difficult
  to read:
  - `getNavigationState(events, selectedEventId, tokenScopedNavigation)`
  - `findFullEventIndex(events, eventId)`
- A small helper is easy to cover with component tests, but it should remain
  local to the Event Encoder timeline unless another Sequence Models surface
  adopts the same behavior.

## 10. Tests

### Frontend component coverage

Add or extend a focused test for the navigation helper or
`EventEncoderTimelinePanel` behavior if the existing setup supports it:

- Toggle starts inactive.
- In normal mode, next from event 1 selects event 2 even when token ids differ.
- After toggling on at event 1 with token `T17`, next skips to the next `T17`
  event.
- Previous in token-scoped mode skips back to the previous `T17` event.
- Buttons disable at same-token boundaries.
- Clicking an event with token `T42` while scoped keeps mode on and changes the
  active scope to `T42`.
- Changing k resets the mode to off.

### Playwright coverage

Update `frontend/e2e/sequence-models/event-encoder.spec.ts` with mocked
timeline data containing at least three events where the first and third share a
token and the second has a different token.

Assertions:

- The selected token badge renders as a toggle with `aria-pressed="false"`.
- Normal `D` navigation moves from event 1 to event 2.
- Returning to event 1, toggling the badge on, then pressing `D` moves to event
  3 and keeps the selected token label `T17`.
- The toolbar next button follows the same token-scoped behavior.
- Toggling off restores normal event navigation.
- Selecting another k turns the toggle off.

### Verification commands

- `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
- `cd frontend && npx vitest run src/components/sequence-models/EventEncoderTokenOverlay.test.tsx`
- `cd frontend && npx tsc --noEmit`

No backend tests are required for this change because the API response contract
already contains the token fields needed by the frontend.

## 11. Risks And Follow-Ups

- The selected token badge becomes both a data display and a mode toggle. The
  active styling and `aria-pressed` state need to make that clear without adding
  bulky explanatory text to the toolbar.
- Very large Event Encoder jobs still load all displayed events. This is
  inherited from the existing timeline endpoint and is acceptable for this
  frontend-only navigation change.
- Future time-window pagination would require server-assisted same-token
  navigation or explicit "next matching event outside current window" behavior.
- A future cluster-analysis panel may want a stronger token filter or highlight
  mode. This spec deliberately keeps the current timeline in context and changes
  only prev/next targeting.
