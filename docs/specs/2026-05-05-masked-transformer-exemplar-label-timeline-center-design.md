# Masked Transformer Exemplar Label Click-to-Center — Design

**Status:** design
**Date:** 2026-05-05
**Scope:** Frontend-only UX change on the Masked Transformer detail page (`/app/sequence-models/masked-transformer/:jobId`).

## 1. Problem

The Masked Transformer Exemplars panel shows vocalization label badges for each
exemplar via `ExemplarEventTypeChips`. Those badges are useful for spotting
which token exemplars align with labeled call-parsing events, but clicking one
currently navigates away to Classify Review when `event_id` is present.

For interpretation work, the more immediate need is local context: when a
reviewer clicks a vocal label badge, the Token Timeline should recenter on the
corresponding labeled exemplar/event so the reviewer can inspect the token
strip, spectrogram, and motif overlays around that moment without leaving the
Masked Transformer page.

## 2. Goals

- Clicking a vocal label badge in the Masked Transformer Exemplars panel recenters
  the Token Timeline on that labeled exemplar.
- The timeline center is the exemplar midpoint:
  `(exemplar.start_timestamp + exemplar.end_timestamp) / 2`.
- The page scrolls the Token Timeline card into view after the seek so the
  action is visible even when the user clicked from the lower Exemplars panel.
- The action uses the existing `TimelinePlaybackHandle.seekTo(...)` path. No
  duplicate timeline state or standalone audio/viewer logic is introduced.
- Background chips remain inert because they have no labeled event.

## 3. Non-Goals

- Do not change Masked Transformer artifact generation.
- Do not add a backend endpoint or fetch Classify typed-event bounds for this v1.
- Do not change HMM exemplar behavior.
- Do not start playback when the badge is clicked.
- Do not change label-distribution semantics or regenerate artifacts.

## 4. Existing Code

- `MaskedTransformerDetailPage` already owns
  `timelineHandleRef: React.RefObject<TimelinePlaybackHandle>` and passes it to
  `TimelineSection` / `MotifSection`.
- `TimelinePlaybackHandle.seekTo(epoch)` already recenters the provider state and
  clamps to `jobStart` / `jobEnd`.
- `ExemplarsSection` renders exemplar cards from `useMaskedTransformerExemplars`.
- `ExemplarEventTypeChips` renders each `extras.event_types` entry as a badge;
  today, if `extras.event_id` is present, the badge is wrapped in a `Link` to
  Classify Review.
- `ExemplarRecord.start_timestamp` and `end_timestamp` are already absolute
  epoch timestamps on current sequence-model artifacts.

## 5. Design

### 5.1 Lift a Timeline Seek Callback

Add a handler in `MaskedTransformerDetailPage`:

- `handleCenterExemplar(exemplar: ExemplarRecord)`
- Compute `center = (exemplar.start_timestamp + exemplar.end_timestamp) / 2`.
- Call `timelineHandleRef.current?.seekTo(center)`.
- Call `timelineCardRef.current?.scrollIntoView({ behavior: "smooth", block: "center" })`.

Add `timelineCardRef = useRef<HTMLDivElement | null>(null)` at the page level
and pass it to `TimelineSection`, where it attaches to the root element around
`data-testid="mt-timeline-viewer"` or the containing timeline `Card`.

### 5.2 Pass the Callback to Exemplars

Thread an optional callback through the exemplar components:

- `ExemplarsSection`
  - new prop: `onCenterExemplar?: (exemplar: ExemplarRecord) => void`
- `ExemplarList`
  - forwards `onCenterExemplar`
- `ExemplarEventTypeChips`
  - new prop: `onCenterExemplar?: (exemplar: ExemplarRecord) => void`

Only `MaskedTransformerDetailPage` supplies the callback. This keeps the change
page-local and avoids touching HMM or shared Sequence Models primitives.

### 5.3 Badge Behavior

For non-background labels:

- Render each label chip as a button-like badge when `onCenterExemplar` is
  present.
- On click, call `event.preventDefault()` and `onCenterExemplar(exemplar)`.
- Add accessible text via `aria-label`, for example:
  `Center timeline on Moan exemplar at 123.45 seconds`.
- Add a compact visual hint via `title="Center in timeline"`.
- Keep the badge styling dense and chip-like; do not add explanatory text inside
  the panel.

The existing Classify Review navigation should not remain the primary badge
click behavior because it conflicts with the requested interaction. If preserving
navigation is desired, add a separate small external-link icon next to the chip
in a later follow-up.

For background chips:

- Leave the current inert chip behavior. It represents no labeled event and
  should not seek.

### 5.4 Target Timestamp Choice

Use the exemplar midpoint for v1.

Reasoning:

- It is available in the current `ExemplarRecord` without another request.
- The label attached to the exemplar is assigned because the exemplar window
  center falls inside an effective event, so centering the exemplar window lands
  inside the labeled event.
- It works for regenerated artifacts and existing artifacts without adding new
  `extras.event_start_timestamp` / `extras.event_end_timestamp` fields.

If later users need exact event centering rather than exemplar-window centering,
the follow-up should extend generated exemplar extras with absolute
`event_start_timestamp` and `event_end_timestamp`, then use their midpoint when
present. That path requires backend artifact changes and regeneration, so it is
out of scope for this small UI request.

## 6. Affected Files

- `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`
  - add `timelineCardRef`
  - add `handleCenterExemplar`
  - pass `timelineCardRef` to `TimelineSection`
  - pass `handleCenterExemplar` to `ExemplarsSection`
  - thread `onCenterExemplar` through `ExemplarList` and `ExemplarEventTypeChips`
- `frontend/e2e/sequence-models/masked-transformer.spec.ts`
  - add or extend an exemplar-panel test for click-to-center behavior

## 7. Alternatives Considered

### A. Frontend-only seek to exemplar midpoint

This is the recommended approach.

Pros:

- Smallest implementation.
- Uses existing `TimelinePlaybackHandle.seekTo`.
- Works with current artifacts and test fixtures.
- No artifact regeneration or API migration.

Cons:

- Centers the exemplar window, not necessarily the exact full event bounds.

### B. Fetch typed-event bounds by `event_id` and center the event

Pros:

- Better matches the literal phrase "labeled event is centered."
- Reuses the existing typed-events API shape, which includes `event_id`,
  `start_sec`, and `end_sec`.

Cons:

- Requires converting region-relative event seconds to absolute epoch time.
- Needs another query keyed by the bound Classify job.
- Saved boundary corrections and added events make exact resolution less simple.
- Adds latency to a local UI gesture.

### C. Add event bounds to exemplar artifact extras

Pros:

- Exact event centering without a runtime lookup.
- Makes future exemplar interactions richer.

Cons:

- Backend artifact change.
- Existing artifacts remain missing the new fields until regenerated.
- Larger than the requested UI feature.

## 8. Test Plan

Frontend unit/component testing:

- If the existing component test setup can render `MaskedTransformerDetailPage`
  with mocked queries, assert that clicking a label chip calls the timeline
  ref's `seekTo` with the exemplar midpoint and calls `scrollIntoView`.

Playwright:

- Extend `frontend/e2e/sequence-models/masked-transformer.spec.ts` with a
  fixture exemplar whose `start_timestamp` / `end_timestamp` are away from the
  initial timeline viewport.
- Navigate to the complete Masked Transformer detail page.
- Open/locate the Exemplars panel.
- Click a vocal label chip (`data-testid="mt-exemplar-type-chip"`).
- Assert the Token Timeline is visible.
- Assert the timeline viewport recenters around the exemplar midpoint. If direct
  provider state is not exposed, assert via visible token/spectrogram position
  or a narrowly-scoped test-only marker rather than coupling to implementation
  internals.
- Assert clicking `(background)` does not seek.

Verification:

- `cd frontend && npm run test -- --run`
- `cd frontend && npx playwright test e2e/sequence-models/masked-transformer.spec.ts`
- Full project gates per CLAUDE.md §10.2 before session end.

## 9. Acceptance Criteria

- Clicking a non-background vocal label badge in the Masked Transformer Exemplars
  panel keeps the user on the Masked Transformer detail page.
- The Token Timeline center moves to the clicked exemplar midpoint.
- The Token Timeline card is scrolled into view after the click.
- Background chips do not trigger timeline movement.
- Existing motif timeline controls continue to recenter through the same
  `TimelinePlaybackHandle` path.
- No backend files, database migrations, or artifact formats are changed.
