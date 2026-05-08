# Event Encoder Token Frequency Piano Roll — Implementation Plan

**Goal:** Add a dedicated full-viewport piano roll viewer for Event Encoder
jobs that encodes token identity, tonal range, dominant frequency, and ridge
slope direction on a time × frequency canvas with audio playback.

**Spec:** `docs/specs/2026-05-08-event-encoder-piano-roll-design.md`

**Prototype:** `/tmp/token-swimlane-prototype.html` (Canvas rendering logic,
visual encoding, interaction patterns)

**Primary domain:** sequence-models

**Neighbor domains:** signal-timeline, frontend-shell

---

### Task 1: Piano Roll Page Component and Route

**Files:**
- Create: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`
- Modify: `frontend/src/App.tsx`

**Acceptance criteria:**
- [x] New route `/app/sequence-models/event-encoder/:jobId/piano-roll`
  registered in App.tsx
- [x] Page component loads the job via `useEventEncoderJob`, shows loading and
  error states
- [x] When the job is complete, fetches timeline data via
  `useEventEncoderTimeline` with the selected k
- [x] Top toolbar renders: back link to detail page, job ID, event count,
  duration, token count, Y-mode selector, frequency max selector, unvoiced
  mode selector, play/pause button, k-value selector
- [x] Status bar renders at bottom with placeholder cursor info and keyboard
  shortcut hints
- [x] Full-viewport layout: toolbar + canvas area + status bar, no scrolling

**Tests needed:**
- Route renders and shows toolbar when given a valid complete job
- Loading and error states render correctly
- K-value selector triggers data refetch

---

### Task 2: Canvas Renderer — Grid, Events, and Visual Encoding

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`

**Acceptance criteria:**
- [x] Canvas element fills the available space between toolbar and status bar,
  scaled by `devicePixelRatio` for Retina
- [x] Frequency grid lines drawn with Hz labels on the Y axis
- [x] Time grid lines drawn with mm:ss labels relative to recording start
- [x] Each event rendered as a rectangle: X from start/end timestamps, Y
  center from median F0 (or peak frequency based on Y-mode), height from F0
  range, minimum 4 px height and 2 px width
- [x] Rectangle fill color from `labelColor(token_id, k)` with opacity scaled
  by voicing fraction
- [x] Unvoiced events (voicing ≤ 0.3) rendered according to the selected
  unvoiced mode: at peak frequency with solid border and light fill, in a
  bottom lane, or hidden
- [x] Ridge slope line drawn inside rectangles at least 12 px wide and 6 px
  tall, tilting by `ridge_log_frequency_slope`, clamped to ±40% of rect
  height, with arrowhead when rect ≥ 30 px wide and slope magnitude > 0.3
- [x] Token label with opaque background pill rendered centered inside
  rectangles at least 20 px wide and 10 px tall
- [x] Events outside the visible viewport are frustum-culled (not drawn)

**Tests needed:**
- Canvas renders without error when timeline data is loaded
- Visual regression not required for prototype phase; verify canvas element
  exists with correct dimensions in Playwright

---

### Task 3: Pan, Zoom, and Keyboard Interaction

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`

**Acceptance criteria:**
- [x] Scroll (no modifier) zooms the time axis centered on cursor
- [x] Shift+Scroll zooms the frequency axis centered on cursor
- [x] Drag pans the time axis only (no frequency panning)
- [x] Zoom has minimum and maximum bounds (time: 2 s to full recording,
  frequency: clamped 0–5000 Hz)
- [x] F key fits all events in view
- [x] +/= zooms in, - zooms out (time axis)
- [x] ArrowLeft/ArrowRight pans by 10% of viewport span
- [x] Escape clears selection and token filter
- [x] All keyboard shortcuts suppressed when focus is in input/textarea/select
  or when meta/ctrl/alt modifiers are held
- [x] Status bar updates with cursor time, cursor frequency, and current zoom
  span on mouse move

**Tests needed:**
- Keyboard shortcuts suppressed when focus is in an input element
- Escape clears selection state

---

### Task 4: Selection, Tooltip, and Token Legend

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`

**Acceptance criteria:**
- [x] Click on an event selects it (white border highlight), does not activate
  token filtering
- [x] Hover shows a tooltip with: token label (colored), duration, median F0,
  F0 range, peak frequency, voicing fraction, ridge slope with direction
  indicator, pulse rate, gap from previous, position number
- [x] Double-click zooms to fit the clicked event with padding
- [x] Token legend panel (top-right, collapsible) lists all token families
  with color swatch, label, count, and mean F0 or "noise"
- [x] Clicking a token in the legend dims all non-matching events; clicking
  again clears the filter
- [x] Escape clears both selection and filter

**Tests needed:**
- Selection state toggles on click
- Legend toggle collapses and expands

---

### Task 5: Minimap

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`

**Acceptance criteria:**
- [x] A 240 × 40 px canvas in the bottom-right corner of the main canvas area
- [x] Shows all events as 2 px marks at their time × frequency position,
  colored by token
- [x] White rectangle outline shows the current viewport bounds
- [x] Clicking the minimap centers the viewport on the clicked time position
- [x] Minimap redraws when viewport or data changes

**Tests needed:**
- Minimap canvas element is rendered
- Minimap click updates the main viewport (verify via state change, not visual)

---

### Task 6: Audio Playback

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`

**Acceptance criteria:**
- [x] Play/pause button in the toolbar and Space keyboard shortcut control an
  HTML audio element
- [x] When an event is selected, playback uses `regionAudioSliceUrl` with the
  event's start timestamp and duration (end - start), bounded to a minimum of
  0.1 s
- [x] When no event is selected, playback uses `regionAudioSliceUrl` with the
  viewport start and the viewport span capped at 30 s
- [x] Play/pause state is reflected in the button icon (Play/Pause from
  lucide-react)
- [x] Audio element stops and resets when a new event is selected or k changes
- [x] Playback does not scroll the canvas

**Tests needed:**
- Play button toggles between play and pause icons
- Audio element is created with correct src URL for selected event

---

### Task 7: Detail Page Link and Navigation

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderDetailPage.tsx`

**Acceptance criteria:**
- [x] A link or button on the Event Encoder detail page navigates to
  `/app/sequence-models/event-encoder/{jobId}/piano-roll`
- [x] Link is only rendered when the job status is "complete"
- [x] Piano roll back link navigates to
  `/app/sequence-models/event-encoder/{jobId}`

**Tests needed:**
- Link is visible on the detail page for a complete job
- Link is not visible for non-complete jobs

---

### Task 8: Playwright End-to-End Tests

**Files:**
- Create: `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts`
- Modify: `frontend/e2e/sequence-models/event-encoder.spec.ts`

**Acceptance criteria:**
- [x] Piano roll spec uses mocked timeline data with at least 5 events
  spanning two token families and both voiced/unvoiced events
- [x] Asserts: route renders, toolbar shows job stats, canvas element present,
  k-value selector present when multiple k values, back link navigates to
  detail page, play button exists
- [x] Existing event-encoder spec extended: detail page shows piano roll link
  for a complete job
- [x] All tests pass alongside existing event encoder tests

**Tests needed:**
- This task IS the test task

---

### Verification

Run in order after all tasks:

1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test e2e/sequence-models/event-encoder-piano-roll.spec.ts`
3. `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
4. `cd frontend && npx playwright test`
