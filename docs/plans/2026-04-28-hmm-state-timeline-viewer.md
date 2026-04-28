# HMM State Timeline Viewer Implementation Plan

**Goal:** Add an interactive timeline viewer panel to the HMM Sequence detail page with PCEN spectrogram, HMM state bar, span navigation, zoom/pan, and audio playback.
**Spec:** [docs/specs/2026-04-28-hmm-state-timeline-viewer-design.md](../specs/2026-04-28-hmm-state-timeline-viewer-design.md)

---

### Task 1: Backend — add `region_detection_job_id` to HMM job detail response

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/api/routers/sequence_models.py`

**Acceptance criteria:**
- [ ] `HMMSequenceJobOut` schema includes `region_detection_job_id: str`
- [ ] `get_hmm_sequence` endpoint resolves `region_detection_job_id` by loading the parent CEJ via `continuous_embedding_job_id`
- [ ] The `_hmm_to_out` helper (or the endpoint itself) populates the field
- [ ] Existing tests still pass — no breaking changes to other HMM endpoints

**Tests needed:**
- Test that `GET /sequence-models/hmm-sequences/{jobId}` response includes `region_detection_job_id` matching the CEJ's value

---

### Task 2: Frontend — extract STATE_COLORS to shared constants

**Files:**
- Create: `frontend/src/components/sequence-models/constants.ts`
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`

**Acceptance criteria:**
- [ ] `STATE_COLORS` palette moved to `constants.ts` and exported
- [ ] `HMMSequenceDetailPage.tsx` imports `STATE_COLORS` from `constants.ts` instead of defining it inline
- [ ] No visual change to existing page behavior

**Tests needed:**
- Existing page tests still pass after the import change

---

### Task 3: Frontend — update API types for `region_detection_job_id`

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`

**Acceptance criteria:**
- [ ] `HMMSequenceJobOut` (or equivalent TypeScript interface) includes `region_detection_job_id: string`
- [ ] TypeScript compiles cleanly (`npx tsc --noEmit`)

**Tests needed:**
- Type check passes

---

### Task 4: Frontend — create SpanNavBar component

**Files:**
- Create: `frontend/src/components/sequence-models/SpanNavBar.tsx`

**Acceptance criteria:**
- [ ] Accepts props: `spans` (array of `{id, startSec, endSec}`), `activeIndex`, `onPrev`, `onNext`
- [ ] Left-justified layout: `‹` `›` chevron buttons + label `Span {idx+1}/{total} · {startTime} – {endTime}`
- [ ] Prev button disabled when `activeIndex === 0`, Next disabled when `activeIndex === spans.length - 1`
- [ ] Time formatting uses `formatRecordingTime` from `@/utils/format`
- [ ] Uses `ChevronLeft`/`ChevronRight` icons from lucide-react, matching `ReviewToolbar` styling

**Tests needed:**
- Renders correct span label for given activeIndex
- Prev/Next buttons disabled at boundaries
- Clicking Prev/Next calls the corresponding callback

---

### Task 5: Frontend — create HMMStateBar component

**Files:**
- Create: `frontend/src/components/sequence-models/HMMStateBar.tsx`

**Acceptance criteria:**
- [ ] Accepts props: `items` (Viterbi window array for active span), `nStates`
- [ ] Consumes `useTimelineContext()` for viewport sync (`viewStart`, `viewEnd`, `pxPerSec`, `centerTimestamp`)
- [ ] Left offset matches spectrogram canvas via `FREQ_AXIS_WIDTH_PX`
- [ ] Canvas-based rendering: each window is a colored horizontal bar at its state's Y position
- [ ] Y-axis: state 0 at bottom, state n-1 at top
- [ ] Y-axis label strip on the left with state numbers, matching frequency axis width
- [ ] Colors from `STATE_COLORS` palette imported from `constants.ts`
- [ ] Fixed height of 60px for the canvas area
- [ ] Playhead: vertical red line at `centerTimestamp`, aligned with spectrogram playhead
- [ ] Responsive width via `ResizeObserver`
- [ ] Hover tooltip: positioned `<div>` showing `State {n} · {startTime}–{endTime} · prob {value}`
- [ ] Hover hit detection via binary search on sorted window start times

**Tests needed:**
- Canvas renders correct number of visible bars for a given viewport range
- Hover hit detection returns correct window for a given X coordinate
- Empty items array renders without error
- Playhead position matches centerTimestamp

---

### Task 6: Frontend — compose the HMM State Timeline Viewer panel

**Files:**
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`

**Acceptance criteria:**
- [ ] New "HMM State Timeline Viewer" Card panel inserted between Job Config and existing State Timeline
- [ ] Panel only renders when job status is `complete` and states data is loaded
- [ ] Computes span list with `{id, startSec, endSec}` from states data (min/max times per merged_span_id)
- [ ] SpanNavBar above TimelineProvider, with prev/next updating the active span
- [ ] TimelineProvider configured with: `key` including span ID, `jobStart`/`jobEnd` from active span, `REVIEW_ZOOM` presets, `playback="slice"`, `audioUrlBuilder` using `regionAudioSliceUrl`
- [ ] `defaultZoom` selects the best-fit preset for the span duration
- [ ] Spectrogram renders PCEN tiles via `regionTileUrl(regionDetectionJobId, ...)` with `freqRange=[0, 3000]`
- [ ] HMMStateBar receives filtered items for the active span
- [ ] ZoomSelector and PlaybackControls rendered below HMMStateBar
- [ ] `region_detection_job_id` read from the updated HMM job detail response
- [ ] Existing State Timeline (Plotly) panel and all subsequent panels unchanged, just shifted down
- [ ] Spectrogram height set appropriately (160px or similar, matching other consumers)

**Tests needed:**
- Panel renders when job is complete with states data
- Panel does not render when job is running or has no states
- Span navigation updates the active span and re-keys the TimelineProvider
- All 9 panels appear in correct order

---

### Task 7: Playwright — end-to-end test for the timeline viewer panel

**Files:**
- Create or modify: `frontend/tests/hmm-sequence-detail.spec.ts` (or appropriate test file)

**Acceptance criteria:**
- [ ] Test loads the HMM Sequence detail page for a completed job
- [ ] Verifies the "HMM State Timeline Viewer" panel is visible
- [ ] Verifies the spectrogram viewport is present within the panel
- [ ] Verifies the HMMStateBar canvas is present
- [ ] Verifies span navigation buttons are visible and functional (click next, verify span label changes)
- [ ] Verifies zoom preset buttons are present

**Tests needed:**
- Playwright spec covering the above assertions

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py`
2. `uv run ruff check src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py`
3. `uv run pyright src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
