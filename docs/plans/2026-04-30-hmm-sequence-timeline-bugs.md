# HMM Sequence Timeline Bugs Implementation Plan

**Goal:** Fix HMM Sequence timeline navigation, playback synchronization, and zoomed-out region/state rendering.
**Spec:** Bug fix; no separate design spec.

---

### Task 1: Shared Drag Panning

**Files:**
- Modify: `frontend/src/components/timeline/provider/TimelineProvider.tsx`
- Modify: `frontend/src/components/timeline/provider/types.ts`
- Modify: `frontend/src/components/timeline/spectrogram/Spectrogram.tsx`
- Modify: `frontend/src/components/sequence-models/HMMStateBar.tsx`
- Modify: `frontend/src/components/timeline/provider/TimelineProvider.test.tsx`

**Acceptance criteria:**
- [x] Timeline drag panning is exposed through the shared timeline context.
- [x] Dragging the spectrogram still pans horizontally.
- [x] Dragging the HMM state bar also pans horizontally.
- [x] Dragging does not start while playback is active.

**Tests needed:**
- Frontend unit coverage for drag delta to center timestamp behavior.

---

### Task 2: Playback and Playhead Synchronization

**Files:**
- Modify: `frontend/src/components/timeline/provider/TimelineProvider.tsx`
- Modify: `frontend/src/components/sequence-models/HMMStateBar.tsx`
- Modify: `frontend/src/components/timeline/provider/TimelineProvider.test.tsx`

**Acceptance criteria:**
- [x] Playback time is the single source of truth for playhead position while audio is playing.
- [x] The timeline scrolls with playback at every zoom level.
- [x] The playhead remains aligned with current audio time instead of drifting to the viewport center after zoom or pan updates.
- [x] Playback epoch is cleared when playback ends or pauses.

**Tests needed:**
- Frontend unit coverage for playback epoch updates and reset semantics.

---

### Task 3: All Visible Region State Bars and Region Highlighting

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/HMMStateBar.tsx`
- Modify: `frontend/src/components/timeline/overlays/RegionBoundaryMarkers.tsx`
- Modify: `frontend/src/components/timeline/overlays/overlays.test.ts`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Create: `frontend/src/components/sequence-models/HMMStateBar.test.ts`

**Acceptance criteria:**
- [x] HMM state bar receives all decoded windows, not just the selected region/span.
- [x] At any zoom level, all visible regions render their state bars.
- [x] Current region is highlighted with white left/right bounding lines.
- [x] Timeline tiles outside the current region are not darkened or shaded.
- [x] Existing region boundary overlay no longer shades out-of-region tiles in the HMM Sequence timeline.

**Tests needed:**
- Frontend unit coverage for visible window filtering and current-region boundary drawing helpers.

---

### Task 4: Browser Mockup and Visual Check

**Files:**
- Create: `docs/mockups/hmm-sequence-timeline-bugs.html`

**Acceptance criteria:**
- [x] Mockup demonstrates drag-scrollable timeline, all visible region bars, synchronized playhead, and white current-region bounds.
- [x] Mockup opens in the Codex in-app browser for review.

**Tests needed:**
- Manual browser inspection only.

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx vitest run src/components/timeline/provider/TimelineProvider.test.tsx src/components/sequence-models/HMMStateBar.test.tsx`
2. `cd frontend && npx tsc --noEmit`
3. `uv run pytest tests/`
