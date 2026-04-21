# Review Workspace Playback Unification — Implementation Plan

**Goal:** Eliminate dual audio systems in review workspaces by migrating playback into TimelineProvider's ref handle, fix zoom reset on navigation, and restyle ZoomSelector for light theme.
**Spec:** [docs/specs/2026-04-20-review-playback-unification-design.md](../specs/2026-04-20-review-playback-unification-design.md)

---

### Task 1: TimelineProvider — forwardRef + playback handle + new props

Extend TimelineProvider to expose its playback system upward via ref and accept new configuration props.

**Files:**
- Modify: `frontend/src/components/timeline/provider/types.ts`
- Modify: `frontend/src/components/timeline/provider/TimelineProvider.tsx`

**Acceptance criteria:**
- [ ] `TimelinePlaybackHandle` interface exported from `types.ts` with `play(startEpoch, duration?)`, `pause()`, `isPlaying`
- [ ] `TimelineProviderProps` extended with `disableKeyboardShortcuts?: boolean`, `onZoomChange?: (zoomKey: string) => void`, `onPlayStateChange?: (playing: boolean) => void`
- [ ] TimelineProvider wrapped with `React.forwardRef` and exposes handle via `useImperativeHandle`
- [ ] `disableKeyboardShortcuts` when `true` skips the keyboard `useEffect` registration entirely
- [ ] `onZoomChange` fires inside `setZoomLevel` with the new preset's `key` string
- [ ] `onPlayStateChange` fires on play/pause state transitions (both user-initiated and onEnded)
- [ ] Existing callers (ClassifierTimeline, RegionDetectionTimeline) continue working without changes (all new props are optional)

**Tests needed:**
- Unit test: TimelineProvider renders without ref (backwards compatible)
- Unit test: `onZoomChange` callback fires when zoom level changes
- Unit test: `onPlayStateChange` fires on play and pause transitions
- Unit test: keyboard shortcuts do not register when `disableKeyboardShortcuts={true}`

---

### Task 2: ZoomSelector light theme styling

Replace hardcoded dark colors with semantic Tailwind classes.

**Files:**
- Modify: `frontend/src/components/timeline/controls/ZoomSelector.tsx`

**Acceptance criteria:**
- [ ] `COLORS` import removed
- [ ] All `style={{...}}` props removed from buttons
- [ ] Active button uses Tailwind classes: `bg-primary/10 border border-primary/30 text-primary`
- [ ] Inactive button uses Tailwind classes: `bg-muted border border-transparent text-muted-foreground hover:text-foreground`
- [ ] Layout classes preserved: `flex justify-center gap-1 py-1` container, `px-2 py-0.5 rounded text-[10px] font-mono transition-colors` buttons
- [ ] Visually correct in both review workspaces (light theme) and full timeline viewers (dark theme container)

**Tests needed:**
- Visual verification in browser: ZoomSelector in SegmentReviewWorkspace renders with light theme styling
- Visual verification in browser: ZoomSelector in ClassifierTimeline still renders correctly in dark context

---

### Task 3: SegmentReviewWorkspace — migrate to playback handle + zoom persistence

Remove the workspace's own audio system and wire everything through the provider ref.

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] `audioRef`, `<audio>` element, `playbackOriginSec` state removed
- [ ] `startPlayback` and `stopPlayback` callbacks removed
- [ ] `isPlaying` state synced via `onPlayStateChange` callback from provider
- [ ] `playbackRef = useRef<TimelinePlaybackHandle>(null)` created and passed to TimelineProvider
- [ ] `togglePlayback` calls `playbackRef.current?.play(event.startSec, duration)` / `.pause()`
- [ ] `EventDetailPanel.onPlaySlice` uses the same ref
- [ ] `ReviewToolbar.onPlay` uses the same ref
- [ ] `disableKeyboardShortcuts={true}` passed to TimelineProvider
- [ ] Workspace keydown handler handles `Space`, `A`, `D` only (no zoom/pan keys)
- [ ] `userZoom` state tracked, passed as `defaultZoom`, updated via `onZoomChange`
- [ ] Zoom level preserved when navigating across regions

**Tests needed:**
- Playwright test: spacebar plays selected event audio (not region audio)
- Playwright test: spacebar when no event selected plays from region start (capped at 30s)
- Playwright test: navigate across regions with A/D, verify zoom level unchanged
- Playwright test: EventDetailPanel play button plays event slice
- Playwright test: pressing spacebar again pauses playback

---

### Task 4: SegmentViewerBody — zoom/pan keyboard shortcuts

Move zoom and pan keyboard handling into the ViewerBody (which has context access).

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx` (SegmentViewerBody inner component)

**Acceptance criteria:**
- [ ] SegmentViewerBody registers its own keydown listener for `+`/`=`/`-`/`ArrowLeft`/`ArrowRight`
- [ ] `+`/`=` calls `ctx.zoomIn()`, `-` calls `ctx.zoomOut()`
- [ ] `ArrowLeft`/`ArrowRight` calls `ctx.pan(...)` with 10% viewport offset
- [ ] Handler skips events targeting input/textarea/select elements
- [ ] No conflict with workspace's Space/A/D/Delete handler (disjoint key sets)

**Tests needed:**
- Playwright test: pressing `+` zooms in on the segment spectrogram
- Playwright test: pressing `-` zooms out
- Playwright test: arrow keys pan the spectrogram

---

### Task 5: ClassifyReviewWorkspace — migrate to playback handle + zoom persistence

Same migration as Task 3 but for the classify workspace.

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] `audioRef`, `<audio>` element, `playbackOriginSec` state removed
- [ ] `startPlayback` and `stopPlayback` callbacks removed
- [ ] `isPlaying` state synced via `onPlayStateChange` callback
- [ ] `playbackRef = useRef<TimelinePlaybackHandle>(null)` created and passed to TimelineProvider
- [ ] `togglePlayback` calls `playbackRef.current?.play(displayEvent.startSec, duration)` / `.pause()`
- [ ] Toolbar play button uses the same ref
- [ ] `disableKeyboardShortcuts={true}` passed to TimelineProvider
- [ ] Workspace keydown handler handles `Space`, `A`/`D`/`ArrowLeft`/`ArrowRight`/`BracketLeft`/`BracketRight`, `Delete`/`Backspace` only
- [ ] `userZoom` state tracked, passed as `defaultZoom`, updated via `onZoomChange`
- [ ] Zoom level preserved when navigating across regions

**Tests needed:**
- Playwright test: spacebar plays current event audio
- Playwright test: navigate events, verify zoom preserved across region boundaries
- Playwright test: toolbar play button plays event, stop button pauses

---

### Task 6: ClassifyViewerBody — zoom/pan keyboard shortcuts

Same as Task 4 for the classify viewer body.

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx` (ClassifyViewerBody inner component)

**Acceptance criteria:**
- [ ] ClassifyViewerBody registers its own keydown listener for `+`/`=`/`-`/`ArrowLeft`/`ArrowRight`
- [ ] Calls `ctx.zoomIn()`, `ctx.zoomOut()`, `ctx.pan(...)` as appropriate
- [ ] Handler skips events targeting input/textarea/select elements
- [ ] No conflict with workspace's handler

**Tests needed:**
- Playwright test: zoom and pan keys work in classify review spectrogram

---

### Task 7: Behavioral constraint + CLAUDE.md pointer

Document the timeline compound-component architecture rules.

**Files:**
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `CLAUDE.md`

**Acceptance criteria:**
- [ ] New "Timeline Compound-Component Architecture" section appended to behavioral-constraints.md with all 6 rules from the spec
- [ ] CLAUDE.md §8 gains a one-line pointer: `### 8.10 Timeline Compound-Component Architecture` linking to the behavioral constraints section
- [ ] Rules reference the specific patterns: ref handle, useTimelineContext, disableKeyboardShortcuts, onZoomChange

**Tests needed:**
- None (documentation only)

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test`
3. Manual verification: open Segment Review, navigate events across regions — zoom holds, spacebar plays event, no dual audio
4. Manual verification: open Classify Review, same checks
5. Manual verification: ZoomSelector renders correctly in both light (review) and dark (full timeline) contexts
