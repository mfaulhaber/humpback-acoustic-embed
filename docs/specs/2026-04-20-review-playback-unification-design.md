# Review Workspace Playback Unification & Timeline Bug Fixes

**Date:** 2026-04-20
**Status:** Draft
**Scope:** SegmentReviewWorkspace, ClassifyReviewWorkspace, TimelineProvider, ZoomSelector

---

## Problem Statement

Four issues in the Call Parsing Segment and Classifier timeline viewers:

1. **Dual audio systems** — Review workspaces maintain their own `<audio>` element and playback state alongside the TimelineProvider's `usePlayback` system. Spacebar fires both handlers, causing two audio sources to play simultaneously (one plays the selected event, the other plays from center timestamp).

2. **Zoom resets on navigation** — TimelineProvider `key` includes the region ID. Cross-region event navigation re-mounts the provider, resetting zoom to the default level.

3. **ZoomSelector dark theme** — Hardcoded marine-dark `COLORS` constant looks wrong in review workspaces which use the app's standard light theme.

4. **Vertical dashed lines** — Confirmed working-as-designed. They mark unpadded region boundaries (start_sec/end_sec) to distinguish them from the padded spectrogram extent. No change needed.

---

## Design

### 1. Playback Handle (ref-based upward communication)

**New interface:**

```typescript
export interface TimelinePlaybackHandle {
  play: (startEpoch: number, duration?: number) => void;
  pause: () => void;
  isPlaying: boolean;
}
```

**TimelineProvider changes:**
- Wrap with `React.forwardRef`
- Expose `{play, pause, isPlaying}` via `useImperativeHandle`, delegating to the internal `usePlayback` hook
- Accept optional `onPlayStateChange?: (playing: boolean) => void` callback, fired on every play/pause state transition (enables parent re-renders for UI updates)

**Workspace changes (both Segment and Classify):**
- Hold `useRef<TimelinePlaybackHandle>(null)`, pass to `<TimelineProvider ref={...}>`
- Remove: `audioRef`, `<audio>` element, `isPlaying` state, `startPlayback`, `stopPlayback`, `playbackOriginSec`
- Add: `const [isPlaying, setIsPlaying] = useState(false)` synced via `onPlayStateChange`
- `togglePlayback` calls `ref.current?.play(event.startSec, duration)` or `ref.current?.pause()`
- `EventDetailPanel.onPlaySlice` and `ReviewToolbar.onPlay` call through the same ref
- `onEnded` handling: `usePlayback` already calls `onEnded` → provider dispatches `SET_PLAYING(false)` → `onPlayStateChange(false)` fires → workspace updates UI

### 2. Keyboard Shortcut Control

**TimelineProvider changes:**
- Accept `disableKeyboardShortcuts?: boolean` prop (default `false`)
- When `true`, skip the `useEffect` that registers the keydown listener

**Workspace changes:**
- Pass `disableKeyboardShortcuts={true}` to TimelineProvider
- Keyboard handling splits cleanly between two listeners:
  - **ViewerBody** (child, has context access): registers its own keydown for `+`/`-`/arrows, calling `ctx.zoomIn()`, `ctx.zoomOut()`, `ctx.pan(...)` directly
  - **Workspace** (parent, has event selection state): handles `Space`, `A`/`D`, `Delete` — calls playback ref and navigation callbacks
- No conflict: provider shortcuts are off, each listener handles disjoint keys

### 3. Zoom Persistence

**TimelineProvider changes:**
- Accept `onZoomChange?: (zoomKey: string) => void` callback
- Fire it inside `setZoomLevel` with the new preset's `key` string

**Workspace changes:**
- Hold `const [userZoom, setUserZoom] = useState<string>("1m")` (Segment) or `"30s"` (Classify)
- Pass `defaultZoom={userZoom}` and `onZoomChange={setUserZoom}`
- When provider re-mounts (key change on region navigation), it initializes with `userZoom`

### 4. ZoomSelector Theme

**Replace in ZoomSelector.tsx:**
- Remove `import { COLORS }` and all `style={{...}}` props
- Active button classes: `bg-primary/10 border border-primary/30 text-primary`
- Inactive button classes: `bg-muted border border-transparent text-muted-foreground hover:text-foreground`
- Container: keep existing `flex justify-center gap-1 py-1` layout

This makes ZoomSelector theme-aware via Tailwind's CSS variable system. Dark-themed parent containers (full timeline viewers) resolve these to dark values; light-themed parents (review workspaces) resolve to light values.

---

## Behavioral Constraint (new section in docs/reference/behavioral-constraints.md)

### Timeline Compound-Component Architecture

- **TimelineProvider owns all playback, zoom, pan, and viewport state.** No consumer may create parallel audio elements or duplicate zoom/playback state. External play/pause triggers go through the provider's ref handle.
- **Parent-to-provider communication** uses a forwarded ref (`TimelinePlaybackHandle`). Parents hold a ref to trigger play/pause — they never create their own `<audio>` elements.
- **Child-to-provider communication** uses `useTimelineContext()`. Overlays, controls, and viewer bodies consume context — they never receive `centerTimestamp`, `zoomLevel`, `width`, `height`, or `jobStart` as props.
- **Keyboard shortcuts are opt-out per consumer.** Generic viewers use the provider's built-in shortcuts. Workspaces that manage their own key handlers set `disableKeyboardShortcuts={true}` and call provider methods via context or ref.
- **Overlays are children of Spectrogram**, receiving coordinates from `useOverlayContext()`. They never position themselves relative to external containers.
- **Zoom persistence is the workspace's responsibility.** When a workspace re-mounts the provider (e.g., region navigation), it passes the user's last zoom preference via `defaultZoom` and receives changes via `onZoomChange`.

---

## Files Changed

| File | Change |
|------|--------|
| `frontend/src/components/timeline/provider/TimelineProvider.tsx` | Add `forwardRef`, `useImperativeHandle`, `disableKeyboardShortcuts`, `onZoomChange`, `onPlayStateChange` props |
| `frontend/src/components/timeline/provider/types.ts` | Export `TimelinePlaybackHandle` interface |
| `frontend/src/components/timeline/controls/ZoomSelector.tsx` | Replace `COLORS` with Tailwind classes |
| `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx` | Remove own audio system, use ref handle, track zoom, disable shortcuts |
| `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx` | Same as Segment |
| `docs/reference/behavioral-constraints.md` | Add "Timeline Compound-Component Architecture" section |
| `CLAUDE.md` | Add §8 pointer to behavioral constraint |

---

## Non-Goals

- Migrating the full timeline viewers (ClassifierTimeline, RegionDetectionTimeline) — they already use the provider correctly
- Changing RegionBoundaryMarkers behavior (vertical dashed lines are working as designed)
- Adding tooltip/label to explain dashed lines (could be done separately if desired)
