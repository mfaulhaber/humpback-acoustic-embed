# Call Parsing Review Workspaces — Epoch Timestamp Alignment Implementation Plan

**Goal:** Bring `SegmentReviewWorkspace`, `ClassifyReviewWorkspace`, and `WindowClassifyReviewWorkspace` into the same absolute-epoch coordinate system as the HMM and Masked Transformer detail pages, fixing the `400` from `/call-parsing/region-jobs/{id}/audio-slice` and consolidating the conversion seam into one shared hook plus one shared wrapper.

**Spec:** [docs/specs/2026-05-04-call-parsing-review-epoch-timestamps-design.md](../specs/2026-05-04-call-parsing-review-epoch-timestamps-design.md)

---

### Task 1: Test fixture helper

**Files:**
- Create: `frontend/src/components/call-parsing/__test-helpers__/regionEpoch.ts`

**Acceptance criteria:**
- [ ] Exports a `REGION_EPOCH_BASE` constant equal to `1635638400` (`2021-10-31T00:00:00Z`).
- [ ] Exports a `REGION_EPOCH_END` constant equal to `REGION_EPOCH_BASE + 86400` (24-hour window).
- [ ] Module is referenced only by test files; no production code imports from `__test-helpers__/`.
- [ ] No additional helpers added in this task — keep the surface minimal.

**Tests needed:**
- None for this file directly. It is a fixture helper consumed by Tasks 2–7.

---

### Task 2: `useRegionEpoch` hook

**Files:**
- Create: `frontend/src/components/call-parsing/useRegionEpoch.ts`
- Create: `frontend/src/components/call-parsing/useRegionEpoch.test.tsx`

**Acceptance criteria:**
- [ ] Hook signature: `useRegionEpoch(regionDetectionJobId: string | null): RegionEpoch | null`.
- [ ] Returned `RegionEpoch` interface exposes `regionStartTimestamp: number`, `regionEndTimestamp: number`, and `toEpoch: (relativeSec: number) => number`.
- [ ] Implementation reads from `useRegionDetectionJobs()` (no new API endpoint, no new query key).
- [ ] Returns `null` when `regionDetectionJobId` is `null`, when no matching job is in the cache, or when the matching job has `start_timestamp == null` or `end_timestamp == null`.
- [ ] Result is memoized so referential identity is stable across renders that don't change the underlying job.
- [ ] `RegionEpoch` is exported from the hook's module so callers can type props.

**Tests needed:**
- Returns `null` for each of the three null/missing branches above (jobId null, no match, missing timestamps).
- Returns a fully populated `RegionEpoch` when the job has both timestamps; verify `toEpoch(0)` and `toEpoch(7916.1)` produce the expected absolute-epoch values using `REGION_EPOCH_BASE`.
- Result identity is stable across an unrelated re-render of the consumer.

---

### Task 3: `RegionAudioTimeline` wrapper component

**Files:**
- Create: `frontend/src/components/call-parsing/RegionAudioTimeline.tsx`
- Create: `frontend/src/components/call-parsing/RegionAudioTimeline.test.tsx`

**Acceptance criteria:**
- [ ] Component is a `forwardRef` that forwards a `TimelinePlaybackHandle` ref to the inner `TimelineProvider`.
- [ ] Required props: `regionDetectionJobId: string`, `regionEpoch: RegionEpoch`, `children: ReactNode`.
- [ ] Optional passthrough props: `zoomLevels`, `defaultZoom`, `disableKeyboardShortcuts`, `scrollOnPlayback`, `onZoomChange`, `onPlayStateChange`, `resetKey` (used as the inner provider's `key` so workspaces can force a remount on region/event navigation).
- [ ] Internally configures `playback="slice"` and an `audioUrlBuilder` that calls `regionAudioSliceUrl(regionDetectionJobId, startEpoch, durationSec)`. The `audioUrlBuilder` is memoized on `regionDetectionJobId`.
- [ ] Sets `jobStart={regionEpoch.regionStartTimestamp}` and `jobEnd={regionEpoch.regionEndTimestamp}` on the inner `TimelineProvider`.
- [ ] Does not modify, re-export, or wrap any shared timeline primitive other than `TimelineProvider`.
- [ ] Co-located in `frontend/src/components/call-parsing/`; not exported from `components/timeline/`.

**Tests needed:**
- Inner `TimelineProvider` receives epoch `jobStart` and `jobEnd` matching the passed `regionEpoch`.
- The wrapper's internal `audioUrlBuilder` produces a URL whose `start_timestamp` query param equals the epoch passed in (asserted against a captured fetch / spy).
- The forwarded `TimelinePlaybackHandle` ref is reachable from the outer ref (call `play(epoch, dur)` via the outer ref and assert the inner handle was invoked with the same arguments).
- Passthrough props (`disableKeyboardShortcuts`, `scrollOnPlayback`, `onZoomChange`, `onPlayStateChange`, `defaultZoom`, `zoomLevels`) reach the inner provider.

---

### Task 4: Migrate `SegmentReviewWorkspace`

**Files:**
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`
- Create: `frontend/src/components/call-parsing/SegmentReviewWorkspace.test.ts`

**Acceptance criteria:**
- [ ] Workspace calls `useRegionEpoch(regionDetectionJobId)` and renders a loading placeholder consistent with the existing "Select a region" empty-state when it returns `null` while a job is selected.
- [ ] The render gate that today reads `selectedJob && selectedRegion && timelineExtent` is extended to also require `regionEpoch != null` before mounting the timeline.
- [ ] Replaces the inline `<TimelineProvider>` with `<RegionAudioTimeline>` from Task 3, passing `regionDetectionJobId`, `regionEpoch`, and the existing `key`/zoom/keyboard/scroll/`onZoomChange`/`onPlayStateChange` props as passthroughs.
- [ ] Removes the local `jobStart = 0` constant and the inline `audioUrlBuilder` definition (the wrapper owns it).
- [ ] All `playbackRef.current?.play(startSec, duration)` call sites are converted to `playbackRef.current?.play(regionEpoch.toEpoch(startSec), duration)` — covers the toolbar Play button, the event-detail Play button, and any other play sites in the workspace.
- [ ] Recentering math is converted: targets become `regionEpoch.toEpoch(...)` and comparisons normalize the relative event coordinate to epoch before comparing against the (now-epoch) `viewStart`. Three sites to update.
- [ ] `jobStartEpoch` props passed to `ReviewToolbar`, `EventDetailPanel`, and both `RegionTable` instances receive `regionEpoch.regionStartTimestamp`.
- [ ] Overlay `jobStart` props inside `SegmentViewerBody` (e.g. `RegionEditOverlay`, `RegionOverlay`) receive `regionEpoch.regionStartTimestamp`.
- [ ] `tileUrlBuilder` definition is unchanged.
- [ ] Workspace state, correction payloads, and the saved-corrections POST body still carry job-relative `start_sec` / `end_sec`. No state shape changes.

**Tests needed:**
- Region-job fixture uses `REGION_EPOCH_BASE` / `REGION_EPOCH_END`.
- Selecting an event with `startSec = 7916.1` and clicking the toolbar Play button captures a `playbackRef.play(REGION_EPOCH_BASE + 7916.1, …)` call.
- Selecting an event with `startSec = 7916.1` and clicking the event-detail Play button captures the same epoch-translated argument.
- Event navigation triggers `setScrollToCenter` with a target equal to `regionEpoch.toEpoch(<expected relative target>)`.
- Saving a correction issues a POST whose `corrected_start_sec` is the relative seconds value (no epoch addition).
- `ReviewToolbar` and `EventDetailPanel` rendered text reflects the absolute UTC time-of-day for the selected event.
- When `regionEpoch` is unavailable (e.g. region-jobs fixture omits the matching job), the loading placeholder is rendered and `<RegionAudioTimeline>` is not mounted.

---

### Task 5: Migrate `ClassifyReviewWorkspace`

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.test.ts`

**Acceptance criteria:**
- [ ] Workspace calls `useRegionEpoch(regionDetectionJobId)` and gates the timeline render on `regionEpoch != null`.
- [ ] Replaces the inline `<TimelineProvider jobStart={0}>` with `<RegionAudioTimeline>` and removes the local `audioUrlBuilder` if any.
- [ ] All `playbackRef.current?.play(...)` sites — including the `displayEvent.startSec` site — use `regionEpoch.toEpoch(...)`.
- [ ] Recentering math (one site) is converted as in Task 4.
- [ ] Overlay `jobStart` props receive `regionEpoch.regionStartTimestamp`.
- [ ] `<ClassifyDetailPanel jobStartEpoch={…}>` receives `regionEpoch.regionStartTimestamp`.
- [ ] `tileUrlBuilder` definition is unchanged.
- [ ] Workspace state, `event_boundary_corrections` payload, and `vocalization_corrections` payload still carry job-relative `start_sec` / `end_sec`.

**Tests needed:**
- Add a region-job fixture using `REGION_EPOCH_BASE` / `REGION_EPOCH_END`.
- New case: `displayEvent` with `startSec = 7916.1` plus a Play click captures `playbackRef.play(REGION_EPOCH_BASE + 7916.1, …)`.
- Existing correction round-trip assertions remain green and assert relative `corrected_start_sec`/`corrected_end_sec` in the POST.
- `ClassifyDetailPanel` rendered text reflects absolute UTC time-of-day.
- Loading placeholder rendered when `regionEpoch` unavailable.

---

### Task 6: Migrate `WindowClassifyReviewWorkspace`

**Files:**
- Modify: `frontend/src/components/call-parsing/WindowClassifyReviewWorkspace.tsx`
- Create: `frontend/src/components/call-parsing/WindowClassifyReviewWorkspace.test.ts`

**Acceptance criteria:**
- [ ] Workspace calls `useRegionEpoch(regionDetectionJobId)` and gates the timeline render on `regionEpoch != null`.
- [ ] Replaces the inline `<TimelineProvider jobStart={0}>` with `<RegionAudioTimeline>`.
- [ ] Both `playbackRef.current?.play(...)` call sites — `selectedEvent.start_sec` and `selectedAddedEvent.startSec` — use `regionEpoch.toEpoch(...)`.
- [ ] Overlay `jobStart` props receive `regionEpoch.regionStartTimestamp`.
- [ ] Both `jobStartEpoch={0}` prop sites are replaced (one on the inline event-row block and any others currently in the file).
- [ ] All inline `formatRecordingTime(_, jobStartEpoch)` calls receive `regionEpoch.regionStartTimestamp` via the same prop chain.
- [ ] `tileUrlBuilder` definition is unchanged.
- [ ] Workspace state and `vocalization_corrections` / `event_boundary_corrections` payloads still carry job-relative `start_sec` / `end_sec`.

**Tests needed:**
- Region-job fixture uses `REGION_EPOCH_BASE` / `REGION_EPOCH_END`.
- `selectedEvent.start_sec = 7916.1` plus Play captures `play(REGION_EPOCH_BASE + 7916.1, …)`.
- `selectedAddedEvent.startSec = 7950.0` plus Play captures `play(REGION_EPOCH_BASE + 7950.0, …)`.
- Inline event-row block rendered text reflects absolute UTC time-of-day.
- Correction round-trip assertions issue relative `start_sec` / `end_sec` payloads.
- Loading placeholder rendered when `regionEpoch` unavailable.

---

### Task 7: Playwright fixture audit (conditional)

**Files:**
- Modify: any existing `frontend/e2e/call-parsing/*.spec.ts` whose mocks include the audio-slice path for one of the three workspaces.

**Acceptance criteria:**
- [ ] Audit `frontend/e2e/call-parsing/` for specs that mock `/call-parsing/region-jobs` and exercise an audio-slice request in Segment, Classify, or Window Classify review.
- [ ] For each such spec, set `start_timestamp = REGION_EPOCH_BASE` and `end_timestamp = REGION_EPOCH_END` on the region-job fixture.
- [ ] For each such spec, add an assertion that any captured audio-slice request URL has `start_timestamp` falling within `[REGION_EPOCH_BASE, REGION_EPOCH_END]`.
- [ ] If no specs currently exercise the audio-slice path in these workspaces, mark this task complete with a one-line note in the PR description and add no new specs.

**Tests needed:**
- N/A — this task is itself a test update.

---

### Verification

Run in order after all tasks:

1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx vitest run src/components/call-parsing/`
3. `cd frontend && npx playwright test call-parsing/` (skip if Task 7 found no specs to update)
4. `uv run pytest tests/` (no-op safety net; this PR has no backend changes)
5. Manual browser verification:
   1. Open a Segment review page for a job whose region has a non-zero `start_timestamp`. Confirm spectrogram tiles render for the actual recording window, the timeline header shows real UTC times, and clicking Play on a selected event returns `200` with audio matching the visible spectrogram region.
   2. Same for Classify review.
   3. Same for Window Classify review.
   4. Save a correction in Segment review and confirm the network payload contains relative `start_sec` / `end_sec` (e.g. `7920.0`), not absolute epoch.
