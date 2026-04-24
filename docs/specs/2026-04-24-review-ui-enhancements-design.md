# Review UI Enhancements — Design Spec

**Date:** 2026-04-24
**Scope:** All Call Parsing review pages (Segment, Classify, Window Classify)

## Overview

Four enhancements to improve consistency and usability across Call Parsing review pages:

1. Recording-based time display (`HH:MM:SS.d`) replacing offset-seconds
2. Unified correction indicators on text metadata, including a three-state type indicator (inferred / approved / corrected) with distinct ring colors
3. Three-state ring treatment on TypePalette badges and timeline EventTypeBadges
4. Drag-to-pan fix for timeline spectrograms

## 1. Recording-Based Time Display

### Current behavior

Event times are displayed as offset-from-zero in `m:ss.s` format (e.g., `2:05.4`) via `formatTimeDecimal()`. This does not match the `TimeAxis` which shows absolute UTC time (`HH:MM:SS`), forcing the user to mentally convert.

### New behavior

A new utility `formatRecordingTime(offsetSec: number, jobStartEpoch: number): string` adds the offset to the job's start epoch and formats as `HH:MM:SS.d` — zero-padded hours, one decimal second, UTC. Example: `14:32:05.4`.

Duration stays as offset-based (e.g., `1.7s`) since it is a relative measure.

### Where it applies

| Component | Current | New |
|---|---|---|
| `EventDetailPanel` | `formatTimeDecimal(event.startSec)` | `formatRecordingTime(event.startSec, jobStartEpoch)` |
| `ClassifyDetailPanel` | `event.startSec.toFixed(2)s` | `formatRecordingTime(event.startSec, jobStartEpoch)` |
| `WindowClassifyReviewWorkspace` local `EventDetailPanel` | `selectedEvent.start_sec.toFixed(1)s` | `formatRecordingTime(selectedEvent.start_sec, jobStartEpoch)` |
| `ReviewToolbar` | `formatTime(region.start_sec)` | `formatRecordingTime(region.start_sec, jobStartEpoch)` |
| `RegionTable` | `formatTime()` | `formatRecordingTime()` |

### Data flow

Each review workspace already accesses the timeline context (`ctx.jobStart`). Detail panels receive `jobStartEpoch` as a new prop threaded from the workspace.

### Format rules

- Always `HH:MM:SS.d` — zero-padded hours, one decimal second
- UTC, matching TimeAxis labels at small viewport spans
- "Was" annotations for adjusted boundaries use the same format

## 2. Unified Correction Indicators on Text Metadata

### Current state

Correction indicators are inconsistent across review pages:
- `EventDetailPanel` (Segment Review): purple/red/green pill badges for "adjusted"/"deleted"/"added", purple "was" annotations for original values
- `ClassifyDetailPanel` (Classify Review): green "corrected" text label for type corrections, but no boundary correction badges
- `WindowClassifyReviewWorkspace` local `EventDetailPanel`: `Plus`/`X` icons on label badges, yellow pending outlines, no boundary badges

### Unified treatment

**Boundary corrections:** Purple pill badge for "adjusted", red for "deleted", green for "added". The `EventDetailPanel` style becomes the standard. Applied consistently wherever boundary corrections are possible.

**Type corrections:** Text label next to the type badge — "corrected" (green) or "approved" (lime). See §3 for the three-state model.

**"Was" annotations:** Purple text showing original values when boundaries have been adjusted, using the new `HH:MM:SS.d` format. Example: `Start: 14:32:05.4 (was 14:32:05.1)`.

### Changes per page

**Segment Review (`SegmentReviewWorkspace`):**
- Uses shared `EventDetailPanel` — already has boundary badges
- Time format update only; no new indicators needed
- No type corrections (Pass 2 is boundary-only)

**Classify Review (`ClassifyReviewWorkspace`):**
- `ClassifyDetailPanel`: add boundary correction badges ("adjusted"/"deleted"/"added") — currently missing despite Classify Review supporting boundary editing
- `ClassifyDetailPanel`: add "was" annotations for adjusted boundaries
- Keep existing type correction indicator, upgrade to three-state (§3)
- Thread `correctionType` from `EffectiveEvent` into the panel (currently only receives `AggregatedEvent`)

**Window Classify Review (`WindowClassifyReviewWorkspace`):**
- Local `EventDetailPanel`: add boundary correction badges
- Label badges: keep `Plus`/`X` correction icons and yellow pending outlines (these serve a different purpose — showing add/remove actions on multi-label classification)
- Add three-state ring treatment to label badges where applicable (§3)

## 3. Three-State Type Indicator

### The three states

When a human interacts with a type assignment, the result falls into one of three states relative to the model prediction:

| State | Condition | Meaning |
|---|---|---|
| **Inferred** | No human correction exists | Model prediction, no human action |
| **Approved** | `correctedType === predictedType` | Human confirmed the model's prediction (hard positive for training) |
| **Corrected** | `correctedType !== predictedType` | Human overrode the model's prediction |

### Data model change

`resolveEventType()` in `ClassifyReviewWorkspace.tsx` currently returns `typeSource: "inference" | "correction" | "negative" | null`. It gains a fifth value: `"approved"`.

Logic change: when `correctedType` is a string and equals `predictedType`, return `typeSource: "approved"` instead of `"correction"`. The `EffectiveEvent.typeSource` type updates to include `"approved"`.

No backend changes — the distinction is determinable from existing data (`correctedType` vs `predictedType` comparison).

### Visual constants

```
APPROVED_RING_COLOR  = "hsl(85, 80%, 45%)"   // lime green
CORRECTED_RING_COLOR = "rgb(74, 222, 128)"    // green (existing "corrected" text color)
```

### TypePalette (`TypePalette.tsx`)

New prop: `typeSource: EffectiveEvent["typeSource"]`

Active badge ring color by state:
- `"inference"` or `null`: type-colored ring (current behavior, `--tw-ring-color: typeColor(name)`)
- `"approved"`: lime ring (`box-shadow: 0 0 0 2.5px hsl(85, 80%, 45%)`)
- `"correction"`: green ring (`box-shadow: 0 0 0 2.5px rgb(74, 222, 128)`)
- `"negative"`: no change (red styling on Negative button, current behavior)

### EventTypeBadge (`EventBarOverlay.tsx`)

Ring treatment by `typeSource`:
- `"inference"`: no ring (current: white background, colored text)
- `"approved"`: filled background, white text + lime ring (`box-shadow: 0 0 0 1.5px hsl(85, 80%, 45%)`)
- `"correction"`: filled background, white text + green ring (`box-shadow: 0 0 0 1.5px rgb(74, 222, 128)`)
- `"negative"`: no change (red background, current behavior)

The existing fill-vs-outline distinction is preserved — the ring adds a second signal on top.

### Detail panel type badges

All detail panels showing a type badge apply the same ring treatment:
- `ClassifyDetailPanel`: `Badge` component gets `box-shadow` style when approved/corrected
- `WindowClassifyReviewWorkspace` local `EventDetailPanel`: label badges in the "above threshold" list get the ring based on correction state. A label with `correction === "add"` whose `typeName` matches an inferred (above-threshold) label from the model = approved (lime ring). A label with `correction === "add"` for a type not in the model's above-threshold set = corrected (green ring). Labels with no correction get no ring.

### Text labels

Next to the type badge:
- `"approved"`: lime-colored "approved" text (`hsl(85, 80%, 45%)`)
- `"corrected"`: green "corrected" text (`rgb(74, 222, 128)`)
- `"inference"`: score display (current behavior)

## 4. Drag-to-Pan Fix

### Root cause

`EventBarOverlay` renders as `absolute inset-0` with `pointerEvents: "auto"` and calls `e.stopPropagation()` unconditionally in `handleMouseMove` (line 122). This prevents mouse events from reaching `Spectrogram`'s drag-to-pan handlers.

### Fix

Make `stopPropagation` conditional in `EventBarOverlay`:

**`handleMouseMove`:** Only call `e.stopPropagation()` when:
- `dragRef.current` is set (edge drag in progress), OR
- `addMode` is true (tracking ghost cursor)

Otherwise, let the event bubble to `Spectrogram`'s pan handler.

**`handleMouseUp`:** Only call `e.stopPropagation()` when `dragRef.current` is set.

**`handleMouseLeave`:** Same — only stop propagation during active drag.

**`handleContainerClick`:** Keeps `stopPropagation` unconditionally — clicks select/deselect events or place new events; panning is mouse-move-only.

### No changes to Spectrogram.tsx

The drag-to-pan implementation in `Spectrogram.tsx` (lines 73–110) is correct. It just needs events to reach it.

### Edge case: pan through event bars

When a user mousedowns on an event bar (not on an edge), the bar's `onClick` fires on mouseup to select it. If they drag, the mouse-move events bubble to `Spectrogram` for panning. This matches expected behavior: click selects, drag pans.

### Scope

All three review pages use `Spectrogram` + `EventBarOverlay` — the fix applies to all of them from a single code change in `EventBarOverlay.tsx`.

## 5. Component Changes Summary

| File | Changes |
|---|---|
| `frontend/src/utils/format.ts` | Add `formatRecordingTime(offsetSec, jobStartEpoch)` |
| `frontend/src/components/call-parsing/EventDetailPanel.tsx` | New `jobStartEpoch` prop; use `formatRecordingTime` |
| `frontend/src/components/call-parsing/ClassifyDetailPanel.tsx` | New `jobStartEpoch` prop; use `formatRecordingTime`; add boundary correction badges; add three-state ring on type badge |
| `frontend/src/components/call-parsing/WindowClassifyReviewWorkspace.tsx` (local `EventDetailPanel`) | New `jobStartEpoch` prop; use `formatRecordingTime`; add boundary badges; three-state ring on label badges |
| `frontend/src/components/call-parsing/TypePalette.tsx` | New `typeSource` prop; ring color by state |
| `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx` | Thread `jobStartEpoch` and `typeSource` to children; update `resolveEventType` to return `"approved"` |
| `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx` | Thread `jobStartEpoch` to `EventDetailPanel` and `ReviewToolbar` |
| `frontend/src/components/call-parsing/ReviewToolbar.tsx` | Use `formatRecordingTime`; accept `jobStartEpoch` prop |
| `frontend/src/components/call-parsing/RegionTable.tsx` | Use `formatRecordingTime`; accept `jobStartEpoch` prop |
| `frontend/src/components/timeline/overlays/EventBarOverlay.tsx` | Three-state ring on `EventTypeBadge`; conditional `stopPropagation` in mouse handlers; add `"approved"` to `typeSource` union |

## 6. Testing

- Unit test for `formatRecordingTime`: verify UTC formatting, zero-padding, decimal precision
- Unit test for updated `resolveEventType`: verify `"approved"` when correctedType === predictedType
- Playwright tests per review page:
  - Verify time format matches `HH:MM:SS.d` pattern in detail panels
  - Verify correction badges appear for boundary and type corrections
  - Verify drag-to-pan works (mousedown + mousemove on empty spectrogram area scrolls the timeline)
  - Verify edge-drag still works on selected event boundaries
  - Verify ring color distinction on type badges (approved vs corrected)

## 7. Non-Goals

- No backend changes
- No new API endpoints
- No changes to correction persistence logic
- No changes to the TimeAxis component itself
- No changes to detection review pages (outside Call Parsing review scope)
