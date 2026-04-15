# Classify Review: Deleted Event UX Improvements

**Date:** 2026-04-15
**Status:** Approved

## Problem

When reviewing corrected events in the classify review workspace, two UX issues
arise from deleted events:

1. **Ghost overlap**: Deleted events render as faint red ghost bars at their
   original positions. When an adjacent active event has been adjusted to extend
   into the deleted event's time range (e.g., the user merged two events by
   extending one and deleting the other), the ghost bar overlaps the active bar.
   This is visually confusing — the user sees two bars in the same space.

2. **Navigation includes deleted events**: The event navigation counter
   ("Event N of M") steps through all events including deleted ones. Deleted
   events are nearly invisible (0.3 opacity, very short durations), so the user
   lands on invisible events and the numbering has gaps when mentally counting
   visible bars on the spectrogram.

## Design

### Change 1: Suppress overlapping deleted-event ghosts

In `EventBarOverlay`, filter the `deletedEvents` list to exclude any deleted
event whose time range is fully covered by an active (non-deleted) event. A
deleted event is "covered" when there exists an active event whose `startSec <=
deleted.startSec` and `endSec >= deleted.endSec`.

Deleted events that are NOT fully covered by an active event continue to render
as ghost bars (unchanged behavior).

### Change 2: Filter deleted events from navigation

In `ClassifyReviewWorkspace`, derive a `navigableEvents` list that excludes
events with saved boundary-deletion corrections. Use this filtered list for:

- `currentEventIndex` bounds and navigation (goPrev/goNext)
- The "Event N of M" counter
- `currentEvent` derivation

The full `events` list (including deleted) is still passed to
`regionEffectiveEvents` so that ghost bars can render when appropriate.

The numbering is contiguous — the user sees "Event 1, 2, 3, ..." with no gaps,
because the navigation list only contains active events.

### What does NOT change

- The `regionEffectiveEvents` computation (still includes deleted events for
  ghost rendering)
- The boundary correction save/load path
- The data in `event_boundary_corrections` table
- Deleted events that don't overlap an active event still render as ghosts
