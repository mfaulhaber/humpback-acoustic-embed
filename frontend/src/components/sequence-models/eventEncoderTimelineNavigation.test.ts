import { describe, expect, it } from "vitest";

import {
  buildTimelineNavigationState,
  type TimelineNavigationEvent,
} from "./eventEncoderTimelineNavigation";

const events: TimelineNavigationEvent[] = [
  { event_id: "evt-1", token_id: 17 },
  { event_id: "evt-2", token_id: 42 },
  { event_id: "evt-3", token_id: 17 },
  { event_id: "evt-4", token_id: 99 },
  { event_id: "evt-5", token_id: 17 },
];

describe("buildTimelineNavigationState", () => {
  it("navigates through all events when token scope is off", () => {
    const state = buildTimelineNavigationState({
      events,
      selectedEventId: "evt-2",
      tokenScopedNavigation: false,
    });

    expect(state.tokenScoped).toBe(false);
    expect(state.eventCounterIndex).toBe(1);
    expect(state.eventCounterTotal).toBe(5);
    expect(state.previous?.event.event_id).toBe("evt-1");
    expect(state.previous?.fullIndex).toBe(0);
    expect(state.next?.event.event_id).toBe("evt-3");
    expect(state.next?.fullIndex).toBe(2);
  });

  it("skips to matching token events when token scope is on", () => {
    const state = buildTimelineNavigationState({
      events,
      selectedEventId: "evt-3",
      tokenScopedNavigation: true,
    });

    expect(state.tokenScoped).toBe(true);
    expect(state.tokenOccurrenceIndex).toBe(1);
    expect(state.tokenOccurrenceTotal).toBe(3);
    expect(state.previous?.event.event_id).toBe("evt-1");
    expect(state.previous?.fullIndex).toBe(0);
    expect(state.next?.event.event_id).toBe("evt-5");
    expect(state.next?.fullIndex).toBe(4);
  });

  it("disables previous and next at same-token boundaries", () => {
    const first = buildTimelineNavigationState({
      events,
      selectedEventId: "evt-1",
      tokenScopedNavigation: true,
    });
    const last = buildTimelineNavigationState({
      events,
      selectedEventId: "evt-5",
      tokenScopedNavigation: true,
    });

    expect(first.previous).toBeNull();
    expect(first.next?.event.event_id).toBe("evt-3");
    expect(last.previous?.event.event_id).toBe("evt-3");
    expect(last.next).toBeNull();
  });

  it("disables both directions for a single token occurrence", () => {
    const state = buildTimelineNavigationState({
      events,
      selectedEventId: "evt-2",
      tokenScopedNavigation: true,
    });

    expect(state.tokenScoped).toBe(true);
    expect(state.tokenOccurrenceIndex).toBe(0);
    expect(state.tokenOccurrenceTotal).toBe(1);
    expect(state.previous).toBeNull();
    expect(state.next).toBeNull();
  });

  it("falls back safely when the selected event is missing", () => {
    const state = buildTimelineNavigationState({
      events,
      selectedEventId: "missing",
      fallbackIndex: 2,
      tokenScopedNavigation: true,
    });

    expect(state.selectedEvent).toBeNull();
    expect(state.tokenScoped).toBe(false);
    expect(state.eventCounterIndex).toBe(2);
    expect(state.previous?.event.event_id).toBe("evt-2");
    expect(state.next?.event.event_id).toBe("evt-4");
  });

  it("handles an empty event list", () => {
    const state = buildTimelineNavigationState({
      events: [],
      selectedEventId: null,
      tokenScopedNavigation: true,
    });

    expect(state.selectedEvent).toBeNull();
    expect(state.selectedFullIndex).toBe(-1);
    expect(state.eventCounterIndex).toBe(-1);
    expect(state.eventCounterTotal).toBe(0);
    expect(state.previous).toBeNull();
    expect(state.next).toBeNull();
  });

  it("preserves the original event ordering", () => {
    const originalIds = events.map((event) => event.event_id);

    buildTimelineNavigationState({
      events,
      selectedEventId: "evt-3",
      tokenScopedNavigation: true,
    });

    expect(events.map((event) => event.event_id)).toEqual(originalIds);
  });
});
