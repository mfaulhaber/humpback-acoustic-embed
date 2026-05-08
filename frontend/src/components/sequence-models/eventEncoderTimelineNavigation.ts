import type { EventEncoderTimelineEvent } from "@/api/sequenceModels";

export type TimelineNavigationEvent = Pick<
  EventEncoderTimelineEvent,
  "event_id" | "token_id"
>;

export interface TimelineNavigationTarget<
  TEvent extends TimelineNavigationEvent = TimelineNavigationEvent,
> {
  event: TEvent;
  fullIndex: number;
}

export interface TimelineNavigationState<
  TEvent extends TimelineNavigationEvent = TimelineNavigationEvent,
> {
  selectedEvent: TEvent | null;
  selectedFullIndex: number;
  eventCounterIndex: number;
  eventCounterTotal: number;
  tokenScoped: boolean;
  tokenOccurrenceIndex: number;
  tokenOccurrenceTotal: number;
  previous: TimelineNavigationTarget<TEvent> | null;
  next: TimelineNavigationTarget<TEvent> | null;
}

interface BuildTimelineNavigationStateArgs<
  TEvent extends TimelineNavigationEvent = TimelineNavigationEvent,
> {
  events: TEvent[];
  selectedEventId: string | null;
  fallbackIndex?: number;
  tokenScopedNavigation: boolean;
}

export function buildTimelineNavigationState<
  TEvent extends TimelineNavigationEvent,
>({
  events,
  selectedEventId,
  fallbackIndex = 0,
  tokenScopedNavigation,
}: BuildTimelineNavigationStateArgs<TEvent>): TimelineNavigationState<TEvent> {
  const selectedFullIndex = events.findIndex(
    (event) => event.event_id === selectedEventId,
  );
  const eventCounterIndex =
    selectedFullIndex >= 0 ? selectedFullIndex : boundedIndex(events, fallbackIndex);
  const selectedEvent =
    selectedFullIndex >= 0 ? events[selectedFullIndex] ?? null : null;
  const sameTokenEvents = selectedEvent
    ? events.filter((event) => event.token_id === selectedEvent.token_id)
    : [];
  const tokenScoped = tokenScopedNavigation && selectedEvent != null;
  const navigationEvents = tokenScoped ? sameTokenEvents : events;
  const navigationIndex = tokenScoped
    ? navigationEvents.findIndex((event) => event.event_id === selectedEventId)
    : eventCounterIndex;
  const tokenOccurrenceIndex = selectedEvent
    ? sameTokenEvents.findIndex((event) => event.event_id === selectedEvent.event_id)
    : -1;

  return {
    selectedEvent,
    selectedFullIndex,
    eventCounterIndex,
    eventCounterTotal: events.length,
    tokenScoped,
    tokenOccurrenceIndex,
    tokenOccurrenceTotal: sameTokenEvents.length,
    previous: targetAt(events, navigationEvents, navigationIndex - 1),
    next: targetAt(events, navigationEvents, navigationIndex + 1),
  };
}

function boundedIndex<TEvent extends TimelineNavigationEvent>(
  events: TEvent[],
  index: number,
): number {
  if (events.length === 0) return -1;
  return Math.max(0, Math.min(events.length - 1, index));
}

function targetAt<TEvent extends TimelineNavigationEvent>(
  allEvents: TEvent[],
  navigationEvents: TEvent[],
  navigationIndex: number,
): TimelineNavigationTarget<TEvent> | null {
  const event = navigationEvents[navigationIndex];
  if (!event) return null;
  const fullIndex = allEvents.findIndex((item) => item.event_id === event.event_id);
  if (fullIndex < 0) return null;
  return { event, fullIndex };
}
