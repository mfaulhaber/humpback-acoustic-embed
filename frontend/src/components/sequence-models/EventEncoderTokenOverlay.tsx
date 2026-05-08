import type { EventEncoderTimelineEvent } from "@/api/sequenceModels";
import { useOverlayContext } from "@/components/timeline/overlays/OverlayContext";
import { cn } from "@/lib/utils";

import { labelColor } from "./constants";

interface EventEncoderTokenOverlayProps {
  events: EventEncoderTimelineEvent[];
  selectedEventId: string | null;
  selectedK: number;
  onSelectEvent: (eventId: string) => void;
}

const MIN_BAR_WIDTH_PX = 3;
const BADGE_HEIGHT_PX = 16;

export function EventEncoderTokenOverlay({
  events,
  selectedEventId,
  selectedK,
  onSelectEvent,
}: EventEncoderTokenOverlayProps) {
  const ctx = useOverlayContext();
  const barTop = Math.max(0, ctx.canvasHeight * 0.08);
  const barHeight = Math.max(24, ctx.canvasHeight * 0.84);

  return (
    <div
      className="absolute inset-0"
      data-testid="eej-token-overlay"
      style={{ pointerEvents: "none" }}
    >
      {events.map((event) => {
        const x = ctx.epochToX(event.start_timestamp);
        const endX = ctx.epochToX(event.end_timestamp);
        const width = Math.max(MIN_BAR_WIDTH_PX, endX - x);
        const color = labelColor(event.token_id, Math.max(selectedK, 1));
        const selected = event.event_id === selectedEventId;

        return (
          <button
            key={event.event_id}
            type="button"
            className={cn(
              "absolute overflow-visible border text-left transition-[opacity,box-shadow]",
              selected ? "opacity-100" : "opacity-75 hover:opacity-95",
            )}
            style={{
              left: x,
              top: barTop,
              width,
              height: barHeight,
              backgroundColor: `color-mix(in srgb, ${color} 35%, transparent)`,
              borderColor: color,
              boxShadow: selected ? "0 0 0 2px rgba(255,255,255,0.9)" : undefined,
              pointerEvents: "auto",
            }}
            onClick={(e) => {
              e.stopPropagation();
              onSelectEvent(event.event_id);
            }}
            onMouseDown={(e) => e.stopPropagation()}
            data-testid={`eej-token-bar-${event.event_id}`}
            data-selected={selected ? "true" : "false"}
            aria-label={`${event.token_label} event ${event.event_id}`}
          >
            <span
              className="absolute left-0 top-0 rounded-[2px] border bg-background px-1 text-center font-mono text-[9px] font-bold leading-none"
              style={{
                minWidth: Math.max(24, event.token_label.length * 7 + 8),
                height: BADGE_HEIGHT_PX,
                lineHeight: `${BADGE_HEIGHT_PX - 2}px`,
                color,
                borderColor: color,
              }}
              data-testid={`eej-token-badge-${event.event_id}`}
            >
              {event.token_label}
            </span>
          </button>
        );
      })}
    </div>
  );
}
