import React, { useCallback, useMemo, useRef, useState } from "react";
import { useOverlayContext } from "./RegionSpectrogramViewer";
import { cn } from "@/lib/utils";
import type { BoundaryCorrection } from "@/api/types";

const SNAP = 0.1;
const EDGE_HIT_PX = 8;
const DEFAULT_ADD_DURATION = 1.0;
const BAR_HEIGHT_FRAC = 1.0; // fraction of canvas height

function snap(sec: number): number {
  return Math.round(sec / SNAP) * SNAP;
}

/** An event with effective (possibly corrected) boundaries for rendering. */
export interface EffectiveEvent {
  eventId: string;
  regionId: string;
  startSec: number;
  endSec: number;
  originalStartSec: number;
  originalEndSec: number;
  confidence: number;
  correctionType: "adjust" | "add" | "delete" | null;
}

interface EventBarOverlayProps {
  events: EffectiveEvent[];
  selectedEventId: string | null;
  onSelectEvent: (eventId: string | null) => void;
  onAdjust: (eventId: string, startSec: number, endSec: number) => void;
  onAdd: (regionId: string, startSec: number, endSec: number) => void;
  addMode: boolean;
  /** The region_id for add-mode events. */
  activeRegionId: string;
}

type DragEdge = "left" | "right";

interface DragState {
  eventId: string;
  edge: DragEdge;
  /** Opposite edge position (fixed anchor). */
  anchor: number;
  /** Starting mouse position in seconds. */
  startMouseSec: number;
  /** Original edge value at drag start. */
  originalEdgeSec: number;
}

export function EventBarOverlay({
  events,
  selectedEventId,
  onSelectEvent,
  onAdjust,
  onAdd,
  addMode,
  activeRegionId,
}: EventBarOverlayProps) {
  const ctx = useOverlayContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const [dragPreview, setDragPreview] = useState<{
    eventId: string;
    startSec: number;
    endSec: number;
  } | null>(null);
  const [ghostSec, setGhostSec] = useState<number | null>(null);

  const sortedEvents = useMemo(
    () =>
      [...events]
        .filter((e) => e.correctionType !== "delete")
        .sort((a, b) => a.startSec - b.startSec),
    [events],
  );

  // Coordinate transforms
  const secToX = useCallback(
    (sec: number) => (sec - ctx.viewStart) * ctx.pxPerSec,
    [ctx.viewStart, ctx.pxPerSec],
  );

  const xToSec = useCallback(
    (x: number) => x / ctx.pxPerSec + ctx.viewStart,
    [ctx.viewStart, ctx.pxPerSec],
  );

  const getMouseSec = useCallback(
    (e: React.MouseEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return 0;
      return xToSec(e.clientX - rect.left);
    },
    [xToSec],
  );

  // Clamp edge to avoid crossing the anchor or overlapping neighbors
  const clampEdge = useCallback(
    (candidateSec: number, edge: DragEdge, anchor: number, eventId: string) => {
      let clamped = snap(candidateSec);
      const minGap = SNAP;
      if (edge === "left") {
        clamped = Math.min(clamped, anchor - minGap);
        // Don't overlap previous event
        for (const ev of sortedEvents) {
          if (ev.eventId === eventId) continue;
          if (ev.endSec <= anchor && ev.endSec > clamped) {
            clamped = ev.endSec;
          }
        }
      } else {
        clamped = Math.max(clamped, anchor + minGap);
        // Don't overlap next event
        for (const ev of sortedEvents) {
          if (ev.eventId === eventId) continue;
          if (ev.startSec >= anchor && ev.startSec < clamped) {
            clamped = ev.startSec;
          }
        }
      }
      return clamped;
    },
    [sortedEvents],
  );

  // Mouse move: drag handling + ghost tracking
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const sec = getMouseSec(e);

      if (dragRef.current) {
        const d = dragRef.current;
        const delta = sec - d.startMouseSec;
        const newEdge = clampEdge(
          d.originalEdgeSec + delta,
          d.edge,
          d.anchor,
          d.eventId,
        );
        const startSec = d.edge === "left" ? newEdge : d.anchor;
        const endSec = d.edge === "left" ? d.anchor : newEdge;
        setDragPreview({ eventId: d.eventId, startSec, endSec });
        return;
      }

      if (addMode) {
        setGhostSec(sec);
      }
    },
    [getMouseSec, clampEdge, addMode],
  );

  const handleMouseUp = useCallback(() => {
    if (dragRef.current && dragPreview) {
      onAdjust(dragPreview.eventId, dragPreview.startSec, dragPreview.endSec);
    }
    dragRef.current = null;
    setDragPreview(null);
  }, [dragPreview, onAdjust]);

  const handleMouseLeave = useCallback(() => {
    setGhostSec(null);
    if (dragRef.current && dragPreview) {
      onAdjust(dragPreview.eventId, dragPreview.startSec, dragPreview.endSec);
    }
    dragRef.current = null;
    setDragPreview(null);
  }, [dragPreview, onAdjust]);

  // Click on empty space: deselect or add
  const handleContainerClick = useCallback(
    (e: React.MouseEvent) => {
      if (addMode && ghostSec !== null) {
        const start = snap(ghostSec - DEFAULT_ADD_DURATION / 2);
        const end = snap(ghostSec + DEFAULT_ADD_DURATION / 2);
        onAdd(activeRegionId, start, end);
        return;
      }
      // Only deselect on click on the container itself (not on a bar)
      if (e.target === e.currentTarget) {
        onSelectEvent(null);
      }
    },
    [addMode, ghostSec, activeRegionId, onAdd, onSelectEvent],
  );

  // Start edge drag
  const startEdgeDrag = useCallback(
    (e: React.MouseEvent, eventId: string, edge: DragEdge, ev: EffectiveEvent) => {
      e.stopPropagation();
      const sec = getMouseSec(e);
      dragRef.current = {
        eventId,
        edge,
        anchor: edge === "left" ? ev.endSec : ev.startSec,
        startMouseSec: sec,
        originalEdgeSec: edge === "left" ? ev.startSec : ev.endSec,
      };
    },
    [getMouseSec],
  );

  // Detect if mouse is near an edge of the selected bar
  const [hoverEdge, setHoverEdge] = useState<DragEdge | null>(null);

  const handleBarMouseMove = useCallback(
    (e: React.MouseEvent, ev: EffectiveEvent) => {
      if (ev.eventId !== selectedEventId || dragRef.current) return;
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const mouseX = e.clientX - rect.left;
      const leftX = secToX(ev.startSec);
      const rightX = secToX(ev.endSec);
      if (Math.abs(mouseX - leftX) <= EDGE_HIT_PX) {
        setHoverEdge("left");
      } else if (Math.abs(mouseX - rightX) <= EDGE_HIT_PX) {
        setHoverEdge("right");
      } else {
        setHoverEdge(null);
      }
    },
    [selectedEventId, secToX],
  );

  // Render bar Y position
  const barTop = ctx.canvasHeight * (1 - BAR_HEIGHT_FRAC);
  const barHeight = ctx.canvasHeight * BAR_HEIGHT_FRAC;

  // Deleted events (shown separately with strikethrough)
  const deletedEvents = events.filter((e) => e.correctionType === "delete");

  return (
    <div
      ref={containerRef}
      className="absolute inset-0"
      style={{
        pointerEvents: "auto",
        cursor: addMode ? "crosshair" : undefined,
      }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      onClick={handleContainerClick}
      data-testid="event-bar-overlay"
    >
      {/* Active (non-deleted) event bars */}
      {sortedEvents.map((ev) => {
        const isDragging = dragPreview?.eventId === ev.eventId;
        const renderStart = isDragging ? dragPreview!.startSec : ev.startSec;
        const renderEnd = isDragging ? dragPreview!.endSec : ev.endSec;
        const x = secToX(renderStart);
        const w = (renderEnd - renderStart) * ctx.pxPerSec;
        const isSelected = ev.eventId === selectedEventId;

        return (
          <div
            key={ev.eventId}
            className={cn(
              "absolute transition-opacity",
              isSelected && "ring-2 ring-white/80",
            )}
            style={{
              left: x,
              top: barTop,
              width: Math.max(w, 2),
              height: barHeight,
              cursor:
                isSelected && hoverEdge
                  ? "col-resize"
                  : addMode
                    ? "crosshair"
                    : "pointer",
              ...barStyle(ev.correctionType, isDragging),
            }}
            onClick={(e) => {
              e.stopPropagation();
              onSelectEvent(ev.eventId);
            }}
            onMouseMove={(e) => handleBarMouseMove(e, ev)}
            onMouseDown={(e) => {
              if (isSelected && hoverEdge) {
                startEdgeDrag(e, ev.eventId, hoverEdge, ev);
              }
            }}
            data-testid={`event-bar-${ev.eventId}`}
            data-correction={ev.correctionType ?? "none"}
          />
        );
      })}

      {/* Deleted event bars (dimmed with strikethrough) */}
      {deletedEvents.map((ev) => {
        const x = secToX(ev.startSec);
        const w = (ev.endSec - ev.startSec) * ctx.pxPerSec;
        return (
          <div
            key={ev.eventId}
            className="absolute"
            style={{
              left: x,
              top: barTop,
              width: Math.max(w, 2),
              height: barHeight,
              background: "rgba(239, 68, 68, 0.15)",
              opacity: 0.3,
              borderTop: "2px solid rgba(239, 68, 68, 0.6)",
            }}
            onClick={(e) => {
              e.stopPropagation();
              onSelectEvent(ev.eventId);
            }}
            data-testid={`event-bar-${ev.eventId}`}
            data-correction="delete"
          >
            {/* Strikethrough line */}
            <div
              className="absolute left-0 right-0"
              style={{
                top: "50%",
                height: 2,
                background: "rgba(239, 68, 68, 0.8)",
              }}
            />
          </div>
        );
      })}

      {/* Add-mode ghost */}
      {addMode && ghostSec !== null && (
        <div
          className="absolute pointer-events-none"
          style={{
            left: secToX(snap(ghostSec - DEFAULT_ADD_DURATION / 2)),
            top: barTop,
            width: DEFAULT_ADD_DURATION * ctx.pxPerSec,
            height: barHeight,
            background: "rgba(34, 197, 94, 0.2)",
            border: "1px dashed rgba(34, 197, 94, 0.7)",
          }}
        />
      )}
    </div>
  );
}

function barStyle(
  correctionType: EffectiveEvent["correctionType"],
  isDragging: boolean,
): React.CSSProperties {
  if (isDragging) {
    return {
      background: "rgba(168, 130, 220, 0.3)",
      border: "1px dashed rgba(168, 130, 220, 0.8)",
    };
  }
  switch (correctionType) {
    case "adjust":
      return {
        background: "rgba(168, 130, 220, 0.5)",
        border: "1px dashed rgba(168, 130, 220, 0.9)",
      };
    case "add":
      return {
        background: "rgba(34, 197, 94, 0.35)",
        border: "1px solid rgba(34, 197, 94, 0.8)",
      };
    default:
      return {
        background: "rgba(168, 130, 220, 0.4)",
      };
  }
}
