import React, { useCallback, useMemo, useRef, useState } from "react";
import { useOverlayContext } from "./OverlayContext";
import { cn } from "@/lib/utils";
import type { BoundaryCorrection } from "@/api/types";
import { typeColor } from "@/components/call-parsing/TypePalette";

const BADGE_WIDTH_PX = 22;
const BADGE_HEIGHT_PX = 14;
const NEGATIVE_COLOR = "hsl(0, 70%, 50%)";

const SNAP = 0.1;
const EDGE_HIT_PX = 8;
const DEFAULT_ADD_DURATION = 1.0;
const BAR_HEIGHT_FRAC = 1.0;

function snap(sec: number): number {
  return Math.round(sec / SNAP) * SNAP;
}

export interface EffectiveEvent {
  eventId: string;
  regionId: string;
  startSec: number;
  endSec: number;
  originalStartSec: number;
  originalEndSec: number;
  confidence: number;
  correctionType: "adjust" | "add" | "delete" | null;
  effectiveType: string | null;
  typeSource: "inference" | "correction" | "negative" | null;
}

interface EventBarOverlayProps {
  events: EffectiveEvent[];
  selectedEventId: string | null;
  onSelectEvent: (eventId: string | null) => void;
  onAdjust: (eventId: string, startSec: number, endSec: number) => void;
  onAdd: (regionId: string, startSec: number, endSec: number) => void;
  addMode: boolean;
  activeRegionId: string;
}

type DragEdge = "left" | "right";

interface DragState {
  eventId: string;
  edge: DragEdge;
  anchor: number;
  startMouseSec: number;
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

  const clampEdge = useCallback(
    (candidateSec: number, edge: DragEdge, anchor: number, eventId: string) => {
      let clamped = snap(candidateSec);
      const minGap = SNAP;
      if (edge === "left") {
        clamped = Math.min(clamped, anchor - minGap);
        for (const ev of sortedEvents) {
          if (ev.eventId === eventId) continue;
          if (ev.endSec <= anchor && ev.endSec > clamped) clamped = ev.endSec;
        }
      } else {
        clamped = Math.max(clamped, anchor + minGap);
        for (const ev of sortedEvents) {
          if (ev.eventId === eventId) continue;
          if (ev.startSec >= anchor && ev.startSec < clamped) clamped = ev.startSec;
        }
      }
      return clamped;
    },
    [sortedEvents],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      const sec = getMouseSec(e);

      if (dragRef.current) {
        const d = dragRef.current;
        const delta = sec - d.startMouseSec;
        const newEdge = clampEdge(d.originalEdgeSec + delta, d.edge, d.anchor, d.eventId);
        const startSec = d.edge === "left" ? newEdge : d.anchor;
        const endSec = d.edge === "left" ? d.anchor : newEdge;
        setDragPreview({ eventId: d.eventId, startSec, endSec });
        return;
      }

      if (addMode) setGhostSec(sec);
    },
    [getMouseSec, clampEdge, addMode],
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (dragRef.current && dragPreview) {
        onAdjust(dragPreview.eventId, dragPreview.startSec, dragPreview.endSec);
      }
      dragRef.current = null;
      setDragPreview(null);
    },
    [dragPreview, onAdjust],
  );

  const handleMouseLeave = useCallback(() => {
    setGhostSec(null);
    if (dragRef.current && dragPreview) {
      onAdjust(dragPreview.eventId, dragPreview.startSec, dragPreview.endSec);
    }
    dragRef.current = null;
    setDragPreview(null);
  }, [dragPreview, onAdjust]);

  const handleContainerClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (addMode && ghostSec !== null) {
        const start = snap(ghostSec - DEFAULT_ADD_DURATION / 2);
        const end = snap(ghostSec + DEFAULT_ADD_DURATION / 2);
        onAdd(activeRegionId, start, end);
        return;
      }
      if (e.target === e.currentTarget) onSelectEvent(null);
    },
    [addMode, ghostSec, activeRegionId, onAdd, onSelectEvent],
  );

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

  const barTop = ctx.canvasHeight * (1 - BAR_HEIGHT_FRAC);
  const barHeight = ctx.canvasHeight * BAR_HEIGHT_FRAC;

  const deletedEvents = useMemo(() => {
    const deleted = events.filter((e) => e.correctionType === "delete");
    return deleted.filter(
      (d) => !sortedEvents.some((a) => a.startSec <= d.startSec && a.endSec >= d.endSec),
    );
  }, [events, sortedEvents]);

  return (
    <div
      ref={containerRef}
      className="absolute inset-0"
      style={{ pointerEvents: "auto", cursor: addMode ? "crosshair" : undefined }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      onClick={handleContainerClick}
      data-testid="event-bar-overlay"
    >
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
            className={cn("absolute transition-opacity", isSelected && "ring-2 ring-white/80")}
            style={{
              left: x,
              top: barTop,
              width: Math.max(w, 2),
              height: barHeight,
              cursor: isSelected && hoverEdge ? "col-resize" : addMode ? "crosshair" : "pointer",
              ...barStyle(ev.correctionType, isDragging),
            }}
            onClick={(e) => {
              e.stopPropagation();
              onSelectEvent(ev.eventId);
            }}
            onMouseMove={(e) => handleBarMouseMove(e, ev)}
            onMouseDown={(e) => {
              if (isSelected && hoverEdge) startEdgeDrag(e, ev.eventId, hoverEdge, ev);
            }}
            data-testid={`event-bar-${ev.eventId}`}
            data-correction={ev.correctionType ?? "none"}
          >
            <EventTypeBadge eventId={ev.eventId} effectiveType={ev.effectiveType} typeSource={ev.typeSource} />
          </div>
        );
      })}

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
            <div className="absolute left-0 right-0" style={{ top: "50%", height: 2, background: "rgba(239, 68, 68, 0.8)" }} />
          </div>
        );
      })}

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

function EventTypeBadge({
  eventId,
  effectiveType,
  typeSource,
}: {
  eventId: string;
  effectiveType: EffectiveEvent["effectiveType"];
  typeSource: EffectiveEvent["typeSource"];
}) {
  if (typeSource === null) return null;

  let label: string;
  let background: string;
  let color: string;
  let borderColor: string;

  if (typeSource === "negative") {
    label = "—";
    background = NEGATIVE_COLOR;
    color = "#fff";
    borderColor = NEGATIVE_COLOR;
  } else {
    const type = effectiveType ?? "";
    label = type.slice(0, 2).toUpperCase();
    const tc = typeColor(type);
    borderColor = tc;
    if (typeSource === "correction") {
      background = tc;
      color = "#fff";
    } else {
      background = "#fff";
      color = tc;
    }
  }

  return (
    <div
      data-testid={`event-badge-${eventId}`}
      data-source={typeSource}
      style={{
        position: "absolute",
        left: 0,
        top: 0,
        width: BADGE_WIDTH_PX,
        height: BADGE_HEIGHT_PX,
        lineHeight: `${BADGE_HEIGHT_PX - 2}px`,
        textAlign: "center",
        fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
        fontSize: 9,
        fontWeight: 700,
        letterSpacing: "0.02em",
        background,
        color,
        border: `1px solid ${borderColor}`,
        borderRadius: 2,
        pointerEvents: "none",
        overflow: "visible",
      }}
    >
      {label}
    </div>
  );
}

function barStyle(correctionType: EffectiveEvent["correctionType"], isDragging: boolean): React.CSSProperties {
  if (isDragging) {
    return { background: "rgba(168, 130, 220, 0.3)", border: "1px dashed rgba(168, 130, 220, 0.8)" };
  }
  switch (correctionType) {
    case "adjust":
      return { background: "rgba(168, 130, 220, 0.5)", border: "1px dashed rgba(168, 130, 220, 0.9)" };
    case "add":
      return { background: "rgba(34, 197, 94, 0.35)", border: "1px solid rgba(34, 197, 94, 0.8)" };
    default:
      return { background: "rgba(168, 130, 220, 0.4)" };
  }
}
