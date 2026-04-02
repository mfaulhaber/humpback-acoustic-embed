// frontend/src/components/timeline/LabelEditor.tsx
import React, { useState, useRef, useMemo, useCallback, useEffect } from "react";
import type { DetectionRow } from "@/api/types";
import type { ZoomLevel } from "@/api/types";
import { VIEWPORT_SPAN, LABEL_COLORS, COLORS, type LabelType } from "./constants";
import type { Action } from "@/hooks/queries/useLabelEdits";

export interface LabelEditorProps {
  mergedRows: DetectionRow[];
  mode: "select" | "add";
  selectedLabel: LabelType | null;
  selectedId: string | null;
  dispatch: React.Dispatch<Action>;
  jobStart: number;
  jobDuration: number;
  centerTimestamp: number;
  zoomLevel: ZoomLevel;
  width: number;
  height: number;
}

const SNAP_GRID = 0.5;
const WINDOW_DURATION = 5;

function snapToGrid(sec: number): number {
  return Math.round(sec / SNAP_GRID) * SNAP_GRID;
}

function rowIdKey(row: DetectionRow): string {
  return row.row_id;
}

function getRowLabel(row: DetectionRow): LabelType | null {
  if (row.humpback === 1) return "humpback";
  if (row.orca === 1) return "orca";
  if (row.ship === 1) return "ship";
  if (row.background === 1) return "background";
  return null;
}

function isLabeled(row: DetectionRow): boolean {
  return getRowLabel(row) !== null;
}

interface DragState {
  rowId: string;
  startUtc: number;
  endUtc: number;
  originalStartUtc: number;
  dragStartUtc: number;
  duration: number;
}

export function LabelEditor({
  mergedRows,
  mode,
  selectedLabel,
  selectedId,
  dispatch,
  jobStart,
  jobDuration,
  centerTimestamp,
  zoomLevel,
  width,
  height,
}: LabelEditorProps) {
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const [ghostUtc, setGhostUtc] = useState<number | null>(null);
  const [dragOffset, setDragOffset] = useState<number | null>(null); // visual offset during drag (absolute UTC)

  const dragRef = useRef<DragState | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const pxPerSec = width / VIEWPORT_SPAN[zoomLevel];
  const jobEnd = jobStart + jobDuration;

  // Sorted labeled rows for overlap checks
  const labeledOthers = useMemo(() => {
    return mergedRows
      .filter((r) => isLabeled(r))
      .sort((a, b) => a.start_utc - b.start_utc);
  }, [mergedRows]);

  // --- Coordinate transforms (absolute UTC ↔ pixels) ---
  const utcToX = useCallback(
    (utcSec: number): number => {
      return (utcSec - centerTimestamp) * pxPerSec + width / 2;
    },
    [centerTimestamp, pxPerSec, width],
  );

  const pxToUtc = useCallback(
    (px: number): number => {
      return (px - width / 2) / pxPerSec + centerTimestamp;
    },
    [width, pxPerSec, centerTimestamp],
  );

  const getMouseUtc = useCallback(
    (e: React.MouseEvent): number => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return 0;
      return pxToUtc(e.clientX - rect.left);
    },
    [pxToUtc],
  );

  // --- Overlap clamping for drag ---
  const clampDrag = useCallback(
    (candidateStart: number, duration: number, dragStartUtc: number, dragEndUtc: number): number => {
      const others = labeledOthers.filter(
        (r) => !(r.start_utc === dragStartUtc && r.end_utc === dragEndUtc),
      );
      let cs = candidateStart;
      // Clamp to job bounds (absolute UTC)
      cs = Math.max(jobStart, Math.min(jobEnd - duration, cs));
      // Clamp against neighbors
      for (const other of others) {
        if (cs < other.end_utc && cs + duration > other.start_utc) {
          const leftGap = other.start_utc - duration;
          const rightGap = other.end_utc;
          if (Math.abs(cs - leftGap) < Math.abs(cs - rightGap)) {
            cs = leftGap;
          } else {
            cs = rightGap;
          }
        }
      }
      // Final bounds clamp after neighbor adjustments
      cs = Math.max(jobStart, Math.min(jobEnd - duration, cs));
      return cs;
    },
    [labeledOthers, jobStart, jobEnd],
  );

  // --- Ghost (add mode) overlap checks ---
  const ghostStart = ghostUtc !== null ? snapToGrid(ghostUtc - WINDOW_DURATION / 2) : null;
  const ghostEnd = ghostStart !== null ? ghostStart + WINDOW_DURATION : null;

  const ghostOverlapsLabeled = useMemo(() => {
    if (ghostStart === null || ghostEnd === null) return false;
    return labeledOthers.some(
      (r) => ghostStart < r.end_utc && ghostEnd > r.start_utc,
    );
  }, [ghostStart, ghostEnd, labeledOthers]);

  // Set of UTC keys of unlabeled rows overlapped by ghost
  const ghostOverlappedUnlabeled = useMemo(() => {
    if (ghostStart === null || ghostEnd === null) return new Set<string>();
    const keys = new Set<string>();
    for (const r of mergedRows) {
      if (!isLabeled(r) && ghostStart < r.end_utc && ghostEnd > r.start_utc) {
        keys.add(rowIdKey(r));
      }
    }
    return keys;
  }, [ghostStart, ghostEnd, mergedRows]);

  // --- Event handlers ---

  const handleContainerMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const utc = getMouseUtc(e);

      // Drag handling
      if (dragRef.current) {
        const drag = dragRef.current;
        const delta = utc - drag.dragStartUtc;
        const candidate = snapToGrid(drag.originalStartUtc + delta);
        const clamped = clampDrag(candidate, drag.duration, drag.startUtc, drag.endUtc);
        setDragOffset(clamped);
        return;
      }

      // Ghost tracking in add mode
      if (mode === "add") {
        setGhostUtc(utc);
      }
    },
    [getMouseUtc, mode, clampDrag],
  );

  const handleContainerMouseUp = useCallback(() => {
    if (dragRef.current && dragOffset !== null) {
      const drag = dragRef.current;
      const duration = drag.duration;
      dispatch({
        type: "move",
        row_id: drag.rowId,
        start_utc: dragOffset,
        end_utc: dragOffset + duration,
      });
    }
    dragRef.current = null;
    setDragOffset(null);
  }, [dragOffset, dispatch]);

  const handleContainerMouseLeave = useCallback(() => {
    setGhostUtc(null);
    // If dragging, commit on leave
    if (dragRef.current && dragOffset !== null) {
      const drag = dragRef.current;
      dispatch({
        type: "move",
        row_id: drag.rowId,
        start_utc: dragOffset,
        end_utc: dragOffset + drag.duration,
      });
    }
    dragRef.current = null;
    setDragOffset(null);
  }, [dragOffset, dispatch]);

  const handleContainerClick = useCallback(
    (e: React.MouseEvent) => {
      // Add mode click
      if (mode === "add" && ghostStart !== null && ghostEnd !== null && !ghostOverlapsLabeled && selectedLabel) {
        dispatch({ type: "add", start_utc: ghostStart, end_utc: ghostEnd, label: selectedLabel });
        return;
      }
      // Select mode: click on empty space → deselect
      if (mode === "select" && e.target === e.currentTarget) {
        dispatch({ type: "select", id: null });
      }
    },
    [mode, ghostStart, ghostEnd, ghostOverlapsLabeled, dispatch, selectedLabel],
  );

  const handleBarMouseDown = useCallback(
    (row: DetectionRow, e: React.MouseEvent) => {
      if (mode !== "select") return;
      e.stopPropagation();

      // Select on click
      dispatch({ type: "select", id: row.row_id });

      // Start drag only for labeled rows (unlabeled rows are selectable but not draggable)
      if (isLabeled(row)) {
        const utc = getMouseUtc(e);
        const duration = row.end_utc - row.start_utc;
        dragRef.current = {
          rowId: row.row_id,
          startUtc: row.start_utc,
          endUtc: row.end_utc,
          originalStartUtc: row.start_utc,
          dragStartUtc: utc,
          duration,
        };
        setDragOffset(row.start_utc);
      }
    },
    [mode, dispatch, getMouseUtc],
  );

  // --- Global drag listeners for select mode ---
  useEffect(() => {
    if (mode !== "select") return;

    const handleGlobalMove = (e: MouseEvent) => {
      if (!dragRef.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const utc = (px - width / 2) / pxPerSec + centerTimestamp;
      const drag = dragRef.current;
      const delta = utc - drag.dragStartUtc;
      const candidate = snapToGrid(drag.originalStartUtc + delta);
      const clamped = clampDrag(candidate, drag.duration, drag.startUtc, drag.endUtc);
      setDragOffset(clamped);
    };

    const handleGlobalUp = () => {
      if (dragRef.current && dragOffset !== null) {
        const drag = dragRef.current;
        dispatch({
          type: "move",
          row_id: drag.rowId,
          start_utc: dragOffset,
          end_utc: dragOffset + drag.duration,
        });
      }
      dragRef.current = null;
      setDragOffset(null);
    };

    window.addEventListener("mousemove", handleGlobalMove);
    window.addEventListener("mouseup", handleGlobalUp);
    return () => {
      window.removeEventListener("mousemove", handleGlobalMove);
      window.removeEventListener("mouseup", handleGlobalUp);
    };
  }, [mode, width, pxPerSec, centerTimestamp, clampDrag, dragOffset, dispatch]);

  if (width <= 0 || height <= 0) return null;

  // --- Build bar data ---
  const bars: {
    row: DetectionRow;
    x: number;
    w: number;
    label: LabelType | null;
    isSelected: boolean;
    isManual: boolean;
    dimmed: boolean;
    key: string;
  }[] = [];

  for (const row of mergedRows) {
    const label = getRowLabel(row);
    const key = rowIdKey(row);

    // During drag, use drag offset for the dragged row
    let startUtc = row.start_utc;
    let endUtc = row.end_utc;
    if (dragRef.current && dragOffset !== null && dragRef.current.rowId === row.row_id) {
      startUtc = dragOffset;
      endUtc = dragOffset + dragRef.current.duration;
    }

    const x = utcToX(startUtc);
    const w = Math.max(2, (endUtc - startUtc) * pxPerSec);

    // Skip if entirely out of view
    if (x + w < 0 || x > width) continue;

    const isSelected = key === selectedId;
    const isManual = row.avg_confidence === null;
    const dimmed = !label && ghostOverlappedUnlabeled.has(key);

    bars.push({ row, x, w, label, isSelected, isManual, dimmed, key });
  }

  // Ghost bar rendering data
  let ghostBar: { x: number; w: number; color: string } | null = null;
  if (mode === "add" && ghostStart !== null && ghostEnd !== null && selectedLabel) {
    const gx = utcToX(ghostStart);
    const gw = Math.max(2, (ghostEnd - ghostStart) * pxPerSec);
    const color = ghostOverlapsLabeled
      ? "rgba(239, 68, 68, 0.4)"
      : LABEL_COLORS[selectedLabel].fill;
    ghostBar = { x: gx, w: gw, color };
  }

  const isDragging = dragRef.current !== null;

  const containerInteractive = mode === "add";

  return (
    <div
      ref={containerRef}
      data-testid="label-editor"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width,
        height,
        pointerEvents: containerInteractive ? "auto" : "none",
        zIndex: 10,
        overflow: "hidden",
        cursor: mode === "add" ? "crosshair" : isDragging ? "grabbing" : "default",
      }}
      onMouseMove={containerInteractive ? handleContainerMouseMove : undefined}
      onMouseUp={containerInteractive ? handleContainerMouseUp : undefined}
      onMouseLeave={containerInteractive ? handleContainerMouseLeave : undefined}
      onClick={containerInteractive ? handleContainerClick : undefined}
    >
      {/* Render all bars */}
      {bars.map(({ row, x, w, label, isSelected, isManual, dimmed, key }) => {
        const isHovered = hoveredKey === key;
        const isBeingDragged = dragRef.current !== null &&
          dragRef.current.rowId === row.row_id &&
          dragOffset !== null;

        let bg: string;
        let opacity = 1;

        if (label) {
          bg = isHovered ? LABEL_COLORS[label].hover : LABEL_COLORS[label].fill;
        } else {
          // Unlabeled: subtle confidence-based
          const conf = row.avg_confidence ?? 0;
          const alpha = 0.08 + conf * 0.17;
          bg = `rgba(64, 224, 192, ${alpha.toFixed(3)})`;
          if (dimmed) opacity = 0.3;
        }

        return (
          <div
            key={key}
            style={{
              position: "absolute",
              top: 0,
              left: x,
              width: w,
              height,
              background: bg,
              opacity,
              border: isSelected ? "2px solid white" : "none",
              boxShadow: isSelected ? "0 0 8px rgba(255, 255, 255, 0.4)" : "none",
              borderTop:
                isManual && label && !isSelected
                  ? "2px dashed rgba(255, 255, 255, 0.5)"
                  : undefined,
              cursor:
                mode === "add"
                  ? "crosshair"
                  : label
                    ? isBeingDragged
                      ? "grabbing"
                      : "grab"
                    : "pointer",
              pointerEvents: "auto",
              boxSizing: "border-box",
              transition: isBeingDragged ? "none" : "opacity 0.15s",
            }}
            onMouseEnter={() => setHoveredKey(key)}
            onMouseLeave={() => setHoveredKey(null)}
            onMouseDown={(e) => {
              if (mode === "select") {
                handleBarMouseDown(row, e);
              }
            }}
            onClick={(e) => {
              if (mode === "select") {
                e.stopPropagation();
              }
            }}
          />
        );
      })}

      {/* Ghost preview in add mode */}
      {ghostBar && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: ghostBar.x,
            width: ghostBar.w,
            height,
            background: ghostBar.color,
            pointerEvents: "none",
            boxSizing: "border-box",
            border: ghostOverlapsLabeled
              ? "2px dashed rgba(239, 68, 68, 0.7)"
              : `2px dashed ${selectedLabel ? LABEL_COLORS[selectedLabel].border : COLORS.border}`,
          }}
        />
      )}
    </div>
  );
}
