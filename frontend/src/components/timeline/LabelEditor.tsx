// frontend/src/components/timeline/LabelEditor.tsx
import React, { useState, useRef, useMemo, useCallback, useEffect } from "react";
import type { DetectionRow } from "@/api/types";
import type { ZoomLevel } from "@/api/types";
import { VIEWPORT_SPAN, LABEL_COLORS, type LabelType } from "./constants";
import type { Action } from "@/hooks/queries/useLabelEdits";

export interface LabelEditorProps {
  mergedRows: DetectionRow[];
  mode: "select" | "add";
  selectedLabel: LabelType;
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
  originalStartSec: number;
  dragStartSec: number;
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
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [ghostSec, setGhostSec] = useState<number | null>(null);
  const [dragOffset, setDragOffset] = useState<number | null>(null); // visual offset during drag (start_sec)

  const dragRef = useRef<DragState | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const pxPerSec = width / VIEWPORT_SPAN[zoomLevel];

  // Sorted labeled rows for overlap checks
  const labeledOthers = useMemo(() => {
    return mergedRows
      .filter((r) => isLabeled(r))
      .sort((a, b) => a.start_sec - b.start_sec);
  }, [mergedRows]);

  // --- Coordinate transforms ---
  const secToX = useCallback(
    (sec: number): number => {
      return (jobStart + sec - centerTimestamp) * pxPerSec + width / 2;
    },
    [jobStart, centerTimestamp, pxPerSec, width],
  );

  const pxToSec = useCallback(
    (px: number): number => {
      return (px - width / 2) / pxPerSec + centerTimestamp - jobStart;
    },
    [width, pxPerSec, centerTimestamp, jobStart],
  );

  const getMouseSec = useCallback(
    (e: React.MouseEvent): number => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return 0;
      return pxToSec(e.clientX - rect.left);
    },
    [pxToSec],
  );

  // --- Overlap clamping for drag ---
  const clampDrag = useCallback(
    (candidateStart: number, duration: number, dragRowId: string): number => {
      const others = labeledOthers.filter(
        (r) => r.row_id !== dragRowId,
      );
      let cs = candidateStart;
      // Clamp to job bounds
      cs = Math.max(0, Math.min(jobDuration - duration, cs));
      // Clamp against neighbors
      for (const other of others) {
        if (cs < other.end_sec && cs + duration > other.start_sec) {
          const leftGap = other.start_sec - duration;
          const rightGap = other.end_sec;
          if (Math.abs(cs - leftGap) < Math.abs(cs - rightGap)) {
            cs = leftGap;
          } else {
            cs = rightGap;
          }
        }
      }
      // Final bounds clamp after neighbor adjustments
      cs = Math.max(0, Math.min(jobDuration - duration, cs));
      return cs;
    },
    [labeledOthers, jobDuration],
  );

  // --- Ghost (add mode) overlap checks ---
  const ghostStart = ghostSec !== null ? snapToGrid(ghostSec - WINDOW_DURATION / 2) : null;
  const ghostEnd = ghostStart !== null ? ghostStart + WINDOW_DURATION : null;

  const ghostOverlapsLabeled = useMemo(() => {
    if (ghostStart === null || ghostEnd === null) return false;
    return labeledOthers.some(
      (r) => ghostStart < r.end_sec && ghostEnd > r.start_sec,
    );
  }, [ghostStart, ghostEnd, labeledOthers]);

  // Set of row_ids of unlabeled rows overlapped by ghost
  const ghostOverlappedUnlabeled = useMemo(() => {
    if (ghostStart === null || ghostEnd === null) return new Set<string>();
    const ids = new Set<string>();
    for (const r of mergedRows) {
      if (!isLabeled(r) && r.row_id && ghostStart < r.end_sec && ghostEnd > r.start_sec) {
        ids.add(r.row_id);
      }
    }
    return ids;
  }, [ghostStart, ghostEnd, mergedRows]);

  // --- Event handlers ---

  const handleContainerMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const sec = getMouseSec(e);

      // Drag handling
      if (dragRef.current) {
        const drag = dragRef.current;
        const delta = sec - drag.dragStartSec;
        const candidate = snapToGrid(drag.originalStartSec + delta);
        const clamped = clampDrag(candidate, drag.duration, drag.rowId);
        setDragOffset(clamped);
        return;
      }

      // Ghost tracking in add mode
      if (mode === "add") {
        setGhostSec(sec);
      }
    },
    [getMouseSec, mode, clampDrag],
  );

  const handleContainerMouseUp = useCallback(() => {
    if (dragRef.current && dragOffset !== null) {
      const drag = dragRef.current;
      const duration = drag.duration;
      dispatch({
        type: "move",
        row_id: drag.rowId,
        new_start_sec: dragOffset,
        new_end_sec: dragOffset + duration,
      });
    }
    dragRef.current = null;
    setDragOffset(null);
  }, [dragOffset, dispatch]);

  const handleContainerMouseLeave = useCallback(() => {
    setGhostSec(null);
    // If dragging, commit on leave
    if (dragRef.current && dragOffset !== null) {
      const drag = dragRef.current;
      dispatch({
        type: "move",
        row_id: drag.rowId,
        new_start_sec: dragOffset,
        new_end_sec: dragOffset + drag.duration,
      });
    }
    dragRef.current = null;
    setDragOffset(null);
  }, [dragOffset, dispatch]);

  const handleContainerClick = useCallback(
    (e: React.MouseEvent) => {
      // Add mode click
      if (mode === "add" && ghostStart !== null && ghostEnd !== null && !ghostOverlapsLabeled) {
        dispatch({ type: "add", start_sec: ghostStart, end_sec: ghostEnd, label: selectedLabel });
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
      const rowId = row.row_id;
      if (!rowId) return;

      // Select on click
      dispatch({ type: "select", id: rowId });

      // Start drag only if already selected (or just-selected)
      const sec = getMouseSec(e);
      const duration = row.end_sec - row.start_sec;
      dragRef.current = {
        rowId,
        originalStartSec: row.start_sec,
        dragStartSec: sec,
        duration,
      };
      setDragOffset(row.start_sec);
    },
    [mode, dispatch, getMouseSec],
  );

  // --- Global drag listeners for select mode ---
  // In select mode the container has pointerEvents: "none" so viewport pan
  // works, but we still need to track mousemove/mouseup during bar drags.
  useEffect(() => {
    if (mode !== "select") return;

    const handleGlobalMove = (e: MouseEvent) => {
      if (!dragRef.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const sec = (px - width / 2) / pxPerSec + centerTimestamp - jobStart;
      const drag = dragRef.current;
      const delta = sec - drag.dragStartSec;
      const candidate = snapToGrid(drag.originalStartSec + delta);
      const clamped = clampDrag(candidate, drag.duration, drag.rowId);
      setDragOffset(clamped);
    };

    const handleGlobalUp = () => {
      if (dragRef.current && dragOffset !== null) {
        const drag = dragRef.current;
        dispatch({
          type: "move",
          row_id: drag.rowId,
          new_start_sec: dragOffset,
          new_end_sec: dragOffset + drag.duration,
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
  }, [mode, width, pxPerSec, centerTimestamp, jobStart, clampDrag, dragOffset, dispatch]);

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
  }[] = [];

  for (const row of mergedRows) {
    const label = getRowLabel(row);
    const rowId = row.row_id ?? "";

    // During drag, use drag offset for the dragged row
    let startSec = row.start_sec;
    let endSec = row.end_sec;
    if (dragRef.current && dragOffset !== null && dragRef.current.rowId === rowId) {
      startSec = dragOffset;
      endSec = dragOffset + dragRef.current.duration;
    }

    const x = secToX(startSec);
    const w = Math.max(2, (endSec - startSec) * pxPerSec);

    // Skip if entirely out of view
    if (x + w < 0 || x > width) continue;

    const isSelected = rowId === selectedId;
    const isManual = row.avg_confidence === null;
    const dimmed = !label && ghostOverlappedUnlabeled.has(rowId);

    bars.push({ row, x, w, label, isSelected, isManual, dimmed });
  }

  // Ghost bar rendering data
  let ghostBar: { x: number; w: number; color: string } | null = null;
  if (mode === "add" && ghostStart !== null && ghostEnd !== null) {
    const gx = secToX(ghostStart);
    const gw = Math.max(2, (ghostEnd - ghostStart) * pxPerSec);
    const color = ghostOverlapsLabeled
      ? "rgba(239, 68, 68, 0.4)"
      : LABEL_COLORS[selectedLabel].fill;
    ghostBar = { x: gx, w: gw, color };
  }

  const isDragging = dragRef.current !== null;

  // In select mode: container is transparent to pointer events so viewport
  // pan still works.  Only individual label bars capture clicks/drags.
  // In add mode: container captures events for ghost tracking and placement,
  // but only when NOT panning (no drag in progress on the viewport).
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
      {bars.map(({ row, x, w, label, isSelected, isManual, dimmed }) => {
        const rowId = row.row_id ?? "";
        const isHovered = hoveredId === rowId;
        const isBeingDragged = dragRef.current?.rowId === rowId && dragOffset !== null;

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
            key={rowId || `${row.start_sec}-${row.end_sec}`}
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
                    : "default",
              pointerEvents: "auto",
              boxSizing: "border-box",
              transition: isBeingDragged ? "none" : "opacity 0.15s",
            }}
            onMouseEnter={() => setHoveredId(rowId || null)}
            onMouseLeave={() => setHoveredId(null)}
            onMouseDown={(e) => {
              if (label && mode === "select") {
                handleBarMouseDown(row, e);
              }
            }}
            onClick={(e) => {
              if (mode === "select" && label) {
                e.stopPropagation();
                // Selection handled in mousedown
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
              : `2px dashed ${LABEL_COLORS[selectedLabel].border}`,
          }}
        />
      )}
    </div>
  );
}
