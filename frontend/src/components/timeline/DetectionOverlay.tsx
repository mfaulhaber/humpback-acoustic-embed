// frontend/src/components/timeline/DetectionOverlay.tsx
import React, { useState, useCallback, useRef, useLayoutEffect } from "react";
import type { DetectionRow, ZoomLevel } from "@/api/types";
import { VIEWPORT_SPAN, LABEL_COLORS, type LabelType } from "./constants";

export interface DetectionOverlayProps {
  detections: DetectionRow[];
  jobStart: number;
  centerTimestamp: number;
  zoomLevel: ZoomLevel;
  width: number;
  height: number;
  visible: boolean;
  onDetectionClick?: (row: DetectionRow, x: number, y: number) => void;
}

// Unlabeled detections use confidence-based subtle coloring
function confidenceColor(conf: number): string {
  // Low confidence → barely visible, high confidence → moderately visible
  const alpha = 0.08 + conf * 0.17; // range 0.08–0.25
  return `rgba(64, 224, 192, ${alpha.toFixed(3)})`;
}

function confidenceHoverColor(conf: number): string {
  const alpha = 0.2 + conf * 0.3;
  return `rgba(64, 224, 192, ${alpha.toFixed(3)})`;
}

function getLabel(row: DetectionRow): LabelType | null {
  if (row.humpback === 1) return "humpback";
  if (row.orca === 1) return "orca";
  if (row.ship === 1) return "ship";
  if (row.background === 1) return "background";
  return null;
}

function getDisplayLabel(row: DetectionRow): string {
  return getLabel(row) ?? "detection";
}

function formatTime(epoch: number): string {
  const d = new Date(epoch * 1000);
  return d.toISOString().slice(11, 19) + " UTC";
}

const TOOLTIP_OFFSET = 12;

interface TooltipState {
  mouseX: number;
  mouseY: number;
  label: string;
  startTime: string;
  endTime: string;
  avgConfidence: number;
  peakConfidence: number;
}

export function DetectionOverlay({
  detections,
  jobStart,
  centerTimestamp,
  zoomLevel,
  width,
  height,
  visible,
  onDetectionClick,
}: DetectionOverlayProps) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ left: number; top: number }>({ left: 0, top: 0 });
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Clamp tooltip position after render so we can measure its dimensions
  useLayoutEffect(() => {
    if (!tooltip || !tooltipRef.current) return;
    const el = tooltipRef.current;
    const tw = el.offsetWidth;
    const th = el.offsetHeight;

    let left = tooltip.mouseX + TOOLTIP_OFFSET;
    let top = tooltip.mouseY + TOOLTIP_OFFSET;

    // Flip left if overflowing right
    if (left + tw > width) {
      left = tooltip.mouseX - TOOLTIP_OFFSET - tw;
    }
    // Flip above if overflowing bottom
    if (top + th > height) {
      top = tooltip.mouseY - TOOLTIP_OFFSET - th;
    }
    // Final clamp to container
    left = Math.max(0, Math.min(left, width - tw));
    top = Math.max(0, Math.min(top, height - th));

    setTooltipPos({ left, top });
  }, [tooltip, width, height]);

  const handleMouseEnter = useCallback(
    (row: DetectionRow, label: string, idx: number, e: React.MouseEvent) => {
      const rect = e.currentTarget.parentElement?.getBoundingClientRect();
      const mouseX = e.clientX - (rect?.left ?? 0);
      const mouseY = e.clientY - (rect?.top ?? 0);
      setHoveredIdx(idx);
      setTooltip({
        mouseX,
        mouseY,
        label,
        startTime: formatTime(row.start_utc),
        endTime: formatTime(row.end_utc),
        avgConfidence: row.avg_confidence ?? 0,
        peakConfidence: row.peak_confidence ?? 0,
      });
    },
    [],
  );

  const handleMouseLeave = useCallback(() => {
    setHoveredIdx(null);
    setTooltip(null);
  }, []);

  const handleClick = useCallback(
    (row: DetectionRow, e: React.MouseEvent) => {
      if (!onDetectionClick) return;
      e.stopPropagation();
      const rect = e.currentTarget.parentElement?.getBoundingClientRect();
      const barRect = e.currentTarget.getBoundingClientRect();
      const x = barRect.left - (rect?.left ?? 0) + barRect.width / 2;
      const y = barRect.top - (rect?.top ?? 0) + barRect.height;
      onDetectionClick(row, x, y);
    },
    [onDetectionClick],
  );

  if (!visible || width <= 0 || height <= 0) return null;

  const pxPerSec = width / VIEWPORT_SPAN[zoomLevel];

  // Build bars for all detections
  const bars: {
    row: DetectionRow;
    label: string;
    x: number;
    w: number;
    idx: number;
    color: string;
    hoverColor: string;
  }[] = [];

  for (let i = 0; i < detections.length; i++) {
    const row = detections[i];

    const x = (row.start_utc - centerTimestamp) * pxPerSec + width / 2;
    const w = Math.max(2, (row.end_utc - row.start_utc) * pxPerSec);

    // Skip if entirely out of view
    if (x + w < 0 || x > width) continue;

    const labelType = getLabel(row);
    const label = getDisplayLabel(row);
    const conf = row.avg_confidence ?? 0;

    let color: string;
    let hoverColor: string;

    if (labelType) {
      // Labeled row — use warm/cool palette from LABEL_COLORS
      color = LABEL_COLORS[labelType].fill;
      hoverColor = LABEL_COLORS[labelType].hover;
    } else {
      // Unlabeled — subtle confidence-based coloring
      color = confidenceColor(conf);
      hoverColor = confidenceHoverColor(conf);
    }

    bars.push({ row, label, x, w, idx: i, color, hoverColor });
  }

  return (
    <div
      data-testid="detection-overlay"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width,
        height,
        pointerEvents: "none",
        zIndex: 5,
        overflow: "hidden",
      }}
    >
      {bars.map(({ row, label, x, w, idx, color, hoverColor }) => (
        <div
          key={`${row.start_utc}:${row.end_utc}`}
          style={{
            position: "absolute",
            top: 0,
            left: x,
            width: w,
            height,
            background: hoveredIdx === idx ? hoverColor : color,
            pointerEvents: "auto",
            cursor: onDetectionClick ? "pointer" : "default",
          }}
          onMouseEnter={(e) => handleMouseEnter(row, label, idx, e)}
          onMouseLeave={handleMouseLeave}
          onClick={(e) => handleClick(row, e)}
        />
      ))}

      {/* Tooltip */}
      {tooltip && (
        <div
          ref={tooltipRef}
          style={{
            position: "absolute",
            left: tooltipPos.left,
            top: tooltipPos.top,
            background: "rgba(6, 13, 20, 0.95)",
            border: "1px solid rgba(64, 224, 192, 0.3)",
            borderRadius: 6,
            padding: "6px 10px",
            fontSize: 11,
            lineHeight: "1.5",
            color: "#a0c8c0",
            whiteSpace: "nowrap",
            pointerEvents: "none",
            zIndex: 20,
          }}
        >
          <div style={{ fontWeight: 600, textTransform: "capitalize", marginBottom: 2 }}>
            {tooltip.label}
          </div>
          <div>
            {tooltip.startTime} &ndash; {tooltip.endTime}
          </div>
          <div>
            Confidence: avg {(tooltip.avgConfidence * 100).toFixed(1)}% / peak{" "}
            {(tooltip.peakConfidence * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  );
}
