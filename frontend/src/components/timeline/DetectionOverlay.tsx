// frontend/src/components/timeline/DetectionOverlay.tsx
import React, { useState, useCallback } from "react";
import type { DetectionRow, ZoomLevel } from "@/api/types";
import { TILE_DURATION, TILE_WIDTH_PX } from "./constants";

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

// Positive labels get bright, distinctive colors
const POSITIVE_COLORS: Record<string, string> = {
  humpback: "rgba(64, 224, 192, 0.35)",
  orca: "rgba(224, 176, 64, 0.35)",
};

const POSITIVE_HOVER_COLORS: Record<string, string> = {
  humpback: "rgba(64, 224, 192, 0.6)",
  orca: "rgba(224, 176, 64, 0.6)",
};

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

function getPositiveLabel(row: DetectionRow): string | null {
  if (row.humpback === 1) return "humpback";
  if (row.orca === 1) return "orca";
  return null;
}

function isNegativeLabel(row: DetectionRow): boolean {
  return row.ship === 1 || row.background === 1;
}

function getDisplayLabel(row: DetectionRow): string {
  const pos = getPositiveLabel(row);
  if (pos) return pos;
  if (row.ship === 1) return "ship";
  if (row.background === 1) return "background";
  return "detection";
}

function formatTime(epoch: number): string {
  const d = new Date(epoch * 1000);
  return d.toISOString().slice(11, 19) + " UTC";
}

interface TooltipState {
  x: number;
  y: number;
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

  const handleMouseEnter = useCallback(
    (row: DetectionRow, label: string, idx: number, e: React.MouseEvent) => {
      const rect = e.currentTarget.parentElement?.getBoundingClientRect();
      const barRect = e.currentTarget.getBoundingClientRect();
      const x = barRect.left - (rect?.left ?? 0) + barRect.width / 2;
      const y = barRect.top - (rect?.top ?? 0);
      setHoveredIdx(idx);
      setTooltip({
        x,
        y,
        label,
        startTime: formatTime(jobStart + row.start_sec),
        endTime: formatTime(jobStart + row.end_sec),
        avgConfidence: row.avg_confidence,
        peakConfidence: row.peak_confidence,
      });
    },
    [jobStart],
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

  const pxPerSec = TILE_WIDTH_PX / TILE_DURATION[zoomLevel];

  // Build bars for all detections (skip negatives)
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

    // Skip explicitly negative-labeled rows (ship, background)
    if (isNegativeLabel(row)) continue;

    const startEpoch = jobStart + row.start_sec;
    const endEpoch = jobStart + row.end_sec;

    const x = (startEpoch - centerTimestamp) * pxPerSec + width / 2;
    const w = Math.max(2, (endEpoch - startEpoch) * pxPerSec);

    // Skip if entirely out of view
    if (x + w < 0 || x > width) continue;

    const positiveLabel = getPositiveLabel(row);
    const label = getDisplayLabel(row);
    const conf = row.avg_confidence ?? 0;

    let color: string;
    let hoverColor: string;

    if (positiveLabel) {
      // Labeled positive — bright distinctive color
      color = POSITIVE_COLORS[positiveLabel];
      hoverColor = POSITIVE_HOVER_COLORS[positiveLabel];
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
          key={row.row_id ?? idx}
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
          style={{
            position: "absolute",
            left: tooltip.x,
            top: Math.max(0, tooltip.y - 4),
            transform: "translate(-50%, -100%)",
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
