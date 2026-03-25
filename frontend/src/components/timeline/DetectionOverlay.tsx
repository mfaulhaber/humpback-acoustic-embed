// frontend/src/components/timeline/DetectionOverlay.tsx
import type { DetectionRow, ZoomLevel } from "@/api/types";
import { COLORS, TILE_DURATION, TILE_WIDTH_PX } from "./constants";

export interface DetectionOverlayProps {
  detections: DetectionRow[];
  jobStart: number;
  centerTimestamp: number;
  zoomLevel: ZoomLevel;
  width: number;
  height: number;
  visible: boolean;
}

function labelColor(row: DetectionRow): string | null {
  if (row.humpback === 1) return COLORS.labelHumpback;
  if (row.orca === 1) return COLORS.labelOrca;
  if (row.ship === 1) return COLORS.labelShip;
  if (row.background === 1) return COLORS.labelBackground;
  return null;
}

export function DetectionOverlay({
  detections,
  jobStart,
  centerTimestamp,
  zoomLevel,
  width,
  height,
  visible,
}: DetectionOverlayProps) {
  if (!visible || width <= 0 || height <= 0) return null;

  const pxPerSec = TILE_WIDTH_PX / TILE_DURATION[zoomLevel];

  return (
    <div
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
      {detections.map((row, i) => {
        const color = labelColor(row);
        if (color === null) return null;

        // start_sec/end_sec are relative to jobStart for hydrophone jobs
        const startEpoch = jobStart + row.start_sec;
        const endEpoch = jobStart + row.end_sec;

        const x = (startEpoch - centerTimestamp) * pxPerSec + width / 2;
        const w = (endEpoch - startEpoch) * pxPerSec;

        // Skip if entirely out of view
        if (x + w < 0 || x > width) return null;

        return (
          <div
            key={row.row_id ?? i}
            style={{
              position: "absolute",
              top: 0,
              left: x,
              width: Math.max(1, w),
              height,
              background: color,
            }}
          />
        );
      })}
    </div>
  );
}
