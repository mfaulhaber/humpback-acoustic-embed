import type { ZoomLevel } from "@/api/types";
import { VIEWPORT_SPAN } from "./constants";

export interface RegionOverlayProps {
  regions: {
    start_sec: number;
    end_sec: number;
    padded_start_sec: number;
    padded_end_sec: number;
    max_score: number;
  }[];
  jobStart: number;
  centerTimestamp: number;
  zoomLevel: ZoomLevel;
  width: number;
  height: number;
  visible: boolean;
}

export function RegionOverlay({
  regions,
  jobStart,
  centerTimestamp,
  zoomLevel,
  width,
  height,
  visible,
}: RegionOverlayProps) {
  if (!visible || width <= 0 || height <= 0) return null;

  const pxPerSec = width / VIEWPORT_SPAN[zoomLevel];

  const bars: { x: number; w: number; alpha: number; key: string }[] = [];

  for (let i = 0; i < regions.length; i++) {
    const r = regions[i];
    const startEpoch = jobStart + r.padded_start_sec;
    const endEpoch = jobStart + r.padded_end_sec;

    const x = (startEpoch - centerTimestamp) * pxPerSec + width / 2;
    const w = Math.max(2, (endEpoch - startEpoch) * pxPerSec);

    if (x + w < 0 || x > width) continue;

    const alpha = 0.10 + r.max_score * 0.25;
    bars.push({ x, w, alpha, key: `${r.padded_start_sec}:${r.padded_end_sec}` });
  }

  return (
    <div
      data-testid="region-overlay"
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
      {bars.map(({ x, w, alpha, key }) => (
        <div
          key={key}
          style={{
            position: "absolute",
            top: 0,
            left: x,
            width: w,
            height,
            background: `rgba(64, 224, 192, ${alpha.toFixed(3)})`,
          }}
        />
      ))}
    </div>
  );
}
