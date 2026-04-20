import type { RegionCorrectionResponse } from "@/api/types";
import { useOverlayContext } from "./OverlayContext";

export interface RegionOverlayProps {
  regions: {
    region_id?: string;
    start_sec: number;
    end_sec: number;
    padded_start_sec: number;
    padded_end_sec: number;
    max_score: number;
  }[];
  jobStart: number;
  visible: boolean;
  corrections?: RegionCorrectionResponse[];
  onRegionClick?: (regionId: string) => void;
}

interface Bar {
  x: number;
  w: number;
  alpha: number;
  key: string;
  regionId?: string;
  corrected: boolean;
}

export function RegionOverlay({ regions, jobStart, visible, corrections, onRegionClick }: RegionOverlayProps) {
  const { epochToX, canvasWidth, canvasHeight } = useOverlayContext();

  if (!visible || canvasWidth <= 0 || canvasHeight <= 0) return null;

  const interactive = !!onRegionClick;

  const correctionMap = new Map<string, RegionCorrectionResponse>();
  if (corrections) {
    for (const c of corrections) {
      correctionMap.set(c.region_id, c);
    }
  }

  const bars: Bar[] = [];
  for (const r of regions) {
    const c = r.region_id ? correctionMap.get(r.region_id) : undefined;

    if (c?.correction_type === "delete") continue;

    let startSec: number;
    let endSec: number;
    if (c?.correction_type === "adjust" && c.start_sec != null && c.end_sec != null) {
      startSec = c.start_sec;
      endSec = c.end_sec;
    } else {
      startSec = r.padded_start_sec;
      endSec = r.padded_end_sec;
    }

    const startEpoch = jobStart + startSec;
    const endEpoch = jobStart + endSec;
    const x = epochToX(startEpoch);
    const w = Math.max(2, epochToX(endEpoch) - x);
    if (x + w < 0 || x > canvasWidth) continue;
    const alpha = 0.1 + r.max_score * 0.25;
    bars.push({ x, w, alpha, key: `${startSec}:${endSec}`, regionId: r.region_id, corrected: !!c });
  }

  if (correctionMap.size > 0) {
    correctionMap.forEach((c) => {
      if (c.correction_type === "add" && c.start_sec != null && c.end_sec != null) {
        const x = epochToX(jobStart + c.start_sec);
        const w = Math.max(2, epochToX(jobStart + c.end_sec) - x);
        if (x + w >= 0 && x <= canvasWidth) {
          bars.push({ x, w, alpha: 0.2, key: `add:${c.region_id}`, regionId: c.region_id, corrected: true });
        }
      }
    });
  }

  return (
    <div
      data-testid="region-overlay"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: canvasWidth,
        height: canvasHeight,
        pointerEvents: interactive ? "auto" : "none",
        zIndex: 5,
        overflow: "hidden",
      }}
    >
      {bars.map(({ x, w, alpha, key, regionId, corrected }) => (
        <div
          key={key}
          style={{
            position: "absolute",
            top: 0,
            left: x,
            width: w,
            height: canvasHeight,
            background: corrected
              ? `rgba(255, 200, 64, ${alpha.toFixed(3)})`
              : `rgba(64, 224, 192, ${alpha.toFixed(3)})`,
            border: corrected ? "1px dashed rgba(255, 200, 64, 0.7)" : undefined,
            boxSizing: "border-box",
            cursor: interactive ? "pointer" : undefined,
          }}
          onClick={interactive && regionId ? () => onRegionClick(regionId) : undefined}
        />
      ))}
    </div>
  );
}
