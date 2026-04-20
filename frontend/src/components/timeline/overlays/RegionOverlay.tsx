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
  onRegionClick?: (regionId: string) => void;
}

export function RegionOverlay({ regions, jobStart, visible, onRegionClick }: RegionOverlayProps) {
  const { epochToX, canvasWidth, canvasHeight } = useOverlayContext();

  if (!visible || canvasWidth <= 0 || canvasHeight <= 0) return null;

  const interactive = !!onRegionClick;

  const bars: { x: number; w: number; alpha: number; key: string; regionId?: string }[] = [];
  for (const r of regions) {
    const startEpoch = jobStart + r.padded_start_sec;
    const endEpoch = jobStart + r.padded_end_sec;
    const x = epochToX(startEpoch);
    const w = Math.max(2, epochToX(endEpoch) - x);
    if (x + w < 0 || x > canvasWidth) continue;
    const alpha = 0.1 + r.max_score * 0.25;
    bars.push({ x, w, alpha, key: `${r.padded_start_sec}:${r.padded_end_sec}`, regionId: r.region_id });
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
      {bars.map(({ x, w, alpha, key, regionId }) => (
        <div
          key={key}
          style={{
            position: "absolute",
            top: 0,
            left: x,
            width: w,
            height: canvasHeight,
            background: `rgba(64, 224, 192, ${alpha.toFixed(3)})`,
            cursor: interactive ? "pointer" : undefined,
          }}
          onClick={interactive && regionId ? () => onRegionClick(regionId) : undefined}
        />
      ))}
    </div>
  );
}
