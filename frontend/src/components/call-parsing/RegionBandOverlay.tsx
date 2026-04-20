import type { Region } from "@/api/types";
import { useOverlayContext } from "./RegionSpectrogramViewer";

interface RegionBandOverlayProps {
  regions: Region[];
  activeRegionId: string;
  onSelectRegion: (regionId: string) => void;
}

export function RegionBandOverlay({
  regions,
  activeRegionId,
  onSelectRegion,
}: RegionBandOverlayProps) {
  const { viewStart, viewEnd, pxPerSec, canvasHeight } = useOverlayContext();

  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
      {regions.map((r, idx) => {
        const bandStart = r.padded_start_sec;
        const bandEnd = r.padded_end_sec;

        // Skip bands fully outside viewport
        if (bandEnd < viewStart || bandStart > viewEnd) return null;

        const x = (bandStart - viewStart) * pxPerSec;
        const w = (bandEnd - bandStart) * pxPerSec;
        const isActive = r.region_id === activeRegionId;

        return (
          <div
            key={r.region_id}
            style={{
              position: "absolute",
              left: x,
              top: 0,
              width: w,
              height: canvasHeight,
              background: isActive
                ? "rgba(59, 130, 246, 0.15)"
                : "rgba(148, 163, 184, 0.1)",
              borderLeft: isActive
                ? "2px solid rgba(59, 130, 246, 0.6)"
                : "1px solid rgba(148, 163, 184, 0.3)",
              borderRight: isActive
                ? "2px solid rgba(59, 130, 246, 0.6)"
                : "1px solid rgba(148, 163, 184, 0.3)",
              pointerEvents: isActive ? "none" : "auto",
              cursor: isActive ? "default" : "pointer",
              zIndex: 1,
            }}
            onClick={
              isActive
                ? undefined
                : (e) => {
                    e.stopPropagation();
                    onSelectRegion(r.region_id);
                  }
            }
          >
            <span
              style={{
                position: "absolute",
                top: 2,
                left: 4,
                fontSize: 9,
                color: isActive
                  ? "rgba(59, 130, 246, 0.9)"
                  : "rgba(148, 163, 184, 0.8)",
                background: "rgba(0, 0, 0, 0.6)",
                padding: "1px 3px",
                borderRadius: 2,
                whiteSpace: "nowrap",
                pointerEvents: "none",
              }}
            >
              R{idx + 1}
            </span>
          </div>
        );
      })}
    </div>
  );
}
