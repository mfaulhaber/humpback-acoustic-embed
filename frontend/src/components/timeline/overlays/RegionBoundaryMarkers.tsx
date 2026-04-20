import { useOverlayContext } from "./OverlayContext";

interface RegionBoundaryMarkersProps {
  startEpoch: number;
  endEpoch: number;
}

export function RegionBoundaryMarkers({ startEpoch, endEpoch }: RegionBoundaryMarkersProps) {
  const { epochToX, canvasWidth, canvasHeight } = useOverlayContext();

  const startX = epochToX(startEpoch);
  const endX = epochToX(endEpoch);

  return (
    <div
      style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden" }}
      data-testid="region-boundary-markers"
    >
      {/* Dimmed area before region */}
      {startX > 0 && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: Math.max(0, startX),
            height: canvasHeight,
            background: "rgba(0, 0, 0, 0.4)",
          }}
        />
      )}

      {/* Dimmed area after region */}
      {endX < canvasWidth && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: Math.max(0, endX),
            width: Math.max(0, canvasWidth - endX),
            height: canvasHeight,
            background: "rgba(0, 0, 0, 0.4)",
          }}
        />
      )}

      {/* Start boundary line */}
      {startX >= 0 && startX <= canvasWidth && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: startX,
            width: 0,
            height: canvasHeight,
            borderLeft: "1.5px dashed rgba(59, 130, 246, 0.7)",
          }}
        />
      )}

      {/* End boundary line */}
      {endX >= 0 && endX <= canvasWidth && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: endX,
            width: 0,
            height: canvasHeight,
            borderLeft: "1.5px dashed rgba(59, 130, 246, 0.7)",
          }}
        />
      )}
    </div>
  );
}
