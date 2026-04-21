import { useTimelineContext } from "../provider/useTimelineContext";
import { COLORS } from "../constants";

interface PlayheadProps {
  canvasWidth: number;
  canvasHeight: number;
}

export function Playhead({ canvasWidth, canvasHeight }: PlayheadProps) {
  const ctx = useTimelineContext();

  let leftPx: number;
  if (ctx.playbackEpoch !== null) {
    const pxPerSec = canvasWidth > 0 ? canvasWidth / ctx.viewportSpan : 1;
    leftPx = (ctx.playbackEpoch - ctx.viewStart) * pxPerSec;
  } else {
    leftPx = canvasWidth / 2;
  }

  if (leftPx < 0 || leftPx > canvasWidth) return null;

  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left: leftPx,
        top: 0,
        width: 0,
        height: canvasHeight,
        borderLeft: `1.5px solid ${COLORS.accent}`,
        zIndex: 10,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: -1,
          left: -5,
          width: 0,
          height: 0,
          borderLeft: "5px solid transparent",
          borderRight: "5px solid transparent",
          borderTop: `6px solid ${COLORS.accent}`,
        }}
      />
    </div>
  );
}
