import { COLORS } from "../constants";

interface PlayheadProps {
  canvasWidth: number;
  canvasHeight: number;
}

export function Playhead({ canvasWidth, canvasHeight }: PlayheadProps) {
  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left: canvasWidth / 2,
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
