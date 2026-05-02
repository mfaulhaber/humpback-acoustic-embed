import { type MotifOccurrence } from "@/api/sequenceModels";
import { labelColor } from "@/components/sequence-models/constants";
import { useOverlayContext } from "./OverlayContext";

interface MotifHighlightOverlayProps {
  occurrences: MotifOccurrence[];
  activeOccurrenceIndex: number;
  colorIndex: number;
  numLabels: number;
}

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return null;
  const v = parseInt(m[1], 16);
  return { r: (v >> 16) & 0xff, g: (v >> 8) & 0xff, b: v & 0xff };
}

function withAlpha(color: string, alpha: number): string {
  const rgb = hexToRgb(color);
  if (rgb) return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
  // Fallback for hsl(...) palette entries (>30 labels)
  const hslMatch = /^hsl\((.+)\)$/.exec(color);
  if (hslMatch) return `hsla(${hslMatch[1]}, ${alpha})`;
  return color;
}

export function MotifHighlightOverlay({
  occurrences,
  activeOccurrenceIndex,
  colorIndex,
  numLabels,
}: MotifHighlightOverlayProps) {
  const { viewStart, viewEnd, pxPerSec, canvasHeight } = useOverlayContext();

  if (occurrences.length === 0) return null;

  const baseColor = labelColor(colorIndex, Math.max(numLabels, 1));
  const inactiveFill = withAlpha(baseColor, 0.15);
  const inactiveBorder = withAlpha(baseColor, 0.4);
  const activeFill = withAlpha(baseColor, 0.35);
  const activeBorder = withAlpha(baseColor, 0.8);

  return (
    <div
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
      data-testid="mt-motif-highlight-layer"
    >
      {occurrences.map((occ, idx) => {
        const start = occ.start_timestamp;
        const end = occ.end_timestamp;
        if (end < viewStart || start > viewEnd) return null;

        const x = (start - viewStart) * pxPerSec;
        const w = Math.max(1, (end - start) * pxPerSec);
        const isActive = idx === activeOccurrenceIndex;

        return (
          <div
            key={occ.occurrence_id}
            data-testid="mt-motif-highlight-band"
            data-active={isActive ? "true" : "false"}
            data-occurrence-index={idx}
            style={{
              position: "absolute",
              left: x,
              top: 0,
              width: w,
              height: canvasHeight,
              background: isActive ? activeFill : inactiveFill,
              borderLeft: isActive
                ? `2px solid ${activeBorder}`
                : `1px solid ${inactiveBorder}`,
              pointerEvents: "none",
              zIndex: 1,
            }}
          />
        );
      })}
    </div>
  );
}
