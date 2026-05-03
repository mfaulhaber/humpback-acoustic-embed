import { type MotifOccurrence } from "@/api/sequenceModels";
import { labelColor } from "@/components/sequence-models/constants";
import type { MotifColor } from "@/lib/motifColor";
import { useOverlayContext } from "./OverlayContext";

interface MotifHighlightOverlayProps {
  occurrences: MotifOccurrence[];
  activeOccurrenceIndex: number;
  /** Used only when ``colorForMotifKey`` is not supplied (single-motif mode). */
  colorIndex: number;
  /** Used only when ``colorForMotifKey`` is not supplied (single-motif mode). */
  numLabels: number;
  /**
   * When provided, each occurrence is colored by its ``motif_key`` using
   * the supplied mapper — used by the masked-transformer page to render
   * many length-N motifs at once with distinct hues. When omitted, the
   * overlay falls back to the legacy single-color behavior driven by
   * ``colorIndex``/``numLabels``.
   */
  colorForMotifKey?: (motifKey: string) => MotifColor;
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
  colorForMotifKey,
}: MotifHighlightOverlayProps) {
  const { viewStart, viewEnd, pxPerSec, canvasHeight } = useOverlayContext();

  if (occurrences.length === 0) return null;

  const fallbackBase = labelColor(colorIndex, Math.max(numLabels, 1));
  const fallbackFill = withAlpha(fallbackBase, 0.15);
  const fallbackBorder = withAlpha(fallbackBase, 0.4);

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

        const color = colorForMotifKey?.(occ.motif_key);
        const fill = color ? color.fill : fallbackFill;
        const border = color ? color.border : fallbackBorder;

        // Active-occurrence indicator. With a per-motif color mapper we
        // keep the fill at the same uniform alpha across all rectangles
        // and indicate the active occurrence with a separate dashed
        // outline ring drawn outside the box. Without a mapper we
        // preserve the legacy two-level fill emphasis so single-motif
        // mode keeps its existing appearance.
        const fillForRender = color
          ? fill
          : isActive
            ? withAlpha(fallbackBase, 0.35)
            : fill;
        const borderForRender = color
          ? `1px solid ${border}`
          : isActive
            ? `2px solid ${withAlpha(fallbackBase, 0.8)}`
            : `1px solid ${border}`;

        return (
          <div
            key={occ.occurrence_id}
            data-testid="mt-motif-highlight-band"
            data-active={isActive ? "true" : "false"}
            data-occurrence-index={idx}
            data-motif-key={occ.motif_key}
            style={{
              position: "absolute",
              left: x,
              top: 0,
              width: w,
              height: canvasHeight,
              background: fillForRender,
              borderLeft: borderForRender,
              outline:
                color && isActive ? "2px dashed rgba(15, 23, 42, 0.85)" : "none",
              outlineOffset: color && isActive ? 1 : 0,
              pointerEvents: "none",
              zIndex: 1,
            }}
          />
        );
      })}
    </div>
  );
}
