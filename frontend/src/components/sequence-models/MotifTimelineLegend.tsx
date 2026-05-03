import type { ReactNode } from "react";
import { LABEL_COLORS, labelColor } from "./constants";

export interface MotifTimelineLegendProps {
  selectedMotifKey: string | null;
  selectedStates: number[];
  numLabels: number;
  occurrencesTotal: number;
  activeOccurrenceIndex: number;
  onPrev: () => void;
  onNext: () => void;
  /**
   * Optional palette override. Defaults to ``LABEL_COLORS``. The exact
   * lookup is ``palette[state % palette.length]`` so callers can supply
   * shorter or longer palettes; for ``numLabels > 30`` ``labelColor`` is
   * still preferred to match the rest of the page's coloring.
   */
  palette?: readonly string[];
  /**
   * Optional content rendered to the right of the prev/next controls.
   * Used by the masked-transformer detail page to host the Token Count
   * selector. When omitted (e.g., on the HMM detail page) the legend
   * renders identically to its prior shape.
   */
  tokenSelector?: ReactNode;
  /**
   * Optional motif-bounded Play handler. When provided, a Play button is
   * rendered after Next; clicking it plays the active occurrence's
   * bounded span. Used in byLength mode where the panel-level Play
   * buttons are not visible.
   */
  onPlay?: () => void;
}

const HSL_PALETTE_THRESHOLD = 30;

function colorForState(
  state: number,
  numLabels: number,
  palette?: readonly string[],
): string {
  if (palette) {
    return palette[state % palette.length];
  }
  if (numLabels <= HSL_PALETTE_THRESHOLD) {
    return LABEL_COLORS[state % LABEL_COLORS.length];
  }
  return labelColor(state, numLabels);
}

export function MotifTimelineLegend({
  selectedMotifKey,
  selectedStates,
  numLabels,
  occurrencesTotal,
  activeOccurrenceIndex,
  onPrev,
  onNext,
  palette,
  tokenSelector,
  onPlay,
}: MotifTimelineLegendProps) {
  const hasSingleMotif = selectedMotifKey != null;
  // Render the legend whenever there's something to show: either a
  // single-motif selection, navigable occurrences (byLength mode), or a
  // tokenSelector slot supplied by the caller.
  const hasNavigation = occurrencesTotal > 0;
  if (!hasSingleMotif && !hasNavigation && !tokenSelector) return null;

  const prevDisabled = !hasNavigation || activeOccurrenceIndex <= 0;
  const nextDisabled =
    !hasNavigation || activeOccurrenceIndex >= occurrencesTotal - 1;
  const counterText = hasNavigation
    ? `${activeOccurrenceIndex + 1} / ${occurrencesTotal}`
    : `0 in view`;

  return (
    <div
      className="flex flex-wrap items-center gap-2 text-xs"
      data-testid="motif-timeline-legend"
    >
      {hasSingleMotif && (
        <>
          <span className="text-muted-foreground">Selected motif:</span>
          <div className="flex items-center gap-1">
            {selectedStates.map((state, idx) => (
              <span key={`${state}-${idx}`} className="flex items-center gap-1">
                <span
                  className="inline-block h-3 w-3 rounded-sm border border-border"
                  style={{ backgroundColor: colorForState(state, numLabels, palette) }}
                  data-testid={`motif-timeline-legend-swatch-${idx}`}
                  aria-label={`state ${state}`}
                />
                {idx < selectedStates.length - 1 && (
                  <span className="text-muted-foreground">→</span>
                )}
              </span>
            ))}
          </div>
        </>
      )}
      <span
        className="ml-2 tabular-nums text-muted-foreground"
        data-testid="motif-timeline-legend-counter"
      >
        {counterText}
      </span>
      <button
        type="button"
        onClick={onPrev}
        disabled={prevDisabled}
        data-testid="motif-timeline-legend-prev"
        className="rounded border px-2 py-0.5 text-xs hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
      >
        ← prev
      </button>
      <button
        type="button"
        onClick={onNext}
        disabled={nextDisabled}
        data-testid="motif-timeline-legend-next"
        className="rounded border px-2 py-0.5 text-xs hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
      >
        next →
      </button>
      {onPlay && (
        <button
          type="button"
          onClick={onPlay}
          disabled={!hasNavigation}
          data-testid="motif-timeline-legend-play"
          className="rounded border px-2 py-0.5 text-xs hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
        >
          ▶ play
        </button>
      )}
      {tokenSelector && (
        <div
          className="ml-auto"
          data-testid="motif-timeline-legend-token-selector"
        >
          {tokenSelector}
        </div>
      )}
    </div>
  );
}
