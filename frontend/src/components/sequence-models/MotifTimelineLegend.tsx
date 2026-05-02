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
}: MotifTimelineLegendProps) {
  if (selectedMotifKey == null) return null;

  const prevDisabled = activeOccurrenceIndex <= 0;
  const nextDisabled = activeOccurrenceIndex >= occurrencesTotal - 1;
  const counterText = `${activeOccurrenceIndex + 1} / ${occurrencesTotal}`;

  return (
    <div
      className="flex flex-wrap items-center gap-2 text-xs"
      data-testid="motif-timeline-legend"
    >
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
    </div>
  );
}
