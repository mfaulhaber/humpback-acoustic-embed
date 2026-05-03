export type MotifTokenCount = 2 | 3 | 4;

const TOKEN_COUNT_VALUES: readonly MotifTokenCount[] = [2, 3, 4] as const;

export interface MotifTokenCountSelectorProps {
  value: MotifTokenCount | null;
  onChange: (next: MotifTokenCount | null) => void;
  /**
   * Lengths for which at least one motif exists in the current job.
   * Buttons whose value is not in this set are rendered disabled with
   * a "no length-N motifs" tooltip.
   */
  availableLengths: ReadonlySet<number>;
  /** While the underlying ``useMotifs`` query is pending. */
  isMotifsLoading: boolean;
}

export function MotifTokenCountSelector({
  value,
  onChange,
  availableLengths,
  isMotifsLoading,
}: MotifTokenCountSelectorProps) {
  return (
    <div
      className="flex items-center gap-1"
      data-testid="motif-token-count-selector"
      role="group"
      aria-label="Token Count"
    >
      <span className="text-muted-foreground">Token Count:</span>
      {TOKEN_COUNT_VALUES.map((n) => {
        const isActive = value === n;
        const hasAny = availableLengths.has(n);
        const disabled = isMotifsLoading || !hasAny;
        const title = isMotifsLoading
          ? "Loading motifs…"
          : hasAny
            ? undefined
            : `No length-${n} motifs`;
        return (
          <button
            key={n}
            type="button"
            disabled={disabled}
            onClick={() => onChange(isActive ? null : n)}
            title={title}
            data-testid={`motif-token-count-${n}`}
            data-active={isActive ? "true" : "false"}
            aria-pressed={isActive}
            className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors border disabled:cursor-not-allowed disabled:opacity-40 ${
              isActive
                ? "bg-primary/10 border-primary/30 text-primary"
                : "bg-muted border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {n}
          </button>
        );
      })}
      {isMotifsLoading && (
        <span
          className="ml-1 text-[10px] text-muted-foreground"
          data-testid="motif-token-count-spinner"
        >
          …
        </span>
      )}
    </div>
  );
}
