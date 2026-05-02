import { useSearchParams } from "react-router-dom";

export interface KPickerProps {
  kValues: number[];
  /**
   * Optional override for the URL search-param name. Defaults to ``k``.
   * Tests can pass their own to avoid clashing with other search params.
   */
  paramName?: string;
}

/**
 * URL-synced k-picker for masked-transformer detail pages.
 *
 * Persists the chosen ``k`` value in ``?k=N``; defaults to the first entry
 * of ``kValues`` when no value is set or the URL value isn't in the list.
 * Switching the picker updates the URL without a page navigation, so
 * TanStack Query keys keyed on ``(jobId, k)`` re-fetch automatically.
 */
export function KPicker({ kValues, paramName = "k" }: KPickerProps) {
  const [searchParams, setSearchParams] = useSearchParams();

  const fallback = kValues[0];
  const raw = searchParams.get(paramName);
  const parsed = raw != null ? Number.parseInt(raw, 10) : NaN;
  const current = kValues.includes(parsed) ? parsed : fallback;

  if (!kValues.length) {
    return (
      <div className="text-xs text-muted-foreground" data-testid="k-picker">
        No k-values configured
      </div>
    );
  }

  return (
    <div
      className="flex items-center gap-1 text-sm"
      data-testid="k-picker"
      role="tablist"
      aria-label="Tokenization k-value"
    >
      <span className="text-xs text-muted-foreground mr-1">k=</span>
      {kValues.map((k) => {
        const active = k === current;
        return (
          <button
            key={k}
            type="button"
            role="tab"
            aria-selected={active}
            data-testid={`k-picker-tab-${k}`}
            onClick={() => {
              setSearchParams(
                (prev) => {
                  const next = new URLSearchParams(prev);
                  next.set(paramName, String(k));
                  return next;
                },
                { replace: true },
              );
            }}
            className={
              active
                ? "rounded-md border bg-primary text-primary-foreground px-2 py-1 text-xs"
                : "rounded-md border px-2 py-1 text-xs hover:bg-accent"
            }
          >
            {k}
          </button>
        );
      })}
    </div>
  );
}

/**
 * Read the currently selected ``k`` (URL-synced) outside of the
 * ``KPicker`` component so siblings (charts, hooks) can react.
 */
export function useSelectedK(
  kValues: number[],
  paramName = "k",
): number | null {
  const [searchParams] = useSearchParams();
  if (!kValues.length) return null;
  const raw = searchParams.get(paramName);
  const parsed = raw != null ? Number.parseInt(raw, 10) : NaN;
  return kValues.includes(parsed) ? parsed : kValues[0];
}
