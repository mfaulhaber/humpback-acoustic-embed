// Label palette shared by HMM state bars, motif visualizations, and the
// masked-transformer token strip (ADR-061). Renamed from ``STATE_COLORS``
// so the palette name reflects its source-agnostic role.
export const LABEL_COLORS = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
  "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
];

const HSL_PALETTE_THRESHOLD = 30;

/**
 * Pick a color for label index ``idx`` out of ``total`` labels.
 *
 * For ``total`` ≤ ``HSL_PALETTE_THRESHOLD`` we wrap the categorical palette
 * (``palette[idx % palette.length]``). For larger ``total`` the categorical
 * palette becomes ambiguous, so we generate a deterministic HSL ramp
 * spanning the full hue circle.
 */
export function labelColor(idx: number, total: number): string {
  const palette = LABEL_COLORS;
  if (total <= HSL_PALETTE_THRESHOLD) {
    return palette[idx % palette.length];
  }
  const hue = (idx * 360) / Math.max(1, total);
  return `hsl(${hue.toFixed(1)}, 65%, 50%)`;
}

// Backwards-compatible alias kept temporarily so any stale imports fail
// loudly during tests rather than silently picking the wrong palette.
// New code MUST use ``LABEL_COLORS``; this alias may be removed.
export const STATE_COLORS = LABEL_COLORS;
