import Plot from "react-plotly.js";
import { labelColor } from "./constants";

export interface TokenRunLengthHistogramsProps {
  /** Map keyed by token-index strings to per-token run-length samples. */
  runLengths: Record<string, number[]>;
  k: number;
  /** Optional cap for the number of histograms shown (sorted by count). */
  maxTokens?: number;
}

/**
 * Per-token run-length histograms for masked-transformer detail pages.
 *
 * Renders one tiny Plotly histogram per token, mirroring the existing
 * HMM dwell-histogram grid layout but relabeled (token index instead of
 * HMM state index, "run length" instead of "dwell").
 */
export function TokenRunLengthHistograms({
  runLengths,
  k,
  maxTokens = 30,
}: TokenRunLengthHistogramsProps) {
  const entries = Object.entries(runLengths)
    .map(([key, samples]) => ({ key, samples }))
    .filter((e) => e.samples.length > 0)
    .sort((a, b) => b.samples.length - a.samples.length)
    .slice(0, maxTokens);

  if (entries.length === 0) {
    return (
      <div
        className="border rounded-md p-3 text-xs text-muted-foreground"
        data-testid="token-run-length-histograms-empty"
      >
        No run-length samples available.
      </div>
    );
  }

  return (
    <div
      className="border rounded-md p-2"
      data-testid="token-run-length-histograms"
    >
      <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">
        Token run lengths (k={k})
      </div>
      <div
        className="grid gap-2"
        style={{
          gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
        }}
      >
        {entries.map((entry) => {
          const tokenIdx = Number.parseInt(entry.key, 10);
          const color = labelColor(
            Number.isFinite(tokenIdx) ? tokenIdx : 0,
            k,
          );
          return (
            <div
              key={entry.key}
              className="border rounded-md p-1"
              data-testid={`token-run-length-${entry.key}`}
            >
              <div className="text-[10px] text-muted-foreground px-1">
                Token {entry.key} · n={entry.samples.length}
              </div>
              <Plot
                data={[
                  {
                    x: entry.samples,
                    type: "histogram",
                    marker: { color },
                  },
                ]}
                layout={{
                  autosize: true,
                  height: 110,
                  margin: { l: 22, r: 4, t: 4, b: 22 },
                  xaxis: { title: { text: "" }, tickfont: { size: 8 } },
                  yaxis: { title: { text: "" }, tickfont: { size: 8 } },
                  showlegend: false,
                  bargap: 0.05,
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
                useResizeHandler
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
