import { useMemo } from "react";
import Plot from "react-plotly.js";
import { useMetrics } from "@/hooks/queries/useClustering";

interface LabelDotPlotProps {
  jobId: string;
}

export function LabelDotPlot({ jobId }: LabelDotPlotProps) {
  const { data: metrics, isLoading } = useMetrics(jobId);

  const { trace, layout } = useMemo(() => {
    const cm = metrics?.confusion_matrix as
      | Record<string, Record<string, number>>
      | undefined;
    if (!cm) return { trace: null, layout: null };

    const categories = Object.keys(cm).sort();
    const clusterSet = new Set<string>();
    for (const counts of Object.values(cm)) {
      for (const cl of Object.keys(counts)) clusterSet.add(cl);
    }
    const clusterLabels = [...clusterSet].sort(
      (a, b) => Number(a) - Number(b),
    );

    // Build parallel arrays for scatter
    const xs: string[] = [];
    const ys: string[] = [];
    const sizes: number[] = [];
    const counts: number[] = [];

    for (const cat of categories) {
      for (const cl of clusterLabels) {
        const count = cm[cat]?.[cl] ?? 0;
        if (count === 0) continue;
        xs.push(cat);
        ys.push(cl);
        sizes.push(count);
        counts.push(count);
      }
    }

    if (sizes.length === 0) return { trace: null, layout: null };

    // Compute log-proportions per category, then z-score for color
    const catTotals = new Map<string, number>();
    for (let i = 0; i < xs.length; i++) {
      catTotals.set(xs[i], (catTotals.get(xs[i]) ?? 0) + counts[i]);
    }

    const logProps = counts.map((c, i) =>
      Math.log1p(c / (catTotals.get(xs[i]) ?? 1)),
    );

    const mean = logProps.reduce((a, b) => a + b, 0) / logProps.length;
    const std =
      Math.sqrt(
        logProps.reduce((a, b) => a + (b - mean) ** 2, 0) / logProps.length,
      ) || 1;
    const zScores = logProps.map((v) => (v - mean) / std);

    // Scale dot sizes: largest dot ~40px, smallest ~6px
    const maxCount = Math.max(...sizes);
    const scaledSizes = sizes.map((s) => 6 + (s / maxCount) * 34);

    const trace: Plotly.Data = {
      x: xs,
      y: ys,
      mode: "markers",
      type: "scatter",
      marker: {
        size: scaledSizes,
        color: zScores,
        colorscale: [
          [0, "#f2f0f7"],
          [0.5, "#9e9ac8"],
          [1, "#3f007d"],
        ],
        showscale: true,
        colorbar: { title: { text: "Z-score" }, thickness: 12, len: 0.6 },
      },
      text: counts.map(
        (c, i) => `Category: ${xs[i]}<br>Cluster: ${ys[i]}<br>Count: ${c}`,
      ),
      hoverinfo: "text",
    };

    const layout: Partial<Plotly.Layout> = {
      xaxis: {
        title: { text: "Category" },
        type: "category",
        tickangle: -45,
      },
      yaxis: {
        title: { text: "Cluster" },
        type: "category",
        autorange: "reversed",
      },
      margin: { l: 60, r: 40, t: 20, b: 120 },
      height: Math.max(300, clusterLabels.length * 28 + 160),
      hovermode: "closest",
    };

    return { trace, layout };
  }, [metrics]);

  if (isLoading)
    return (
      <p className="text-sm text-muted-foreground">Loading label data...</p>
    );
  if (!trace || !layout)
    return (
      <p className="text-sm text-muted-foreground">
        No category label data available.
      </p>
    );

  return (
    <Plot
      data={[trace]}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
