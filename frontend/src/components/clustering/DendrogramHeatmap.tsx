import { useMemo } from "react";
import Plot from "react-plotly.js";
import { useDendrogram } from "@/hooks/queries/useClustering";
import type { DendrogramData } from "@/api/types";

interface DendrogramHeatmapProps {
  jobId: string;
}

function buildDendrogramTraces(
  data: DendrogramData,
): { traces: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const { categories, cluster_labels, values, raw_counts, row_dendrogram, col_dendrogram } = data;

  const nRows = cluster_labels.length;
  const nCols = categories.length;

  // -- Column dendrogram traces (top) --
  const colDendroTraces: Plotly.Data[] = col_dendrogram.icoord.map((ic, idx) => ({
    x: ic,
    y: col_dendrogram.dcoord[idx],
    type: "scatter" as const,
    mode: "lines" as const,
    line: { color: "#64748b", width: 1 },
    xaxis: "x" as const,
    yaxis: "y2" as const,
    hoverinfo: "skip" as const,
    showlegend: false,
  }));

  // -- Row dendrogram traces (left) — rotated: swap x/y and flip x direction --
  const rowDendroTraces: Plotly.Data[] = row_dendrogram.icoord.map((ic, idx) => ({
    x: row_dendrogram.dcoord[idx].map((d) => -d),
    y: ic,
    type: "scatter" as const,
    mode: "lines" as const,
    line: { color: "#64748b", width: 1 },
    xaxis: "x3" as const,
    yaxis: "y3" as const,
    hoverinfo: "skip" as const,
    showlegend: false,
  }));

  // -- Heatmap customdata for hover (raw counts per cell) --
  const customdata = raw_counts as unknown as Plotly.Datum[][];

  // -- Heatmap trace --
  const heatmapTrace: Plotly.Data = {
    z: values,
    x: categories,
    y: cluster_labels,
    type: "heatmap" as const,
    colorscale: [
      [0, "#f0f4ff"],
      [0.2, "#c7d7fe"],
      [0.4, "#818cf8"],
      [0.6, "#6366f1"],
      [0.8, "#4338ca"],
      [1, "#312e81"],
    ],
    xaxis: "x4" as const,
    yaxis: "y4" as const,
    customdata,
    hovertemplate:
      "Category: %{x}<br>Cluster: %{y}<br>Count: %{customdata}<br>Proportion: %{z:.3f}<extra></extra>",
    showscale: true,
    colorbar: {
      title: { text: "Proportion" },
      thickness: 12,
      len: 0.55,
      y: 0.25,
      yanchor: "middle" as const,
    },
    zmin: 0,
    zmax: 1,
  };

  // Compute dendrogram axis ranges
  const colIAll = col_dendrogram.icoord.flat();
  const colDAll = col_dendrogram.dcoord.flat();
  const rowIAll = row_dendrogram.icoord.flat();
  const rowDAll = row_dendrogram.dcoord.flat();

  const colXRange = [Math.min(...colIAll) - 5, Math.max(...colIAll) + 5];
  const colYRange = [0, Math.max(...colDAll) * 1.05];
  const rowYRange = [Math.min(...rowIAll) - 5, Math.max(...rowIAll) + 5];
  const rowXRange = [-Math.max(...rowDAll) * 1.05, 0];

  // Layout with manual domain positioning
  // Left 15% = row dendrogram, right 85% = heatmap + col dendrogram
  // Top 18% = col dendrogram, bottom 82% = heatmap + row dendrogram
  const dendroLeft = 0;
  const dendroRight = 0.13;
  const heatLeft = 0.15;
  const heatRight = 0.92;
  const dendroTop = 1;
  const dendroBottom = 0.82;
  const heatTop = 0.8;
  const heatBottom = 0;

  const heatHeight = Math.max(350, nRows * 28 + 200);

  const layout: Partial<Plotly.Layout> = {
    // Col dendrogram x-axis (top)
    xaxis: {
      domain: [heatLeft, heatRight],
      range: colXRange,
      showticklabels: false,
      showgrid: false,
      zeroline: false,
      showline: false,
    },
    // Col dendrogram y-axis (top)
    yaxis2: {
      domain: [dendroBottom, dendroTop],
      range: colYRange,
      showticklabels: false,
      showgrid: false,
      zeroline: false,
      showline: false,
    },
    // Row dendrogram x-axis (left)
    xaxis3: {
      domain: [dendroLeft, dendroRight],
      range: rowXRange,
      showticklabels: false,
      showgrid: false,
      zeroline: false,
      showline: false,
    },
    // Row dendrogram y-axis (left)
    yaxis3: {
      domain: [heatBottom, heatTop],
      range: rowYRange,
      showticklabels: false,
      showgrid: false,
      zeroline: false,
      showline: false,
    },
    // Heatmap x-axis (bottom)
    xaxis4: {
      domain: [heatLeft, heatRight],
      tickangle: -45,
    },
    // Heatmap y-axis (left side of heatmap)
    yaxis4: {
      domain: [heatBottom, heatTop],
      autorange: "reversed" as const,
    },
    height: heatHeight,
    margin: { l: 100, r: 60, t: 30, b: Math.min(150, nCols * 10 + 60) },
    hovermode: "closest" as const,
    showlegend: false,
  };

  return {
    traces: [...colDendroTraces, ...rowDendroTraces, heatmapTrace],
    layout,
  };
}

export function DendrogramHeatmap({ jobId }: DendrogramHeatmapProps) {
  const { data, isLoading, isError } = useDendrogram(jobId);

  const plotData = useMemo(() => {
    if (!data) return null;
    return buildDendrogramTraces(data);
  }, [data]);

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading dendrogram data...</p>;
  }
  if (isError || !plotData) {
    return (
      <p className="text-sm text-muted-foreground">
        Dendrogram not available (need at least 2 clusters and 2 categories).
      </p>
    );
  }

  return (
    <Plot
      data={plotData.traces}
      layout={plotData.layout}
      config={{ responsive: true, displayModeBar: false }}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
