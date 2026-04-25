import { useMemo } from "react";
import Plot from "react-plotly.js";
import { useVocClusteringVisualization } from "@/hooks/queries/useVocalization";

const PALETTE = [
  "#3a86ff", "#e63946", "#2a9d8f", "#e9c46a", "#264653",
  "#f4a261", "#6c5ce7", "#00b894", "#fd79a8", "#636e72",
  "#d63031", "#74b9ff", "#a29bfe", "#ffeaa7", "#55efc4",
];

interface VocalizationUmapPlotProps {
  jobId: string;
}

export function VocalizationUmapPlot({ jobId }: VocalizationUmapPlotProps) {
  const { data: viz, isLoading } = useVocClusteringVisualization(jobId);

  const { traces, layout } = useMemo(() => {
    if (!viz) return { traces: [] as Plotly.Data[], layout: {} as Partial<Plotly.Layout> };

    // Group by category (vocalization label) for coloring
    const categorySet = new Set(viz.category);
    const categoryList = [...categorySet].sort();
    const catColorMap = new Map<string, string>();
    categoryList.forEach((cat, i) => {
      catColorMap.set(cat, cat === "unlabeled" ? "#b2bec3" : PALETTE[i % PALETTE.length]);
    });

    // Group by cluster label for trace structure
    const groups = new Map<number, number[]>();
    for (let i = 0; i < viz.cluster_label.length; i++) {
      const label = viz.cluster_label[i];
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label)!.push(i);
    }

    const sortedLabels = [...groups.keys()].sort((a, b) => a - b);

    const traces = sortedLabels.map((label) => {
      const indices = groups.get(label)!;
      const isNoise = label === -1;
      return {
        x: indices.map((i) => viz.x[i]),
        y: indices.map((i) => viz.y[i]),
        mode: "markers" as const,
        type: "scatter" as const,
        name: isNoise ? "Noise" : `Cluster ${label}`,
        marker: {
          color: indices.map((i) => catColorMap.get(viz.category[i]) ?? "#b2bec3"),
          size: isNoise ? 4 : 7,
          opacity: isNoise ? 0.4 : 0.8,
        },
        text: indices.map((i) => viz.category[i] ?? ""),
        hoverinfo: "text+name" as const,
      };
    }) as unknown as Plotly.Data[];

    // Add a legend trace per category
    const legendTraces = categoryList.map((cat) => ({
      x: [null],
      y: [null],
      mode: "markers" as const,
      type: "scatter" as const,
      name: cat,
      marker: { color: catColorMap.get(cat), size: 8 },
      showlegend: true,
      hoverinfo: "skip" as const,
    })) as unknown as Plotly.Data[];

    // Hide cluster trace legends to avoid duplication
    traces.forEach((t) => {
      (t as Record<string, unknown>).showlegend = false;
    });

    const layout: Partial<Plotly.Layout> = {
      xaxis: { title: { text: "UMAP 1" } },
      yaxis: { title: { text: "UMAP 2" } },
      legend: { x: 1.02, y: 1, orientation: "v" as const },
      margin: { l: 50, r: 120, t: 20, b: 50 },
      hovermode: "closest" as const,
      height: 500,
    };

    return { traces: [...traces, ...legendTraces], layout };
  }, [viz]);

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading UMAP data...</p>;
  if (!viz) return <p className="text-sm text-muted-foreground">No visualization data available.</p>;

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
