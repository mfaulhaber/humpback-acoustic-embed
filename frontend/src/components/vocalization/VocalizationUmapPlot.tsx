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

    // Group by vocalization label for coloring
    const groups = new Map<string, number[]>();
    for (let i = 0; i < viz.category.length; i++) {
      const cat = viz.category[i];
      if (!groups.has(cat)) groups.set(cat, []);
      groups.get(cat)!.push(i);
    }

    const sortedCategories = [...groups.keys()].sort((a, b) => {
      if (a === "unlabeled") return 1;
      if (b === "unlabeled") return -1;
      return a.localeCompare(b);
    });

    const traces = sortedCategories.map((cat, catIdx) => {
      const indices = groups.get(cat)!;
      const isUnlabeled = cat === "unlabeled";
      return {
        x: indices.map((i) => viz.x[i]),
        y: indices.map((i) => viz.y[i]),
        mode: "markers" as const,
        type: "scatter" as const,
        name: cat,
        marker: {
          color: isUnlabeled ? "#b2bec3" : PALETTE[catIdx % PALETTE.length],
          size: isUnlabeled ? 4 : 7,
          opacity: isUnlabeled ? 0.4 : 0.8,
        },
        text: indices.map((i) => `${cat} (cluster ${viz.cluster_label[i]})`),
        hoverinfo: "text" as const,
      };
    }) as unknown as Plotly.Data[];

    const layout: Partial<Plotly.Layout> = {
      xaxis: { title: { text: "UMAP 1" } },
      yaxis: { title: { text: "UMAP 2" } },
      legend: { x: 1.02, y: 1, orientation: "v" as const },
      margin: { l: 50, r: 150, t: 20, b: 50 },
      hovermode: "closest" as const,
      height: 500,
    };

    return { traces, layout };
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
