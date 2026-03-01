import { useMemo, useCallback, useRef } from "react";
import Plot from "react-plotly.js";
import { useVisualization } from "@/hooks/queries/useClustering";
import { audioWindowUrl } from "@/api/client";

const PALETTE = [
  "#3a86ff", "#e63946", "#2a9d8f", "#e9c46a", "#264653",
  "#f4a261", "#6c5ce7", "#00b894", "#fd79a8", "#636e72",
  "#d63031", "#74b9ff", "#a29bfe", "#ffeaa7", "#55efc4",
];

interface PointCustomData {
  audioFileId: string;
  rowIndex: number;
  windowSize: number;
  filename: string;
}

interface UmapPlotProps {
  jobId: string;
}

export function UmapPlot({ jobId }: UmapPlotProps) {
  const { data: viz, isLoading } = useVisualization(jobId);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const { traces, layout } = useMemo(() => {
    if (!viz) return { traces: [] as Plotly.Data[], layout: {} as Partial<Plotly.Layout> };

    // Group by cluster label
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
          color: isNoise ? "#b2bec3" : PALETTE[label % PALETTE.length],
          size: isNoise ? 4 : 7,
          opacity: isNoise ? 0.4 : 0.8,
        },
        text: indices.map((i) => viz.category[i] ?? ""),
        hoverinfo: "text+name" as const,
        customdata: indices.map((i) => ({
          audioFileId: viz.audio_file_id[i],
          rowIndex: viz.embedding_row_index[i],
          windowSize: viz.window_size_seconds[i],
          filename: viz.audio_filename[i],
        })),
      };
    }) as unknown as Plotly.Data[];

    const layout: Partial<Plotly.Layout> = {
      xaxis: { title: { text: "UMAP 1" } },
      yaxis: { title: { text: "UMAP 2" } },
      legend: { x: 1.02, y: 1, orientation: "v" as const },
      margin: { l: 50, r: 120, t: 20, b: 50 },
      hovermode: "closest" as const,
      height: 500,
    };

    return { traces, layout };
  }, [viz]);

  const handleClick = useCallback(
    (event: Plotly.PlotMouseEvent) => {
      const point = event.points[0];
      if (!point || !point.customdata) return;
      const cd = point.customdata as unknown as PointCustomData;
      if (!cd.audioFileId) return;

      if (!audioRef.current) {
        audioRef.current = new Audio();
      }
      const start = cd.rowIndex * cd.windowSize;
      audioRef.current.src = audioWindowUrl(cd.audioFileId, start, cd.windowSize);
      audioRef.current.play().catch(() => {});
    },
    [],
  );

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading UMAP data...</p>;
  if (!viz) return <p className="text-sm text-muted-foreground">No visualization data available.</p>;

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      onClick={handleClick}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
