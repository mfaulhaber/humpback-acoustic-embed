import { useMemo, useCallback } from "react";
import Plot from "react-plotly.js";
import type { EmbeddingSimilarity } from "@/api/types";

interface SimilarityMatrixProps {
  data: EmbeddingSimilarity;
  currentWindow: number;
  onWindowClick: (windowIndex: number) => void;
}

export function SimilarityMatrix({ data, currentWindow, onWindowClick }: SimilarityMatrixProps) {
  const labels = useMemo(
    () => Array.from({ length: data.num_windows }, (_, i) => `W${i}`),
    [data.num_windows],
  );

  const crosshairShapes = useMemo((): Partial<Plotly.Shape>[] => {
    if (currentWindow < 0 || currentWindow >= data.num_windows) return [];
    return [
      {
        type: "line",
        x0: -0.5,
        x1: data.num_windows - 0.5,
        y0: currentWindow,
        y1: currentWindow,
        line: { color: "red", width: 1, dash: "dot" },
      },
      {
        type: "line",
        x0: currentWindow,
        x1: currentWindow,
        y0: -0.5,
        y1: data.num_windows - 0.5,
        line: { color: "red", width: 1, dash: "dot" },
      },
    ];
  }, [currentWindow, data.num_windows]);

  const plotData = useMemo((): Plotly.Data[] => {
    return [
      {
        z: data.similarity_matrix,
        type: "heatmap" as const,
        colorscale: "RdBu" as unknown as Plotly.ColorScale,
        zmin: -1,
        zmax: 1,
        x: labels,
        y: labels,
        colorbar: { title: { text: "Cosine Sim" }, thickness: 12 },
      },
    ];
  }, [data.similarity_matrix, labels]);

  const layout = useMemo(
    (): Partial<Plotly.Layout> => ({
      title: { text: "Pairwise Cosine Similarity", font: { size: 13 } },
      yaxis: { autorange: "reversed" },
      margin: { t: 30, b: 50, l: 50, r: 20 },
      height: 400,
      shapes: crosshairShapes as Plotly.Layout["shapes"],
    }),
    [crosshairShapes],
  );

  const handleClick = useCallback(
    (event: Plotly.PlotMouseEvent) => {
      const point = event.points[0];
      if (point && typeof point.y === "number") {
        onWindowClick(point.y);
      } else if (point && typeof point.pointIndex === "object") {
        // Heatmap returns [row, col]
        const idx = (point.pointIndex as unknown as number[])[0];
        if (typeof idx === "number") onWindowClick(idx);
      }
    },
    [onWindowClick],
  );

  return (
    <Plot
      data={plotData}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      onClick={handleClick}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
