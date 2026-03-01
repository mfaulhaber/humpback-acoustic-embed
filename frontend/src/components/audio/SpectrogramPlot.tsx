import { useMemo } from "react";
import Plot from "react-plotly.js";
import type { SpectrogramData } from "@/api/types";

const COLORSCALE: [number, string][] = [
  [0.0, "#0d0829"],
  [0.1, "#1b0c41"],
  [0.2, "#330a5f"],
  [0.3, "#52076c"],
  [0.4, "#721a6e"],
  [0.5, "#932567"],
  [0.6, "#b6365a"],
  [0.7, "#d44d46"],
  [0.8, "#e76e33"],
  [0.9, "#f49622"],
  [1.0, "#fcffa4"],
];

interface SpectrogramPlotProps {
  data: SpectrogramData;
}

export function SpectrogramPlot({ data }: SpectrogramPlotProps) {
  const plotData = useMemo(() => {
    const trace: Plotly.Data = {
      z: data.data,
      type: "heatmap" as const,
      zsmooth: "best" as const,
      colorscale: COLORSCALE,
      zmin: data.min_db,
      zmax: 0,
      colorbar: {
        thickness: 12,
        ticksuffix: " dB",
        tickmode: "linear" as const,
        dtick: 10,
      },
      x: data.x_axis_seconds.length > 0 ? data.x_axis_seconds : undefined,
      y: data.y_axis_hz.length > 0 ? data.y_axis_hz : undefined,
    };
    return [trace];
  }, [data]);

  const layout = useMemo(
    () => ({
      title: { text: "Mel Spectrogram", font: { size: 13 } },
      xaxis: { title: { text: "Time" } },
      yaxis: {
        title: { text: "Hz" },
        type: data.y_axis_hz.length > 0 ? ("log" as const) : undefined,
      },
      margin: { t: 30, b: 40, l: 50, r: 20 },
      height: 300,
    }),
    [data.y_axis_hz.length],
  );

  return (
    <Plot
      data={plotData}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
