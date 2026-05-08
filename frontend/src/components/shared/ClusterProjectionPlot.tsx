import { useMemo } from "react";
import Plot from "react-plotly.js";

export interface ClusterProjectionPlotPoint<CustomData = unknown> {
  id: string;
  x: number;
  y: number;
  groupKey: string;
  groupLabel: string;
  color: string;
  hoverText?: string;
  markerSize?: number;
  markerOpacity?: number;
  customData?: CustomData;
}

interface ClusterProjectionPlotProps<CustomData = unknown> {
  points: ClusterProjectionPlotPoint<CustomData>[];
  xAxisTitle: string;
  yAxisTitle: string;
  height?: number;
  selectedPointId?: string | null;
  emptyMessage?: string;
  testId?: string;
  onPointClick?: (point: ClusterProjectionPlotPoint<CustomData>) => void;
}

export function ClusterProjectionPlot<CustomData = unknown>({
  points,
  xAxisTitle,
  yAxisTitle,
  height = 360,
  selectedPointId,
  emptyMessage = "No projection points are available.",
  testId,
  onPointClick,
}: ClusterProjectionPlotProps<CustomData>) {
  const { traces, layout } = useMemo(() => {
    const groups = new Map<string, ClusterProjectionPlotPoint<CustomData>[]>();
    for (const point of points) {
      if (!groups.has(point.groupKey)) groups.set(point.groupKey, []);
      groups.get(point.groupKey)!.push(point);
    }

    const data = [...groups.entries()].map(([, groupPoints]) => ({
      x: groupPoints.map((point) => point.x),
      y: groupPoints.map((point) => point.y),
      mode: "markers" as const,
      type: "scatter" as const,
      name: groupPoints[0]?.groupLabel ?? "",
      marker: {
        color: groupPoints.map((point) => point.color),
        size: groupPoints.map((point) =>
          point.id === selectedPointId ? 12 : point.markerSize ?? 7,
        ),
        opacity: groupPoints.map((point) => point.markerOpacity ?? 0.82),
        line: {
          color: groupPoints.map((point) =>
            point.id === selectedPointId ? "#0f172a" : "#ffffff",
          ),
          width: groupPoints.map((point) =>
            point.id === selectedPointId ? 2 : 0.5,
          ),
        },
      },
      text: groupPoints.map((point) => point.hoverText ?? point.id),
      hoverinfo: "text+name" as const,
      customdata: groupPoints,
    })) as unknown as Plotly.Data[];

    const plotLayout: Partial<Plotly.Layout> = {
      xaxis: { title: { text: xAxisTitle }, zeroline: false },
      yaxis: { title: { text: yAxisTitle }, zeroline: false },
      legend: { x: 1.02, y: 1, orientation: "v" as const },
      margin: { l: 50, r: 120, t: 12, b: 50 },
      hovermode: "closest" as const,
      height,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    };

    return { traces: data, layout: plotLayout };
  }, [height, points, selectedPointId, xAxisTitle, yAxisTitle]);

  if (!points.length) {
    return (
      <div
        className="rounded-md border border-dashed p-4 text-sm text-muted-foreground"
        data-testid={testId ? `${testId}-empty` : undefined}
      >
        {emptyMessage}
      </div>
    );
  }

  return (
    <div data-testid={testId}>
      <Plot
        data={traces}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
        onClick={(event) => {
          const point = event.points[0];
          if (!point?.customdata || !onPointClick) return;
          onPointClick(
            point.customdata as unknown as ClusterProjectionPlotPoint<CustomData>,
          );
        }}
        useResizeHandler
        style={{ width: "100%" }}
      />
    </div>
  );
}
