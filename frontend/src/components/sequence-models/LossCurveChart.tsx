import Plot from "react-plotly.js";
import type { LossCurveResponse } from "@/api/sequenceModels";

export interface LossCurveChartProps {
  data: LossCurveResponse;
  height?: number;
}

/**
 * Train + validation loss line plot for the masked-transformer training run.
 */
export function LossCurveChart({ data, height = 240 }: LossCurveChartProps) {
  const epochs = data.epochs;
  const train = data.train_loss;
  const val = data.val_loss;

  return (
    <div className="border rounded-md p-2" data-testid="loss-curve-chart">
      <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">
        Loss curve
      </div>
      <Plot
        data={[
          {
            x: epochs,
            y: train,
            name: "train",
            mode: "lines+markers",
            type: "scatter",
            line: { color: "#1f77b4" },
          },
          {
            x: epochs,
            y: val,
            name: "val",
            mode: "lines+markers",
            type: "scatter",
            line: { color: "#ff7f0e" },
          },
        ]}
        layout={{
          autosize: true,
          height,
          margin: { l: 40, r: 12, t: 8, b: 32 },
          xaxis: { title: { text: "epoch" } },
          yaxis: { title: { text: "loss" } },
          legend: { orientation: "h", y: -0.3 },
          showlegend: true,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
