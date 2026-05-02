import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { LossCurveChart } from "./LossCurveChart";

vi.mock("react-plotly.js", () => ({
  default: ({ data }: { data: unknown }) => (
    <div data-testid="plotly-mock" data-trace-count={(data as unknown[]).length} />
  ),
}));

describe("LossCurveChart", () => {
  it("renders with two traces (train + val)", () => {
    render(
      <LossCurveChart
        data={{
          epochs: [1, 2, 3],
          train_loss: [0.5, 0.3, 0.2],
          val_loss: [0.6, 0.4, 0.3],
          val_metrics: { final_val_loss: 0.3 },
        }}
      />,
    );
    expect(screen.getByTestId("loss-curve-chart")).not.toBeNull();
    const plot = screen.getByTestId("plotly-mock");
    expect(plot.getAttribute("data-trace-count")).toBe("2");
  });
});
