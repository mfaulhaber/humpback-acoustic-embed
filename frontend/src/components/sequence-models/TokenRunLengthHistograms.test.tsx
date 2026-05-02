import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TokenRunLengthHistograms } from "./TokenRunLengthHistograms";

vi.mock("react-plotly.js", () => ({
  default: () => <div data-testid="plotly-mock" />,
}));

describe("TokenRunLengthHistograms", () => {
  it("renders one panel per token with samples", () => {
    render(
      <TokenRunLengthHistograms
        runLengths={{ "0": [1, 2, 3], "1": [4, 5] }}
        k={100}
      />,
    );
    expect(
      screen.getByTestId("token-run-length-histograms"),
    ).not.toBeNull();
    expect(screen.getByTestId("token-run-length-0")).not.toBeNull();
    expect(screen.getByTestId("token-run-length-1")).not.toBeNull();
  });

  it("renders empty placeholder when no samples present", () => {
    render(<TokenRunLengthHistograms runLengths={{}} k={100} />);
    expect(
      screen.getByTestId("token-run-length-histograms-empty"),
    ).not.toBeNull();
  });

  it("caps panels by maxTokens", () => {
    const runLengths: Record<string, number[]> = {};
    for (let i = 0; i < 10; i++) runLengths[String(i)] = [i + 1];
    render(
      <TokenRunLengthHistograms runLengths={runLengths} k={100} maxTokens={3} />,
    );
    const grid = screen.getByTestId("token-run-length-histograms");
    const items = grid.querySelectorAll('[data-testid^="token-run-length-"]');
    // grid item testids are token-run-length-<key>; the wrapping container
    // also matches token-run-length-histograms but is excluded by the
    // ``-`` prefix expectations from above.
    const nested = Array.from(items).filter(
      (el) => el.getAttribute("data-testid") !== "token-run-length-histograms",
    );
    expect(nested.length).toBe(3);
  });
});
