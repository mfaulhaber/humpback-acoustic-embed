import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  MTAnalysisReportTables,
  classifyAnalysisMetric,
} from "./MTAnalysisReportTables";

const REPORT = {
  job: { job_id: "mt-1", k: 100 },
  options: { include_geometry_report: true },
  artifacts: { contextual_path: "/tmp/contextual.parquet" },
  label_coverage: { labeled_query_count: 12 },
  results: {
    unrestricted: {
      raw_l2: {
        same_human_label: 0.72,
        adjacent_1s: 0.12,
        avg_cosine: 0.91,
      },
    },
    exclude_same_event: {
      raw_l2: {
        same_human_label: 0.2,
        adjacent_1s: 0.7,
      },
    },
  },
  event_level_results: {
    unrestricted: {
      raw_l2: {
        exact_human_label_set: 0.6,
      },
    },
  },
  representative_good_queries: [
    {
      query_idx: 1,
      query_human_types: "Moan",
      verdict: "good",
      same_human_label_rate: 0.8,
      adjacent_1s_rate: 0.1,
    },
  ],
  representative_risky_queries: [],
  query_rows: [],
  neighbor_rows: [],
  geometry_report: {
    spaces: {
      "contextual.raw_l2": {
        available: true,
        mean_vector_band: "healthy",
        effective_rank_band: "plausible",
        warnings: [],
      },
      "retrieval.raw_l2": {
        available: true,
        mean_vector_band: "saturated",
        effective_rank_band: "severe_collapse",
        warnings: ["low_effective_rank"],
      },
    },
    summary: { lambda_sweeps_blocked: false },
  },
};

describe("MTAnalysisReportTables", () => {
  it("classifies only directional analysis metrics", () => {
    expect(classifyAnalysisMetric("same_human_label", 0.7)).toBe("good");
    expect(classifyAnalysisMetric("same_human_label", 0.3)).toBe("warn");
    expect(classifyAnalysisMetric("same_human_label", 0.1)).toBe("bad");
    expect(classifyAnalysisMetric("adjacent_1s", 0.1)).toBe("good");
    expect(classifyAnalysisMetric("adjacent_1s", 0.7)).toBe("bad");
    expect(classifyAnalysisMetric("same_event", 0.7)).toBe("bad");
    expect(classifyAnalysisMetric("avg_cosine", 0.9)).toBe("none");
  });

  it("renders metric, event-level, query, and geometry tables with indicators", () => {
    render(<MTAnalysisReportTables report={REPORT} />);

    expect(screen.getByText("Aggregate Retrieval Metrics")).not.toBeNull();
    expect(screen.getByText("Event-Level Metrics")).not.toBeNull();
    expect(screen.getByTestId("mt-analysis-geometry-table")).not.toBeNull();
    expect(screen.getByText("Representative Good Queries")).not.toBeNull();
    expect(screen.getByText("Moan")).not.toBeNull();

    const sameLabel = screen.getAllByText("same_human_label")[0].closest("tr");
    const adjacent = screen.getAllByText("adjacent_1s")[0].closest("tr");
    const avgCosine = screen.getByText("avg_cosine").closest("tr");
    const collapsed = screen.getByText("severe_collapse").closest("tr");

    expect(sameLabel?.className).toContain("bg-emerald-50");
    expect(adjacent?.className).toContain("bg-emerald-50");
    expect(avgCosine?.className).not.toContain("bg-emerald-50");
    expect(collapsed?.className).toContain("bg-red-50");
  });
});
