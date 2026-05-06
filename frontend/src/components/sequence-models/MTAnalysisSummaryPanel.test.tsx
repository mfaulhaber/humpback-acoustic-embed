import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  MTAnalysisSummaryPanel,
  buildMTAnalysisSummary,
} from "./MTAnalysisSummaryPanel";

const REPORT = {
  job: {
    job_id: "mt-1",
    k: 100,
    total_sequences: 109,
    total_chunks: 25273,
    final_train_loss: 0.159,
    final_val_loss: 0.148,
  },
  options: { include_geometry_report: true },
  artifacts: {},
  label_coverage: {
    human_labeled_query_pool_rows: 500,
  },
  results: {
    unrestricted: {
      raw_l2: {
        same_human_label: 0.7,
        exact_human_label_set: 0.6,
        same_token: 0.95,
        similar_duration: 0.62,
        same_region: 0.36,
        adjacent_1s: 0.02,
        nearby_5s: 0.04,
      },
      whiten_pca: {
        same_human_label: 0.8,
        same_token: 0.83,
        similar_duration: 0.68,
        same_region: 0.51,
        adjacent_1s: 0.08,
        nearby_5s: 0.12,
      },
    },
  },
  event_level_results: null,
  representative_good_queries: [],
  representative_risky_queries: [],
  query_rows: [],
  neighbor_rows: [],
  geometry_report: {
    spaces: {
      "contextual.raw_l2": {
        available: true,
        vector_dim: 256,
        random_pair_percentiles: { p75: 0.967, p95: 0.998 },
        mean_vector_band: "collapse_risk",
        effective_rank: 24.3,
        effective_rank_band: "weak",
        pca_explained_variance: { pc1: 0.89 },
        warnings: ["pc1_dominant", "mean_norm_collapse_risk"],
      },
      "contextual.whiten_pca": {
        available: true,
        vector_dim: 256,
        random_pair_percentiles: { p75: 0.102, p95: 0.209 },
        mean_vector_band: "good",
        effective_rank: 121.8,
        effective_rank_band: "broad",
        pca_explained_variance: { pc1: 0.05 },
        warnings: [],
      },
      "retrieval.remove_pc10": {
        available: true,
        vector_dim: 256,
        random_pair_percentiles: { p75: 0.205, p95: 0.377 },
        mean_vector_band: "okay",
        effective_rank: 60.4,
        effective_rank_band: "plausible",
        pca_explained_variance: { pc1: 0.11 },
        warnings: [],
      },
    },
    summary: {
      retrieval_raw_saturated: true,
      lambda_sweeps_blocked: true,
      warnings: ["retrieval_raw_saturated"],
    },
  },
};

describe("MTAnalysisSummaryPanel", () => {
  it("builds a deterministic non-human-label summary", () => {
    const summary = buildMTAnalysisSummary(REPORT);
    const text = [
      ...summary.run,
      ...summary.geometry,
      ...summary.spaceDiagnostics.flatMap((space) => [
        { text: space.title },
        ...space.lines.map((line) => ({ text: line })),
      ]),
      ...summary.neighborhood,
    ]
      .map((line) => line.text)
      .join(" ");

    expect(text).toContain("Raw spaces need caution");
    expect(text).toContain("Raw Contextual Space");
    expect(text).toContain(
      "Very saturated: random-pair cosine p75 0.967, p95 0.998.",
    );
    expect(text).toContain("Mean vector band: collapse_risk.");
    expect(text).toContain("Effective rank: 24.3, weak for 256 dims.");
    expect(text).toContain("PC1 alone explains 89% of variance.");
    expect(text).toContain("Not a good space to trust directly.");
    expect(text).toContain("Whitened Contextual Space");
    expect(text).toContain("Good candidate for inference and motif search.");
    expect(text).toContain("Same-token neighbor rate range: 0.830-0.950");
    expect(text).not.toContain("same_human_label");
    expect(text).not.toContain("human_labeled_query_pool_rows");
  });

  it("renders the summary panel above the detailed tables", () => {
    render(<MTAnalysisSummaryPanel report={REPORT} />);

    expect(screen.getByTestId("mt-analysis-summary-panel")).not.toBeNull();
    expect(screen.getByText("Embedding Spaces")).not.toBeNull();
    expect(screen.getByText("Raw Contextual Space")).not.toBeNull();
    expect(screen.getByText("Neighborhood Behavior")).not.toBeNull();
    expect(screen.getByText(/Raw spaces need caution/)).not.toBeNull();
  });
});
