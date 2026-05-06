import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MTTrainingDetailPage } from "./MTTrainingDetailPage";
import { useRunMaskedTransformerAnalysis } from "@/api/sequenceModels";

const runAnalysis = vi.fn();

vi.mock("react-plotly.js", () => ({
  default: () => <div data-testid="plotly-mock" />,
}));

vi.mock("@/api/sequenceModels", () => ({
  useMaskedTransformerJob: vi.fn(() => ({
    isLoading: false,
    data: {
      job: {
        id: "mt-training-complete",
        status: "complete",
        status_reason: null,
        continuous_embedding_job_id: "cej-1",
        event_classification_job_id: "cls-1",
        training_signature: "sig-1",
        preset: "default",
        mask_fraction: 0.2,
        span_length_min: 2,
        span_length_max: 6,
        dropout: 0.1,
        mask_weight_bias: true,
        cosine_loss_weight: 0,
        batch_size: 8,
        retrieval_head_enabled: true,
        retrieval_dim: 128,
        retrieval_hidden_dim: 512,
        retrieval_l2_normalize: true,
        retrieval_head_arch: "mlp",
        sequence_construction_mode: "region",
        event_centered_fraction: 0,
        pre_event_context_sec: null,
        post_event_context_sec: null,
        contrastive_loss_weight: 0,
        contrastive_temperature: 0.07,
        contrastive_label_source: "none",
        contrastive_min_events_per_label: 4,
        contrastive_min_regions_per_label: 2,
        require_cross_region_positive: true,
        related_label_policy_json: null,
        contrastive_sampler_enabled: true,
        contrastive_labels_per_batch: 4,
        contrastive_events_per_label: 4,
        contrastive_max_unlabeled_fraction: 0.25,
        contrastive_region_balance: true,
        training_freeze_mode: "none",
        source_masked_transformer_job_id: null,
        negative_label_family_policy_json: null,
        max_epochs: 30,
        early_stop_patience: 3,
        val_split: 0.1,
        seed: 42,
        k_values: [50, 100],
        chosen_device: "mps",
        fallback_reason: null,
        final_train_loss: 0.12,
        final_val_loss: 0.18,
        total_epochs: 8,
        job_dir: "/tmp/mt-training-complete",
        total_sequences: 3,
        total_chunks: 36,
        error_message: null,
        created_at: "2026-05-06T12:00:00Z",
        updated_at: "2026-05-06T12:10:00Z",
      },
      sources: [
        {
          id: "source-1",
          masked_transformer_job_id: "mt-training-complete",
          source_order: 0,
          continuous_embedding_job_id: "cej-1",
          event_classification_job_id: "cls-1",
          source_alias: "north",
          created_at: "2026-05-06T12:00:00Z",
          updated_at: "2026-05-06T12:00:00Z",
        },
        {
          id: "source-2",
          masked_transformer_job_id: "mt-training-complete",
          source_order: 1,
          continuous_embedding_job_id: "cej-2",
          event_classification_job_id: "cls-2",
          source_alias: "south",
          created_at: "2026-05-06T12:00:00Z",
          updated_at: "2026-05-06T12:00:00Z",
        },
      ],
      region_detection_job_id: null,
      region_start_timestamp: null,
      region_end_timestamp: null,
      tier_composition: null,
      source_kind: "region_crnn",
    },
  })),
  useMaskedTransformerLossCurve: vi.fn(() => ({
    data: {
      epochs: [1, 2],
      train_loss: [0.4, 0.3],
      val_loss: [0.5, 0.35],
      val_metrics: {},
    },
  })),
  useMaskedTransformerRunLengths: vi.fn(() => ({
    data: { k: 50, tau: 1.5, run_lengths: { "1": [1, 2] } },
  })),
  useMaskedTransformerOverlay: vi.fn(() => ({
    data: {
      total: 1,
      items: [
        {
          sequence_id: "0:region-a",
          position_in_sequence: 0,
          start_timestamp: 10,
          end_timestamp: 10.25,
          pca_x: 0.1,
          pca_y: 0.2,
          umap_x: 1,
          umap_y: 2,
          viterbi_state: 1,
          max_state_probability: 0.9,
        },
      ],
    },
  })),
  useMaskedTransformerExemplars: vi.fn(() => ({
    data: {
      n_states: 50,
      states: {
        "1": [
          {
            sequence_id: "0:region-a",
            position_in_sequence: 0,
            audio_file_id: 1,
            start_timestamp: 10,
            end_timestamp: 10.25,
            max_state_probability: 0.9,
            exemplar_type: "high_confidence",
            extras: { tier: "event_core", event_types: ["song"] },
          },
        ],
      },
    },
  })),
  useRunMaskedTransformerAnalysis: vi.fn(() => ({
    isPending: false,
    mutate: runAnalysis,
  })),
}));

function renderPage() {
  render(
    <MemoryRouter initialEntries={["/app/sequence-models/mt-training/mt-training-complete"]}>
      <Routes>
        <Route
          path="/app/sequence-models/mt-training/:jobId"
          element={<MTTrainingDetailPage />}
        />
        <Route
          path="/app/sequence-models/mt-training/:jobId/analysis"
          element={<div data-testid="analysis-route" />}
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe("MTTrainingDetailPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    runAnalysis.mockImplementation((_vars, options) => {
      options?.onSuccess?.();
    });
  });

  it("renders training analysis without motif-only panels", () => {
    renderPage();

    expect(screen.getByTestId("mt-training-detail-page")).not.toBeNull();
    expect(screen.getByTestId("mt-training-source-table").textContent).toContain(
      "cej-2",
    );
    expect(screen.getByTestId("loss-curve-chart")).not.toBeNull();
    expect(screen.getByTestId("token-run-length-histograms")).not.toBeNull();
    expect(screen.getByTestId("mt-training-overlay")).not.toBeNull();
    expect(screen.getByTestId("mt-training-exemplars")).not.toBeNull();
    expect(screen.queryByTestId("mt-timeline-viewer")).toBeNull();
    expect(screen.queryByTestId("mt-label-distribution")).toBeNull();
    expect(screen.queryByTestId("motif-extraction-panel")).toBeNull();
    expect(screen.queryByText("song")).toBeNull();
  });

  it("runs the full Phase 0 analysis report and navigates to its child page", async () => {
    renderPage();

    fireEvent.click(screen.getByTestId("mt-training-analysis-button"));

    const mutation = vi.mocked(useRunMaskedTransformerAnalysis).mock.results[0].value;
    expect(mutation.mutate).toHaveBeenCalledWith(
      {
        jobId: "mt-training-complete",
        body: expect.objectContaining({
          k: 50,
          include_event_level: true,
          include_geometry_report: true,
          include_query_rows: true,
          include_neighbor_rows: false,
          retrieval_modes: [
            "unrestricted",
            "exclude_same_event",
            "exclude_same_event_and_region",
          ],
        }),
      },
      expect.any(Object),
    );
    await waitFor(() => expect(screen.getByTestId("analysis-route")).not.toBeNull());
  });
});
