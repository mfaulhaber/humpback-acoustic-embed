import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MTTrainingCreateForm } from "./MTTrainingCreateForm";
import { useCreateMaskedTransformerJob } from "@/api/sequenceModels";

const mutate = vi.fn();

vi.mock("@/api/sequenceModels", () => ({
  continuousEmbeddingSourceKind: () => "region_crnn",
  useContinuousEmbeddingJobs: () => ({
    data: [
      {
        id: "cej-1",
        status: "complete",
        event_segmentation_job_id: "seg-1",
        model_version: "crnn-call-parsing-pytorch",
        vector_dim: 64,
        chunk_size_seconds: 0.25,
        chunk_hop_seconds: 0.25,
        projection_kind: "identity",
        projection_dim: 64,
        crnn_checkpoint_sha256: "ckpt",
        total_chunks: 12,
      },
      {
        id: "cej-2",
        status: "complete",
        event_segmentation_job_id: "seg-2",
        model_version: "crnn-call-parsing-pytorch",
        vector_dim: 64,
        chunk_size_seconds: 0.25,
        chunk_hop_seconds: 0.25,
        projection_kind: "identity",
        projection_dim: 64,
        crnn_checkpoint_sha256: "ckpt",
        total_chunks: 24,
      },
      {
        id: "cej-bad",
        status: "complete",
        event_segmentation_job_id: "seg-bad",
        model_version: "crnn-call-parsing-pytorch",
        vector_dim: 128,
        chunk_size_seconds: 0.25,
        chunk_hop_seconds: 0.25,
        projection_kind: "identity",
        projection_dim: 128,
        crnn_checkpoint_sha256: "ckpt",
        total_chunks: 24,
      },
    ],
  }),
  useCreateMaskedTransformerJob: vi.fn(() => ({
    isPending: false,
    mutate,
  })),
  useEventClassificationJobsForSegmentation: (segmentationJobId: string | null) => ({
    isLoading: false,
    data:
      segmentationJobId === "seg-2"
        ? [{ id: "cls-2", model_name: "second", n_events_classified: 24 }]
        : segmentationJobId === "seg-bad"
          ? [{ id: "cls-bad", model_name: "bad", n_events_classified: 24 }]
          : [{ id: "cls-1", model_name: "first", n_events_classified: 12 }],
  }),
}));

function renderForm() {
  render(
    <MemoryRouter>
      <MTTrainingCreateForm />
    </MemoryRouter>,
  );
}

describe("MTTrainingCreateForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mutate.mockReset();
  });

  it("submits ordered source pairs with contrastive disabled", async () => {
    renderForm();

    fireEvent.change(screen.getByTestId("mt-training-source-0"), {
      target: { value: "cej-1" },
    });
    await waitFor(() =>
      expect(
        (screen.getByTestId("mt-training-classify-0") as HTMLSelectElement)
          .value,
      ).toBe("cls-1"),
    );
    fireEvent.click(screen.getByTestId("mt-training-add-source"));
    fireEvent.change(screen.getByTestId("mt-training-source-1"), {
      target: { value: "cej-2" },
    });
    await waitFor(() =>
      expect(
        (screen.getByTestId("mt-training-classify-1") as HTMLSelectElement)
          .value,
      ).toBe("cls-2"),
    );

    fireEvent.click(screen.getByTestId("mt-training-submit"));

    const mutation = vi.mocked(useCreateMaskedTransformerJob).mock.results[0].value;
    expect(mutation.mutate).toHaveBeenCalledWith(
      expect.objectContaining({
        sources: [
          {
            continuous_embedding_job_id: "cej-1",
            event_classification_job_id: "cls-1",
            source_alias: null,
          },
          {
            continuous_embedding_job_id: "cej-2",
            event_classification_job_id: "cls-2",
            source_alias: null,
          },
        ],
        contrastive_loss_weight: 0,
        contrastive_label_source: "none",
        training_freeze_mode: "none",
      }),
      expect.any(Object),
    );
  });

  it("filters classify choices by selected embedding segmentation", async () => {
    renderForm();
    fireEvent.change(screen.getByTestId("mt-training-source-0"), {
      target: { value: "cej-2" },
    });

    await waitFor(() =>
      expect(
        (screen.getByTestId("mt-training-classify-0") as HTMLSelectElement)
          .value,
      ).toBe("cls-2"),
    );
  });

  it("blocks incompatible selected sources before submit", async () => {
    renderForm();
    fireEvent.change(screen.getByTestId("mt-training-source-0"), {
      target: { value: "cej-1" },
    });
    await waitFor(() =>
      expect(
        (screen.getByTestId("mt-training-classify-0") as HTMLSelectElement)
          .value,
      ).toBe("cls-1"),
    );
    fireEvent.click(screen.getByTestId("mt-training-add-source"));
    fireEvent.change(screen.getByTestId("mt-training-source-1"), {
      target: { value: "cej-bad" },
    });

    expect(await screen.findByTestId("mt-training-compat-error")).toBeTruthy();
    fireEvent.click(screen.getByTestId("mt-training-submit"));
    expect(mutate).not.toHaveBeenCalled();
  });

  it("does not render contrastive controls", () => {
    renderForm();
    expect(screen.queryByTestId("mt-contrastive-enabled")).toBeNull();
  });
});
