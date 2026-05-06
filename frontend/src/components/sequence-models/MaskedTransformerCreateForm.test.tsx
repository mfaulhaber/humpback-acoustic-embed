import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MaskedTransformerCreateForm } from "./MaskedTransformerCreateForm";
import {
  useCreateMaskedTransformerJob,
  useEventClassificationJobsForSegmentation,
} from "@/api/sequenceModels";

vi.mock("@/api/sequenceModels", () => {
  const mutate = vi.fn();
  return {
    continuousEmbeddingSourceKind: () => "region_crnn",
    useContinuousEmbeddingJobs: () => ({
      data: [
        {
          id: "cej-1",
          status: "complete",
          event_segmentation_job_id: "seg-1",
          model_version: "crnn-call-parsing-pytorch",
          vector_dim: 256,
          total_chunks: 24,
          region_detection_job_id: "region-job-1",
        },
      ],
    }),
    useCreateMaskedTransformerJob: vi.fn(() => ({
      isPending: false,
      mutate,
    })),
    useEventClassificationJobsForSegmentation: vi.fn(() => ({
      isLoading: false,
      data: [
        {
          id: "cls-1",
          model_name: "classifier",
          n_events_classified: 12,
        },
      ],
    })),
  };
});

function renderForm() {
  render(
    <MemoryRouter>
      <MaskedTransformerCreateForm />
    </MemoryRouter>,
  );
}

function selectReadySource() {
  fireEvent.change(screen.getByTestId("mt-source-select"), {
    target: { value: "cej-1" },
  });
}

function openAdvancedAndEnableRetrieval() {
  fireEvent.click(screen.getByTestId("mt-show-advanced"));
  fireEvent.click(screen.getByTestId("mt-retrieval-head-enabled"));
}

describe("MaskedTransformerCreateForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("defaults retrieval-head architecture to MLP", () => {
    renderForm();
    openAdvancedAndEnableRetrieval();

    const mlp = screen
      .getByTestId("mt-retrieval-head-arch-mlp")
      .querySelector("input");
    const linear = screen
      .getByTestId("mt-retrieval-head-arch-linear")
      .querySelector("input");

    expect((mlp as HTMLInputElement | null)?.checked).toBe(true);
    expect((linear as HTMLInputElement | null)?.checked).toBe(false);
    expect(screen.queryByTestId("mt-adv-retrieval_hidden_dim")).not.toBeNull();
  });

  it("hides hidden dimension when Linear is selected", () => {
    renderForm();
    openAdvancedAndEnableRetrieval();

    fireEvent.click(
      screen
        .getByTestId("mt-retrieval-head-arch-linear")
        .querySelector("input") as HTMLInputElement,
    );

    expect(screen.queryByTestId("mt-adv-retrieval_hidden_dim")).toBeNull();
  });

  it("submits a linear retrieval-head payload", async () => {
    renderForm();
    selectReadySource();
    await waitFor(() =>
      expect(useEventClassificationJobsForSegmentation).toHaveBeenCalled(),
    );
    openAdvancedAndEnableRetrieval();
    fireEvent.click(
      screen
        .getByTestId("mt-retrieval-head-arch-linear")
        .querySelector("input") as HTMLInputElement,
    );

    fireEvent.click(screen.getByTestId("mt-create-submit"));

    const mutation = vi.mocked(useCreateMaskedTransformerJob).mock.results[0]
      .value;
    expect(mutation.mutate).toHaveBeenCalledTimes(1);
    expect(mutation.mutate).toHaveBeenCalledWith(
      expect.objectContaining({
        continuous_embedding_job_id: "cej-1",
        event_classification_job_id: "cls-1",
        retrieval_head_enabled: true,
        retrieval_head_arch: "linear",
        retrieval_dim: 128,
        retrieval_hidden_dim: null,
      }),
      expect.any(Object),
    );
  });
});
