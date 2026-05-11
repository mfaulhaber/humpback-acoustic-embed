import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { TrainingDatasetTable } from "./TrainingDatasetTable";

const mocks = vi.hoisted(() => ({
  deleteDataset: vi.fn(),
  train: vi.fn(),
  toast: vi.fn(),
}));

vi.mock("@/components/ui/use-toast", () => ({
  toast: mocks.toast,
}));

vi.mock("@/hooks/queries/useCallParsing", () => ({
  useSegmentationTrainingDatasets: () => ({
    data: [
      {
        id: "dataset-1",
        name: "curated corrections",
        sample_count: 12,
        source_job_count: 2,
        created_at: "2026-05-11T00:00:00Z",
      },
    ],
  }),
  useCreateSegmentationTrainingJob: () => ({
    isPending: false,
    mutate: mocks.train,
  }),
  useDeleteSegmentationTrainingDataset: () => ({
    isPending: false,
    mutateAsync: mocks.deleteDataset,
  }),
  useSegmentationJobsWithCorrectionCounts: () => ({ data: [] }),
}));

describe("TrainingDatasetTable", () => {
  beforeEach(() => {
    mocks.deleteDataset.mockReset();
    mocks.deleteDataset.mockResolvedValue(undefined);
    mocks.train.mockReset();
    mocks.toast.mockReset();
  });

  it("confirms before deleting a training dataset", async () => {
    render(<TrainingDatasetTable />);

    const deleteButton = screen.getByRole("button", { name: "Delete" });
    expect(deleteButton.className).toContain("bg-red-600");

    fireEvent.click(deleteButton);
    expect(
      screen.getByRole("heading", { name: "Delete training dataset" }),
    ).toBeTruthy();
    expect(mocks.deleteDataset).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() =>
      expect(mocks.deleteDataset).toHaveBeenCalledWith("dataset-1"),
    );
  });
});
