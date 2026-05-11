import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { ContinuousEmbeddingJobTable } from "./ContinuousEmbeddingJobTable";
import type { ContinuousEmbeddingJob } from "@/api/sequenceModels";

const mocks = vi.hoisted(() => ({
  deleteJob: vi.fn(),
  cancelJob: vi.fn(),
}));

vi.mock("@/api/sequenceModels", () => ({
  continuousEmbeddingSourceKind: (job: ContinuousEmbeddingJob) =>
    job.region_detection_job_id ? "region_crnn" : "surfperch",
  useCancelContinuousEmbeddingJob: () => ({
    mutate: mocks.cancelJob,
    isPending: false,
  }),
  useDeleteContinuousEmbeddingJob: () => ({
    mutateAsync: mocks.deleteJob,
    isPending: false,
  }),
}));

const job = {
  id: "continuous-job-1",
  status: "complete",
  created_at: "2026-05-11T00:00:00Z",
  region_detection_job_id: "region-job-1",
  event_segmentation_job_id: null,
  event_source_mode: "raw",
  model_version: "surfperch-test",
  total_regions: 2,
  total_chunks: 3,
  merged_spans: null,
  total_windows: null,
} as ContinuousEmbeddingJob;

function renderTable() {
  return render(
    <MemoryRouter>
      <ContinuousEmbeddingJobTable jobs={[job]} mode="previous" />
    </MemoryRouter>,
  );
}

describe("ContinuousEmbeddingJobTable", () => {
  beforeEach(() => {
    mocks.deleteJob.mockReset();
    mocks.deleteJob.mockResolvedValue(undefined);
    mocks.cancelJob.mockReset();
  });

  it("confirms before deleting a row", async () => {
    renderTable();

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(
      screen.getByRole("heading", {
        name: "Delete continuous embedding job",
      }),
    ).toBeTruthy();
    expect(mocks.deleteJob).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() =>
      expect(mocks.deleteJob).toHaveBeenCalledWith("continuous-job-1"),
    );
  });

  it("confirms before bulk deleting selected rows", async () => {
    renderTable();

    fireEvent.click(screen.getAllByRole("checkbox")[1]);
    fireEvent.click(screen.getByRole("button", { name: "Delete (1)" }));
    expect(
      screen.getByRole("heading", {
        name: "Delete continuous embedding job",
      }),
    ).toBeTruthy();
    expect(mocks.deleteJob).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() =>
      expect(mocks.deleteJob).toHaveBeenCalledWith("continuous-job-1"),
    );
  });
});
