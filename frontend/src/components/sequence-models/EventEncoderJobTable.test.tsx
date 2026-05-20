import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { EventEncoderJobTable } from "./EventEncoderJobTable";
import type { EventEncoderJob } from "@/api/sequenceModels";

const mocks = vi.hoisted(() => ({
  deleteJob: vi.fn(),
  cancelJob: vi.fn(),
}));

vi.mock("@/api/sequenceModels", () => ({
  useCancelEventEncoderJob: () => ({
    mutate: mocks.cancelJob,
    isPending: false,
  }),
  useDeleteEventEncoderJob: () => ({
    mutateAsync: mocks.deleteJob,
    isPending: false,
  }),
  usePianoRollNotesStatus: () => ({ data: { status: "absent" } }),
}));

const job = {
  id: "event-encoder-job-1",
  status: "complete",
  created_at: "2026-05-11T00:00:00Z",
  event_segmentation_job_id: "segmentation-job-1",
  event_source_mode: "raw",
  encoded_events: 12,
  total_events: 12,
  event_vector_dim: 32,
  k_values_json: "[8,12]",
} as EventEncoderJob;

describe("EventEncoderJobTable", () => {
  beforeEach(() => {
    mocks.deleteJob.mockReset();
    mocks.deleteJob.mockResolvedValue(undefined);
    mocks.cancelJob.mockReset();
  });

  it("confirms before deleting an Event Encoder job", async () => {
    render(
      <MemoryRouter>
        <EventEncoderJobTable jobs={[job]} mode="previous" />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(
      screen.getByRole("heading", { name: "Delete event encoder job" }),
    ).toBeTruthy();
    expect(mocks.deleteJob).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() =>
      expect(mocks.deleteJob).toHaveBeenCalledWith("event-encoder-job-1"),
    );
  });
});
