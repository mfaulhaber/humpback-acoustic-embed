import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import {
  DeleteActionButton,
  DeleteConfirmationDialog,
  DeleteConfirmButton,
} from "./DeleteConfirmationDialog";

describe("DeleteConfirmationDialog", () => {
  it("renders Cloudscape-style delete copy for a single resource", () => {
    render(
      <DeleteConfirmationDialog
        open
        onOpenChange={() => {}}
        resourceType="region detection job"
        resourceName="abc12345"
        consequence="The job and cached region artifacts will be removed."
        onConfirm={() => {}}
      />,
    );

    expect(screen.getByRole("heading", { name: "Delete region detection job" })).toBeTruthy();
    expect(screen.getByText(/Permanently delete/).textContent ?? "").toContain(
      "abc12345",
    );
    expect(
      screen.getByText("The job and cached region artifacts will be removed."),
    ).toBeTruthy();
  });

  it("uses plural title and count for multi-resource deletes", () => {
    render(
      <DeleteConfirmationDialog
        open
        onOpenChange={() => {}}
        resourceType="segmentation job"
        count={3}
        consequence="Selected jobs and artifacts will be removed."
        onConfirm={() => {}}
      />,
    );

    expect(screen.getByRole("heading", { name: "Delete segmentation jobs" })).toBeTruthy();
    expect(screen.getByText(/Permanently delete/).textContent ?? "").toContain("3");
  });

  it("cancels without confirming", () => {
    const onOpenChange = vi.fn();
    const onConfirm = vi.fn();
    render(
      <DeleteConfirmationDialog
        open
        onOpenChange={onOpenChange}
        resourceType="event encoder job"
        resourceName="job-1"
        consequence="The job artifacts will be removed."
        onConfirm={onConfirm}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("confirms deletion from the dialog action", () => {
    const onConfirm = vi.fn();
    render(
      <DeleteConfirmationDialog
        open
        onOpenChange={() => {}}
        resourceType="event encoder job"
        resourceName="job-1"
        consequence="The job artifacts will be removed."
        onConfirm={onConfirm}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("provides a shared red rounded delete button style", () => {
    render(<DeleteActionButton>Delete</DeleteActionButton>);
    const button = screen.getByRole("button", { name: "Delete" });
    expect(button.className).toContain("rounded-md");
    expect(button.className).toContain("bg-red-600");
    expect(button.className).toContain("text-white");
  });

  it("opens from DeleteConfirmButton and waits for confirmation", async () => {
    const onConfirm = vi.fn();
    render(
      <DeleteConfirmButton
        resourceType="training dataset"
        resourceName="dataset-a"
        consequence="The dataset samples will be removed."
        onConfirm={onConfirm}
      >
        Delete
      </DeleteConfirmButton>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(screen.getByRole("heading", { name: "Delete training dataset" })).toBeTruthy();
    expect(onConfirm).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(1));
  });
});
