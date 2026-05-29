import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { PianoRollNotesStatusPill } from "./PianoRollNotesStatusPill";
import type { PianoRollNotesStatus } from "@/api/sequenceModels";

function makeStatus(
  overrides: Partial<PianoRollNotesStatus> = {},
): PianoRollNotesStatus {
  return {
    id: "row-1",
    event_encoder_job_id: "ee-1",
    extractor_version: "v3",
    status: "complete",
    started_at: "2026-05-22T19:48:47.853013",
    finished_at: "2026-05-22T19:52:21.044439",
    error_message: null,
    notes_path: "/tmp/event_notes_v3.parquet",
    n_events: 100,
    n_notes: 1000,
    compute_seconds: 10.0,
    params_json: "{}",
    created_at: "2026-05-22T19:48:47.738035",
    updated_at: "2026-05-22T19:52:21.044915",
    ...overrides,
  } as PianoRollNotesStatus;
}

describe("PianoRollNotesStatusPill upgrade badge", () => {
  it("renders the v4-available badge when a v3 row is complete and v4 is the latest", () => {
    const handler = vi.fn();
    render(
      <PianoRollNotesStatusPill
        status={makeStatus({ extractor_version: "v3" })}
        latestExtractorVersion="v4"
        onRequestUpgrade={handler}
      />,
    );
    const badge = screen.getByTestId("piano-roll-notes-upgrade-badge");
    expect(badge.textContent).toBe("v4 available");
    fireEvent.click(badge);
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("renders the v6-available badge when a v5 row is complete and v6 is the latest", () => {
    const handler = vi.fn();
    render(
      <PianoRollNotesStatusPill
        status={makeStatus({ extractor_version: "v5" })}
        latestExtractorVersion="v6"
        onRequestUpgrade={handler}
      />,
    );
    const badge = screen.getByTestId("piano-roll-notes-upgrade-badge");
    expect(badge.textContent).toBe("v6 available");
    fireEvent.click(badge);
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("does not render the upgrade badge when the row is already at the latest version", () => {
    render(
      <PianoRollNotesStatusPill
        status={makeStatus({ extractor_version: "v4" })}
        latestExtractorVersion="v4"
        onRequestUpgrade={vi.fn()}
      />,
    );
    expect(
      screen.queryByTestId("piano-roll-notes-upgrade-badge"),
    ).toBeNull();
  });

  it("does not render the upgrade badge when the row is not complete", () => {
    render(
      <PianoRollNotesStatusPill
        status={makeStatus({ extractor_version: "v3", status: "running" })}
        latestExtractorVersion="v4"
        onRequestUpgrade={vi.fn()}
      />,
    );
    expect(
      screen.queryByTestId("piano-roll-notes-upgrade-badge"),
    ).toBeNull();
  });

  it("does not render the upgrade badge when no handler is provided", () => {
    render(
      <PianoRollNotesStatusPill
        status={makeStatus({ extractor_version: "v3" })}
        latestExtractorVersion="v4"
      />,
    );
    expect(
      screen.queryByTestId("piano-roll-notes-upgrade-badge"),
    ).toBeNull();
  });

  it("does not render the upgrade badge when no latest version is provided", () => {
    render(
      <PianoRollNotesStatusPill
        status={makeStatus({ extractor_version: "v3" })}
        onRequestUpgrade={vi.fn()}
      />,
    );
    expect(
      screen.queryByTestId("piano-roll-notes-upgrade-badge"),
    ).toBeNull();
  });
});
