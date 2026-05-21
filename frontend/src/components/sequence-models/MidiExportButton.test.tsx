import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

import type {
  PianoRollMidiExportRead,
  PianoRollMidiExportStatus,
  PianoRollNotesJobRead,
  PianoRollNotesStatus,
} from "@/api/sequenceModels";

import { MidiExportButton } from "./MidiExportButton";

vi.mock("@tanstack/react-query", () => ({
  useQuery: vi.fn(),
  useMutation: vi.fn(),
  useQueryClient: vi.fn(() => ({ invalidateQueries: vi.fn() })),
}));

vi.mock("@/api/sequenceModels", async () => {
  const actual = await vi.importActual<
    typeof import("@/api/sequenceModels")
  >("@/api/sequenceModels");
  return {
    ...actual,
    usePianoRollMidiExportStatus: vi.fn(),
    useCreatePianoRollMidiExport: vi.fn(),
  };
});

import {
  useCreatePianoRollMidiExport,
  usePianoRollMidiExportStatus,
} from "@/api/sequenceModels";

const mockedExportStatus = vi.mocked(usePianoRollMidiExportStatus);
const mockedCreateExport = vi.mocked(useCreatePianoRollMidiExport);

function completeNotesStatus(): PianoRollNotesStatus {
  const row: PianoRollNotesJobRead = {
    id: "notes-1",
    event_encoder_job_id: "eej-1",
    extractor_version: "v2",
    status: "complete",
    notes_path: "event_encoders/eej-1/event_notes_v2.parquet",
    n_events: 5,
    n_notes: 9,
    finished_at: "2026-05-21T12:00:00Z",
    started_at: "2026-05-21T11:59:30Z",
    error_message: null,
    compute_seconds: 0.1,
    params_json: "{}",
    created_at: "2026-05-21T11:59:00Z",
    updated_at: "2026-05-21T12:00:00Z",
  };
  return row as unknown as PianoRollNotesStatus;
}

function completeExportRow(
  overrides: Partial<PianoRollMidiExportRead> = {},
): PianoRollMidiExportRead {
  return {
    id: "midi-1",
    event_encoder_job_id: "eej-1",
    extractor_version: "v2",
    status: "complete",
    started_at: "2026-05-21T12:01:00Z",
    finished_at: "2026-05-21T12:01:05Z",
    error_message: null,
    midi_path: "exports/event_encoders/eej-1/notes_v2.mid",
    n_notes: 9,
    n_bytes: 4096,
    compute_seconds: 0.5,
    params_json: "{}",
    window_start_utc: 1_750_000_000,
    window_end_utc: 1_750_000_060,
    audio_path: "exports/event_encoders/eej-1/audio_v2.flac",
    audio_size_bytes: 1024 * 1024,
    audio_sample_rate: 32_000,
    audio_duration_s: 60.0,
    created_at: "2026-05-21T12:01:00Z",
    updated_at: "2026-05-21T12:01:05Z",
    ...overrides,
  };
}

function setExportStatus(data: PianoRollMidiExportStatus | undefined) {
  mockedExportStatus.mockReturnValue({ data } as never);
}

function setMutation({
  mutate = vi.fn(),
  isPending = false,
}: { mutate?: ReturnType<typeof vi.fn>; isPending?: boolean } = {}) {
  mockedCreateExport.mockReturnValue({ mutate, isPending } as never);
  return mutate;
}

const baseProps = {
  jobId: "eej-1",
  windowStartUtc: 1_750_000_000,
  windowEndUtc: 1_750_000_060,
};

beforeEach(() => {
  mockedExportStatus.mockReset();
  mockedCreateExport.mockReset();
  setMutation();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("MidiExportButton", () => {
  it("disables export when notes are not complete", () => {
    setExportStatus({ status: "absent" });
    render(<MidiExportButton {...baseProps} notesStatus={undefined} />);
    const button = screen.getByTestId(
      "eej-piano-roll-midi-export-button",
    ) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
  });

  it("submits the current window when export is absent", () => {
    setExportStatus({ status: "absent" });
    const mutate = setMutation();
    render(
      <MidiExportButton
        {...baseProps}
        notesStatus={completeNotesStatus()}
      />,
    );
    const button = screen.getByTestId("eej-piano-roll-midi-export-button");
    expect(button.textContent).toContain("Export view");
    fireEvent.click(button);
    expect(mutate).toHaveBeenCalledWith({
      window_start_utc: baseProps.windowStartUtc,
      window_end_utc: baseProps.windowEndUtc,
    });
  });

  it("shows the progress label while the export is running", () => {
    setExportStatus(completeExportRow({ status: "running" }));
    render(
      <MidiExportButton
        {...baseProps}
        notesStatus={completeNotesStatus()}
      />,
    );
    expect(
      screen.getByTestId("eej-piano-roll-midi-export-progress").textContent,
    ).toContain("Exporting MIDI and audio…");
  });

  it("renders both download links and re-export when complete", () => {
    setExportStatus(completeExportRow());
    render(
      <MidiExportButton
        {...baseProps}
        notesStatus={completeNotesStatus()}
      />,
    );

    const midi = screen.getByTestId("eej-piano-roll-midi-export-download");
    expect(midi.getAttribute("href")).toBe(
      "/sequence-models/event-encoders/eej-1/midi-export",
    );
    const audio = screen.getByTestId("eej-piano-roll-audio-export-download");
    expect(audio.getAttribute("href")).toBe(
      "/sequence-models/event-encoders/eej-1/audio-export",
    );
    expect(
      screen.getByTestId("eej-piano-roll-midi-export-window").textContent,
    ).toContain("Exported window:");

    const reExport = screen.getByTestId("eej-piano-roll-midi-export-button");
    expect(reExport.getAttribute("data-window-match")).toBe("true");
  });

  it("flags re-export as window-mismatch when the view changes", () => {
    setExportStatus(completeExportRow());
    render(
      <MidiExportButton
        {...baseProps}
        windowStartUtc={baseProps.windowStartUtc + 30}
        windowEndUtc={baseProps.windowEndUtc + 30}
        notesStatus={completeNotesStatus()}
      />,
    );
    const reExport = screen.getByTestId("eej-piano-roll-midi-export-button");
    expect(reExport.getAttribute("data-window-match")).toBe("false");
  });

  it("disables the button when the requested window exceeds the cap", () => {
    setExportStatus({ status: "absent" });
    render(
      <MidiExportButton
        {...baseProps}
        windowEndUtc={baseProps.windowStartUtc + 1801}
        notesStatus={completeNotesStatus()}
      />,
    );
    const button = screen.getByTestId(
      "eej-piano-roll-midi-export-button",
    ) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(button.getAttribute("title") || "").toMatch(/cap/i);
  });

  it("re-export sends force=true", () => {
    setExportStatus(completeExportRow());
    const mutate = setMutation();
    render(
      <MidiExportButton
        {...baseProps}
        notesStatus={completeNotesStatus()}
      />,
    );
    fireEvent.click(screen.getByTestId("eej-piano-roll-midi-export-button"));
    expect(mutate).toHaveBeenCalledWith({
      window_start_utc: baseProps.windowStartUtc,
      window_end_utc: baseProps.windowEndUtc,
      force: true,
    });
  });
});
