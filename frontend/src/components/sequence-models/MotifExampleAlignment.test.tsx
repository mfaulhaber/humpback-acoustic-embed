import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { MotifOccurrence } from "@/api/sequenceModels";
import { MotifExampleAlignment } from "./MotifExampleAlignment";

function makeOccurrence(idx: number): MotifOccurrence {
  return {
    occurrence_id: `occ-${idx}`,
    motif_key: "1-2",
    states: [1, 2],
    source_kind: "region_crnn",
    group_key: `region-${idx}`,
    event_source_key: `e${idx}`,
    audio_source_key: null,
    token_start_index: 0,
    token_end_index: 1,
    raw_start_index: 0,
    raw_end_index: 1,
    start_timestamp: 1_700_000_000 + idx * 10,
    end_timestamp: 1_700_000_001 + idx * 10,
    duration_seconds: 1,
    event_core_fraction: 1.0,
    background_fraction: 0.0,
    mean_call_probability: null,
    anchor_event_id: `e${idx}`,
    anchor_timestamp: 1_700_000_000 + idx * 10,
    relative_start_seconds: 0.0,
    relative_end_seconds: 1.0,
    anchor_strategy: "event_midpoint",
  };
}

describe("MotifExampleAlignment", () => {
  it("highlights the row matching activeOccurrenceIndex", () => {
    const rows = [makeOccurrence(0), makeOccurrence(1), makeOccurrence(2)];
    render(
      <MotifExampleAlignment
        occurrences={rows}
        regionDetectionJobId="rdj"
        onJumpToTimestamp={() => {}}
        activeOccurrenceIndex={2}
      />,
    );
    expect(
      screen.getByTestId("motif-example-row-0").getAttribute("data-active"),
    ).toBe("false");
    expect(
      screen.getByTestId("motif-example-row-2").getAttribute("data-active"),
    ).toBe("true");
  });

  it("clicking Jump on a row reports that row's index via onActiveOccurrenceChange", () => {
    const rows = [makeOccurrence(0), makeOccurrence(1), makeOccurrence(2)];
    const onActive = vi.fn();
    const onJump = vi.fn();
    render(
      <MotifExampleAlignment
        occurrences={rows}
        regionDetectionJobId="rdj"
        onJumpToTimestamp={onJump}
        activeOccurrenceIndex={0}
        onActiveOccurrenceChange={onActive}
      />,
    );
    const jumpButtons = screen.getAllByText("Jump");
    fireEvent.click(jumpButtons[1]);
    expect(onActive).toHaveBeenCalledWith(1);
    expect(onJump).toHaveBeenCalledTimes(1);
  });

  it("renders empty state when occurrences are empty", () => {
    render(
      <MotifExampleAlignment
        occurrences={[]}
        regionDetectionJobId="rdj"
        onJumpToTimestamp={() => {}}
      />,
    );
    expect(screen.getByTestId("motif-examples-empty")).toBeTruthy();
  });
});
