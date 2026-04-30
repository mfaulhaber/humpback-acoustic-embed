import { describe, expect, it } from "vitest";
import {
  regionBoundaryXPositions,
  visibleWindows,
  type ViterbiWindow,
} from "./HMMStateBar";

const windows: ViterbiWindow[] = [
  {
    start_timestamp: 0,
    end_timestamp: 5,
    viterbi_state: 0,
    max_state_probability: 0.9,
  },
  {
    start_timestamp: 10,
    end_timestamp: 15,
    viterbi_state: 1,
    max_state_probability: 0.8,
  },
  {
    start_timestamp: 20,
    end_timestamp: 25,
    viterbi_state: 2,
    max_state_probability: 0.7,
  },
];

describe("visibleWindows", () => {
  it("keeps windows that overlap the current view", () => {
    expect(visibleWindows(windows, 4, 21)).toEqual(windows);
  });

  it("drops windows fully outside the current view", () => {
    expect(visibleWindows(windows, 6, 19)).toEqual([windows[1]]);
  });
});

describe("regionBoundaryXPositions", () => {
  it("maps region edges into viewport pixels", () => {
    expect(
      regionBoundaryXPositions(
        { startTimestamp: 110, endTimestamp: 140 },
        100,
        2,
      ),
    ).toEqual({ startX: 20, endX: 80 });
  });
});
