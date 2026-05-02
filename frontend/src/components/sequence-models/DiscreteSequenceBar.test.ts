import { describe, expect, it } from "vitest";
import {
  regionBoundaryXPositions,
  visibleItems,
  type DiscreteSequenceItem,
} from "./DiscreteSequenceBar";
import { LABEL_COLORS, labelColor } from "./constants";

const items: DiscreteSequenceItem[] = [
  { start_timestamp: 0, end_timestamp: 5, label: 0, confidence: 0.9 },
  { start_timestamp: 10, end_timestamp: 15, label: 1, confidence: 0.8 },
  { start_timestamp: 20, end_timestamp: 25, label: 2, confidence: 0.7 },
];

describe("visibleItems", () => {
  it("keeps items that overlap the current view", () => {
    expect(visibleItems(items, 4, 21)).toEqual(items);
  });

  it("drops items fully outside the current view", () => {
    expect(visibleItems(items, 6, 19)).toEqual([items[1]]);
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

describe("labelColor", () => {
  it("wraps the categorical palette under the threshold", () => {
    expect(labelColor(0, 5)).toBe(LABEL_COLORS[0]);
    expect(labelColor(1, 5)).toBe(LABEL_COLORS[1]);
    // 25 wraps back to index 5 in the 20-wide palette.
    expect(labelColor(25, 30)).toBe(LABEL_COLORS[25 % LABEL_COLORS.length]);
  });

  it("generates a deterministic HSL ramp above the threshold", () => {
    const c0 = labelColor(0, 100);
    const c50 = labelColor(50, 100);
    const c0_again = labelColor(0, 100);
    expect(c0).toBe(c0_again);
    expect(c0).not.toBe(c50);
    expect(c0.startsWith("hsl(")).toBe(true);
  });
});
