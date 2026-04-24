import { describe, expect, it } from "vitest";

import { formatRecordingTime } from "./format";

describe("formatRecordingTime", () => {
  const jobStart = Date.UTC(2026, 3, 24, 14, 30, 0) / 1000;

  it("formats offset as HH:MM:SS.d", () => {
    expect(formatRecordingTime(125.4, jobStart)).toBe("14:32:05.4");
  });

  it("zero-pads hours, minutes, seconds", () => {
    // jobStart at 01:02:00 UTC = epoch for that time
    const earlyStart = Date.UTC(2026, 0, 1, 1, 2, 0) / 1000;
    expect(formatRecordingTime(3.7, earlyStart)).toBe("01:02:03.7");
  });

  it("handles zero offset", () => {
    const midnight = Date.UTC(2026, 0, 1, 0, 0, 0) / 1000;
    expect(formatRecordingTime(0, midnight)).toBe("00:00:00.0");
  });

  it("handles offsets crossing hour boundaries", () => {
    const start = Date.UTC(2026, 0, 1, 23, 59, 0) / 1000;
    expect(formatRecordingTime(65.3, start)).toBe("00:00:05.3");
  });

  it("rounds sub-second to one decimal", () => {
    const start = Date.UTC(2026, 0, 1, 12, 0, 0) / 1000;
    expect(formatRecordingTime(0.06, start)).toBe("12:00:00.1");
    expect(formatRecordingTime(0.04, start)).toBe("12:00:00.0");
  });
});
