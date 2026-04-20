import { describe, it, expect } from "vitest";
import { renderHook } from "@testing-library/react";
import { useEpochRegions, epochToJobRelative } from "./useEpochRegions";
import { useEpochEvents } from "./useEpochEvents";
import type { JobRelativeRegion } from "./useEpochRegions";
import type { JobRelativeEvent } from "./useEpochEvents";

describe("useEpochRegions", () => {
  it("adds jobStartEpoch offset to region timestamps", () => {
    const regions: JobRelativeRegion[] = [
      { region_id: "r1", start_sec: 10, end_sec: 20, padded_start_sec: 8, padded_end_sec: 22, max_score: 0.9 },
    ];
    const jobStart = 1000;

    const { result } = renderHook(() => useEpochRegions(regions, jobStart));

    expect(result.current[0].start_sec).toBe(1010);
    expect(result.current[0].end_sec).toBe(1020);
    expect(result.current[0].padded_start_sec).toBe(1008);
    expect(result.current[0].padded_end_sec).toBe(1022);
  });

  it("round-trips: add offset then subtract gives original", () => {
    const regions: JobRelativeRegion[] = [
      { region_id: "r1", start_sec: 42.5, end_sec: 55.3, padded_start_sec: 40, padded_end_sec: 58, max_score: 0.7 },
    ];
    const jobStart = 5000;

    const { result } = renderHook(() => useEpochRegions(regions, jobStart));

    expect(epochToJobRelative(result.current[0].start_sec, jobStart)).toBeCloseTo(42.5);
    expect(epochToJobRelative(result.current[0].end_sec, jobStart)).toBeCloseTo(55.3);
  });

  it("returns empty array with stable reference for empty input", () => {
    const { result, rerender } = renderHook(
      ({ regions, jobStart }) => useEpochRegions(regions, jobStart),
      { initialProps: { regions: [] as JobRelativeRegion[], jobStart: 1000 } },
    );

    const first = result.current;
    expect(first).toEqual([]);

    rerender({ regions: [], jobStart: 1000 });
    expect(result.current).toBe(first);
  });
});

describe("useEpochEvents", () => {
  it("adds offset to event timestamps while preserving other fields", () => {
    const events: JobRelativeEvent[] = [
      {
        eventId: "e1",
        regionId: "r1",
        startSec: 5,
        endSec: 8,
        originalStartSec: 5,
        originalEndSec: 8,
        confidence: 0.95,
        correctionType: null,
        effectiveType: "song",
        typeSource: "inference",
      },
    ];
    const jobStart = 2000;

    const { result } = renderHook(() => useEpochEvents(events, jobStart));

    expect(result.current[0].startSec).toBe(2005);
    expect(result.current[0].endSec).toBe(2008);
    expect(result.current[0].originalStartSec).toBe(2005);
    expect(result.current[0].originalEndSec).toBe(2008);
    expect(result.current[0].confidence).toBe(0.95);
    expect(result.current[0].effectiveType).toBe("song");
  });

  it("handles mixed correction types", () => {
    const events: JobRelativeEvent[] = [
      {
        eventId: "e1", regionId: "r1", startSec: 5, endSec: 8,
        originalStartSec: 5, originalEndSec: 8, confidence: 0.9,
        correctionType: "adjust", effectiveType: "call", typeSource: "correction",
      },
      {
        eventId: "e2", regionId: "r1", startSec: 10, endSec: 12,
        originalStartSec: 10, originalEndSec: 12, confidence: 0,
        correctionType: "add", effectiveType: null, typeSource: null,
      },
      {
        eventId: "e3", regionId: "r1", startSec: 15, endSec: 18,
        originalStartSec: 15, originalEndSec: 18, confidence: 0.8,
        correctionType: "delete", effectiveType: "song", typeSource: "negative",
      },
    ];
    const jobStart = 3000;

    const { result } = renderHook(() => useEpochEvents(events, jobStart));

    expect(result.current).toHaveLength(3);
    expect(result.current[0].correctionType).toBe("adjust");
    expect(result.current[1].correctionType).toBe("add");
    expect(result.current[2].correctionType).toBe("delete");
    expect(result.current[0].startSec).toBe(3005);
    expect(result.current[1].startSec).toBe(3010);
    expect(result.current[2].startSec).toBe(3015);
  });

  it("returns empty array with stable reference for empty input", () => {
    const { result, rerender } = renderHook(
      ({ events, jobStart }) => useEpochEvents(events, jobStart),
      { initialProps: { events: [] as JobRelativeEvent[], jobStart: 1000 } },
    );

    const first = result.current;
    expect(first).toEqual([]);

    rerender({ events: [], jobStart: 1000 });
    expect(result.current).toBe(first);
  });
});

describe("epochToJobRelative utility", () => {
  it("subtracts job start from epoch", () => {
    expect(epochToJobRelative(5042.5, 5000)).toBeCloseTo(42.5);
  });

  it("returns 0 when epoch equals jobStart", () => {
    expect(epochToJobRelative(1000, 1000)).toBe(0);
  });
});
