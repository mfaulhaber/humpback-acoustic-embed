import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";

import { useRegionEpoch } from "./useRegionEpoch";
import {
  REGION_EPOCH_BASE,
  REGION_EPOCH_END,
} from "./__test-helpers__/regionEpoch";
import type { RegionDetectionJob } from "@/api/types";

const mockUseRegionDetectionJobs = vi.fn();

vi.mock("@/hooks/queries/useCallParsing", () => ({
  useRegionDetectionJobs: () => mockUseRegionDetectionJobs(),
}));

function makeJob(overrides: Partial<RegionDetectionJob> = {}): RegionDetectionJob {
  return {
    id: "rd-1",
    status: "complete",
    audio_file_id: null,
    hydrophone_id: "hp-1",
    start_timestamp: REGION_EPOCH_BASE,
    end_timestamp: REGION_EPOCH_END,
    model_config_id: "mc-1",
    classifier_model_id: "cl-1",
    config_json: null,
    parent_run_id: null,
    error_message: null,
    chunks_total: null,
    chunks_completed: null,
    windows_detected: null,
    trace_row_count: null,
    region_count: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    started_at: null,
    completed_at: null,
    ...overrides,
  };
}

beforeEach(() => {
  mockUseRegionDetectionJobs.mockReset();
});

describe("useRegionEpoch", () => {
  it("returns null when regionDetectionJobId is null", () => {
    mockUseRegionDetectionJobs.mockReturnValue({ data: [makeJob()] });
    const { result } = renderHook(() => useRegionEpoch(null));
    expect(result.current).toBeNull();
  });

  it("returns null when no matching job is in cache", () => {
    mockUseRegionDetectionJobs.mockReturnValue({ data: [makeJob({ id: "other" })] });
    const { result } = renderHook(() => useRegionEpoch("rd-1"));
    expect(result.current).toBeNull();
  });

  it("returns null when matching job has start_timestamp null", () => {
    mockUseRegionDetectionJobs.mockReturnValue({
      data: [makeJob({ start_timestamp: null })],
    });
    const { result } = renderHook(() => useRegionEpoch("rd-1"));
    expect(result.current).toBeNull();
  });

  it("returns null when matching job has end_timestamp null", () => {
    mockUseRegionDetectionJobs.mockReturnValue({
      data: [makeJob({ end_timestamp: null })],
    });
    const { result } = renderHook(() => useRegionEpoch("rd-1"));
    expect(result.current).toBeNull();
  });

  it("returns RegionEpoch with toEpoch translation when fully populated", () => {
    mockUseRegionDetectionJobs.mockReturnValue({ data: [makeJob()] });
    const { result } = renderHook(() => useRegionEpoch("rd-1"));
    expect(result.current).not.toBeNull();
    expect(result.current!.regionStartTimestamp).toBe(REGION_EPOCH_BASE);
    expect(result.current!.regionEndTimestamp).toBe(REGION_EPOCH_END);
    expect(result.current!.toEpoch(0)).toBe(REGION_EPOCH_BASE);
    expect(result.current!.toEpoch(7916.1)).toBe(REGION_EPOCH_BASE + 7916.1);
  });

  it("returns referentially stable result across re-renders with same inputs", () => {
    const jobs = [makeJob()];
    mockUseRegionDetectionJobs.mockReturnValue({ data: jobs });
    const { result, rerender } = renderHook(() => useRegionEpoch("rd-1"));
    const first = result.current;
    rerender();
    expect(result.current).toBe(first);
  });

  it("falls back to empty array when data is undefined", () => {
    mockUseRegionDetectionJobs.mockReturnValue({ data: undefined });
    const { result } = renderHook(() => useRegionEpoch("rd-1"));
    expect(result.current).toBeNull();
  });
});
