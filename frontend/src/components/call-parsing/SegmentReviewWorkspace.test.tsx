import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import {
  REGION_EPOCH_BASE,
  REGION_EPOCH_END,
} from "./__test-helpers__/regionEpoch";
import type { RegionDetectionJob } from "@/api/types";

// Hoisted spies so vi.mock factories can reference them.
const {
  mockUseSegmentationJobs,
  mockUseRegionDetectionJobs,
  mockUseRegionJobRegions,
  mockUseSegmentationJobEvents,
  mockUseEventBoundaryCorrections,
  mockUseSegmentationModels,
  mockUseHydrophones,
  regionAudioTimelineMock,
} = vi.hoisted(() => ({
  mockUseSegmentationJobs: vi.fn(),
  mockUseRegionDetectionJobs: vi.fn(),
  mockUseRegionJobRegions: vi.fn(),
  mockUseSegmentationJobEvents: vi.fn(),
  mockUseEventBoundaryCorrections: vi.fn(),
  mockUseSegmentationModels: vi.fn(),
  mockUseHydrophones: vi.fn(),
  regionAudioTimelineMock: vi.fn(),
}));

vi.mock("@/hooks/queries/useCallParsing", () => ({
  useSegmentationJobs: () => mockUseSegmentationJobs(),
  useRegionDetectionJobs: () => mockUseRegionDetectionJobs(),
  useRegionJobRegions: () => mockUseRegionJobRegions(),
  useSegmentationJobEvents: () => mockUseSegmentationJobEvents(),
  useEventBoundaryCorrections: () => mockUseEventBoundaryCorrections(),
  useUpsertEventBoundaryCorrections: () => ({ mutate: vi.fn(), isPending: false }),
  useCreateSegmentationJob: () => ({ mutate: vi.fn(), isPending: false }),
  useSegmentationModels: () => mockUseSegmentationModels(),
  useQuickRetrain: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("@/hooks/queries/useClassifier", () => ({
  useHydrophones: () => mockUseHydrophones(),
}));

// Replace RegionAudioTimeline with a forwardRef stub that records the props
// it received without rendering its children (rendering them would require a
// TimelineContext).
import { forwardRef } from "react";
vi.mock("./RegionAudioTimeline", () => ({
  RegionAudioTimeline: forwardRef<unknown, Record<string, unknown>>(
    function MockRegionAudioTimeline(props, _ref) {
      regionAudioTimelineMock(props);
      return <div data-testid="region-audio-timeline" />;
    },
  ),
}));

vi.mock("./ReviewToolbar", () => ({
  ReviewToolbar: (props: { jobStartEpoch: number }) => (
    <div
      data-testid="review-toolbar"
      data-job-start-epoch={String(props.jobStartEpoch)}
    />
  ),
}));
vi.mock("./EventDetailPanel", () => ({
  EventDetailPanel: (props: { jobStartEpoch: number }) => (
    <div
      data-testid="event-detail-panel"
      data-job-start-epoch={String(props.jobStartEpoch)}
    />
  ),
}));
vi.mock("./RegionTable", () => ({
  RegionTable: (props: { jobStartEpoch: number }) => (
    <div
      data-testid="region-table"
      data-job-start-epoch={String(props.jobStartEpoch)}
    />
  ),
}));

import { SegmentReviewWorkspace } from "./SegmentReviewWorkspace";

const SEG_JOB_ID = "seg-1";
const REGION_JOB_ID = "rd-1";

function makeRegionJob(
  overrides: Partial<RegionDetectionJob> = {},
): RegionDetectionJob {
  return {
    id: REGION_JOB_ID,
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

function setupHooks({
  regionJob,
  regions = [],
  events = [],
  corrections = [],
}: {
  regionJob: RegionDetectionJob | null;
  regions?: unknown[];
  events?: unknown[];
  corrections?: unknown[];
}) {
  const segJob = {
    id: SEG_JOB_ID,
    status: "complete",
    region_detection_job_id: REGION_JOB_ID,
    segmentation_model_id: "sm-1",
    config_json: null,
    parent_run_id: null,
    event_count: events.length,
    compute_device: null,
    gpu_fallback_reason: null,
    error_message: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    started_at: null,
    completed_at: null,
  };
  mockUseSegmentationJobs.mockReturnValue({ data: [segJob] });
  mockUseRegionDetectionJobs.mockReturnValue({
    data: regionJob ? [regionJob] : [],
  });
  mockUseRegionJobRegions.mockReturnValue({ data: regions });
  mockUseSegmentationJobEvents.mockReturnValue({ data: events });
  mockUseEventBoundaryCorrections.mockReturnValue({ data: corrections });
  mockUseSegmentationModels.mockReturnValue({ data: [], refetch: vi.fn() });
  mockUseHydrophones.mockReturnValue({ data: [] });
}

beforeEach(() => {
  regionAudioTimelineMock.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

const defaultRegions = [
  {
    region_id: "r-1",
    start_sec: 100,
    end_sec: 200,
    padded_start_sec: 90,
    padded_end_sec: 210,
    max_score: 0.8,
    mean_score: 0.5,
    n_windows: 12,
  },
];

describe("SegmentReviewWorkspace — epoch wiring", () => {
  it("renders loading placeholder and does not mount RegionAudioTimeline when region job is missing timestamps", () => {
    setupHooks({
      regionJob: makeRegionJob({ start_timestamp: null, end_timestamp: null }),
      regions: defaultRegions,
    });
    render(<SegmentReviewWorkspace initialJobId={SEG_JOB_ID} />);

    expect(screen.queryByTestId("region-audio-timeline")).toBeNull();
    expect(screen.getByText("Loading region…")).toBeTruthy();
    expect(regionAudioTimelineMock).not.toHaveBeenCalled();
  });

  it("mounts RegionAudioTimeline with epoch-anchored regionEpoch when region job has timestamps", () => {
    setupHooks({
      regionJob: makeRegionJob(),
      regions: defaultRegions,
    });
    render(<SegmentReviewWorkspace initialJobId={SEG_JOB_ID} />);

    expect(screen.getByTestId("region-audio-timeline")).toBeTruthy();
    expect(regionAudioTimelineMock).toHaveBeenCalled();
    const calls = regionAudioTimelineMock.mock.calls;
    const props = calls[calls.length - 1][0] as {
      regionDetectionJobId: string;
      regionEpoch: {
        regionStartTimestamp: number;
        regionEndTimestamp: number;
        toEpoch: (s: number) => number;
      };
    };
    expect(props.regionDetectionJobId).toBe(REGION_JOB_ID);
    expect(props.regionEpoch.regionStartTimestamp).toBe(REGION_EPOCH_BASE);
    expect(props.regionEpoch.regionEndTimestamp).toBe(REGION_EPOCH_END);
    expect(props.regionEpoch.toEpoch(7916.1)).toBe(REGION_EPOCH_BASE + 7916.1);
  });

  it("passes regionEpoch.regionStartTimestamp (not 0) into ReviewToolbar / EventDetailPanel / RegionTable", () => {
    setupHooks({
      regionJob: makeRegionJob(),
      regions: defaultRegions,
    });
    render(<SegmentReviewWorkspace initialJobId={SEG_JOB_ID} />);

    const expected = String(REGION_EPOCH_BASE);
    expect(
      screen.getByTestId("review-toolbar").getAttribute("data-job-start-epoch"),
    ).toBe(expected);
    expect(
      screen
        .getByTestId("event-detail-panel")
        .getAttribute("data-job-start-epoch"),
    ).toBe(expected);
    expect(
      screen.getByTestId("region-table").getAttribute("data-job-start-epoch"),
    ).toBe(expected);
  });
});
