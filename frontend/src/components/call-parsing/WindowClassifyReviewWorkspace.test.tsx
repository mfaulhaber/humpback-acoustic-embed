import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { forwardRef } from "react";

import {
  REGION_EPOCH_BASE,
  REGION_EPOCH_END,
} from "./__test-helpers__/regionEpoch";
import type { RegionDetectionJob } from "@/api/types";

const {
  mockUseWindowClassificationJobs,
  mockUseSegmentationJobs,
  mockUseRegionDetectionJobs,
  mockUseRegionJobRegions,
  mockUseSegmentationJobEvents,
  mockUseWindowScores,
  mockUseVocalizationCorrections,
  mockUseEventBoundaryCorrections,
  mockUseVocClassifierModel,
  mockUseHydrophones,
  regionAudioTimelineMock,
} = vi.hoisted(() => ({
  mockUseWindowClassificationJobs: vi.fn(),
  mockUseSegmentationJobs: vi.fn(),
  mockUseRegionDetectionJobs: vi.fn(),
  mockUseRegionJobRegions: vi.fn(),
  mockUseSegmentationJobEvents: vi.fn(),
  mockUseWindowScores: vi.fn(),
  mockUseVocalizationCorrections: vi.fn(),
  mockUseEventBoundaryCorrections: vi.fn(),
  mockUseVocClassifierModel: vi.fn(),
  mockUseHydrophones: vi.fn(),
  regionAudioTimelineMock: vi.fn(),
}));

vi.mock("@/hooks/queries/useCallParsing", () => ({
  useWindowClassificationJobs: () => mockUseWindowClassificationJobs(),
  useSegmentationJobs: () => mockUseSegmentationJobs(),
  useRegionDetectionJobs: () => mockUseRegionDetectionJobs(),
  useRegionJobRegions: () => mockUseRegionJobRegions(),
  useSegmentationJobEvents: () => mockUseSegmentationJobEvents(),
  useWindowScores: () => mockUseWindowScores(),
  useVocalizationCorrections: () => mockUseVocalizationCorrections(),
  useUpsertVocalizationCorrections: () => ({ mutate: vi.fn(), isPending: false }),
  useEventBoundaryCorrections: () => mockUseEventBoundaryCorrections(),
  useUpsertEventBoundaryCorrections: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("@/hooks/queries/useVocalization", () => ({
  useVocClassifierModel: () => mockUseVocClassifierModel(),
}));

vi.mock("@/hooks/queries/useClassifier", () => ({
  useHydrophones: () => mockUseHydrophones(),
}));

vi.mock("./RegionAudioTimeline", () => ({
  RegionAudioTimeline: forwardRef<unknown, Record<string, unknown>>(
    function MockRegionAudioTimeline(props, _ref) {
      regionAudioTimelineMock(props);
      return <div data-testid="region-audio-timeline" />;
    },
  ),
}));

vi.mock("./TypePalette", () => ({
  TypePalette: () => <div data-testid="type-palette" />,
  typeColor: () => "#000000",
}));

import { WindowClassifyReviewWorkspace } from "./WindowClassifyReviewWorkspace";

const WC_JOB_ID = "wc-1";
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
}: {
  regionJob: RegionDetectionJob | null;
  regions?: unknown[];
  events?: unknown[];
}) {
  const wcJob = {
    id: WC_JOB_ID,
    status: "complete",
    region_detection_job_id: REGION_JOB_ID,
    vocalization_model_id: "vm-1",
    config_json: null,
    window_count: 100,
    vocabulary_snapshot: JSON.stringify(["Pop"]),
    error_message: null,
    started_at: null,
    completed_at: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
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
  mockUseWindowClassificationJobs.mockReturnValue({ data: [wcJob] });
  mockUseSegmentationJobs.mockReturnValue({ data: [segJob] });
  mockUseRegionDetectionJobs.mockReturnValue({
    data: regionJob ? [regionJob] : [],
  });
  mockUseRegionJobRegions.mockReturnValue({ data: regions });
  mockUseSegmentationJobEvents.mockReturnValue({ data: events });
  mockUseWindowScores.mockReturnValue({ data: [] });
  mockUseVocalizationCorrections.mockReturnValue({ data: [] });
  mockUseEventBoundaryCorrections.mockReturnValue({ data: [] });
  mockUseVocClassifierModel.mockReturnValue({
    data: { vocabulary_snapshot: ["Pop"], per_class_thresholds: { Pop: 0.5 } },
  });
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

const defaultEvents = [
  {
    event_id: "ev-1",
    region_id: "r-1",
    start_sec: 110,
    end_sec: 115,
    center_sec: 112.5,
    segmentation_confidence: 0.7,
  },
];

describe("WindowClassifyReviewWorkspace — epoch wiring", () => {
  it("renders loading placeholder and does not mount RegionAudioTimeline when region job is missing timestamps", () => {
    setupHooks({
      regionJob: makeRegionJob({ start_timestamp: null, end_timestamp: null }),
      regions: defaultRegions,
      events: defaultEvents,
    });
    render(<WindowClassifyReviewWorkspace initialJobId={WC_JOB_ID} />);

    expect(screen.queryByTestId("region-audio-timeline")).toBeNull();
    expect(screen.getByText("Loading region…")).toBeTruthy();
    expect(regionAudioTimelineMock).not.toHaveBeenCalled();
  });

  it("mounts RegionAudioTimeline with epoch-anchored regionEpoch when region job has timestamps", () => {
    setupHooks({
      regionJob: makeRegionJob(),
      regions: defaultRegions,
      events: defaultEvents,
    });
    render(<WindowClassifyReviewWorkspace initialJobId={WC_JOB_ID} />);

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
    expect(props.regionEpoch.toEpoch(7950)).toBe(REGION_EPOCH_BASE + 7950);
  });

  it("renders the inline EventDetailPanel with absolute UTC times derived from regionStartTimestamp", () => {
    // Use a non-midnight base so the rendered time-of-day differs from what
    // formatRecordingTime would have produced under the previous
    // jobStartEpoch=0 bug. 2021-10-31T13:45:00Z = 1635687900.
    const NON_MIDNIGHT_BASE = 1635687900;
    setupHooks({
      regionJob: makeRegionJob({
        start_timestamp: NON_MIDNIGHT_BASE,
        end_timestamp: NON_MIDNIGHT_BASE + 3600,
      }),
      regions: defaultRegions,
      events: defaultEvents,
    });
    render(<WindowClassifyReviewWorkspace initialJobId={WC_JOB_ID} />);

    // formatRecordingTime(110, NON_MIDNIGHT_BASE) → 13:46:50.0;
    // formatRecordingTime(115, NON_MIDNIGHT_BASE) → 13:46:55.0.
    // Under the bug (jobStartEpoch=0) these would have been
    // 00:01:50.0 / 00:01:55.0.
    const panelText = document.body.textContent ?? "";
    expect(panelText).toContain("13:46:50.0");
    expect(panelText).toContain("13:46:55.0");
  });
});
