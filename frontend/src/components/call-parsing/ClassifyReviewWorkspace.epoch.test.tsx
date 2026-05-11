import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { forwardRef, type ReactNode } from "react";

import {
  REGION_EPOCH_BASE,
  REGION_EPOCH_END,
} from "./__test-helpers__/regionEpoch";
import type { RegionDetectionJob } from "@/api/types";

const {
  mockUseClassificationJobs,
  mockUseSegmentationJobs,
  mockUseRegionDetectionJobs,
  mockUseRegionJobRegions,
  mockUseTypedEvents,
  mockUseVocalizationCorrections,
  mockUseEventBoundaryCorrections,
  mockUseEventClassifierModels,
  mockUseHydrophones,
  mockTimelineContext,
  regionAudioTimelineMock,
} = vi.hoisted(() => ({
  mockUseClassificationJobs: vi.fn(),
  mockUseSegmentationJobs: vi.fn(),
  mockUseRegionDetectionJobs: vi.fn(),
  mockUseRegionJobRegions: vi.fn(),
  mockUseTypedEvents: vi.fn(),
  mockUseVocalizationCorrections: vi.fn(),
  mockUseEventBoundaryCorrections: vi.fn(),
  mockUseEventClassifierModels: vi.fn(),
  mockUseHydrophones: vi.fn(),
  mockTimelineContext: {
    centerTimestamp: 1_700_000_150,
    zoomLevel: 0,
    isPlaying: false,
    isDraggingTimeline: false,
    speed: 1,
    viewportWidth: 1000,
    viewportHeight: 200,
    playbackEpoch: null,
    viewStart: 1_700_000_135,
    viewEnd: 1_700_000_165,
    pxPerSec: 10,
    viewportSpan: 30,
    activePreset: { key: "30s", span: 30, tileDuration: 5 },
    jobStart: 1_700_000_000,
    jobEnd: 1_700_000_300,
    zoomLevels: [{ key: "30s", span: 30, tileDuration: 5 }],
    pan: vi.fn(),
    setZoomLevel: vi.fn(),
    zoomIn: vi.fn(),
    zoomOut: vi.fn(),
    play: vi.fn(),
    pause: vi.fn(),
    togglePlay: vi.fn(),
    beginDragPan: vi.fn(),
    updateDragPan: vi.fn(),
    endDragPan: vi.fn(),
    seekTo: vi.fn(),
    setSpeed: vi.fn(),
    setViewportDimensions: vi.fn(),
  },
  regionAudioTimelineMock: vi.fn(),
}));

vi.mock("@/hooks/queries/useCallParsing", () => ({
  useClassificationJobs: () => mockUseClassificationJobs(),
  useSegmentationJobs: () => mockUseSegmentationJobs(),
  useRegionDetectionJobs: () => mockUseRegionDetectionJobs(),
  useRegionJobRegions: () => mockUseRegionJobRegions(),
  useTypedEvents: () => mockUseTypedEvents(),
  useVocalizationCorrections: () => mockUseVocalizationCorrections(),
  useUpsertVocalizationCorrections: () => ({ mutate: vi.fn(), isPending: false }),
  useEventBoundaryCorrections: () => mockUseEventBoundaryCorrections(),
  useUpsertEventBoundaryCorrections: () => ({ mutate: vi.fn(), isPending: false }),
  useEventClassifierModels: () => mockUseEventClassifierModels(),
  useCreateClassifierTrainingJob: () => ({ mutate: vi.fn(), isPending: false }),
  useCreateClassificationJob: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("@/hooks/queries/useClassifier", () => ({
  useHydrophones: () => mockUseHydrophones(),
}));

vi.mock("@/components/timeline/provider/useTimelineContext", () => ({
  useTimelineContext: () => mockTimelineContext,
}));

vi.mock("@/components/timeline/spectrogram/Spectrogram", () => ({
  Spectrogram: ({ children }: { children?: ReactNode }) => (
    <div data-testid="spectrogram">{children}</div>
  ),
}));

vi.mock("@/components/timeline/overlays/RegionBoundaryMarkers", () => ({
  RegionBoundaryMarkers: () => <div data-testid="region-boundary-markers" />,
}));

vi.mock("@/components/timeline/overlays/EventBarOverlay", () => ({
  EventBarOverlay: ({
    onSelectEvent,
    selectedEventId,
  }: {
    onSelectEvent: (eventId: string | null) => void;
    selectedEventId: string | null;
  }) => (
    <button
      type="button"
      data-testid="blank-event-timeline"
      data-selected-event-id={selectedEventId ?? ""}
      onClick={() => onSelectEvent(null)}
    >
      blank
    </button>
  ),
  APPROVED_RING_COLOR: "hsl(85, 80%, 45%)",
  CORRECTED_RING_COLOR: "rgb(74, 222, 128)",
}));

vi.mock("@/components/timeline/controls/ZoomSelector", () => ({
  ZoomSelector: () => <div data-testid="zoom-selector" />,
}));

vi.mock("./RegionAudioTimeline", () => ({
  RegionAudioTimeline: forwardRef<unknown, Record<string, unknown>>(
    function MockRegionAudioTimeline(props, _ref) {
      regionAudioTimelineMock(props);
      return (
        <div data-testid="region-audio-timeline">
          {props.children as ReactNode}
        </div>
      );
    },
  ),
}));

vi.mock("./TypePalette", () => ({
  TypePalette: () => <div data-testid="type-palette" />,
  typeColor: () => "#000000",
}));

vi.mock("./ClassifyDetailPanel", () => ({
  ClassifyDetailPanel: (props: {
    event: { eventId: string } | null;
    jobStartEpoch: number;
  }) => (
    <div
      data-testid="classify-detail-panel"
      data-job-start-epoch={String(props.jobStartEpoch)}
      data-event-id={props.event?.eventId ?? ""}
    />
  ),
}));

import { ClassifyReviewWorkspace } from "./ClassifyReviewWorkspace";

const CLASSIFY_JOB_ID = "cls-1";
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
  typedEvents = [],
}: {
  regionJob: RegionDetectionJob | null;
  regions?: unknown[];
  typedEvents?: unknown[];
}) {
  const classifyJob = {
    id: CLASSIFY_JOB_ID,
    status: "complete",
    event_segmentation_job_id: SEG_JOB_ID,
    vocalization_model_id: "vm-1",
    typed_event_count: typedEvents.length,
    compute_device: null,
    gpu_fallback_reason: null,
    parent_run_id: null,
    error_message: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    started_at: null,
    completed_at: null,
  };
  const segJob = {
    id: SEG_JOB_ID,
    status: "complete",
    region_detection_job_id: REGION_JOB_ID,
    segmentation_model_id: "sm-1",
    config_json: null,
    parent_run_id: null,
    event_count: typedEvents.length,
    compute_device: null,
    gpu_fallback_reason: null,
    error_message: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    started_at: null,
    completed_at: null,
  };
  mockUseClassificationJobs.mockReturnValue({ data: [classifyJob] });
  mockUseSegmentationJobs.mockReturnValue({ data: [segJob] });
  mockUseRegionDetectionJobs.mockReturnValue({
    data: regionJob ? [regionJob] : [],
  });
  mockUseRegionJobRegions.mockReturnValue({ data: regions });
  mockUseTypedEvents.mockReturnValue({ data: typedEvents });
  mockUseVocalizationCorrections.mockReturnValue({ data: [] });
  mockUseEventBoundaryCorrections.mockReturnValue({ data: [] });
  mockUseEventClassifierModels.mockReturnValue({ data: [], refetch: vi.fn() });
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

const defaultTypedEvents = [
  {
    event_id: "ev-1",
    region_id: "r-1",
    start_sec: 110,
    end_sec: 115,
    type_name: "Pop",
    score: 0.9,
    above_threshold: true,
  },
];

describe("ClassifyReviewWorkspace — epoch wiring", () => {
  it("renders loading placeholder and does not mount RegionAudioTimeline when region job is missing timestamps", () => {
    setupHooks({
      regionJob: makeRegionJob({ start_timestamp: null, end_timestamp: null }),
      regions: defaultRegions,
      typedEvents: defaultTypedEvents,
    });
    render(<ClassifyReviewWorkspace initialJobId={CLASSIFY_JOB_ID} />);

    expect(screen.queryByTestId("region-audio-timeline")).toBeNull();
    expect(screen.getByText("Loading region…")).toBeTruthy();
    expect(regionAudioTimelineMock).not.toHaveBeenCalled();
  });

  it("mounts RegionAudioTimeline with epoch-anchored regionEpoch when region job has timestamps", () => {
    setupHooks({
      regionJob: makeRegionJob(),
      regions: defaultRegions,
      typedEvents: defaultTypedEvents,
    });
    render(<ClassifyReviewWorkspace initialJobId={CLASSIFY_JOB_ID} />);

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

  it("passes regionEpoch.regionStartTimestamp (not 0) into ClassifyDetailPanel", () => {
    setupHooks({
      regionJob: makeRegionJob(),
      regions: defaultRegions,
      typedEvents: defaultTypedEvents,
    });
    render(<ClassifyReviewWorkspace initialJobId={CLASSIFY_JOB_ID} />);

    expect(
      screen
        .getByTestId("classify-detail-panel")
        .getAttribute("data-job-start-epoch"),
    ).toBe(String(REGION_EPOCH_BASE));
  });

  it("keeps selection cleared after a blank timeline click", async () => {
    setupHooks({
      regionJob: makeRegionJob(),
      regions: defaultRegions,
      typedEvents: defaultTypedEvents,
    });
    render(<ClassifyReviewWorkspace initialJobId={CLASSIFY_JOB_ID} />);

    await waitFor(() => {
      expect(
        screen.getByTestId("classify-detail-panel").getAttribute("data-event-id"),
      ).toBe("ev-1");
    });

    fireEvent.click(screen.getByTestId("blank-event-timeline"));

    await waitFor(() => {
      expect(
        screen.getByTestId("classify-detail-panel").getAttribute("data-event-id"),
      ).toBe("");
    });
  });
});
