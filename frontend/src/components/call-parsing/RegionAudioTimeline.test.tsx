import { describe, it, expect, vi, beforeAll, afterAll, afterEach } from "vitest";
import { render, act } from "@testing-library/react";
import { createRef } from "react";

import { RegionAudioTimeline } from "./RegionAudioTimeline";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import type { TimelinePlaybackHandle } from "@/components/timeline/provider/types";
import { REVIEW_ZOOM } from "@/components/timeline/provider/types";
import {
  REGION_EPOCH_BASE,
  REGION_EPOCH_END,
} from "./__test-helpers__/regionEpoch";
import type { RegionEpoch } from "./useRegionEpoch";

beforeAll(() => {
  vi.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockReturnValue(undefined);
});

afterEach(() => {
  vi.clearAllMocks();
});

afterAll(() => {
  vi.restoreAllMocks();
});

const regionEpoch: RegionEpoch = {
  regionStartTimestamp: REGION_EPOCH_BASE,
  regionEndTimestamp: REGION_EPOCH_END,
  toEpoch: (s: number) => REGION_EPOCH_BASE + s,
};

function ContextSpy({ onContext }: { onContext: (ctx: ReturnType<typeof useTimelineContext>) => void }) {
  const ctx = useTimelineContext();
  onContext(ctx);
  return null;
}

describe("RegionAudioTimeline", () => {
  it("mounts inner TimelineProvider with epoch jobStart and jobEnd", () => {
    let captured: ReturnType<typeof useTimelineContext> | null = null;
    render(
      <RegionAudioTimeline
        regionDetectionJobId="rd-1"
        regionEpoch={regionEpoch}
        zoomLevels={REVIEW_ZOOM}
      >
        <ContextSpy onContext={(c) => (captured = c)} />
      </RegionAudioTimeline>,
    );
    expect(captured!.jobStart).toBe(REGION_EPOCH_BASE);
    expect(captured!.jobEnd).toBe(REGION_EPOCH_END);
  });

  it("forwards TimelinePlaybackHandle ref so parents can drive playback", () => {
    const ref = createRef<TimelinePlaybackHandle>();
    render(
      <RegionAudioTimeline
        ref={ref}
        regionDetectionJobId="rd-1"
        regionEpoch={regionEpoch}
        zoomLevels={REVIEW_ZOOM}
      >
        <div />
      </RegionAudioTimeline>,
    );
    expect(typeof ref.current?.play).toBe("function");
    expect(typeof ref.current?.pause).toBe("function");
    expect(typeof ref.current?.seekTo).toBe("function");
  });

  it("audio playback hits regionAudioSliceUrl with absolute start_timestamp", () => {
    const playSpy = vi.spyOn(window.HTMLMediaElement.prototype, "play");
    const srcSetter = vi.fn();
    Object.defineProperty(window.HTMLMediaElement.prototype, "src", {
      configurable: true,
      set: srcSetter,
      get: () => "",
    });

    const ref = createRef<TimelinePlaybackHandle>();
    render(
      <RegionAudioTimeline
        ref={ref}
        regionDetectionJobId="rd-xyz"
        regionEpoch={regionEpoch}
        zoomLevels={REVIEW_ZOOM}
      >
        <div />
      </RegionAudioTimeline>,
    );

    const eventStart = REGION_EPOCH_BASE + 7916.1;
    act(() => {
      ref.current!.play(eventStart, 30);
    });

    expect(playSpy).toHaveBeenCalled();
    const calls = srcSetter.mock.calls;
    const lastSrc = String(calls[calls.length - 1]?.[0] ?? "");
    expect(lastSrc).toContain("/call-parsing/region-jobs/rd-xyz/audio-slice");
    expect(lastSrc).toContain(`start_timestamp=${eventStart}`);
    expect(lastSrc).toContain("duration_sec=30");

    // Property is restored by jsdom between tests; redefine guard not needed.
  });

  it("respects resetKey by remounting the inner provider", () => {
    let firstJobStart = 0;
    let secondJobStart = 0;
    const { rerender } = render(
      <RegionAudioTimeline
        regionDetectionJobId="rd-1"
        regionEpoch={regionEpoch}
        zoomLevels={REVIEW_ZOOM}
        resetKey="A"
      >
        <ContextSpy onContext={(c) => (firstJobStart = c.jobStart)} />
      </RegionAudioTimeline>,
    );

    const otherEpoch: RegionEpoch = {
      regionStartTimestamp: REGION_EPOCH_BASE + 100,
      regionEndTimestamp: REGION_EPOCH_END + 100,
      toEpoch: (s: number) => REGION_EPOCH_BASE + 100 + s,
    };
    rerender(
      <RegionAudioTimeline
        regionDetectionJobId="rd-1"
        regionEpoch={otherEpoch}
        zoomLevels={REVIEW_ZOOM}
        resetKey="B"
      >
        <ContextSpy onContext={(c) => (secondJobStart = c.jobStart)} />
      </RegionAudioTimeline>,
    );

    expect(firstJobStart).toBe(REGION_EPOCH_BASE);
    expect(secondJobStart).toBe(REGION_EPOCH_BASE + 100);
  });
});
