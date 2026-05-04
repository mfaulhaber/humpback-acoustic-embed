import { forwardRef, useCallback, type ReactNode } from "react";

import { regionAudioSliceUrl } from "@/api/client";
import { TimelineProvider } from "@/components/timeline/provider/TimelineProvider";
import type {
  TimelinePlaybackHandle,
  ZoomPreset,
} from "@/components/timeline/provider/types";

import type { RegionEpoch } from "./useRegionEpoch";

export interface RegionAudioTimelineProps {
  regionDetectionJobId: string;
  regionEpoch: RegionEpoch;
  zoomLevels: ZoomPreset[];
  defaultZoom?: string;
  disableKeyboardShortcuts?: boolean;
  scrollOnPlayback?: boolean;
  onZoomChange?: (zoomKey: string) => void;
  onPlayStateChange?: (playing: boolean) => void;
  resetKey?: string;
  children: ReactNode;
}

export const RegionAudioTimeline = forwardRef<
  TimelinePlaybackHandle,
  RegionAudioTimelineProps
>(function RegionAudioTimeline(
  {
    regionDetectionJobId,
    regionEpoch,
    zoomLevels,
    defaultZoom,
    disableKeyboardShortcuts,
    scrollOnPlayback,
    onZoomChange,
    onPlayStateChange,
    resetKey,
    children,
  },
  ref,
) {
  const audioUrlBuilder = useCallback(
    (startEpoch: number, durationSec: number) =>
      regionAudioSliceUrl(regionDetectionJobId, startEpoch, durationSec),
    [regionDetectionJobId],
  );

  return (
    <TimelineProvider
      ref={ref}
      key={resetKey}
      jobStart={regionEpoch.regionStartTimestamp}
      jobEnd={regionEpoch.regionEndTimestamp}
      zoomLevels={zoomLevels}
      defaultZoom={defaultZoom}
      playback="slice"
      audioUrlBuilder={audioUrlBuilder}
      disableKeyboardShortcuts={disableKeyboardShortcuts}
      scrollOnPlayback={scrollOnPlayback}
      onZoomChange={onZoomChange}
      onPlayStateChange={onPlayStateChange}
    >
      {children}
    </TimelineProvider>
  );
});
