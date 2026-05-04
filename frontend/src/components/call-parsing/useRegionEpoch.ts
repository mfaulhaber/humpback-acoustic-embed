import { useMemo } from "react";

import { useRegionDetectionJobs } from "@/hooks/queries/useCallParsing";

export interface RegionEpoch {
  regionStartTimestamp: number;
  regionEndTimestamp: number;
  toEpoch: (relativeSec: number) => number;
}

export function useRegionEpoch(
  regionDetectionJobId: string | null,
): RegionEpoch | null {
  const { data: regionJobs = [] } = useRegionDetectionJobs();
  return useMemo(() => {
    if (!regionDetectionJobId) return null;
    const rj = regionJobs.find((r) => r.id === regionDetectionJobId);
    if (!rj || rj.start_timestamp == null || rj.end_timestamp == null) {
      return null;
    }
    const start = rj.start_timestamp;
    const end = rj.end_timestamp;
    return {
      regionStartTimestamp: start,
      regionEndTimestamp: end,
      toEpoch: (relativeSec: number) => start + relativeSec,
    };
  }, [regionJobs, regionDetectionJobId]);
}
