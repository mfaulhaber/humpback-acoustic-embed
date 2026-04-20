import { useMemo, useRef } from "react";

export interface JobRelativeRegion {
  region_id?: string;
  start_sec: number;
  end_sec: number;
  padded_start_sec: number;
  padded_end_sec: number;
  max_score: number;
}

export interface EpochRegion {
  region_id?: string;
  start_sec: number;
  end_sec: number;
  padded_start_sec: number;
  padded_end_sec: number;
  max_score: number;
}

export function useEpochRegions(regions: JobRelativeRegion[], jobStartEpoch: number): EpochRegion[] {
  const prevRef = useRef<EpochRegion[]>([]);

  return useMemo(() => {
    if (regions.length === 0) {
      if (prevRef.current.length === 0) return prevRef.current;
      prevRef.current = [];
      return prevRef.current;
    }

    const result: EpochRegion[] = regions.map((r) => ({
      region_id: r.region_id,
      start_sec: r.start_sec + jobStartEpoch,
      end_sec: r.end_sec + jobStartEpoch,
      padded_start_sec: r.padded_start_sec + jobStartEpoch,
      padded_end_sec: r.padded_end_sec + jobStartEpoch,
      max_score: r.max_score,
    }));

    prevRef.current = result;
    return result;
  }, [regions, jobStartEpoch]);
}

export function epochToJobRelative(epoch: number, jobStart: number): number {
  return epoch - jobStart;
}
