import { useMemo, useRef } from "react";

export interface JobRelativeEvent {
  eventId: string;
  regionId: string;
  startSec: number;
  endSec: number;
  originalStartSec: number;
  originalEndSec: number;
  confidence: number;
  correctionType: "adjust" | "add" | "delete" | null;
  effectiveType: string | null;
  typeSource: "inference" | "correction" | "negative" | null;
}

export interface EpochEvent {
  eventId: string;
  regionId: string;
  startSec: number;
  endSec: number;
  originalStartSec: number;
  originalEndSec: number;
  confidence: number;
  correctionType: "adjust" | "add" | "delete" | null;
  effectiveType: string | null;
  typeSource: "inference" | "correction" | "negative" | null;
}

export function useEpochEvents(events: JobRelativeEvent[], jobStartEpoch: number): EpochEvent[] {
  const prevRef = useRef<EpochEvent[]>([]);

  return useMemo(() => {
    if (events.length === 0) {
      if (prevRef.current.length === 0) return prevRef.current;
      prevRef.current = [];
      return prevRef.current;
    }

    const result: EpochEvent[] = events.map((e) => ({
      ...e,
      startSec: e.startSec + jobStartEpoch,
      endSec: e.endSec + jobStartEpoch,
      originalStartSec: e.originalStartSec + jobStartEpoch,
      originalEndSec: e.originalEndSec + jobStartEpoch,
    }));

    prevRef.current = result;
    return result;
  }, [events, jobStartEpoch]);
}
