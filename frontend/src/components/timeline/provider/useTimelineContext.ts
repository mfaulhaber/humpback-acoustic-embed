import { useContext } from "react";
import { TimelineContext } from "./TimelineProvider";
import type { TimelineContextValue } from "./types";

export function useTimelineContext(): TimelineContextValue {
  const ctx = useContext(TimelineContext);
  if (!ctx) {
    throw new Error("useTimelineContext must be used within a TimelineProvider");
  }
  return ctx;
}
