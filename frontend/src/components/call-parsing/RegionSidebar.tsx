import { useMemo } from "react";
import type {
  Region,
  SegmentationEvent,
  BoundaryCorrectionResponse,
} from "@/api/types";
import { cn } from "@/lib/utils";
import { formatTime } from "@/utils/format";

interface RegionSidebarProps {
  regions: Region[];
  events: SegmentationEvent[];
  corrections: BoundaryCorrectionResponse[];
  selectedRegionId: string | null;
  onSelectRegion: (regionId: string) => void;
}

type CorrectionStatus = "pending" | "partial" | "reviewed";

interface RegionInfo {
  region: Region;
  eventCount: number;
  correctionCount: number;
  status: CorrectionStatus;
}

export function RegionSidebar({
  regions,
  events,
  corrections,
  selectedRegionId,
  onSelectRegion,
}: RegionSidebarProps) {
  const regionInfos = useMemo(
    () => buildRegionInfos(regions, events, corrections),
    [regions, events, corrections],
  );

  if (regions.length === 0) {
    return (
      <div className="w-48 shrink-0 rounded-md border p-3 text-xs text-muted-foreground">
        No regions found
      </div>
    );
  }

  return (
    <div className="flex w-48 shrink-0 flex-col gap-1 overflow-y-auto rounded-md border p-2">
      <div className="px-2 pb-1 text-xs font-medium text-muted-foreground">
        Regions ({regions.length})
      </div>
      {regionInfos.map((info) => (
        <button
          key={info.region.region_id}
          className={cn(
            "rounded-md px-2 py-1.5 text-left text-xs transition-colors hover:bg-accent",
            selectedRegionId === info.region.region_id && "bg-accent",
          )}
          onClick={() => onSelectRegion(info.region.region_id)}
        >
          <div className="flex items-center gap-1.5">
            <StatusIndicator status={info.status} />
            <span className="font-mono">
              {formatTime(info.region.start_sec)}
            </span>
          </div>
          <div className="mt-0.5 pl-4 text-muted-foreground">
            {info.eventCount} event{info.eventCount !== 1 ? "s" : ""}
            {info.correctionCount > 0 && (
              <span className="ml-1">· {info.correctionCount} edited</span>
            )}
          </div>
        </button>
      ))}
      <div className="mt-2 border-t px-2 pt-2 text-xs text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <StatusIndicator status="reviewed" /> reviewed
        </div>
        <div className="flex items-center gap-1.5">
          <StatusIndicator status="partial" /> partial
        </div>
        <div className="flex items-center gap-1.5">
          <StatusIndicator status="pending" /> pending
        </div>
      </div>
    </div>
  );
}

function StatusIndicator({ status }: { status: CorrectionStatus }) {
  const cls = {
    reviewed: "bg-green-500",
    partial: "bg-yellow-500",
    pending: "bg-muted-foreground/30",
  }[status];
  return <span className={cn("inline-block h-2 w-2 rounded-full", cls)} />;
}

function buildRegionInfos(
  regions: Region[],
  events: SegmentationEvent[],
  corrections: BoundaryCorrectionResponse[],
): RegionInfo[] {
  // Build sets for lookup
  const correctedEventIds = new Set(corrections.map((c) => c.event_id));

  return regions.map((region) => {
    const regionEvents = events.filter(
      (e) => e.region_id === region.region_id,
    );
    const eventCount = regionEvents.length;
    const correctionCount = regionEvents.filter((e) =>
      correctedEventIds.has(e.event_id),
    ).length;

    let status: CorrectionStatus = "pending";
    if (eventCount > 0 && correctionCount === eventCount) {
      status = "reviewed";
    } else if (correctionCount > 0) {
      status = "partial";
    }

    return { region, eventCount, correctionCount, status };
  });
}

