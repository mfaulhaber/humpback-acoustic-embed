import { useMemo } from "react";
import type {
  Region,
  SegmentationEvent,
  EventBoundaryCorrectionResponse,
} from "@/api/types";
import { cn } from "@/lib/utils";
import { formatTime } from "@/utils/format";

interface RegionTableProps {
  regions: Region[];
  events: SegmentationEvent[];
  corrections: EventBoundaryCorrectionResponse[];
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

export function RegionTable({
  regions,
  events,
  corrections,
  selectedRegionId,
  onSelectRegion,
}: RegionTableProps) {
  const regionInfos = useMemo(
    () => buildRegionInfos(regions, events, corrections),
    [regions, events, corrections],
  );

  if (regions.length === 0) {
    return (
      <div className="border-t px-4 py-3 text-xs text-muted-foreground">
        No regions found
      </div>
    );
  }

  return (
    <div className="border-t">
      <div className="flex items-center gap-2 px-4 py-2 text-xs font-medium text-muted-foreground border-b">
        <span>Regions ({regions.length})</span>
        <span className="ml-auto flex items-center gap-3">
          <span className="flex items-center gap-1">
            <StatusDot status="reviewed" /> reviewed
          </span>
          <span className="flex items-center gap-1">
            <StatusDot status="partial" /> partial
          </span>
          <span className="flex items-center gap-1">
            <StatusDot status="pending" /> pending
          </span>
        </span>
      </div>
      <div className="max-h-48 overflow-y-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-3 py-1.5 text-left font-medium w-8" />
              <th className="px-3 py-1.5 text-left font-medium">Time</th>
              <th className="px-3 py-1.5 text-left font-medium">Duration</th>
              <th className="px-3 py-1.5 text-right font-medium">Events</th>
              <th className="px-3 py-1.5 text-right font-medium">Edited</th>
              <th className="px-3 py-1.5 text-right font-medium">Score</th>
            </tr>
          </thead>
          <tbody>
            {regionInfos.map((info) => {
              const isSelected =
                selectedRegionId === info.region.region_id;
              const duration =
                info.region.end_sec - info.region.start_sec;
              return (
                <tr
                  key={info.region.region_id}
                  className={cn(
                    "border-b cursor-pointer hover:bg-accent/50 transition-colors",
                    isSelected && "bg-accent",
                  )}
                  onClick={() => onSelectRegion(info.region.region_id)}
                >
                  <td className="px-3 py-1.5">
                    <StatusDot status={info.status} />
                  </td>
                  <td className="px-3 py-1.5 font-mono">
                    {formatTime(info.region.start_sec)}
                  </td>
                  <td className="px-3 py-1.5 text-muted-foreground">
                    {duration.toFixed(0)}s
                  </td>
                  <td className="px-3 py-1.5 text-right">
                    {info.eventCount}
                  </td>
                  <td className="px-3 py-1.5 text-right text-muted-foreground">
                    {info.correctionCount > 0 ? info.correctionCount : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right text-muted-foreground">
                    {info.region.max_score.toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatusDot({ status }: { status: CorrectionStatus }) {
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
  corrections: EventBoundaryCorrectionResponse[],
): RegionInfo[] {
  const correctionsByRegion = new Map<string, number>();
  for (const c of corrections) {
    correctionsByRegion.set(
      c.region_id,
      (correctionsByRegion.get(c.region_id) ?? 0) + 1,
    );
  }

  return regions.map((region) => {
    const regionEvents = events.filter(
      (e) => e.region_id === region.region_id,
    );
    const eventCount = regionEvents.length;
    const correctionCount = correctionsByRegion.get(region.region_id) ?? 0;

    let status: CorrectionStatus = "pending";
    if (eventCount > 0 && correctionCount === eventCount) {
      status = "reviewed";
    } else if (correctionCount > 0) {
      status = "partial";
    }

    return { region, eventCount, correctionCount, status };
  });
}
