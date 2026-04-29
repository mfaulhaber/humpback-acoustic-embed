import { ChevronLeft, ChevronRight } from "lucide-react";
import { formatRecordingTime } from "@/utils/format";

export interface SpanInfo {
  id: number;
  eventId: string;
  regionId: string;
  startTimestamp: number;
  endTimestamp: number;
}

export interface RegionGroup {
  regionId: string;
  startIndex: number;
  endIndex: number;
}

interface SpanNavBarProps {
  spans: SpanInfo[];
  regions: RegionGroup[];
  activeIndex: number;
  activeRegionIndex: number;
  onPrevEvent: () => void;
  onNextEvent: () => void;
  onPrevRegion: () => void;
  onNextRegion: () => void;
}

export function SpanNavBar({
  spans,
  regions,
  activeIndex,
  activeRegionIndex,
  onPrevEvent,
  onNextEvent,
  onPrevRegion,
  onNextRegion,
}: SpanNavBarProps) {
  const span = spans[activeIndex];
  if (!span) return null;

  const region = regions[activeRegionIndex];

  return (
    <div className="flex items-center gap-4 text-sm" data-testid="hmm-span-nav">
      {regions.length > 1 && (
        <div className="flex items-center gap-1" data-testid="hmm-region-nav">
          <button
            className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
            disabled={activeRegionIndex === 0}
            onClick={onPrevRegion}
            title="Previous region"
            data-testid="hmm-region-prev"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <button
            className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
            disabled={activeRegionIndex === regions.length - 1}
            onClick={onNextRegion}
            title="Next region"
            data-testid="hmm-region-next"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
          <span className="text-muted-foreground text-xs">
            Region {activeRegionIndex + 1}/{regions.length}
            {region ? ` · ${region.regionId.slice(0, 8)}` : ""}
          </span>
        </div>
      )}

      <div className="flex items-center gap-1" data-testid="hmm-event-nav">
        <button
          className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
          disabled={activeIndex === 0}
          onClick={onPrevEvent}
          title="Previous event (A)"
          data-testid="hmm-span-prev"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <button
          className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
          disabled={activeIndex === spans.length - 1}
          onClick={onNextEvent}
          title="Next event (D)"
          data-testid="hmm-span-next"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
        <span className="text-muted-foreground" data-testid="hmm-span-label">
          Event {activeIndex + 1}/{spans.length} ·{" "}
          {formatRecordingTime(0, span.startTimestamp)} –{" "}
          {formatRecordingTime(0, span.endTimestamp)}
        </span>
      </div>
    </div>
  );
}
