import { useState, useMemo, useCallback } from "react";
import { ArrowUp, ArrowDown, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSegmentationJobEvents } from "@/hooks/queries/useCallParsing";
import type { SegmentationEvent } from "@/api/types";

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

interface Stats {
  count: number;
  meanDuration: number;
  medianDuration: number;
  minConfidence: number;
  maxConfidence: number;
}

function computeStats(events: SegmentationEvent[]): Stats {
  if (events.length === 0) {
    return { count: 0, meanDuration: 0, medianDuration: 0, minConfidence: 0, maxConfidence: 0 };
  }
  const durations = events.map((e) => e.end_sec - e.start_sec);
  const confs = events.map((e) => e.segmentation_confidence);
  return {
    count: events.length,
    meanDuration: durations.reduce((a, b) => a + b, 0) / durations.length,
    medianDuration: median(durations),
    minConfidence: Math.min(...confs),
    maxConfidence: Math.max(...confs),
  };
}

type EventSortKey = "start" | "duration" | "confidence";
type SortDir = "asc" | "desc";

interface SegmentJobDetailProps {
  jobId: string;
}

export function SegmentJobDetail({ jobId }: SegmentJobDetailProps) {
  const { data: events, isLoading } = useSegmentationJobEvents(jobId);

  const [sortKey, setSortKey] = useState<EventSortKey>("start");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 20;

  const toggleSort = useCallback(
    (key: EventSortKey) => {
      if (sortKey === key) {
        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      } else {
        setSortKey(key);
        setSortDir("asc");
      }
      setPage(1);
    },
    [sortKey],
  );

  const stats = useMemo(() => computeStats(events ?? []), [events]);

  const sortedEvents = useMemo(() => {
    if (!events) return [];
    const sorted = [...events];
    const dir = sortDir === "asc" ? 1 : -1;
    sorted.sort((a, b) => {
      switch (sortKey) {
        case "start":
          return dir * (a.start_sec - b.start_sec);
        case "duration":
          return (
            dir *
            (a.end_sec - a.start_sec - (b.end_sec - b.start_sec))
          );
        case "confidence":
          return (
            dir *
            (a.segmentation_confidence - b.segmentation_confidence)
          );
        default:
          return 0;
      }
    });
    return sorted;
  }, [events, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(sortedEvents.length / PAGE_SIZE));
  const effectivePage = Math.min(page, totalPages);
  const paginatedEvents = sortedEvents.slice(
    (effectivePage - 1) * PAGE_SIZE,
    effectivePage * PAGE_SIZE,
  );

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-4 px-2 text-xs text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Loading events…
      </div>
    );
  }

  if (!events || events.length === 0) {
    return (
      <div className="py-4 px-2 text-xs text-muted-foreground">
        No events found.
      </div>
    );
  }

  const SortIcon = ({ col }: { col: EventSortKey }) => {
    if (sortKey !== col) return null;
    return sortDir === "asc" ? (
      <ArrowUp className="h-3 w-3" />
    ) : (
      <ArrowDown className="h-3 w-3" />
    );
  };

  const sortableHeader = (label: string, col: EventSortKey) => (
    <th
      className="px-2 py-1.5 text-left font-medium cursor-pointer select-none hover:bg-muted/80"
      onClick={() => toggleSort(col)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <SortIcon col={col} />
      </span>
    </th>
  );

  return (
    <div className="bg-muted/30 rounded-md p-3 my-2 space-y-3">
      {/* Summary stats */}
      <div className="grid grid-cols-5 gap-3 text-center">
        <StatCard label="Events" value={String(stats.count)} />
        <StatCard label="Mean Duration" value={`${stats.meanDuration.toFixed(2)}s`} />
        <StatCard label="Median Duration" value={`${stats.medianDuration.toFixed(2)}s`} />
        <StatCard label="Min Confidence" value={stats.minConfidence.toFixed(2)} />
        <StatCard label="Max Confidence" value={stats.maxConfidence.toFixed(2)} />
      </div>

      {/* Events table */}
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="px-2 py-1.5 text-left font-medium">Region</th>
            {sortableHeader("Start", "start")}
            <th className="px-2 py-1.5 text-left font-medium">End</th>
            {sortableHeader("Duration", "duration")}
            {sortableHeader("Confidence", "confidence")}
          </tr>
        </thead>
        <tbody>
          {paginatedEvents.map((e) => (
            <tr key={e.event_id} className="border-b hover:bg-muted/20">
              <td className="px-2 py-1.5 font-mono text-muted-foreground">
                {e.region_id.slice(0, 4)}…
              </td>
              <td className="px-2 py-1.5">{e.start_sec.toFixed(2)}s</td>
              <td className="px-2 py-1.5">{e.end_sec.toFixed(2)}s</td>
              <td className="px-2 py-1.5">
                {(e.end_sec - e.start_sec).toFixed(2)}s
              </td>
              <td className="px-2 py-1.5">
                {e.segmentation_confidence.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="h-6 px-2"
            disabled={effectivePage <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            <ChevronLeft className="h-3 w-3" />
          </Button>
          <span className="text-xs text-muted-foreground">
            {(effectivePage - 1) * PAGE_SIZE + 1}–
            {Math.min(effectivePage * PAGE_SIZE, sortedEvents.length)} of{" "}
            {sortedEvents.length}
          </span>
          <Button
            variant="outline"
            size="sm"
            className="h-6 px-2"
            disabled={effectivePage >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          >
            <ChevronRight className="h-3 w-3" />
          </Button>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] text-muted-foreground">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}
