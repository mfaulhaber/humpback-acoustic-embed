import { ChevronLeft, ChevronRight } from "lucide-react";
import { formatRecordingTime } from "@/utils/format";

export interface SpanInfo {
  id: number;
  startTimestamp: number;
  endTimestamp: number;
}

interface SpanNavBarProps {
  spans: SpanInfo[];
  activeIndex: number;
  onPrev: () => void;
  onNext: () => void;
}

export function SpanNavBar({ spans, activeIndex, onPrev, onNext }: SpanNavBarProps) {
  const span = spans[activeIndex];
  if (!span) return null;

  return (
    <div className="flex items-center gap-2 text-sm" data-testid="hmm-span-nav">
      <button
        className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
        disabled={activeIndex === 0}
        onClick={onPrev}
        title="Previous span"
        data-testid="hmm-span-prev"
      >
        <ChevronLeft className="h-4 w-4" />
      </button>
      <button
        className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
        disabled={activeIndex === spans.length - 1}
        onClick={onNext}
        title="Next span"
        data-testid="hmm-span-next"
      >
        <ChevronRight className="h-4 w-4" />
      </button>
      <span className="text-muted-foreground ml-1" data-testid="hmm-span-label">
        Span {activeIndex + 1}/{spans.length} ·{" "}
        {formatRecordingTime(0, span.startTimestamp)} –{" "}
        {formatRecordingTime(0, span.endTimestamp)}
      </span>
    </div>
  );
}
