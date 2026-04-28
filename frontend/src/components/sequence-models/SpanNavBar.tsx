import { ChevronLeft, ChevronRight } from "lucide-react";
import { formatRecordingTime } from "@/utils/format";

export interface SpanInfo {
  id: number;
  startSec: number;
  endSec: number;
}

interface SpanNavBarProps {
  spans: SpanInfo[];
  activeIndex: number;
  onPrev: () => void;
  onNext: () => void;
}

export function SpanNavBar({ spans, activeIndex, onPrev, onNext }: SpanNavBarProps) {
  const span = spans[activeIndex];
  const startLabel = formatRecordingTime(span.startSec, 0);
  const endLabel = formatRecordingTime(span.endSec, 0);

  return (
    <div className="flex items-center gap-2 text-sm">
      <button
        className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
        disabled={activeIndex === 0}
        onClick={onPrev}
        title="Previous span"
      >
        <ChevronLeft className="h-4 w-4" />
      </button>
      <button
        className="rounded-md border p-1 hover:bg-accent disabled:opacity-30"
        disabled={activeIndex === spans.length - 1}
        onClick={onNext}
        title="Next span"
      >
        <ChevronRight className="h-4 w-4" />
      </button>
      <span className="text-muted-foreground ml-1">
        Span {activeIndex + 1}/{spans.length} · {startLabel} – {endLabel}
      </span>
    </div>
  );
}
