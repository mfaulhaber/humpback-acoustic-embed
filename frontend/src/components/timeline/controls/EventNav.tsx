import { ChevronLeft, ChevronRight } from "lucide-react";
import { COLORS } from "../constants";

interface EventNavProps {
  currentIndex: number;
  totalCount: number;
  onPrev: () => void;
  onNext: () => void;
}

export function EventNav({ currentIndex, totalCount, onPrev, onNext }: EventNavProps) {
  return (
    <div className="flex items-center gap-1">
      <button
        onClick={onPrev}
        disabled={currentIndex <= 0}
        style={{ color: currentIndex <= 0 ? COLORS.textMuted : COLORS.textBright }}
      >
        <ChevronLeft size={16} />
      </button>
      <span className="text-[10px] font-mono" style={{ color: COLORS.textMuted }}>
        {totalCount > 0 ? `${currentIndex + 1}/${totalCount}` : "0/0"}
      </span>
      <button
        onClick={onNext}
        disabled={currentIndex >= totalCount - 1}
        style={{ color: currentIndex >= totalCount - 1 ? COLORS.textMuted : COLORS.textBright }}
      >
        <ChevronRight size={16} />
      </button>
    </div>
  );
}
