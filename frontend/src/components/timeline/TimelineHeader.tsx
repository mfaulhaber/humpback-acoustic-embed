// frontend/src/components/timeline/TimelineHeader.tsx
import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { COLORS } from "./constants";

interface TimelineHeaderProps {
  hydrophone: string;
  startTimestamp: number;
  endTimestamp: number;
}

export function TimelineHeader({
  hydrophone,
  startTimestamp,
  endTimestamp,
}: TimelineHeaderProps) {
  const navigate = useNavigate();
  const startStr = new Date(startTimestamp * 1000).toISOString().replace("T", " ").slice(0, 19) + "Z";
  const endStr = new Date(endTimestamp * 1000).toISOString().replace("T", " ").slice(0, 19) + "Z";

  return (
    <div
      className="flex items-center justify-between px-4 py-2 shrink-0"
      style={{ background: COLORS.headerBg, borderBottom: `1px solid ${COLORS.border}` }}
    >
      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate("/app/classifier/hydrophone")}
          className="flex items-center gap-1 text-xs hover:opacity-80"
          style={{ color: COLORS.textMuted }}
        >
          <ArrowLeft size={14} /> Back to Jobs
        </button>
        <span className="font-bold text-sm" style={{ color: COLORS.accent }}>
          {hydrophone}
        </span>
        <span className="text-xs" style={{ color: COLORS.textBright }}>
          {startStr} — {endStr}
        </span>
      </div>
    </div>
  );
}
