import React from "react";
import { Play, Pause, SkipBack, SkipForward, Plus, Minus } from "lucide-react";
import { useTimelineContext } from "../provider/useTimelineContext";
import { COLORS } from "../constants";

interface PlaybackControlsProps {
  variant?: "default" | "compact";
  children?: React.ReactNode;
}

const SPEEDS = [0.5, 1, 2];

export function PlaybackControls({ variant = "default", children }: PlaybackControlsProps) {
  const ctx = useTimelineContext();
  const isCompact = variant === "compact";

  const timeStr = new Date(ctx.centerTimestamp * 1000).toISOString().slice(11, 19) + " UTC";

  const cycleSpeed = () => {
    const idx = SPEEDS.indexOf(ctx.speed);
    ctx.setSpeed(SPEEDS[(idx + 1) % SPEEDS.length]);
  };

  return (
    <div className="flex items-center justify-center gap-4">
      <span className="text-[10px] font-mono" data-testid="timeline-center-time" style={{ color: COLORS.textMuted }}>
        {timeStr}
      </span>

      {!isCompact && (
        <button onClick={() => ctx.pan(ctx.centerTimestamp - ctx.viewportSpan * 0.25)} style={{ color: COLORS.textBright }}>
          <SkipBack size={16} />
        </button>
      )}

      <button
        onClick={ctx.togglePlay}
        className="w-9 h-9 rounded-full flex items-center justify-center"
        style={{ border: `1.5px solid ${COLORS.accent}` }}
      >
        {ctx.isPlaying ? (
          <Pause size={16} style={{ color: COLORS.accent }} />
        ) : (
          <Play size={16} style={{ color: COLORS.accent, paddingLeft: 2 }} />
        )}
      </button>

      {!isCompact && (
        <button onClick={() => ctx.pan(ctx.centerTimestamp + ctx.viewportSpan * 0.25)} style={{ color: COLORS.textBright }}>
          <SkipForward size={16} />
        </button>
      )}

      <button onClick={cycleSpeed} className="text-[10px] font-mono" style={{ color: COLORS.textMuted }}>
        {ctx.speed}x
      </button>

      <div className="flex items-center gap-2">
        <button onClick={ctx.zoomOut} style={{ color: COLORS.textBright }}>
          <Minus size={14} />
        </button>
        <span className="text-[10px]" style={{ color: COLORS.accent }}>Zoom</span>
        <button onClick={ctx.zoomIn} style={{ color: COLORS.textBright }}>
          <Plus size={14} />
        </button>
      </div>

      {children}
    </div>
  );
}
