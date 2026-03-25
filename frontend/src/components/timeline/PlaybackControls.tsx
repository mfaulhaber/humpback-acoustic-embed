// frontend/src/components/timeline/PlaybackControls.tsx
import { Play, Pause, SkipBack, SkipForward, Plus, Minus } from "lucide-react";
import { COLORS } from "./constants";

interface PlaybackControlsProps {
  centerTimestamp: number;
  isPlaying: boolean;
  speed: number;
  onPlayPause: () => void;
  onSkipBack: () => void;
  onSkipForward: () => void;
  onSpeedChange: (speed: number) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}

const SPEEDS = [0.5, 1, 2];

export function PlaybackControls({
  centerTimestamp,
  isPlaying,
  speed,
  onPlayPause,
  onSkipBack,
  onSkipForward,
  onSpeedChange,
  onZoomIn,
  onZoomOut,
}: PlaybackControlsProps) {
  const timeStr = new Date(centerTimestamp * 1000).toISOString().slice(11, 19) + " UTC";

  return (
    <div
      className="flex items-center justify-center gap-6 py-2.5 px-4"
      style={{ borderTop: `1px solid ${COLORS.border}` }}
    >
      <span className="text-[10px] font-mono" style={{ color: COLORS.textMuted }}>
        {timeStr}
      </span>
      <div className="flex items-center gap-4">
        <button onClick={onSkipBack} style={{ color: COLORS.textBright }}>
          <SkipBack size={16} />
        </button>
        <button
          onClick={onPlayPause}
          className="w-9 h-9 rounded-full flex items-center justify-center"
          style={{ border: `1.5px solid ${COLORS.accent}` }}
        >
          {isPlaying ? (
            <Pause size={16} style={{ color: COLORS.accent }} />
          ) : (
            <Play size={16} style={{ color: COLORS.accent, paddingLeft: 2 }} />
          )}
        </button>
        <button onClick={onSkipForward} style={{ color: COLORS.textBright }}>
          <SkipForward size={16} />
        </button>
      </div>
      <button
        onClick={() => {
          const idx = SPEEDS.indexOf(speed);
          onSpeedChange(SPEEDS[(idx + 1) % SPEEDS.length]);
        }}
        className="text-[10px] font-mono"
        style={{ color: COLORS.textMuted }}
      >
        {speed}x
      </button>
      <div className="flex items-center gap-2 ml-10">
        <button onClick={onZoomOut} style={{ color: COLORS.textBright }}>
          <Minus size={14} />
        </button>
        <span className="text-[10px]" style={{ color: COLORS.accent }}>Zoom</span>
        <button onClick={onZoomIn} style={{ color: COLORS.textBright }}>
          <Plus size={14} />
        </button>
      </div>
    </div>
  );
}
