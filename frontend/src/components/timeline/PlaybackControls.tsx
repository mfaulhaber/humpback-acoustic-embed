// frontend/src/components/timeline/PlaybackControls.tsx
import { Play, Pause, SkipBack, SkipForward, Plus, Minus, Tag, AudioLines, Pencil } from "lucide-react";
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
  onLabelMode?: () => void;
  labelModeEnabled: boolean;
  labelModeActive: boolean;
  overlayMode: "off" | "detection" | "vocalization";
  onToggleDetection: () => void;
  onToggleVocalization: () => void;
  hasVocalizationData: boolean;
  freqRange: [number, number];
  regionEditMode?: boolean;
  regionEditEnabled?: boolean;
  onRegionEditToggle?: () => void;
  showRegionOverlay?: boolean;
  onToggleRegionOverlay?: () => void;
  pendingCorrectionCount?: number;
  onSaveCorrections?: () => void;
  onCancelCorrections?: () => void;
  isSavingCorrections?: boolean;
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
  onLabelMode,
  labelModeEnabled,
  labelModeActive,
  overlayMode,
  onToggleDetection,
  onToggleVocalization,
  hasVocalizationData,
  freqRange,
  regionEditMode,
  regionEditEnabled,
  onRegionEditToggle,
  showRegionOverlay,
  onToggleRegionOverlay,
  pendingCorrectionCount,
  onSaveCorrections,
  onCancelCorrections,
  isSavingCorrections,
}: PlaybackControlsProps) {
  const timeStr = new Date(centerTimestamp * 1000).toISOString().slice(11, 19) + " UTC";

  return (
    <div
      className="flex items-center py-2.5 px-4"
      style={{ borderTop: `1px solid ${COLORS.border}`, position: "relative" }}
    >
      {/* Center group: playback controls anchored to center of row */}
      <div className="absolute inset-0 flex items-center justify-center gap-6 pointer-events-none">
        <span className="text-[10px] font-mono pointer-events-auto" style={{ color: COLORS.textMuted }}>
          {timeStr}
        </span>
        <div className="flex items-center gap-4 pointer-events-auto">
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
          className="text-[10px] font-mono pointer-events-auto"
          style={{ color: COLORS.textMuted }}
        >
          {speed}x
        </button>
        <div className="flex items-center gap-2 ml-10 pointer-events-auto">
          <button onClick={onZoomOut} style={{ color: COLORS.textBright }}>
            <Minus size={14} />
          </button>
          <span className="text-[10px]" style={{ color: COLORS.accent }}>Zoom</span>
          <button onClick={onZoomIn} style={{ color: COLORS.textBright }}>
            <Plus size={14} />
          </button>
        </div>
        {onRegionEditToggle ? (
          <button
            className="flex items-center gap-1 px-2.5 py-1 rounded text-[10px] font-medium ml-4 pointer-events-auto"
            style={{
              background: regionEditMode ? COLORS.accentDim : "transparent",
              color: regionEditEnabled || regionEditMode ? COLORS.accent : COLORS.textMuted,
              opacity: regionEditEnabled || regionEditMode ? 1 : 0.3,
              border: `1px solid ${regionEditEnabled || regionEditMode ? COLORS.accent : COLORS.border}`,
            }}
            onClick={onRegionEditToggle}
            disabled={!regionEditEnabled && !regionEditMode}
            title={regionEditMode ? "Exit edit mode" : "Edit region boundaries"}
          >
            <Pencil size={12} /> Edit Regions
          </button>
        ) : (
          <button
            className="flex items-center gap-1 px-2.5 py-1 rounded text-[10px] font-medium ml-4 pointer-events-auto"
            style={{
              background: labelModeActive ? COLORS.accentDim : "transparent",
              color: (labelModeEnabled || labelModeActive) ? COLORS.accent : COLORS.textMuted,
              opacity: (labelModeEnabled || labelModeActive) ? 1 : 0.3,
              border: `1px solid ${(labelModeEnabled || labelModeActive) ? COLORS.accent : COLORS.border}`,
            }}
            onClick={onLabelMode}
            disabled={!labelModeEnabled && !labelModeActive}
            title={labelModeActive ? "Exit label mode" : labelModeEnabled ? "Enter label mode" : "Zoom to 5m or closer to edit labels"}
          >
            <Tag size={12} /> Label
          </button>
        )}
      </div>

      {/* Right group: Labels/Regions/Vocalizations/Freq pushed to far right */}
      <div className="flex items-center gap-2 ml-auto">
        {onToggleRegionOverlay ? (
          <button
            onClick={onToggleRegionOverlay}
            className="flex items-center gap-1 px-2 py-1 rounded text-[10px]"
            style={{
              background: showRegionOverlay ? COLORS.accentDim : COLORS.border,
              color: COLORS.accent,
            }}
            title={showRegionOverlay ? "Hide regions" : "Show regions"}
          >
            <Tag size={10} /> Regions
          </button>
        ) : (
          <>
            <button
              onClick={onToggleDetection}
              className="flex items-center gap-1 px-2 py-1 rounded text-[10px]"
              style={{
                background: overlayMode === "detection" ? COLORS.accentDim : COLORS.border,
                color: COLORS.accent,
              }}
            >
              <Tag size={10} /> Labels
            </button>
            <button
              onClick={hasVocalizationData ? onToggleVocalization : undefined}
              disabled={!hasVocalizationData}
              className="flex items-center gap-1 px-2 py-1 rounded text-[10px]"
              style={{
                background: overlayMode === "vocalization" ? "rgba(168, 130, 220, 0.3)" : COLORS.border,
                color: hasVocalizationData ? (overlayMode === "vocalization" ? "#c4a8f0" : COLORS.accent) : COLORS.textMuted,
                opacity: hasVocalizationData ? 1 : 0.4,
              }}
              title={hasVocalizationData ? (overlayMode === "vocalization" ? "Hide vocalization types" : "Show vocalization types") : "No vocalization inference for this job"}
            >
              <AudioLines size={10} /> Vocalizations
            </button>
          </>
        )}
        <span
          className="px-2 py-1 rounded text-[10px]"
          style={{ background: COLORS.border, color: COLORS.accent }}
        >
          Freq: {freqRange[0] / 1000}–{freqRange[1] / 1000} kHz
        </span>
      </div>
    </div>
  );
}
