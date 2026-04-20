import { useTimelineContext } from "../provider/useTimelineContext";
import { COLORS } from "../constants";

export function ZoomSelector() {
  const { zoomLevels, zoomLevel, setZoomLevel } = useTimelineContext();

  return (
    <div className="flex justify-center gap-1 py-1">
      {zoomLevels.map((preset, index) => (
        <button
          key={preset.key}
          onClick={() => setZoomLevel(index)}
          className="px-2 py-0.5 rounded text-[10px] font-mono transition-colors"
          style={{
            background: index === zoomLevel ? "rgba(64, 160, 128, 0.2)" : COLORS.bgDark,
            border: index === zoomLevel ? `1px solid ${COLORS.accentDim}` : "1px solid transparent",
            color: index === zoomLevel ? COLORS.accent : COLORS.textMuted,
          }}
        >
          {preset.key}
        </button>
      ))}
    </div>
  );
}
