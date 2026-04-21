import { useTimelineContext } from "../provider/useTimelineContext";

export function ZoomSelector() {
  const { zoomLevels, zoomLevel, setZoomLevel } = useTimelineContext();

  return (
    <div className="flex justify-center gap-1 py-1">
      {zoomLevels.map((preset, index) => (
        <button
          key={preset.key}
          onClick={() => setZoomLevel(index)}
          className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors border ${
            index === zoomLevel
              ? "bg-primary/10 border-primary/30 text-primary"
              : "bg-muted border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          {preset.key}
        </button>
      ))}
    </div>
  );
}
