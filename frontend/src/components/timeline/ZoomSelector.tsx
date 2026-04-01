// frontend/src/components/timeline/ZoomSelector.tsx
import type { ZoomLevel } from "@/api/types";
import { ZOOM_LEVELS, COLORS } from "./constants";

interface ZoomSelectorProps {
  activeLevel: ZoomLevel;
  onChange: (level: ZoomLevel) => void;
}

export function ZoomSelector({ activeLevel, onChange }: ZoomSelectorProps) {
  return (
    <div className="flex justify-center gap-1 py-1">
      {ZOOM_LEVELS.map((level) => (
        <button
          key={level}
          onClick={() => onChange(level)}
          className="px-2 py-0.5 rounded text-[10px] font-mono transition-colors"
          style={{
            background: level === activeLevel ? "rgba(64, 160, 128, 0.2)" : COLORS.bgDark,
            border: level === activeLevel ? `1px solid ${COLORS.accentDim}` : "1px solid transparent",
            color: level === activeLevel ? COLORS.accent : COLORS.textMuted,
          }}
        >
          {level}
        </button>
      ))}
    </div>
  );
}
