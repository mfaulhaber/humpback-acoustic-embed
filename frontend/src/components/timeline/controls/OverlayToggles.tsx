import { COLORS } from "../constants";

interface OverlayOption {
  key: string;
  label: string;
  active: boolean;
}

interface OverlayTogglesProps {
  options: OverlayOption[];
  onToggle: (key: string) => void;
}

export function OverlayToggles({ options, onToggle }: OverlayTogglesProps) {
  return (
    <div className="flex items-center gap-2">
      {options.map((opt) => (
        <button
          key={opt.key}
          onClick={() => onToggle(opt.key)}
          className="flex items-center gap-1 px-2 py-1 rounded text-[10px]"
          style={{
            background: opt.active ? COLORS.accentDim : COLORS.border,
            color: COLORS.accent,
          }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
