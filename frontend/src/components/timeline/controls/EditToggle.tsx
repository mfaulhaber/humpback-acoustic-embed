import { COLORS } from "../constants";

interface EditToggleProps {
  active: boolean;
  enabled: boolean;
  label: string;
  onToggle: () => void;
  icon?: React.ReactNode;
}

export function EditToggle({ active, enabled, label, onToggle, icon }: EditToggleProps) {
  return (
    <button
      className="flex items-center gap-1 px-2.5 py-1 rounded text-[10px] font-medium"
      style={{
        background: active ? COLORS.accentDim : "transparent",
        color: enabled || active ? COLORS.accent : COLORS.textMuted,
        opacity: enabled || active ? 1 : 0.3,
        border: `1px solid ${enabled || active ? COLORS.accent : COLORS.border}`,
      }}
      onClick={onToggle}
      disabled={!enabled && !active}
    >
      {icon}
      {label}
    </button>
  );
}
