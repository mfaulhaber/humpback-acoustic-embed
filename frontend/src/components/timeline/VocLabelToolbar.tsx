import { Save, X } from "lucide-react";
import { COLORS } from "./constants";

export interface VocLabelToolbarProps {
  onSave: () => void;
  onCancel: () => void;
  isDirty: boolean;
  isSaving: boolean;
  editCount: number;
}

const btnBase: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 4,
  padding: "2px 8px",
  borderRadius: 4,
  fontSize: 10,
  cursor: "pointer",
  border: `1px solid ${COLORS.border}`,
  background: "transparent",
  color: COLORS.text,
  lineHeight: "16px",
};

export function VocLabelToolbar({
  onSave,
  onCancel,
  isDirty,
  isSaving,
  editCount,
}: VocLabelToolbarProps) {
  return (
    <div
      className="flex items-center gap-3 px-3 py-1.5"
      style={{
        borderTop: `1px solid ${COLORS.border}`,
        background: COLORS.headerBg,
        minHeight: 32,
      }}
    >
      {/* Mode indicator */}
      <span style={{ fontSize: 10, color: COLORS.textBright, fontWeight: 600 }}>
        Vocalization Labels
      </span>

      {/* Edit count */}
      {editCount > 0 && (
        <span style={{ fontSize: 10, color: "#fbbf24" }}>
          {editCount} unsaved {editCount === 1 ? "change" : "changes"}
        </span>
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Save */}
      <button
        onClick={onSave}
        disabled={!isDirty || isSaving}
        title="Save vocalization labels"
        style={{
          ...btnBase,
          position: "relative",
          color: isDirty ? COLORS.accent : COLORS.textMuted,
          borderColor: isDirty ? COLORS.accent : COLORS.border,
          cursor: isDirty && !isSaving ? "pointer" : "not-allowed",
          opacity: isDirty && !isSaving ? 1 : 0.5,
        }}
      >
        {isDirty && (
          <span
            style={{
              position: "absolute",
              top: 2,
              right: 2,
              width: 4,
              height: 4,
              borderRadius: "50%",
              background: "#fbbf24",
            }}
          />
        )}
        <Save size={12} />
        {isSaving ? "Saving\u2026" : "Save"}
      </button>

      {/* Cancel */}
      <button
        onClick={onCancel}
        title="Exit vocalization label mode"
        style={{ ...btnBase }}
      >
        <X size={12} />
        Cancel
      </button>
    </div>
  );
}
