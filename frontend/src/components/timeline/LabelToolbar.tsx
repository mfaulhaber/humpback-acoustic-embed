// frontend/src/components/timeline/LabelToolbar.tsx
import { Trash2, Save, Download, X } from "lucide-react";
import { COLORS, LABEL_COLORS } from "./constants";
import type { LabelType } from "./constants";

export interface LabelToolbarProps {
  mode: "select" | "add";
  onModeChange: (mode: "select" | "add") => void;
  selectedLabel: LabelType | null;
  onLabelChange: (label: LabelType) => void;
  onDelete: () => void;
  onSave: () => void;
  onExtract: () => void;
  onCancel: () => void;
  isDirty: boolean;
  isSaving: boolean;
  hasSelection: boolean;
}

const LABEL_OPTIONS: LabelType[] = ["humpback", "orca", "ship", "background"];

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

export function LabelToolbar({
  mode,
  onModeChange,
  selectedLabel,
  onLabelChange,
  onDelete,
  onSave,
  onExtract,
  onCancel,
  isDirty,
  isSaving,
  hasSelection,
}: LabelToolbarProps) {
  return (
    <div
      className="flex items-center gap-3 px-3 py-1.5"
      style={{
        borderTop: `1px solid ${COLORS.border}`,
        background: COLORS.headerBg,
        minHeight: 32,
      }}
    >
      {/* Mode toggle */}
      <div
        className="flex items-center"
        style={{ border: `1px solid ${COLORS.border}`, borderRadius: 4, overflow: "hidden" }}
      >
        {(["select", "add"] as const).map((m) => (
          <button
            key={m}
            onClick={() => onModeChange(m)}
            style={{
              fontSize: 10,
              padding: "2px 8px",
              cursor: "pointer",
              background: mode === m ? COLORS.accent : "transparent",
              color: mode === m ? COLORS.bg : COLORS.text,
              border: "none",
              lineHeight: "16px",
            }}
          >
            {m === "select" ? "Select" : "Add"}
          </button>
        ))}
      </div>

      {/* Label radio group */}
      <div className="flex items-center gap-3">
        <span
          className="flex items-center gap-1"
          style={{ fontSize: 10, color: selectedLabel === null ? COLORS.textBright : COLORS.textMuted, opacity: selectedLabel === null ? 1 : 0.5 }}
        >
          <span
            style={{
              display: "inline-block",
              width: 8,
              height: 8,
              borderRadius: "50%",
              border: `1.5px solid ${COLORS.textMuted}`,
              background: selectedLabel === null ? COLORS.textMuted : "transparent",
              flexShrink: 0,
            }}
          />
          unlabeled
        </span>
        {LABEL_OPTIONS.map((label) => {
          const isSelected = selectedLabel === label;
          const accentColor = LABEL_COLORS[label].border;
          return (
            <label
              key={label}
              className="flex items-center gap-1"
              style={{ cursor: "pointer", fontSize: 10, color: isSelected ? accentColor : COLORS.textMuted }}
            >
              <span
                style={{
                  display: "inline-block",
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  border: `1.5px solid ${accentColor}`,
                  background: isSelected ? accentColor : "transparent",
                  flexShrink: 0,
                }}
              />
              <input
                type="radio"
                name="label-type"
                value={label}
                checked={isSelected}
                onChange={() => onLabelChange(label)}
                style={{ display: "none" }}
              />
              {label}
            </label>
          );
        })}
      </div>

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Action buttons */}
      <button
        onClick={onDelete}
        disabled={!hasSelection}
        title="Delete selected annotation"
        style={{
          ...btnBase,
          color: hasSelection ? "#f87171" : COLORS.textMuted,
          borderColor: hasSelection ? "rgba(248, 113, 113, 0.4)" : COLORS.border,
          cursor: hasSelection ? "pointer" : "not-allowed",
          opacity: hasSelection ? 1 : 0.5,
        }}
      >
        <Trash2 size={12} />
        Delete
      </button>

      <button
        onClick={onSave}
        disabled={!isDirty || isSaving}
        title="Save annotations"
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
        {isSaving ? "Saving…" : "Save"}
      </button>

      <button
        onClick={onExtract}
        title="Extract labeled clips"
        style={{ ...btnBase }}
      >
        <Download size={12} />
        Extract
      </button>

      <button
        onClick={onCancel}
        title="Exit label mode"
        style={{ ...btnBase }}
      >
        <X size={12} />
        Cancel
      </button>
    </div>
  );
}
