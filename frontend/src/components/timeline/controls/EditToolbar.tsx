import React from "react";
import { COLORS } from "../constants";

interface EditToolbarProps {
  pendingCount: number;
  onSave: () => void;
  onCancel: () => void;
  isSaving?: boolean;
  children?: React.ReactNode;
}

export function EditToolbar({ pendingCount, onSave, onCancel, isSaving, children }: EditToolbarProps) {
  return (
    <div className="flex items-center gap-2">
      {children}
      <span className="text-[10px] font-mono" style={{ color: COLORS.textMuted }}>
        {pendingCount} pending
      </span>
      <button
        onClick={onCancel}
        disabled={isSaving}
        className="px-2 py-0.5 rounded text-[10px]"
        style={{
          border: `1px solid ${COLORS.border}`,
          color: COLORS.textBright,
        }}
      >
        Cancel
      </button>
      <button
        onClick={onSave}
        disabled={pendingCount === 0 || isSaving}
        className="px-2 py-0.5 rounded text-[10px] font-medium"
        style={{
          background: pendingCount > 0 && !isSaving ? COLORS.accent : COLORS.border,
          color: pendingCount > 0 && !isSaving ? COLORS.bg : COLORS.textMuted,
        }}
      >
        {isSaving ? "Saving…" : "Save"}
      </button>
    </div>
  );
}
