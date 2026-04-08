import React, { useRef, useEffect, useState } from "react";
import { useVocalizationTypes } from "@/hooks/queries/useVocalization";
import { useLabelVocabulary } from "@/hooks/queries/useLabeling";
import { VOCALIZATION_BADGE_PALETTE } from "./constants";
import type { VocLabelAction, LabelDisplayState } from "@/hooks/queries/useVocLabelEdits";

interface LabelState {
  label: string;
  state: LabelDisplayState;
}

interface VocLabelPopoverProps {
  rowId: string;
  startUtc: number;
  endUtc: number;
  /** Pixel position to anchor the popover */
  anchorX: number;
  anchorY: number;
  /** Viewport dimensions for clamping */
  viewportWidth: number;
  viewportHeight: number;
  effectiveLabels: LabelState[];
  dispatch: React.Dispatch<VocLabelAction>;
  /** Shared type name list for consistent badge coloring with the editor */
  allTypeNames: string[];
}

function formatTime(epoch: number): string {
  const d = new Date(epoch * 1000);
  return d.toISOString().slice(11, 19);
}

function badgeColor(label: string, allTypes: string[]): string {
  const idx = allTypes.indexOf(label);
  if (idx < 0) return "#888";
  return VOCALIZATION_BADGE_PALETTE[idx % VOCALIZATION_BADGE_PALETTE.length];
}

export function VocLabelPopover({
  rowId,
  startUtc,
  endUtc,
  anchorX,
  anchorY,
  viewportWidth,
  viewportHeight,
  effectiveLabels,
  dispatch,
  allTypeNames,
}: VocLabelPopoverProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: anchorX, top: anchorY });
  const [showDropdown, setShowDropdown] = useState(false);

  const { data: vocTypes = [] } = useVocalizationTypes();
  const { data: labelVocab = [] } = useLabelVocabulary();

  // All available types for the dropdown — merge editor's palette types with vocab/types
  const dropdownTypeNames = React.useMemo(() => {
    const names = new Set(allTypeNames);
    for (const vt of vocTypes) names.add(vt.name);
    for (const lv of labelVocab) names.add(lv);
    return Array.from(names).sort();
  }, [allTypeNames, vocTypes, labelVocab]);

  // Clamp popover position to viewport
  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    let left = anchorX;
    let top = anchorY - h - 8; // prefer above the bar
    if (top < 0) top = anchorY + 30; // below if not enough room
    if (left + w > viewportWidth) left = viewportWidth - w - 4;
    if (left < 4) left = 4;
    if (top + h > viewportHeight) top = viewportHeight - h - 4;
    setPos({ left, top });
  }, [anchorX, anchorY, viewportWidth, viewportHeight]);

  // Close on outside click or Escape
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        dispatch({ type: "deselect" });
      }
    }
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        dispatch({ type: "deselect" });
      }
    }
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [dispatch]);

  const activeLabels = effectiveLabels.filter((l) => l.state !== "pending_remove");
  const removedLabels = effectiveLabels.filter((l) => l.state === "pending_remove");

  // Labels available in the "+" dropdown = all types minus active effective labels
  const activeNames = new Set(activeLabels.map((l) => l.label));
  const dropdownTypes = dropdownTypeNames.filter((t) => !activeNames.has(t));

  function handleBadgeClick(label: string, state: LabelState["state"]) {
    if (state === "saved") {
      dispatch({ type: "toggle_remove", row_id: rowId, label });
    } else if (state === "pending_add") {
      dispatch({ type: "toggle_add", row_id: rowId, label });
    } else if (state === "pending_remove") {
      dispatch({ type: "toggle_remove", row_id: rowId, label });
    } else if (state === "inference") {
      // Promote inference label to manual — add a manual version
      dispatch({ type: "toggle_add", row_id: rowId, label });
    }
  }

  function handleAddType(label: string) {
    // Mutual exclusivity: if adding (Negative), queue removal of all saved type labels
    if (label === "(Negative)") {
      for (const el of effectiveLabels) {
        if (el.label !== "(Negative)" && el.state === "saved") {
          dispatch({ type: "toggle_remove", row_id: rowId, label: el.label });
        }
      }
    } else {
      // Adding a type: queue removal of saved (Negative)
      for (const el of effectiveLabels) {
        if (el.label === "(Negative)" && el.state === "saved") {
          dispatch({ type: "toggle_remove", row_id: rowId, label: "(Negative)" });
        }
      }
    }
    dispatch({ type: "toggle_add", row_id: rowId, label });
    setShowDropdown(false);
  }

  return (
    <div
      ref={ref}
      data-testid="voc-label-popover"
      style={{
        position: "absolute",
        left: pos.left,
        top: pos.top,
        background: "rgba(6, 13, 20, 0.96)",
        border: "1px solid rgba(168, 130, 220, 0.5)",
        borderRadius: 8,
        padding: "8px 12px",
        zIndex: 30,
        minWidth: 180,
        maxWidth: 320,
        pointerEvents: "auto",
      }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {/* Header: time range */}
      <div
        style={{
          fontSize: 11,
          color: "#8a7a9a",
          marginBottom: 6,
          fontFamily: "monospace",
        }}
      >
        {formatTime(startUtc)} &ndash; {formatTime(endUtc)} UTC
      </div>

      {/* Active labels as badges */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 4 }}>
        {activeLabels.map(({ label, state }) => {
          const color = badgeColor(label, allTypeNames);
          const isInference = state === "inference";
          const isPending = state === "pending_add";
          return (
            <button
              key={label}
              onClick={() => handleBadgeClick(label, state)}
              title={
                isInference
                  ? "Click to promote to manual label"
                  : isPending
                    ? "Click to undo add"
                    : "Click to remove"
              }
              style={{
                display: "inline-flex",
                alignItems: "center",
                height: 22,
                padding: "0 8px",
                fontSize: 11,
                fontWeight: 600,
                borderRadius: 4,
                border: isInference
                  ? `1.5px dashed ${color}`
                  : isPending
                    ? `2px solid ${color}`
                    : "none",
                background: isInference || isPending ? "transparent" : color,
                color: isInference || isPending ? color : "#fff",
                cursor: "pointer",
                opacity: isInference ? 0.7 : 1,
              }}
            >
              {label}
              {isInference ? (
                <span style={{ marginLeft: 4, fontSize: 9 }}>↑</span>
              ) : (
                <span style={{ marginLeft: 4, fontSize: 9 }}>✕</span>
              )}
            </button>
          );
        })}

        {/* Removed labels (pending removal) shown dimmed with strikethrough */}
        {removedLabels.map(({ label }) => {
          const color = badgeColor(label, allTypeNames);
          return (
            <button
              key={`rm-${label}`}
              onClick={() => handleBadgeClick(label, "pending_remove")}
              title="Click to undo removal"
              style={{
                display: "inline-flex",
                alignItems: "center",
                height: 22,
                padding: "0 8px",
                fontSize: 11,
                fontWeight: 600,
                borderRadius: 4,
                border: "none",
                background: color,
                color: "#fff",
                cursor: "pointer",
                opacity: 0.3,
                textDecoration: "line-through",
              }}
            >
              {label}
            </button>
          );
        })}

        {/* "+" button */}
        <div style={{ position: "relative" }}>
          <button
            onClick={() => setShowDropdown(!showDropdown)}
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              width: 22,
              height: 22,
              fontSize: 14,
              fontWeight: 600,
              borderRadius: 4,
              border: "1px solid rgba(168, 130, 220, 0.5)",
              background: "transparent",
              color: "#a882dc",
              cursor: "pointer",
            }}
            title="Add vocalization type"
          >
            +
          </button>

          {/* Dropdown */}
          {showDropdown && (
            <div
              style={{
                position: "absolute",
                top: 26,
                left: 0,
                background: "rgba(12, 18, 28, 0.98)",
                border: "1px solid rgba(168, 130, 220, 0.4)",
                borderRadius: 6,
                padding: 4,
                zIndex: 40,
                minWidth: 140,
                maxHeight: 200,
                overflowY: "auto",
              }}
            >
              {/* (Negative) option */}
              {!activeNames.has("(Negative)") && (
                <button
                  onClick={() => handleAddType("(Negative)")}
                  style={{
                    display: "block",
                    width: "100%",
                    textAlign: "left",
                    padding: "4px 8px",
                    fontSize: 11,
                    color: "#f87171",
                    background: "transparent",
                    border: "none",
                    borderRadius: 3,
                    cursor: "pointer",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.background = "rgba(168, 130, 220, 0.15)")
                  }
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  (Negative)
                </button>
              )}
              {dropdownTypes
                .filter((t) => t !== "(Negative)")
                .map((typeName) => {
                  const color = badgeColor(typeName, allTypeNames);
                  return (
                    <button
                      key={typeName}
                      onClick={() => handleAddType(typeName)}
                      style={{
                        display: "block",
                        width: "100%",
                        textAlign: "left",
                        padding: "4px 8px",
                        fontSize: 11,
                        color,
                        background: "transparent",
                        border: "none",
                        borderRadius: 3,
                        cursor: "pointer",
                      }}
                      onMouseEnter={(e) =>
                        (e.currentTarget.style.background = "rgba(168, 130, 220, 0.15)")
                      }
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.background = "transparent")
                      }
                    >
                      {typeName}
                    </button>
                  );
                })}
              {dropdownTypes.length === 0 && !activeNames.has("(Negative)") ? null : null}
              {dropdownTypes.length === 0 && activeNames.has("(Negative)") && (
                <div style={{ padding: "4px 8px", fontSize: 11, color: "#666" }}>
                  No types available
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
