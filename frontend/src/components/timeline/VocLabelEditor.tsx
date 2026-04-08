import React, { useMemo, useCallback, useRef } from "react";
import type { DetectionRow, TimelineVocalizationLabel, ZoomLevel } from "@/api/types";
import { VIEWPORT_SPAN, VOCALIZATION_BAR, VOCALIZATION_BADGE_PALETTE } from "./constants";
import { VocLabelPopover } from "./VocLabelPopover";
import {
  computeEffectiveLabels,
  buildSavedLabelMap,
  type VocLabelAction,
} from "@/hooks/queries/useVocLabelEdits";

export interface VocLabelEditorProps {
  /** All detection windows (labeled + unlabeled) */
  detectionRows: DetectionRow[];
  /** Saved vocalization labels from the /all endpoint */
  vocLabels: TimelineVocalizationLabel[];
  /** Current edit state from useVocLabelEdits */
  edits: Map<string, { adds: Set<string>; removes: Set<string> }>;
  selectedRowId: string | null;
  dispatch: React.Dispatch<VocLabelAction>;
  centerTimestamp: number;
  zoomLevel: ZoomLevel;
  width: number;
  height: number;
}

const SELECTED_BAR_BG = "rgba(168, 130, 220, 0.70)";
const BADGE_HEIGHT = 14;
const BADGE_GAP = 2;
const BADGE_PADDING_H = 4;
const BADGE_FONT_SIZE = 9;
const DIRTY_DOT_SIZE = 6;

function badgeColor(label: string, allTypes: string[]): string {
  const idx = allTypes.indexOf(label);
  if (idx < 0) return "#888";
  return VOCALIZATION_BADGE_PALETTE[idx % VOCALIZATION_BADGE_PALETTE.length];
}

export function VocLabelEditor({
  detectionRows,
  vocLabels,
  edits,
  selectedRowId,
  dispatch,
  centerTimestamp,
  zoomLevel,
  width,
  height,
}: VocLabelEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredRowId, setHoveredRowId] = React.useState<string | null>(null);

  const pxPerSec = width / VIEWPORT_SPAN[zoomLevel];

  // Build row_id → saved manual label names
  const rowIdByUtc = useMemo(() => {
    const map = new Map<string, string>();
    for (const row of detectionRows) {
      map.set(`${row.start_utc}_${row.end_utc}`, row.row_id);
    }
    return map;
  }, [detectionRows]);

  const savedLabelMap = useMemo(
    () => buildSavedLabelMap(vocLabels, rowIdByUtc),
    [vocLabels, rowIdByUtc],
  );

  // Collect all vocalization type names for badge coloring
  const allTypeNames = useMemo(() => {
    const names = new Set<string>();
    for (const lbl of vocLabels) names.add(lbl.label);
    for (const rowEdits of edits.values()) {
      for (const lbl of rowEdits.adds) names.add(lbl);
    }
    return Array.from(names).sort();
  }, [vocLabels, edits]);

  // Compute bars to render
  const bars = useMemo(() => {
    const result: {
      row: DetectionRow;
      x: number;
      w: number;
      effectiveLabels: ReturnType<typeof computeEffectiveLabels>;
      isDirty: boolean;
    }[] = [];

    for (const row of detectionRows) {
      const x = (row.start_utc - centerTimestamp) * pxPerSec + width / 2;
      const w = Math.max(2, (row.end_utc - row.start_utc) * pxPerSec);
      if (x + w < 0 || x > width) continue;

      const savedLabels = savedLabelMap.get(row.row_id) ?? [];
      const rowEdits = edits.get(row.row_id);
      const effective = computeEffectiveLabels(savedLabels, rowEdits);
      const isDirty =
        rowEdits !== undefined &&
        (rowEdits.adds.size > 0 || rowEdits.removes.size > 0);

      result.push({ row, x, w, effectiveLabels: effective, isDirty });
    }

    return result;
  }, [detectionRows, centerTimestamp, pxPerSec, width, savedLabelMap, edits]);

  const handleBarClick = useCallback(
    (rowId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      if (selectedRowId === rowId) {
        dispatch({ type: "deselect" });
      } else {
        dispatch({ type: "select", row_id: rowId });
      }
    },
    [selectedRowId, dispatch],
  );

  // Find the selected bar for popover anchor
  const selectedBar = bars.find((b) => b.row.row_id === selectedRowId);

  if (width <= 0 || height <= 0) return null;

  return (
    <div
      ref={containerRef}
      data-testid="voc-label-editor"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width,
        height,
        pointerEvents: "none",
        zIndex: 6,
        overflow: "hidden",
      }}
    >
      {bars.map(({ row, x, w, effectiveLabels, isDirty }) => {
        const isSelected = row.row_id === selectedRowId;
        const isHovered = row.row_id === hoveredRowId;
        const activeLabels = effectiveLabels.filter(
          (l) => l.state !== "pending_remove",
        );

        return (
          <div
            key={row.row_id}
            style={{
              position: "absolute",
              top: 0,
              left: x,
              width: w,
              height,
              background: isSelected
                ? SELECTED_BAR_BG
                : isHovered
                  ? VOCALIZATION_BAR.hover
                  : VOCALIZATION_BAR.fill,
              border: isSelected ? "2px solid white" : "none",
              boxSizing: "border-box",
              pointerEvents: "auto",
              cursor: "pointer",
              display: "flex",
              flexDirection: "column",
              justifyContent: "flex-end",
              alignItems: "flex-start",
              padding: "0 1px 4px 1px",
              gap: BADGE_GAP,
              overflow: "hidden",
            }}
            onClick={(e) => handleBarClick(row.row_id, e)}
            onMouseEnter={() => setHoveredRowId(row.row_id)}
            onMouseLeave={() => setHoveredRowId(null)}
          >
            {/* Dirty indicator dot */}
            {isDirty && (
              <div
                style={{
                  position: "absolute",
                  top: 3,
                  right: 3,
                  width: DIRTY_DOT_SIZE,
                  height: DIRTY_DOT_SIZE,
                  borderRadius: "50%",
                  background: "#facc15",
                }}
              />
            )}

            {/* Vocalization type badges */}
            {w >= 18 &&
              activeLabels.map(({ label, state }) => {
                const color = badgeColor(label, allTypeNames);
                const isInference = state === "inference";
                const isPending = state === "pending_add";
                return (
                  <span
                    key={label}
                    style={{
                      display: "inline-block",
                      height: BADGE_HEIGHT,
                      lineHeight: `${BADGE_HEIGHT}px`,
                      padding: `0 ${BADGE_PADDING_H}px`,
                      fontSize: BADGE_FONT_SIZE,
                      fontWeight: 600,
                      borderRadius: 3,
                      whiteSpace: "nowrap",
                      maxWidth: w - 4,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      color: isInference || isPending ? color : "#fff",
                      background: isInference || isPending ? "transparent" : color,
                      border: isInference ? `1.5px solid ${color}` : isPending ? `2px solid ${color}` : "none",
                      opacity: isInference ? 0.7 : 1,
                      pointerEvents: "none",
                    }}
                  >
                    {label}
                  </span>
                );
              })}
          </div>
        );
      })}

      {/* Popover for selected bar */}
      {selectedBar && (
        <VocLabelPopover
          rowId={selectedBar.row.row_id}
          startUtc={selectedBar.row.start_utc}
          endUtc={selectedBar.row.end_utc}
          anchorX={selectedBar.x + selectedBar.w / 2}
          anchorY={selectedBar.w > 0 ? 40 : 40}
          viewportWidth={width}
          viewportHeight={height}
          effectiveLabels={selectedBar.effectiveLabels}
          dispatch={dispatch}
          allTypeNames={allTypeNames}
        />
      )}
    </div>
  );
}
