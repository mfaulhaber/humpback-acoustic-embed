import React, { useState, useCallback, useRef, useLayoutEffect, useMemo } from "react";
import type { VocalizationLabel, ZoomLevel } from "@/api/types";
import { VIEWPORT_SPAN, VOCALIZATION_BAR, VOCALIZATION_BADGE_PALETTE } from "./constants";

export interface VocalizationOverlayProps {
  labels: VocalizationLabel[];
  centerTimestamp: number;
  zoomLevel: ZoomLevel;
  width: number;
  height: number;
  visible: boolean;
}

interface WindowGroup {
  start_utc: number;
  end_utc: number;
  labels: VocalizationLabel[];
}

function groupByWindow(labels: VocalizationLabel[]): WindowGroup[] {
  const map = new Map<string, WindowGroup>();
  for (const lbl of labels) {
    const key = `${lbl.start_utc}:${lbl.end_utc}`;
    let group = map.get(key);
    if (!group) {
      group = { start_utc: lbl.start_utc, end_utc: lbl.end_utc, labels: [] };
      map.set(key, group);
    }
    group.labels.push(lbl);
  }
  return Array.from(map.values()).sort((a, b) => a.start_utc - b.start_utc);
}

function formatTime(epoch: number): string {
  const d = new Date(epoch * 1000);
  return d.toISOString().slice(11, 19) + " UTC";
}

function buildBadgeColorMap(labels: VocalizationLabel[]): Map<string, string> {
  const types = new Set<string>();
  for (const lbl of labels) types.add(lbl.label);
  const sorted = Array.from(types).sort();
  const map = new Map<string, string>();
  for (let i = 0; i < sorted.length; i++) {
    map.set(sorted[i], VOCALIZATION_BADGE_PALETTE[i % VOCALIZATION_BADGE_PALETTE.length]);
  }
  return map;
}

const TOOLTIP_OFFSET = 12;
const BADGE_HEIGHT = 14;
const BADGE_GAP = 2;
const BADGE_PADDING_H = 4;
const BADGE_FONT_SIZE = 9;

interface TooltipState {
  mouseX: number;
  mouseY: number;
  group: WindowGroup;
}

export function VocalizationOverlay({
  labels,
  centerTimestamp,
  zoomLevel,
  width,
  height,
  visible,
}: VocalizationOverlayProps) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ left: number; top: number }>({ left: 0, top: 0 });
  const tooltipRef = useRef<HTMLDivElement>(null);

  const badgeColorMap = useMemo(() => buildBadgeColorMap(labels), [labels]);
  const groups = useMemo(() => groupByWindow(labels), [labels]);

  useLayoutEffect(() => {
    if (!tooltip || !tooltipRef.current) return;
    const el = tooltipRef.current;
    const tw = el.offsetWidth;
    const th = el.offsetHeight;

    let left = tooltip.mouseX + TOOLTIP_OFFSET;
    let top = tooltip.mouseY + TOOLTIP_OFFSET;

    if (left + tw > width) left = tooltip.mouseX - TOOLTIP_OFFSET - tw;
    if (top + th > height) top = tooltip.mouseY - TOOLTIP_OFFSET - th;
    left = Math.max(0, Math.min(left, width - tw));
    top = Math.max(0, Math.min(top, height - th));

    setTooltipPos({ left, top });
  }, [tooltip, width, height]);

  const handleMouseEnter = useCallback(
    (group: WindowGroup, key: string, e: React.MouseEvent) => {
      const rect = e.currentTarget.parentElement?.getBoundingClientRect();
      const mouseX = e.clientX - (rect?.left ?? 0);
      const mouseY = e.clientY - (rect?.top ?? 0);
      setHoveredKey(key);
      setTooltip({ mouseX, mouseY, group });
    },
    [],
  );

  const handleMouseLeave = useCallback(() => {
    setHoveredKey(null);
    setTooltip(null);
  }, []);

  if (!visible || width <= 0 || height <= 0) return null;

  const pxPerSec = width / VIEWPORT_SPAN[zoomLevel];

  const bars: {
    group: WindowGroup;
    key: string;
    x: number;
    w: number;
  }[] = [];

  for (const group of groups) {
    const x = (group.start_utc - centerTimestamp) * pxPerSec + width / 2;
    const w = Math.max(2, (group.end_utc - group.start_utc) * pxPerSec);
    if (x + w < 0 || x > width) continue;
    const key = `${group.start_utc}:${group.end_utc}`;
    bars.push({ group, key, x, w });
  }

  return (
    <div
      data-testid="vocalization-overlay"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width,
        height,
        pointerEvents: "none",
        zIndex: 5,
        overflow: "hidden",
      }}
    >
      {bars.map(({ group, key, x, w }) => {
        const isHovered = hoveredKey === key;
        return (
          <div
            key={key}
            style={{
              position: "absolute",
              top: 0,
              left: x,
              width: w,
              height,
              background: isHovered ? VOCALIZATION_BAR.hover : VOCALIZATION_BAR.fill,
              pointerEvents: "auto",
              cursor: "default",
              display: "flex",
              flexDirection: "column",
              justifyContent: "flex-end",
              alignItems: "flex-start",
              padding: "0 1px 4px 1px",
              gap: BADGE_GAP,
              overflow: "hidden",
            }}
            onMouseEnter={(e) => handleMouseEnter(group, key, e)}
            onMouseLeave={handleMouseLeave}
          >
            {/* Badges — only render when bar is wide enough */}
            {w >= 18 &&
              group.labels.map((lbl) => {
                const color = badgeColorMap.get(lbl.label) ?? "#ccc";
                const isManual = lbl.source === "manual";
                return (
                  <span
                    key={lbl.id}
                    style={{
                      display: "inline-block",
                      height: BADGE_HEIGHT,
                      lineHeight: `${BADGE_HEIGHT}px`,
                      padding: `0 ${BADGE_PADDING_H}px`,
                      fontSize: BADGE_FONT_SIZE,
                      fontWeight: 600,
                      borderRadius: 3,
                      whiteSpace: "nowrap",
                      maxWidth: w - 2,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      color: isManual ? "#fff" : color,
                      background: isManual ? color : "transparent",
                      border: isManual ? "none" : `1.5px solid ${color}`,
                      pointerEvents: "none",
                    }}
                  >
                    {lbl.label}
                  </span>
                );
              })}
          </div>
        );
      })}

      {/* Tooltip */}
      {tooltip && (
        <div
          ref={tooltipRef}
          style={{
            position: "absolute",
            left: tooltipPos.left,
            top: tooltipPos.top,
            background: "rgba(6, 13, 20, 0.95)",
            border: "1px solid rgba(168, 130, 220, 0.4)",
            borderRadius: 6,
            padding: "6px 10px",
            fontSize: 11,
            lineHeight: "1.5",
            color: "#c4b5d0",
            whiteSpace: "nowrap",
            pointerEvents: "none",
            zIndex: 20,
          }}
        >
          <div style={{ marginBottom: 2 }}>
            {formatTime(tooltip.group.start_utc)} &ndash; {formatTime(tooltip.group.end_utc)}
          </div>
          {tooltip.group.labels.map((lbl) => {
            const color = badgeColorMap.get(lbl.label) ?? "#ccc";
            return (
              <div key={lbl.id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span
                  style={{
                    display: "inline-block",
                    width: 8,
                    height: 8,
                    borderRadius: 2,
                    background: lbl.source === "manual" ? color : "transparent",
                    border: `1.5px solid ${color}`,
                    flexShrink: 0,
                  }}
                />
                <span style={{ fontWeight: 600, color: "#e0d4f0" }}>{lbl.label}</span>
                <span style={{ color: "#8a7a9a" }}>
                  {lbl.source}
                  {lbl.confidence != null && ` ${(lbl.confidence * 100).toFixed(0)}%`}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
