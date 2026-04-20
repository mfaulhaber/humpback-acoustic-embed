// frontend/src/components/timeline/SpectrogramViewport.tsx
import React, { useRef, useState, useEffect, useCallback, useMemo } from "react";
import type { DetectionRow, TimelineVocalizationLabel, ZoomLevel } from "@/api/types";
import { TileCanvas } from "./TileCanvas";
import { DetectionOverlay } from "./DetectionOverlay";
import { VocalizationOverlay } from "./VocalizationOverlay";
import { RegionOverlay } from "./RegionOverlay";
import { DetectionPopover } from "./DetectionPopover";
import {
  FREQ_AXIS_WIDTH_PX,
  VIEWPORT_SPAN,
  CONFIDENCE_GRADIENT,
  COLORS,
} from "./constants";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface SpectrogramViewportProps {
  jobId: string;
  jobStart: number;
  jobEnd: number;
  centerTimestamp: number;
  zoomLevel: ZoomLevel;
  freqRange: [number, number];
  isPlaying: boolean;
  scores: (number | null)[];
  /** Backend-provided window size in seconds for the confidence strip. */
  windowSec?: number;
  showLabels: boolean;
  detections: DetectionRow[];
  onCenterChange: (t: number) => void;
  /** Called on drag-pan instead of onCenterChange when provided. */
  onPan?: (t: number) => void;
  /** When true, replaces DetectionOverlay with a label editor element. */
  labelMode?: boolean;
  /** Which type of label editing is active. */
  labelEditMode?: "detection" | "vocalization" | null;
  /** Render function for the label editor; receives canvas width and height. */
  renderLabelEditor?: (width: number, height: number) => React.ReactNode;
  /** Render function for vocalization label editor; receives canvas width and height. */
  renderVocLabelEditor?: (width: number, height: number) => React.ReactNode;
  /** Which overlay to show: detection labels, vocalization types, or call parsing regions. */
  overlayMode?: "detection" | "vocalization" | "region";
  /** Vocalization labels for the overlay. */
  vocalizationLabels?: TimelineVocalizationLabel[];
  /** Called when a detection bar is clicked in view mode. Return true to suppress popover. */
  onDetectionBarClick?: (row: DetectionRow) => boolean;
  /** Custom tile URL builder. If omitted, uses default classifier detection URLs. */
  tileUrlBuilder?: (jobId: string, zoomLevel: string, tileIndex: number, freqMin: number, freqMax: number) => string;
  /** Custom audio slice URL builder. If omitted, uses default classifier detection URLs. */
  audioSliceUrlBuilder?: (jobId: string, startSec: number, durationSec: number) => string;
  /** Regions for the region overlay (call parsing). */
  regions?: { start_sec: number; end_sec: number; padded_start_sec: number; padded_end_sec: number; max_score: number }[];
}

// ---------------------------------------------------------------------------
// Frequency axis helper
// ---------------------------------------------------------------------------
function freqLabels(freqRange: [number, number]): { hz: number; label: string }[] {
  const [lo, hi] = freqRange;
  const span = hi - lo;
  // Pick a nice step based on range
  let step: number;
  if (span <= 500) step = 100;
  else if (span <= 2000) step = 500;
  else if (span <= 5000) step = 1000;
  else step = 2000;

  const labels: { hz: number; label: string }[] = [];
  const first = Math.ceil(lo / step) * step;
  for (let f = first; f <= hi; f += step) {
    if (f >= 1000) {
      labels.push({ hz: f, label: `${(f / 1000).toFixed(1)}k` });
    } else {
      labels.push({ hz: f, label: `${f}` });
    }
  }
  // Always add "Hz" label at bottom
  if (labels.length === 0 || labels[labels.length - 1].hz !== lo) {
    labels.push({ hz: lo, label: "Hz" });
  }
  return labels;
}

// ---------------------------------------------------------------------------
// Confidence color interpolation
// ---------------------------------------------------------------------------
function confidenceColor(score: number | null): string {
  if (score === null) return COLORS.bgDark;
  const s = Math.max(0, Math.min(1, score));
  // Find bracketing gradient stops
  for (let i = 1; i < CONFIDENCE_GRADIENT.length; i++) {
    const [prevT, prevC] = CONFIDENCE_GRADIENT[i - 1];
    const [curT, curC] = CONFIDENCE_GRADIENT[i];
    if (s <= curT) {
      const t = (s - prevT) / (curT - prevT);
      return lerpColor(prevC, curC, t);
    }
  }
  return CONFIDENCE_GRADIENT[CONFIDENCE_GRADIENT.length - 1][1];
}

function lerpColor(a: string, b: string, t: number): string {
  const pa = parseHex(a);
  const pb = parseHex(b);
  const r = Math.round(pa[0] + (pb[0] - pa[0]) * t);
  const g = Math.round(pa[1] + (pb[1] - pa[1]) * t);
  const bl = Math.round(pa[2] + (pb[2] - pa[2]) * t);
  return `rgb(${r},${g},${bl})`;
}

function parseHex(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}

// ---------------------------------------------------------------------------
// UTC time formatting
// ---------------------------------------------------------------------------
function formatTimeLabel(epoch: number, zoomLevel: ZoomLevel): string {
  const d = new Date(epoch * 1000);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = String(d.getUTCSeconds()).padStart(2, "0");

  // For wider zooms, show date too
  if (zoomLevel === "24h" || zoomLevel === "6h") {
    const mo = String(d.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(d.getUTCDate()).padStart(2, "0");
    return `${mo}-${dd} ${hh}:${mm}`;
  }
  // For narrow zooms, show seconds
  if (zoomLevel === "1m" || zoomLevel === "5m") {
    return `${hh}:${mm}:${ss}`;
  }
  return `${hh}:${mm}`;
}

// ---------------------------------------------------------------------------
// Time label step sizes per zoom
// ---------------------------------------------------------------------------
function timeLabelStepSec(zoomLevel: ZoomLevel): number {
  switch (zoomLevel) {
    case "24h": return 14400;  // 4 hours
    case "6h": return 3600;    // 1 hour
    case "1h": return 600;     // 10 min
    case "15m": return 120;    // 2 min
    case "5m": return 30;      // 30 sec
    case "1m": return 10;      // 10 sec
  }
}

// ---------------------------------------------------------------------------
// Confidence strip height
// ---------------------------------------------------------------------------
const CONFIDENCE_STRIP_HEIGHT = 20;
const TIME_AXIS_HEIGHT = 20;

// ---------------------------------------------------------------------------
// SpectrogramViewport
// ---------------------------------------------------------------------------
export function SpectrogramViewport({
  jobId,
  jobStart,
  jobEnd,
  centerTimestamp,
  zoomLevel,
  freqRange,
  isPlaying,
  scores,
  showLabels,
  detections,
  onCenterChange,
  onPan,
  labelMode,
  labelEditMode,
  renderLabelEditor,
  renderVocLabelEditor,
  overlayMode = "detection",
  vocalizationLabels,
  onDetectionBarClick,
  windowSec: windowSecProp,
  tileUrlBuilder,
  audioSliceUrlBuilder,
  regions,
}: SpectrogramViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);

  // Detection click popover state
  const [selectedRow, setSelectedRow] = useState<DetectionRow | null>(null);
  const [popoverPos, setPopoverPos] = useState({ x: 0, y: 0 });

  // Observe container size
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(Math.floor(entry.contentRect.width));
        setContainerHeight(Math.floor(entry.contentRect.height));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Canvas dimensions (exclude freq axis, confidence strip, time axis)
  const canvasWidth = Math.max(0, containerWidth - FREQ_AXIS_WIDTH_PX);
  const canvasHeight = Math.max(
    0,
    containerHeight - CONFIDENCE_STRIP_HEIGHT - TIME_AXIS_HEIGHT,
  );

  // Pan handling (only when not playing)
  const dragRef = useRef<{
    startX: number;
    startCenter: number;
  } | null>(null);

  const viewportSpan = VIEWPORT_SPAN[zoomLevel];
  const pxPerSec = canvasWidth > 0 ? canvasWidth / viewportSpan : 1;

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (isPlaying) return;
      dragRef.current = {
        startX: e.clientX,
        startCenter: centerTimestamp,
      };
    },
    [isPlaying, centerTimestamp],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragRef.current || isPlaying) return;
      const dx = e.clientX - dragRef.current.startX;
      const dt = dx / pxPerSec;
      const newCenter = dragRef.current.startCenter - dt;
      // Clamp to job bounds
      const clamped = Math.max(jobStart, Math.min(jobEnd, newCenter));
      // Use onPan (pauses playback) if provided, otherwise fall back to onCenterChange
      if (onPan) {
        onPan(clamped);
      } else {
        onCenterChange(clamped);
      }
    },
    [isPlaying, pxPerSec, jobStart, jobEnd, onCenterChange, onPan],
  );

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  useEffect(() => {
    // Clean up drag on mouse up anywhere
    const handleGlobalUp = () => {
      dragRef.current = null;
    };
    window.addEventListener("mouseup", handleGlobalUp);
    return () => window.removeEventListener("mouseup", handleGlobalUp);
  }, []);

  // Click handler: find detection row at click position
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (isPlaying) return;
      // If a drag occurred (pointer moved), skip click
      if (dragRef.current) return;

      const rect = e.currentTarget.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      // Convert pixel position to timestamp
      const clickTimestamp = centerTimestamp + (clickX - canvasWidth / 2) / pxPerSec;

      // Find detection whose [start_utc, end_utc] contains this timestamp
      const hit = detections.find((d) => {
        return clickTimestamp >= d.start_utc && clickTimestamp <= d.end_utc;
      });

      if (hit) {
        setSelectedRow(hit);
        setPopoverPos({ x: clickX, y: 8 });
      } else {
        setSelectedRow(null);
      }
    },
    [isPlaying, centerTimestamp, canvasWidth, pxPerSec, detections, jobStart],
  );

  // Frequency labels
  const fLabels = useMemo(() => freqLabels(freqRange), [freqRange]);

  // Time labels
  const timeLabels = useMemo(() => {
    if (canvasWidth <= 0) return [];
    const halfSpan = viewportSpan / 2;
    const viewStart = centerTimestamp - halfSpan;
    const viewEnd = centerTimestamp + halfSpan;
    const step = timeLabelStepSec(zoomLevel);

    const first = Math.ceil(viewStart / step) * step;
    const labels: { epoch: number; x: number; text: string }[] = [];
    for (let t = first; t <= viewEnd; t += step) {
      const x = (t - centerTimestamp) * pxPerSec + canvasWidth / 2;
      if (x >= -50 && x <= canvasWidth + 50) {
        labels.push({
          epoch: t,
          x,
          text: formatTimeLabel(t, zoomLevel),
        });
      }
    }
    return labels;
  }, [centerTimestamp, viewportSpan, zoomLevel, canvasWidth, pxPerSec]);

  // Confidence strip: compute which scores fall in view using timestamp-based positioning
  const confidenceStrip = useMemo(() => {
    if (scores.length === 0 || canvasWidth <= 0) return null;

    const totalDuration = jobEnd - jobStart;
    const windowSec = windowSecProp ?? totalDuration / scores.length;
    const halfSpan = viewportSpan / 2;
    const viewStart = centerTimestamp - halfSpan;
    const viewEnd = centerTimestamp + halfSpan;

    const barWidthPx = windowSec * pxPerSec;
    const startIdx = Math.max(0, Math.floor((viewStart - jobStart) / windowSec));
    const endIdx = Math.min(
      scores.length - 1,
      Math.ceil((viewEnd - jobStart) / windowSec),
    );

    const bars: { x: number; w: number; color: string }[] = [];
    for (let i = startIdx; i <= endIdx; i++) {
      const windowStart = jobStart + i * windowSec;
      const x = (windowStart - centerTimestamp) * pxPerSec + canvasWidth / 2;
      bars.push({
        x,
        w: Math.max(1, barWidthPx),
        color: confidenceColor(scores[i]),
      });
    }
    return bars;
  }, [scores, canvasWidth, centerTimestamp, viewportSpan, pxPerSec, jobStart, jobEnd, windowSecProp]);

  // Cursor style for pan
  const cursorStyle = isPlaying
    ? "default"
    : dragRef.current
      ? "grabbing"
      : "grab";

  return (
    <div ref={containerRef} className="flex-1 flex flex-col relative select-none min-h-0">
      <div className="flex flex-1 min-h-0">
        {/* Frequency axis */}
        <div
          className="flex flex-col justify-between shrink-0 text-right pr-1 py-1"
          style={{
            width: FREQ_AXIS_WIDTH_PX,
            color: COLORS.textMuted,
            fontSize: "9px",
          }}
        >
          {fLabels.map((l, i) => {
            // Position: top = 0 is high freq, bottom = canvasHeight is low freq
            const frac =
              freqRange[1] === freqRange[0]
                ? 0
                : (freqRange[1] - l.hz) / (freqRange[1] - freqRange[0]);
            return (
              <div
                key={`${l.hz}-${i}`}
                className="absolute"
                style={{
                  top: `${frac * 100}%`,
                  right: canvasWidth > 0 ? undefined : 0,
                  width: FREQ_AXIS_WIDTH_PX - 4,
                  textAlign: "right",
                  transform: "translateY(-50%)",
                }}
              >
                {l.label}
              </div>
            );
          })}
        </div>

        {/* Canvas area + overlays */}
        <div
          className="relative flex-1 min-w-0"
          style={{ cursor: cursorStyle }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={handleCanvasClick}
        >
          {canvasWidth > 0 && canvasHeight > 0 && (
            <TileCanvas
              jobId={jobId}
              jobStart={jobStart}
              jobEnd={jobEnd}
              centerTimestamp={centerTimestamp}
              zoomLevel={zoomLevel}
              freqRange={freqRange}
              width={canvasWidth}
              height={canvasHeight}
              tileUrlBuilder={tileUrlBuilder}
            />
          )}

          {/* Detection/vocalization label overlay (view mode) or label editor (edit mode) */}
          {!labelMode && overlayMode === "detection" && (
            <DetectionOverlay
              detections={detections}
              jobStart={jobStart}
              centerTimestamp={centerTimestamp}
              zoomLevel={zoomLevel}
              width={canvasWidth}
              height={canvasHeight + CONFIDENCE_STRIP_HEIGHT}
              visible={showLabels}
              onDetectionClick={(row, x, y) => {
                if (onDetectionBarClick?.(row)) return;
                setSelectedRow(row);
                setPopoverPos({ x, y });
              }}
            />
          )}
          {!labelMode && overlayMode === "vocalization" && (
            <VocalizationOverlay
              labels={vocalizationLabels ?? []}
              centerTimestamp={centerTimestamp}
              zoomLevel={zoomLevel}
              width={canvasWidth}
              height={canvasHeight + CONFIDENCE_STRIP_HEIGHT}
              visible={showLabels}
            />
          )}
          {overlayMode === "region" && regions && (
            <RegionOverlay
              regions={regions}
              jobStart={jobStart}
              centerTimestamp={centerTimestamp}
              zoomLevel={zoomLevel}
              width={canvasWidth}
              height={canvasHeight + CONFIDENCE_STRIP_HEIGHT}
              visible={true}
            />
          )}
          {labelMode && labelEditMode === "detection" && renderLabelEditor?.(canvasWidth, canvasHeight + CONFIDENCE_STRIP_HEIGHT)}
          {labelMode && labelEditMode === "vocalization" && renderVocLabelEditor?.(canvasWidth, canvasHeight + CONFIDENCE_STRIP_HEIGHT)}

          {/* Detection click popover */}
          {selectedRow !== null && (
            <DetectionPopover
              row={selectedRow}
              jobStart={jobStart}
              position={popoverPos}
              onClose={() => setSelectedRow(null)}
            />
          )}

          {/* Playhead — vertical center line */}
          <div
            className="absolute pointer-events-none"
            style={{
              left: canvasWidth / 2,
              top: 0,
              width: 0,
              height: canvasHeight + CONFIDENCE_STRIP_HEIGHT,
              borderLeft: `1.5px solid ${COLORS.accent}`,
              zIndex: 10,
            }}
          >
            {/* Small downward triangle at top */}
            <div
              style={{
                position: "absolute",
                top: -1,
                left: -5,
                width: 0,
                height: 0,
                borderLeft: "5px solid transparent",
                borderRight: "5px solid transparent",
                borderTop: `6px solid ${COLORS.accent}`,
              }}
            />
          </div>

          {/* Confidence strip below canvas (hidden in vocalization mode) */}
          <div
            style={{
              height: CONFIDENCE_STRIP_HEIGHT,
              position: "relative",
              background: COLORS.bgDark,
              overflow: "hidden",
            }}
          >
            {confidenceStrip && overlayMode !== "vocalization" && (
              <svg
                width={canvasWidth}
                height={CONFIDENCE_STRIP_HEIGHT}
                style={{ display: "block" }}
              >
                {confidenceStrip.map((bar, i) => (
                  <rect
                    key={i}
                    x={bar.x}
                    y={0}
                    width={bar.w}
                    height={CONFIDENCE_STRIP_HEIGHT}
                    fill={bar.color}
                  />
                ))}
              </svg>
            )}
          </div>

          {/* Time axis */}
          <div
            className="relative"
            style={{
              height: TIME_AXIS_HEIGHT,
              color: COLORS.textMuted,
              fontSize: "9px",
              overflow: "hidden",
            }}
          >
            {timeLabels.map((tl) => {
              const isCenter =
                Math.abs(tl.x - canvasWidth / 2) < canvasWidth * 0.05;
              return (
                <span
                  key={tl.epoch}
                  className="absolute whitespace-nowrap"
                  style={{
                    left: tl.x,
                    top: 2,
                    transform: "translateX(-50%)",
                    color: isCenter ? COLORS.accent : COLORS.textMuted,
                    fontWeight: isCenter ? 600 : 400,
                  }}
                >
                  {tl.text}
                </span>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
