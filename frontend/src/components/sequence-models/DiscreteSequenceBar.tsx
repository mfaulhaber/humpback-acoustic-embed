import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import { FREQ_AXIS_WIDTH_PX } from "@/components/timeline/constants";
import { LABEL_COLORS, labelColor } from "./constants";

const ROWS_BAR_HEIGHT = 60;
const SINGLE_ROW_BAR_HEIGHT = 60;

export interface DiscreteSequenceItem {
  start_timestamp: number;
  end_timestamp: number;
  /** Discrete category — HMM state index for HMM jobs, k-means token
   *  index for masked-transformer jobs. */
  label: number;
  /** Confidence, max-state probability, or token confidence in [0,1]. */
  confidence?: number;
}

export interface DiscreteSequenceBarProps {
  items: DiscreteSequenceItem[];
  /** Total number of distinct labels (HMM states or k tokens).
   *  Drives both the row layout (``rows`` mode) and the color palette. */
  numLabels: number;
  /** ``rows``: one row per label (HMM-style); ``single-row``: full-height
   *  fillRect per chunk (masked-transformer-style). */
  mode?: "rows" | "single-row";
  /** Optional region overlay drawn as bright vertical guides. */
  currentRegion?: {
    startTimestamp: number;
    endTimestamp: number;
  } | null;
  /** Optional palette override. Defaults to ``LABEL_COLORS``. */
  colorPalette?: string[];
  /** Optional formatter for hover tooltip. Receives the item and label index. */
  tooltipFormatter?: (item: DiscreteSequenceItem) => string;
  /** Test-id override (defaults to ``discrete-sequence-bar``). */
  testId?: string;
  /** Aria-label override (defaults to ``Discrete sequence timeline``). */
  ariaLabel?: string;
}

function binarySearchItem(
  items: DiscreteSequenceItem[],
  timeSec: number,
): number {
  let lo = 0;
  let hi = items.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (items[mid].end_timestamp <= timeSec) {
      lo = mid + 1;
    } else if (items[mid].start_timestamp > timeSec) {
      hi = mid - 1;
    } else {
      return mid;
    }
  }
  return -1;
}

export function visibleItems(
  items: DiscreteSequenceItem[],
  viewStart: number,
  viewEnd: number,
): DiscreteSequenceItem[] {
  return items.filter(
    (item) => item.end_timestamp >= viewStart && item.start_timestamp <= viewEnd,
  );
}

export function regionBoundaryXPositions(
  region: { startTimestamp: number; endTimestamp: number },
  viewStart: number,
  pxPerSec: number,
): { startX: number; endX: number } {
  return {
    startX: (region.startTimestamp - viewStart) * pxPerSec,
    endX: (region.endTimestamp - viewStart) * pxPerSec,
  };
}

export function DiscreteSequenceBar({
  items,
  numLabels,
  mode = "rows",
  currentRegion,
  colorPalette,
  tooltipFormatter,
  testId,
  ariaLabel,
}: DiscreteSequenceBarProps) {
  const ctx = useTimelineContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    text: string;
  } | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(Math.floor(entry.contentRect.width));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const barHeight = mode === "rows" ? ROWS_BAR_HEIGHT : SINGLE_ROW_BAR_HEIGHT;
  const canvasWidth = Math.max(0, containerWidth - FREQ_AXIS_WIDTH_PX);
  const labelHeight =
    mode === "rows" && numLabels > 0 ? barHeight / numLabels : barHeight;

  const sortedItems = useMemo(
    () => [...items].sort((a, b) => a.start_timestamp - b.start_timestamp),
    [items],
  );

  const colorFor = useCallback(
    (label: number): string => {
      if (colorPalette && colorPalette.length > 0) {
        return colorPalette[label % colorPalette.length];
      }
      return labelColor(label, Math.max(numLabels, 1));
    },
    [colorPalette, numLabels],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvasWidth <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * dpr;
    canvas.height = barHeight * dpr;
    const g = canvas.getContext("2d");
    if (!g) return;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, canvasWidth, barHeight);

    for (const w of visibleItems(sortedItems, ctx.viewStart, ctx.viewEnd)) {
      const x0 = (w.start_timestamp - ctx.viewStart) * ctx.pxPerSec;
      const x1 = (w.end_timestamp - ctx.viewStart) * ctx.pxPerSec;
      if (x1 < 0 || x0 > canvasWidth) continue;
      const px = Math.max(0, x0);
      const pw = Math.min(canvasWidth, x1) - px;
      g.fillStyle = colorFor(w.label);
      if (mode === "rows") {
        const row = numLabels - 1 - w.label;
        const py = row * labelHeight;
        g.fillRect(px, py, Math.max(pw, 1), labelHeight);
      } else {
        g.fillRect(px, 0, Math.max(pw, 1), barHeight);
      }
    }

    const playheadEpoch = ctx.playbackEpoch ?? ctx.centerTimestamp;
    const phX = (playheadEpoch - ctx.viewStart) * ctx.pxPerSec;
    if (phX >= 0 && phX <= canvasWidth) {
      g.strokeStyle = "red";
      g.lineWidth = 1.5;
      g.beginPath();
      g.moveTo(phX, 0);
      g.lineTo(phX, barHeight);
      g.stroke();
    }

    if (currentRegion) {
      const { startX, endX } = regionBoundaryXPositions(
        currentRegion,
        ctx.viewStart,
        ctx.pxPerSec,
      );
      g.strokeStyle = "rgba(255, 255, 255, 0.95)";
      g.lineWidth = 2;
      for (const x of [startX, endX]) {
        if (x < 0 || x > canvasWidth) continue;
        g.beginPath();
        g.moveTo(x, 0);
        g.lineTo(x, barHeight);
        g.stroke();
      }
    }
  }, [
    sortedItems,
    ctx.viewStart,
    ctx.viewEnd,
    ctx.pxPerSec,
    ctx.centerTimestamp,
    ctx.playbackEpoch,
    canvasWidth,
    numLabels,
    labelHeight,
    barHeight,
    mode,
    currentRegion,
    colorFor,
  ]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (ctx.isDraggingTimeline) {
        ctx.updateDragPan(e.clientX);
        setTooltip(null);
        return;
      }
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      const mx = e.clientX - rect.left;
      const timeSec = ctx.viewStart + mx / ctx.pxPerSec;
      const idx = binarySearchItem(sortedItems, timeSec);
      if (idx < 0) {
        setTooltip(null);
        return;
      }
      const w = sortedItems[idx];
      let ty: number;
      if (mode === "rows") {
        const row = numLabels - 1 - w.label;
        ty = row * labelHeight + labelHeight / 2;
      } else {
        ty = barHeight / 2;
      }
      const text = tooltipFormatter
        ? tooltipFormatter(w)
        : `Label ${w.label} · ${w.start_timestamp.toFixed(1)}s–${w.end_timestamp.toFixed(1)}s${
            w.confidence !== undefined
              ? ` · conf ${w.confidence.toFixed(3)}`
              : ""
          }`;
      setTooltip({
        x: mx + FREQ_AXIS_WIDTH_PX,
        y: ty,
        text,
      });
    },
    [sortedItems, ctx, numLabels, labelHeight, barHeight, mode, tooltipFormatter],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (ctx.beginDragPan(e.clientX)) {
        setTooltip(null);
      }
    },
    [ctx],
  );

  const handleMouseUp = useCallback(() => {
    ctx.endDragPan();
  }, [ctx]);

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  const cursor = ctx.isPlaying
    ? "default"
    : ctx.isDraggingTimeline
      ? "grabbing"
      : "grab";

  return (
    <div
      ref={containerRef}
      className="relative"
      data-testid={testId ?? "discrete-sequence-bar"}
    >
      <div className="flex">
        {/* Y-axis label strip */}
        <div
          className="flex flex-col justify-between text-[9px] text-muted-foreground shrink-0"
          style={{ width: FREQ_AXIS_WIDTH_PX, height: barHeight }}
        >
          {mode === "rows" ? (
            Array.from({ length: numLabels }, (_, i) => {
              const labelIdx = numLabels - 1 - i;
              return (
                <div
                  key={labelIdx}
                  className="flex items-center justify-end pr-1"
                  style={{ height: labelHeight }}
                >
                  {labelIdx}
                </div>
              );
            })
          ) : (
            <div
              className="flex items-center justify-end pr-1"
              style={{ height: barHeight }}
            >
              tokens
            </div>
          )}
        </div>
        <canvas
          ref={canvasRef}
          className="block"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          data-testid={`${testId ?? "discrete-sequence-bar"}-canvas`}
          aria-label={ariaLabel ?? "Discrete sequence timeline"}
          role="img"
          style={{ width: canvasWidth, height: barHeight, cursor }}
        />
      </div>
      {tooltip && (
        <div
          className="absolute z-10 rounded bg-popover border px-2 py-1 text-xs text-popover-foreground shadow pointer-events-none whitespace-nowrap"
          style={{
            left: tooltip.x + 8,
            top: tooltip.y - 12,
          }}
        >
          {tooltip.text}
        </div>
      )}
    </div>
  );
}

export { binarySearchItem as _binarySearchItem };
export { LABEL_COLORS as DEFAULT_LABEL_COLORS };
