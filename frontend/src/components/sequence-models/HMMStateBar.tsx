import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import { FREQ_AXIS_WIDTH_PX } from "@/components/timeline/constants";
import { STATE_COLORS } from "./constants";

const BAR_HEIGHT = 60;

export interface ViterbiWindow {
  start_timestamp: number;
  end_timestamp: number;
  viterbi_state: number;
  max_state_probability: number;
}

interface HMMStateBarProps {
  items: ViterbiWindow[];
  nStates: number;
}

function binarySearchWindow(items: ViterbiWindow[], timeSec: number): number {
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

export function HMMStateBar({ items, nStates }: HMMStateBarProps) {
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

  const canvasWidth = Math.max(0, containerWidth - FREQ_AXIS_WIDTH_PX);
  const stateHeight = nStates > 0 ? BAR_HEIGHT / nStates : BAR_HEIGHT;

  const sortedItems = useMemo(
    () => [...items].sort((a, b) => a.start_timestamp - b.start_timestamp),
    [items],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvasWidth <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * dpr;
    canvas.height = BAR_HEIGHT * dpr;
    const g = canvas.getContext("2d");
    if (!g) return;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, canvasWidth, BAR_HEIGHT);

    for (const w of sortedItems) {
      const x0 = (w.start_timestamp - ctx.viewStart) * ctx.pxPerSec;
      const x1 = (w.end_timestamp - ctx.viewStart) * ctx.pxPerSec;
      if (x1 < 0 || x0 > canvasWidth) continue;
      const px = Math.max(0, x0);
      const pw = Math.min(canvasWidth, x1) - px;
      const row = nStates - 1 - w.viterbi_state;
      const py = row * stateHeight;
      g.fillStyle = STATE_COLORS[w.viterbi_state % STATE_COLORS.length];
      g.fillRect(px, py, Math.max(pw, 1), stateHeight);
    }

    const playheadEpoch = ctx.playbackEpoch ?? ctx.centerTimestamp;
    const phX = (playheadEpoch - ctx.viewStart) * ctx.pxPerSec;
    if (phX >= 0 && phX <= canvasWidth) {
      g.strokeStyle = "red";
      g.lineWidth = 1.5;
      g.beginPath();
      g.moveTo(phX, 0);
      g.lineTo(phX, BAR_HEIGHT);
      g.stroke();
    }
  }, [sortedItems, ctx.viewStart, ctx.viewEnd, ctx.pxPerSec, ctx.centerTimestamp, ctx.playbackEpoch, canvasWidth, nStates, stateHeight]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      const mx = e.clientX - rect.left;
      const timeSec = ctx.viewStart + mx / ctx.pxPerSec;
      const idx = binarySearchWindow(sortedItems, timeSec);
      if (idx < 0) {
        setTooltip(null);
        return;
      }
      const w = sortedItems[idx];
      const row = nStates - 1 - w.viterbi_state;
      const ty = row * stateHeight + stateHeight / 2;
      setTooltip({
        x: mx + FREQ_AXIS_WIDTH_PX,
        y: ty,
        text: `State ${w.viterbi_state} · ${w.start_timestamp.toFixed(1)}s–${w.end_timestamp.toFixed(1)}s · prob ${w.max_state_probability.toFixed(3)}`,
      });
    },
    [sortedItems, ctx.viewStart, ctx.pxPerSec, nStates, stateHeight],
  );

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  return (
    <div ref={containerRef} className="relative" data-testid="hmm-state-bar">
      <div className="flex">
        {/* Y-axis label strip */}
        <div
          className="flex flex-col justify-between text-[9px] text-muted-foreground shrink-0"
          style={{ width: FREQ_AXIS_WIDTH_PX, height: BAR_HEIGHT }}
        >
          {Array.from({ length: nStates }, (_, i) => {
            const state = nStates - 1 - i;
            return (
              <div key={state} className="flex items-center justify-end pr-1" style={{ height: stateHeight }}>
                {state}
              </div>
            );
          })}
        </div>
        <canvas
          ref={canvasRef}
          style={{ width: canvasWidth, height: BAR_HEIGHT }}
          className="block"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
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

export { binarySearchWindow as _binarySearchWindow };
