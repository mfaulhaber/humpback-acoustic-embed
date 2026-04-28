import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import { FREQ_AXIS_WIDTH_PX } from "@/components/timeline/constants";
import { STATE_COLORS } from "./constants";

export interface ViterbiWindow {
  merged_span_id: number;
  window_index_in_span: number;
  start_time_sec: number;
  end_time_sec: number;
  viterbi_state: number;
  max_state_probability: number;
}

interface HMMStateBarProps {
  items: ViterbiWindow[];
  nStates: number;
}

const BAR_HEIGHT = 60;

export function HMMStateBar({ items, nStates }: HMMStateBarProps) {
  const ctx = useTimelineContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    text: string;
  } | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setWidth(entry.contentRect.width);
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const canvasWidth = Math.max(0, width - FREQ_AXIS_WIDTH_PX);
  const { viewStart, viewEnd, pxPerSec, centerTimestamp } = ctx;

  const visibleItems = useMemo(() => {
    if (items.length === 0) return [];
    return items.filter(
      (w) => w.end_time_sec > viewStart && w.start_time_sec < viewEnd,
    );
  }, [items, viewStart, viewEnd]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvasWidth <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * dpr;
    canvas.height = BAR_HEIGHT * dpr;
    const ctx2d = canvas.getContext("2d");
    if (!ctx2d) return;
    ctx2d.scale(dpr, dpr);
    ctx2d.clearRect(0, 0, canvasWidth, BAR_HEIGHT);

    const stateHeight = BAR_HEIGHT / nStates;

    for (const w of visibleItems) {
      const x = (w.start_time_sec - viewStart) * pxPerSec;
      const barWidth = (w.end_time_sec - w.start_time_sec) * pxPerSec;
      const y = (nStates - 1 - w.viterbi_state) * stateHeight;
      ctx2d.fillStyle = STATE_COLORS[w.viterbi_state % STATE_COLORS.length];
      ctx2d.fillRect(x, y, Math.max(barWidth, 1), stateHeight);
    }

    // Playhead
    const playheadX = (centerTimestamp - viewStart) * pxPerSec;
    if (playheadX >= 0 && playheadX <= canvasWidth) {
      ctx2d.strokeStyle = "#ef4444";
      ctx2d.lineWidth = 1.5;
      ctx2d.beginPath();
      ctx2d.moveTo(playheadX, 0);
      ctx2d.lineTo(playheadX, BAR_HEIGHT);
      ctx2d.stroke();
    }
  }, [visibleItems, canvasWidth, viewStart, pxPerSec, centerTimestamp, nStates]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (items.length === 0) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const time = viewStart + px / pxPerSec;

      let lo = 0;
      let hi = items.length - 1;
      let found: ViterbiWindow | null = null;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (items[mid].start_time_sec <= time) {
          if (items[mid].end_time_sec > time) {
            found = items[mid];
            break;
          }
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }

      if (found) {
        setTooltip({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
          text: `State ${found.viterbi_state} · ${found.start_time_sec.toFixed(1)}s–${found.end_time_sec.toFixed(1)}s · prob ${found.max_state_probability.toFixed(2)}`,
        });
      } else {
        setTooltip(null);
      }
    },
    [items, viewStart, pxPerSec],
  );

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  return (
    <div ref={containerRef} className="relative flex" style={{ height: BAR_HEIGHT }}>
      {/* Y-axis labels */}
      <div
        className="flex flex-col justify-between text-[10px] text-muted-foreground shrink-0 pr-1"
        style={{ width: FREQ_AXIS_WIDTH_PX }}
      >
        {Array.from({ length: nStates }, (_, i) => nStates - 1 - i).map(
          (s) => (
            <span key={s} className="text-right leading-none">
              {s}
            </span>
          ),
        )}
      </div>
      {/* Canvas */}
      <div className="relative flex-1 overflow-hidden">
        <canvas
          ref={canvasRef}
          className="block"
          style={{ width: canvasWidth, height: BAR_HEIGHT }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />
        {tooltip && (
          <div
            className="pointer-events-none absolute z-50 rounded bg-popover px-2 py-1 text-xs text-popover-foreground shadow-md"
            style={{
              left: tooltip.x + 8,
              top: tooltip.y - 28,
            }}
          >
            {tooltip.text}
          </div>
        )}
      </div>
    </div>
  );
}
