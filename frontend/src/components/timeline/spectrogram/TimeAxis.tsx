import { useMemo } from "react";
import { COLORS } from "../constants";

interface TimeAxisProps {
  viewStart: number;
  viewEnd: number;
  viewportSpan: number;
  pxPerSec: number;
  canvasWidth: number;
  centerTimestamp: number;
}

const TIME_AXIS_HEIGHT = 20;

export function formatTimeLabel(epoch: number, span: number): string {
  const d = new Date(epoch * 1000);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = String(d.getUTCSeconds()).padStart(2, "0");
  const mo = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");

  if (span >= 21600) return `${mo}-${dd} ${hh}:${mm}`;
  if (span <= 300) return `${hh}:${mm}:${ss}`;
  return `${hh}:${mm}`;
}

export function timeLabelStepSec(span: number): number {
  if (span >= 86400) return 14400;
  if (span >= 21600) return 3600;
  if (span >= 3600) return 600;
  if (span >= 900) return 120;
  if (span >= 300) return 30;
  if (span >= 60) return 10;
  return 5;
}

export function TimeAxis({ viewStart, viewEnd, viewportSpan, pxPerSec, canvasWidth, centerTimestamp }: TimeAxisProps) {
  const labels = useMemo(() => {
    if (canvasWidth <= 0) return [];
    const step = timeLabelStepSec(viewportSpan);
    const first = Math.ceil(viewStart / step) * step;
    const result: { epoch: number; x: number; text: string }[] = [];
    for (let t = first; t <= viewEnd; t += step) {
      const x = (t - centerTimestamp) * pxPerSec + canvasWidth / 2;
      if (x >= -50 && x <= canvasWidth + 50) {
        result.push({ epoch: t, x, text: formatTimeLabel(t, viewportSpan) });
      }
    }
    return result;
  }, [viewStart, viewEnd, viewportSpan, pxPerSec, canvasWidth, centerTimestamp]);

  return (
    <div
      className="relative"
      style={{
        height: TIME_AXIS_HEIGHT,
        color: COLORS.textMuted,
        fontSize: "9px",
        overflow: "hidden",
      }}
    >
      {labels.map((tl) => {
        const isCenter = Math.abs(tl.x - canvasWidth / 2) < canvasWidth * 0.05;
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
  );
}
