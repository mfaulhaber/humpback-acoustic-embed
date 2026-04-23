import { useMemo } from "react";
import { COLORS, CONFIDENCE_GRADIENT } from "../constants";

export const DEFAULT_STRIP_HEIGHT = 20;

export type GradientStops = readonly (readonly [number, string])[];

interface ConfidenceStripProps {
  scores: (number | null)[];
  windowSec?: number;
  jobStart: number;
  jobEnd: number;
  viewStart: number;
  viewEnd: number;
  pxPerSec: number;
  canvasWidth: number;
  centerTimestamp: number;
  height?: number;
  gradient?: GradientStops;
  thresholdValue?: number;
  barMode?: boolean;
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

function gradientColor(
  score: number | null,
  gradient: GradientStops,
): string {
  if (score === null) return COLORS.bgDark;
  const s = Math.max(0, Math.min(1, score));
  for (let i = 1; i < gradient.length; i++) {
    const [prevT, prevC] = gradient[i - 1];
    const [curT, curC] = gradient[i];
    if (s <= curT) {
      const t = (s - prevT) / (curT - prevT);
      return lerpColor(prevC, curC, t);
    }
  }
  return gradient[gradient.length - 1][1];
}

export function ConfidenceStrip({
  scores,
  windowSec: windowSecProp,
  jobStart,
  jobEnd,
  viewStart,
  viewEnd,
  pxPerSec,
  canvasWidth,
  centerTimestamp,
  height: heightProp,
  gradient: gradientProp,
  thresholdValue,
  barMode,
}: ConfidenceStripProps) {
  const h = heightProp ?? DEFAULT_STRIP_HEIGHT;
  const grad = gradientProp ?? CONFIDENCE_GRADIENT;

  const bars = useMemo(() => {
    if (scores.length === 0 || canvasWidth <= 0) return null;

    const totalDuration = jobEnd - jobStart;
    const windowSec = windowSecProp ?? totalDuration / scores.length;
    const barWidthPx = windowSec * pxPerSec;
    const startIdx = Math.max(0, Math.floor((viewStart - jobStart) / windowSec));
    const endIdx = Math.min(scores.length - 1, Math.ceil((viewEnd - jobStart) / windowSec));

    const result: { x: number; w: number; color: string; barH: number; barY: number }[] = [];
    for (let i = startIdx; i <= endIdx; i++) {
      const windowStart = jobStart + i * windowSec;
      const x = (windowStart - centerTimestamp) * pxPerSec + canvasWidth / 2;
      const color = gradientColor(scores[i], grad);
      const s = scores[i] ?? 0;
      const barH = barMode ? Math.max(1, s * h) : h;
      const barY = barMode ? h - barH : 0;
      result.push({ x, w: Math.max(1, barWidthPx), color, barH, barY });
    }
    return result;
  }, [scores, canvasWidth, centerTimestamp, viewStart, viewEnd, pxPerSec, jobStart, jobEnd, windowSecProp, grad, barMode, h]);

  const thresholdY =
    thresholdValue != null && barMode
      ? h - thresholdValue * h
      : undefined;

  return (
    <div
      style={{
        height: h,
        position: "relative",
        background: COLORS.bgDark,
        overflow: "hidden",
      }}
    >
      {bars && (
        <svg width={canvasWidth} height={h} style={{ display: "block" }}>
          {bars.map((bar, i) => (
            <rect key={i} x={bar.x} y={bar.barY} width={bar.w} height={bar.barH} fill={bar.color} />
          ))}
          {thresholdY != null && (
            <line
              x1={0}
              y1={thresholdY}
              x2={canvasWidth}
              y2={thresholdY}
              stroke="rgba(255,255,255,0.5)"
              strokeWidth={1}
              strokeDasharray="4 3"
            />
          )}
        </svg>
      )}
    </div>
  );
}
