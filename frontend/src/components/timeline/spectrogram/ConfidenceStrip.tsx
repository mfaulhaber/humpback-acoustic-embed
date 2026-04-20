import { useMemo } from "react";
import { COLORS, CONFIDENCE_GRADIENT } from "../constants";

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
}

const CONFIDENCE_STRIP_HEIGHT = 20;

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

function confidenceColor(score: number | null): string {
  if (score === null) return COLORS.bgDark;
  const s = Math.max(0, Math.min(1, score));
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
}: ConfidenceStripProps) {
  const bars = useMemo(() => {
    if (scores.length === 0 || canvasWidth <= 0) return null;

    const totalDuration = jobEnd - jobStart;
    const windowSec = windowSecProp ?? totalDuration / scores.length;
    const barWidthPx = windowSec * pxPerSec;
    const startIdx = Math.max(0, Math.floor((viewStart - jobStart) / windowSec));
    const endIdx = Math.min(scores.length - 1, Math.ceil((viewEnd - jobStart) / windowSec));

    const result: { x: number; w: number; color: string }[] = [];
    for (let i = startIdx; i <= endIdx; i++) {
      const windowStart = jobStart + i * windowSec;
      const x = (windowStart - centerTimestamp) * pxPerSec + canvasWidth / 2;
      result.push({ x, w: Math.max(1, barWidthPx), color: confidenceColor(scores[i]) });
    }
    return result;
  }, [scores, canvasWidth, centerTimestamp, viewStart, viewEnd, pxPerSec, jobStart, jobEnd, windowSecProp]);

  return (
    <div
      style={{
        height: CONFIDENCE_STRIP_HEIGHT,
        position: "relative",
        background: COLORS.bgDark,
        overflow: "hidden",
      }}
    >
      {bars && (
        <svg width={canvasWidth} height={CONFIDENCE_STRIP_HEIGHT} style={{ display: "block" }}>
          {bars.map((bar, i) => (
            <rect key={i} x={bar.x} y={0} width={bar.w} height={CONFIDENCE_STRIP_HEIGHT} fill={bar.color} />
          ))}
        </svg>
      )}
    </div>
  );
}
