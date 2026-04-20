import { useMemo } from "react";
import { COLORS, FREQ_AXIS_WIDTH_PX } from "../constants";

interface FrequencyAxisProps {
  freqRange: [number, number];
  height: number;
}

export function freqLabels(freqRange: [number, number]): { hz: number; label: string }[] {
  const [lo, hi] = freqRange;
  const span = hi - lo;
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
  if (labels.length === 0 || labels[labels.length - 1].hz !== lo) {
    labels.push({ hz: lo, label: "Hz" });
  }
  return labels;
}

export function FrequencyAxis({ freqRange, height }: FrequencyAxisProps) {
  const labels = useMemo(() => freqLabels(freqRange), [freqRange]);

  return (
    <div
      className="relative shrink-0"
      style={{
        width: FREQ_AXIS_WIDTH_PX,
        height,
        color: COLORS.textMuted,
        fontSize: "9px",
      }}
    >
      {labels.map((l, i) => {
        const frac =
          freqRange[1] === freqRange[0]
            ? 0
            : (freqRange[1] - l.hz) / (freqRange[1] - freqRange[0]);
        return (
          <div
            key={`${l.hz}-${i}`}
            className="absolute text-right"
            style={{
              top: `${frac * 100}%`,
              width: FREQ_AXIS_WIDTH_PX - 4,
              transform: "translateY(-50%)",
            }}
          >
            {l.label}
          </div>
        );
      })}
    </div>
  );
}
