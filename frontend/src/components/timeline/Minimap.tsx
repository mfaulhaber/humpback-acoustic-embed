// frontend/src/components/timeline/Minimap.tsx
import { useRef, useEffect, useCallback } from "react";
import { COLORS, CONFIDENCE_GRADIENT } from "./constants";

interface MinimapProps {
  scores: (number | null)[];
  jobStart: number; // UTC epoch seconds
  jobEnd: number; // UTC epoch seconds
  centerTimestamp: number; // current center position
  viewportSpan: number; // current zoom viewport width in seconds
  onCenterChange: (t: number) => void; // callback when user clicks/drags
}

function confidenceToColor(value: number | null): string {
  if (value === null) return COLORS.bgDark;
  const stops = CONFIDENCE_GRADIENT;
  if (value <= stops[0][0]) return stops[0][1] as string;
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, colorA] = stops[i];
    const [b, colorB] = stops[i + 1];
    if (value <= b) {
      const t = (value - a) / (b - a);
      return t < 0.5 ? (colorA as string) : (colorB as string);
    }
  }
  return stops[stops.length - 1][1] as string;
}

function formatUtcLabel(epochSec: number): string {
  const d = new Date(epochSec * 1000);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return `${hh}:${mm}Z`;
}

export function Minimap({
  scores,
  jobStart,
  jobEnd,
  centerTimestamp,
  viewportSpan,
  onCenterChange,
}: MinimapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDraggingRef = useRef(false);
  const jobDuration = jobEnd - jobStart;

  // Draw heatmap and viewport indicator
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = canvas;

    // Clear background
    ctx.fillStyle = COLORS.bgDark;
    ctx.fillRect(0, 0, width, height);

    // Draw confidence heatmap
    if (scores.length > 0 && jobDuration > 0) {
      const blockWidth = width / scores.length;
      scores.forEach((score, i) => {
        ctx.fillStyle = confidenceToColor(score);
        ctx.fillRect(Math.floor(i * blockWidth), 0, Math.ceil(blockWidth), height);
      });
    }

    // Draw viewport indicator
    if (jobDuration > 0) {
      const vpWidth = Math.max(4, (viewportSpan / jobDuration) * width);
      const vpX = ((centerTimestamp - jobStart) / jobDuration) * width - vpWidth / 2;
      const clampedX = Math.max(0, Math.min(width - vpWidth, vpX));

      // Semi-transparent fill
      ctx.fillStyle = "rgba(112, 224, 192, 0.08)";
      ctx.fillRect(clampedX, 0, vpWidth, height);

      // Border
      ctx.strokeStyle = "#70e0c0";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(clampedX + 0.75, 0.75, vpWidth - 1.5, height - 1.5);
    }
  }, [scores, jobStart, jobEnd, centerTimestamp, viewportSpan, jobDuration]);

  const timestampFromClientX = useCallback(
    (clientX: number): number => {
      const canvas = canvasRef.current;
      if (!canvas || jobDuration <= 0) return centerTimestamp;
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, x / rect.width));
      return jobStart + ratio * jobDuration;
    },
    [jobStart, jobDuration, centerTimestamp],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      isDraggingRef.current = true;
      onCenterChange(timestampFromClientX(e.clientX));
    },
    [onCenterChange, timestampFromClientX],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isDraggingRef.current) return;
      onCenterChange(timestampFromClientX(e.clientX));
    },
    [onCenterChange, timestampFromClientX],
  );

  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = false;
  }, []);

  const handleMouseLeave = useCallback(() => {
    isDraggingRef.current = false;
  }, []);

  return (
    <div className="px-4 pt-1.5">
      <div
        className="relative rounded overflow-hidden"
        style={{ background: COLORS.bgDark }}
      >
        {/* Labels row */}
        <div
          className="flex justify-between items-center px-1 absolute inset-x-0 top-0 pointer-events-none"
          style={{ color: COLORS.textMuted, fontSize: "9px", lineHeight: "14px", zIndex: 1 }}
        >
          <span>{formatUtcLabel(jobStart)}</span>
          <span>24h Overview</span>
          <span>{formatUtcLabel(jobEnd)}</span>
        </div>

        {/* Canvas */}
        <canvas
          ref={canvasRef}
          width={1024}
          height={28}
          className="w-full h-7 block cursor-crosshair"
          style={{ display: "block" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        />
      </div>
    </div>
  );
}
