// frontend/src/components/timeline/DetectionPopover.tsx
import { useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import type { DetectionRow } from "@/api/types";
import { COLORS } from "./constants";

export interface DetectionPopoverProps {
  row: DetectionRow;
  jobStart: number;
  position: { x: number; y: number };
  onClose: () => void;
}

function formatUtcTime(epoch: number): string {
  const d = new Date(epoch * 1000);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = String(d.getUTCSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}Z`;
}

function activeLabels(row: DetectionRow): string[] {
  const labels: string[] = [];
  if (row.humpback === 1) labels.push("humpback");
  if (row.orca === 1) labels.push("orca");
  if (row.ship === 1) labels.push("ship");
  if (row.background === 1) labels.push("background");
  return labels;
}

export function DetectionPopover({
  row,
  jobStart,
  position,
  onClose,
}: DetectionPopoverProps) {
  const navigate = useNavigate();
  const ref = useRef<HTMLDivElement>(null);

  // Close on click outside
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [onClose]);

  // Close on Escape
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const startEpoch = row.start_utc;
  const endEpoch = row.end_utc;
  const labels = activeLabels(row);

  return (
    <div
      ref={ref}
      style={{
        position: "absolute",
        left: position.x,
        top: position.y,
        zIndex: 50,
        background: COLORS.headerBg,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 4,
        padding: "10px 14px",
        minWidth: 220,
        pointerEvents: "auto",
        boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
      }}
    >
      {/* Timestamp range */}
      <div
        style={{
          color: COLORS.accent,
          fontWeight: 600,
          marginBottom: 6,
          fontSize: "11px",
        }}
      >
        {formatUtcTime(startEpoch)} &mdash; {formatUtcTime(endEpoch)}
      </div>

      {/* Confidence */}
      <div
        style={{
          color: COLORS.text,
          fontSize: "10px",
          marginBottom: 2,
        }}
      >
        Avg confidence:{" "}
        <span style={{ color: COLORS.textBright }}>
          {row.avg_confidence != null ? row.avg_confidence.toFixed(3) : "—"}
        </span>
      </div>
      <div
        style={{
          color: COLORS.text,
          fontSize: "10px",
          marginBottom: 6,
        }}
      >
        Peak confidence:{" "}
        <span style={{ color: COLORS.textBright }}>
          {row.peak_confidence != null ? row.peak_confidence.toFixed(3) : "—"}
        </span>
      </div>

      {/* Labels */}
      {labels.length > 0 && (
        <div
          style={{
            color: COLORS.text,
            fontSize: "10px",
            marginBottom: 8,
          }}
        >
          Labels:{" "}
          <span style={{ color: COLORS.accent }}>{labels.join(", ")}</span>
        </div>
      )}

      {/* Navigate link */}
      <button
        onClick={() => {
          navigate("/app/classifier/hydrophone");
          onClose();
        }}
        style={{
          background: "none",
          border: "none",
          padding: 0,
          cursor: "pointer",
          color: COLORS.accentDim,
          fontSize: "10px",
          textDecoration: "underline",
        }}
      >
        View in table &rarr;
      </button>
    </div>
  );
}
