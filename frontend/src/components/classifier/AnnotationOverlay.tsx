import { useCallback, useRef, useState } from "react";
import type { LabelingAnnotation } from "@/api/types";

const LABEL_COLORS: Record<string, string> = {
  whup: "rgba(59,130,246,0.3)",
  moan: "rgba(168,85,247,0.3)",
  shriek: "rgba(239,68,68,0.3)",
  grunt: "rgba(234,179,8,0.3)",
  song: "rgba(34,197,94,0.3)",
};

function colorForLabel(label: string): string {
  if (LABEL_COLORS[label.toLowerCase()]) return LABEL_COLORS[label.toLowerCase()];
  // Deterministic color from hash
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
  }
  const h = Math.abs(hash) % 360;
  return `hsla(${h},60%,55%,0.3)`;
}

function borderForLabel(label: string): string {
  return colorForLabel(label).replace("0.3)", "0.8)");
}

interface Props {
  /** Total window duration in seconds */
  windowDuration: number;
  annotations: LabelingAnnotation[];
  highlightedId: string | null;
  onCreateRegion: (startSec: number, endSec: number) => void;
  onDeleteAnnotation: (id: string) => void;
  onSelectAnnotation: (id: string | null) => void;
}

export function AnnotationOverlay({
  windowDuration,
  annotations,
  highlightedId,
  onCreateRegion,
  onDeleteAnnotation,
  onSelectAnnotation,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dragStart, setDragStart] = useState<number | null>(null);
  const [dragCurrent, setDragCurrent] = useState<number | null>(null);

  const getOffsetFraction = useCallback(
    (clientX: number) => {
      const svg = svgRef.current;
      if (!svg) return 0;
      const rect = svg.getBoundingClientRect();
      return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    },
    [],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      // Don't start drag if clicking on an annotation rect
      if ((e.target as HTMLElement).closest("[data-annotation]")) return;
      const frac = getOffsetFraction(e.clientX);
      setDragStart(frac);
      setDragCurrent(frac);
    },
    [getOffsetFraction],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (dragStart === null) return;
      setDragCurrent(getOffsetFraction(e.clientX));
    },
    [dragStart, getOffsetFraction],
  );

  const handleMouseUp = useCallback(() => {
    if (dragStart !== null && dragCurrent !== null) {
      const startFrac = Math.min(dragStart, dragCurrent);
      const endFrac = Math.max(dragStart, dragCurrent);
      const startSec = startFrac * windowDuration;
      const endSec = endFrac * windowDuration;
      // Minimum 0.1s region
      if (endSec - startSec >= 0.1) {
        onCreateRegion(
          Math.round(startSec * 100) / 100,
          Math.round(endSec * 100) / 100,
        );
      }
    }
    setDragStart(null);
    setDragCurrent(null);
  }, [dragStart, dragCurrent, windowDuration, onCreateRegion]);

  return (
    <svg
      ref={svgRef}
      className="absolute inset-0 w-full h-full cursor-crosshair"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        setDragStart(null);
        setDragCurrent(null);
      }}
    >
      {/* Existing annotations */}
      {annotations.map((ann) => {
        const x = (ann.start_offset_sec / windowDuration) * 100;
        const w =
          ((ann.end_offset_sec - ann.start_offset_sec) / windowDuration) * 100;
        const isHighlighted = ann.id === highlightedId;
        return (
          <g key={ann.id} data-annotation>
            <rect
              x={`${x}%`}
              y="0"
              width={`${w}%`}
              height="100%"
              fill={colorForLabel(ann.label)}
              stroke={borderForLabel(ann.label)}
              strokeWidth={isHighlighted ? 2 : 1}
              opacity={isHighlighted ? 1 : 0.8}
              className="cursor-pointer"
              onClick={(e) => {
                e.stopPropagation();
                onSelectAnnotation(ann.id);
              }}
            />
            {/* Label text */}
            {w > 8 && (
              <text
                x={`${x + w / 2}%`}
                y="14"
                textAnchor="middle"
                fontSize="10"
                fill="white"
                className="pointer-events-none select-none"
                style={{ textShadow: "0 1px 2px rgba(0,0,0,0.8)" }}
              >
                {ann.label}
              </text>
            )}
            {/* Delete button */}
            {isHighlighted && (
              <foreignObject
                x={`${x + w - 2}%`}
                y="2"
                width="16"
                height="16"
              >
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteAnnotation(ann.id);
                  }}
                  className="w-4 h-4 bg-red-500 text-white rounded-full text-[10px] leading-none flex items-center justify-center hover:bg-red-600"
                  title="Delete annotation"
                >
                  &times;
                </button>
              </foreignObject>
            )}
          </g>
        );
      })}

      {/* Drag preview */}
      {dragStart !== null && dragCurrent !== null && (
        <rect
          x={`${Math.min(dragStart, dragCurrent) * 100}%`}
          y="0"
          width={`${Math.abs(dragCurrent - dragStart) * 100}%`}
          height="100%"
          fill="rgba(59,130,246,0.2)"
          stroke="rgba(59,130,246,0.6)"
          strokeWidth={1}
          strokeDasharray="4 2"
          className="pointer-events-none"
        />
      )}
    </svg>
  );
}
