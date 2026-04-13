import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { regionTileUrl } from "@/api/client";
import type { Region } from "@/api/types";
import { formatTime } from "@/utils/format";

/**
 * Zoom level presets for the region spectrogram viewer.
 * Regions shorter than 5 minutes use "1m" zoom for better resolution.
 */
const ZOOM_PRESETS = {
  "5m": { tileDuration: 50, viewportSpan: 300 },
  "1m": { tileDuration: 10, viewportSpan: 60 },
} as const;

const SHORT_REGION_THRESHOLD_SEC = 300;

const FREQ_AXIS_WIDTH = 44;
const TIME_AXIS_HEIGHT = 24;
const FREQ_MIN = 0;
const FREQ_MAX = 3000;

interface RegionSpectrogramViewerProps {
  regionJobId: string;
  region: Region;
  children?: ReactNode;
  onViewStartChange?: (viewStart: number) => void;
}

export function RegionSpectrogramViewer({
  regionJobId,
  region,
  children,
  onViewStartChange,
}: RegionSpectrogramViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);

  // Observe container width
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([entry]) => {
      setContainerWidth(entry.contentRect.width);
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const canvasWidth = Math.max(0, containerWidth - FREQ_AXIS_WIDTH);
  const canvasHeight = 200;

  const regionStart = region.padded_start_sec;
  const regionEnd = region.padded_end_sec;
  const regionDuration = regionEnd - regionStart;

  // Pick zoom level based on region duration
  const zoomLevel = regionDuration < SHORT_REGION_THRESHOLD_SEC ? "1m" : "5m";
  const { tileDuration: TILE_DURATION_SEC, viewportSpan: VIEWPORT_SPAN_SEC } =
    ZOOM_PRESETS[zoomLevel];

  // Center timestamp drives what's visible
  const initialCenter = regionStart + Math.min(regionDuration, VIEWPORT_SPAN_SEC) / 2;
  const [centerTimestamp, setCenterTimestamp] = useState(initialCenter);

  // Reset center when region changes
  useEffect(() => {
    const dur = region.padded_end_sec - region.padded_start_sec;
    const span = dur < SHORT_REGION_THRESHOLD_SEC
      ? ZOOM_PRESETS["1m"].viewportSpan
      : ZOOM_PRESETS["5m"].viewportSpan;
    setCenterTimestamp(region.padded_start_sec + Math.min(dur, span) / 2);
  }, [region.region_id, region.padded_start_sec, region.padded_end_sec]);

  const pxPerSec = canvasWidth > 0 ? canvasWidth / VIEWPORT_SPAN_SEC : 1;

  // Clamp center so the viewport doesn't extend far beyond region bounds
  const clampCenter = useCallback(
    (c: number) => {
      const half = VIEWPORT_SPAN_SEC / 2;
      const minCenter = regionStart + half;
      const maxCenter = regionEnd - half;
      if (maxCenter <= minCenter) {
        return (regionStart + regionEnd) / 2;
      }
      return Math.max(minCenter, Math.min(maxCenter, c));
    },
    [regionStart, regionEnd, VIEWPORT_SPAN_SEC],
  );

  // Drag-to-pan
  const dragRef = useRef<{ startX: number; startCenter: number } | null>(null);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      dragRef.current = { startX: e.clientX, startCenter: centerTimestamp };
    },
    [centerTimestamp],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragRef.current) return;
      const dx = e.clientX - dragRef.current.startX;
      const dt = dx / pxPerSec;
      setCenterTimestamp(clampCenter(dragRef.current.startCenter - dt));
    },
    [pxPerSec, clampCenter],
  );

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  useEffect(() => {
    const up = () => {
      dragRef.current = null;
    };
    window.addEventListener("mouseup", up);
    return () => window.removeEventListener("mouseup", up);
  }, []);

  // Compute visible tiles
  const viewStart = centerTimestamp - VIEWPORT_SPAN_SEC / 2;

  useEffect(() => {
    onViewStartChange?.(viewStart);
  }, [viewStart, onViewStartChange]);
  // Job start for tile indexing — use the job's start_timestamp (approximated
  // from the region detection job's time range). Tiles are indexed from time 0
  // of the job, but we receive region-relative padded bounds. Since the tile
  // endpoint indexes from the job's own start_timestamp, we use regionJobStart=0
  // when the region's padded_start_sec is already in absolute job time.
  // The region start_sec values are in absolute job time (from padded_start_sec).
  const jobStart = 0; // Tiles are indexed from job start = 0

  const tiles = useMemo(() => {
    const first = Math.max(
      0,
      Math.floor((viewStart - jobStart) / TILE_DURATION_SEC) - 1,
    );
    const last = Math.ceil(
      (viewStart + VIEWPORT_SPAN_SEC - jobStart) / TILE_DURATION_SEC,
    );
    const indices: number[] = [];
    for (let i = first; i <= last; i++) {
      indices.push(i);
    }
    return indices;
  }, [viewStart, jobStart, TILE_DURATION_SEC, VIEWPORT_SPAN_SEC]);

  // Frequency axis labels
  const freqLabels = useMemo(() => {
    const labels: { hz: number; y: number }[] = [];
    const step = 500;
    for (let hz = FREQ_MIN; hz <= FREQ_MAX; hz += step) {
      const frac = (hz - FREQ_MIN) / (FREQ_MAX - FREQ_MIN);
      labels.push({ hz, y: canvasHeight * (1 - frac) });
    }
    return labels;
  }, [canvasHeight]);

  // Time axis labels
  const timeLabels = useMemo(() => {
    const labels: { sec: number; x: number }[] = [];
    // Step: every 30 seconds
    const step = 30;
    const first = Math.ceil(viewStart / step) * step;
    for (let t = first; t < viewStart + VIEWPORT_SPAN_SEC; t += step) {
      const x = (t - viewStart) * pxPerSec;
      if (x >= 0 && x <= canvasWidth) {
        labels.push({ sec: t, x });
      }
    }
    return labels;
  }, [viewStart, pxPerSec, canvasWidth]);

  // Coordinate transform for children (overlays)
  const overlayContext = useMemo(
    () => ({
      viewStart,
      viewEnd: viewStart + VIEWPORT_SPAN_SEC,
      pxPerSec,
      canvasWidth,
      canvasHeight,
    }),
    [viewStart, pxPerSec, canvasWidth, canvasHeight],
  );

  return (
    <div ref={containerRef} className="w-full select-none">
      <div className="flex">
        {/* Frequency axis */}
        <div
          className="relative shrink-0 border-r border-border text-[10px] text-muted-foreground"
          style={{ width: FREQ_AXIS_WIDTH, height: canvasHeight }}
        >
          {freqLabels.map((l) => (
            <div
              key={l.hz}
              className="absolute right-1 -translate-y-1/2"
              style={{ top: l.y }}
            >
              {l.hz >= 1000 ? `${(l.hz / 1000).toFixed(1)}k` : l.hz}
            </div>
          ))}
        </div>

        {/* Tile area + overlay container */}
        <div
          className="relative overflow-hidden"
          style={{
            width: canvasWidth,
            height: canvasHeight,
            cursor: dragRef.current ? "grabbing" : "grab",
            background: "#060d14",
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          data-testid="spectrogram-viewport"
        >
          {/* Tile images */}
          {tiles.map((idx) => {
            const tileStart = jobStart + idx * TILE_DURATION_SEC;
            const x = (tileStart - viewStart) * pxPerSec;
            const w = TILE_DURATION_SEC * pxPerSec;
            return (
              <img
                key={idx}
                src={regionTileUrl(regionJobId, zoomLevel, idx)}
                alt=""
                draggable={false}
                style={{
                  position: "absolute",
                  left: x,
                  top: 0,
                  width: w,
                  height: canvasHeight,
                  imageRendering: "auto",
                  pointerEvents: "none",
                }}
              />
            );
          })}

          {/* Overlay children (EventBarOverlay goes here) */}
          {children && (
            <div
              className="absolute inset-0"
              style={{ pointerEvents: "none" }}
              data-testid="spectrogram-overlay"
            >
              <OverlayContext.Provider value={overlayContext}>
                {children}
              </OverlayContext.Provider>
            </div>
          )}
        </div>
      </div>

      {/* Time axis */}
      <div
        className="relative border-t border-border text-[10px] text-muted-foreground"
        style={{
          marginLeft: FREQ_AXIS_WIDTH,
          width: canvasWidth,
          height: TIME_AXIS_HEIGHT,
        }}
      >
        {timeLabels.map((l) => (
          <div
            key={l.sec}
            className="absolute top-1 -translate-x-1/2"
            style={{ left: l.x }}
          >
            {formatTime(l.sec)}
          </div>
        ))}
      </div>
    </div>
  );
}

// Context for overlay components to convert timestamps to pixel positions

interface OverlayContextValue {
  viewStart: number;
  viewEnd: number;
  pxPerSec: number;
  canvasWidth: number;
  canvasHeight: number;
}

const OverlayContext = createContext<OverlayContextValue>({
  viewStart: 0,
  viewEnd: 0,
  pxPerSec: 1,
  canvasWidth: 0,
  canvasHeight: 0,
});

export function useOverlayContext() {
  return useContext(OverlayContext);
}
