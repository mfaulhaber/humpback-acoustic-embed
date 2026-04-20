import {
  createContext,
  type RefObject,
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
import { RegionBandOverlay } from "./RegionBandOverlay";

/**
 * Zoom level presets for the region spectrogram viewer.
 * Auto-selected based on region duration via selectZoomLevel().
 */
const ZOOM_PRESETS = {
  "5m": { tileDuration: 50, viewportSpan: 300, tickInterval: 30 },
  "1m": { tileDuration: 10, viewportSpan: 60, tickInterval: 30 },
  "30s": { tileDuration: 5, viewportSpan: 30, tickInterval: 5 },
  "10s": { tileDuration: 2, viewportSpan: 10, tickInterval: 2 },
} as const;

type ZoomLevel = keyof typeof ZOOM_PRESETS;
const ZOOM_LEVELS: ZoomLevel[] = ["10s", "30s", "1m", "5m"];

function selectZoomLevel(regionDurationSec: number): ZoomLevel {
  if (regionDurationSec >= 300) return "5m";
  if (regionDurationSec >= 30) return "1m";
  if (regionDurationSec >= 10) return "30s";
  return "10s";
}

const FREQ_AXIS_WIDTH = 44;
const TIME_AXIS_HEIGHT = 24;
const FREQ_MIN = 0;
const FREQ_MAX = 3000;

interface RegionSpectrogramViewerProps {
  regionJobId: string;
  region: Region;
  children?: ReactNode;
  onViewStartChange?: (viewStart: number) => void;
  onViewSpanChange?: (span: number) => void;
  /** When changed, scrolls so this timestamp is the viewport center. */
  scrollToCenter?: number;
  /** Full time extent of the run for unrestricted scrolling. */
  timelineExtent?: { start: number; end: number };
  /** All regions in the run (for region band overlay). */
  allRegions?: Region[];
  /** Currently active region id (for band highlight). */
  activeRegionId?: string;
  /** Called when user clicks an inactive region band. */
  onSelectRegion?: (regionId: string) => void;
  audioRef?: RefObject<HTMLAudioElement | null>;
  isPlaying?: boolean;
  playbackOriginSec?: number;
}

export function RegionSpectrogramViewer({
  regionJobId,
  region,
  children,
  onViewStartChange,
  onViewSpanChange,
  scrollToCenter,
  timelineExtent,
  allRegions,
  activeRegionId,
  onSelectRegion,
  audioRef,
  isPlaying = false,
  playbackOriginSec = 0,
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

  // Zoom level as state — auto-selected initially, user can override
  const [zoomLevel, setZoomLevel] = useState<ZoomLevel>(() =>
    selectZoomLevel(regionDuration),
  );
  const {
    tileDuration: TILE_DURATION_SEC,
    viewportSpan: VIEWPORT_SPAN_SEC,
    tickInterval: TICK_INTERVAL_SEC,
  } = ZOOM_PRESETS[zoomLevel];

  // Center timestamp drives what's visible
  const initialCenter = regionStart + Math.min(regionDuration, VIEWPORT_SPAN_SEC) / 2;
  const [centerTimestamp, setCenterTimestamp] = useState(initialCenter);

  // Re-center when region changes but keep current zoom level
  useEffect(() => {
    const dur = region.padded_end_sec - region.padded_start_sec;
    const span = ZOOM_PRESETS[zoomLevel].viewportSpan;
    setCenterTimestamp(region.padded_start_sec + Math.min(dur, span) / 2);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- zoomLevel intentionally excluded
  }, [region.region_id, region.padded_start_sec, region.padded_end_sec]);

  // External scroll request
  useEffect(() => {
    if (scrollToCenter !== undefined) {
      setCenterTimestamp(scrollToCenter);
    }
  }, [scrollToCenter]);

  const pxPerSec = canvasWidth > 0 ? canvasWidth / VIEWPORT_SPAN_SEC : 1;

  // Clamp center — if timelineExtent is given, allow full scrolling across the run
  const clampCenter = useCallback(
    (c: number) => {
      const half = VIEWPORT_SPAN_SEC / 2;
      const extentStart = timelineExtent ? timelineExtent.start : regionStart;
      const extentEnd = timelineExtent ? timelineExtent.end : regionEnd;
      const minCenter = extentStart + half;
      const maxCenter = extentEnd - half;
      if (maxCenter <= minCenter) {
        return (extentStart + extentEnd) / 2;
      }
      return Math.max(minCenter, Math.min(maxCenter, c));
    },
    [regionStart, regionEnd, timelineExtent, VIEWPORT_SPAN_SEC],
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
  useEffect(() => {
    onViewSpanChange?.(VIEWPORT_SPAN_SEC);
  }, [VIEWPORT_SPAN_SEC, onViewSpanChange]);
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
    const step = TICK_INTERVAL_SEC;
    const first = Math.ceil(viewStart / step) * step;
    for (let t = first; t < viewStart + VIEWPORT_SPAN_SEC; t += step) {
      const x = (t - viewStart) * pxPerSec;
      if (x >= 0 && x <= canvasWidth) {
        labels.push({ sec: t, x });
      }
    }
    return labels;
  }, [viewStart, pxPerSec, canvasWidth, TICK_INTERVAL_SEC, VIEWPORT_SPAN_SEC]);

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

  // Playhead: rAF loop that directly manipulates DOM position
  const playheadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isPlaying || !audioRef?.current) return;
    let raf: number;
    const tick = () => {
      const audio = audioRef.current;
      const el = playheadRef.current;
      if (audio && el && !audio.paused) {
        const currentSec = playbackOriginSec + audio.currentTime;
        const x = (currentSec - viewStart) * pxPerSec;
        el.style.left = `${x}px`;
        el.style.display = x >= 0 && x <= canvasWidth ? "" : "none";
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, audioRef, playbackOriginSec, viewStart, pxPerSec, canvasWidth]);

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

          {/* Region boundary indicators */}
          {(() => {
            const startX = (region.start_sec - viewStart) * pxPerSec;
            const endX = (region.end_sec - viewStart) * pxPerSec;
            return (
              <>
                {/* Dimmed overlay before region start */}
                {startX > 0 && (
                  <div
                    data-testid="region-dim-left"
                    style={{
                      position: "absolute",
                      left: 0,
                      top: 0,
                      width: Math.min(startX, canvasWidth),
                      height: canvasHeight,
                      background: "rgba(0,0,0,0.3)",
                      pointerEvents: "none",
                      zIndex: 3,
                    }}
                  />
                )}
                {/* Dimmed overlay after region end */}
                {endX < canvasWidth && (
                  <div
                    data-testid="region-dim-right"
                    style={{
                      position: "absolute",
                      left: Math.max(endX, 0),
                      top: 0,
                      width: canvasWidth - Math.max(endX, 0),
                      height: canvasHeight,
                      background: "rgba(0,0,0,0.3)",
                      pointerEvents: "none",
                      zIndex: 3,
                    }}
                  />
                )}
                {/* Region start boundary line */}
                {startX >= 0 && startX <= canvasWidth && (
                  <div
                    data-testid="region-boundary-start"
                    style={{
                      position: "absolute",
                      left: startX,
                      top: 0,
                      width: 0,
                      height: canvasHeight,
                      borderLeft: "2px dashed rgba(251, 191, 36, 0.8)",
                      pointerEvents: "none",
                      zIndex: 4,
                    }}
                  >
                    <div
                      style={{
                        position: "absolute",
                        top: 2,
                        left: 2,
                        fontSize: 9,
                        color: "rgba(251, 191, 36, 0.9)",
                        whiteSpace: "nowrap",
                        background: "rgba(0,0,0,0.6)",
                        padding: "1px 3px",
                        borderRadius: 2,
                      }}
                    >
                      R start
                    </div>
                  </div>
                )}
                {/* Region end boundary line */}
                {endX >= 0 && endX <= canvasWidth && (
                  <div
                    data-testid="region-boundary-end"
                    style={{
                      position: "absolute",
                      left: endX,
                      top: 0,
                      width: 0,
                      height: canvasHeight,
                      borderLeft: "2px dashed rgba(251, 191, 36, 0.8)",
                      pointerEvents: "none",
                      zIndex: 4,
                    }}
                  >
                    <div
                      style={{
                        position: "absolute",
                        top: 2,
                        right: 2,
                        fontSize: 9,
                        color: "rgba(251, 191, 36, 0.9)",
                        whiteSpace: "nowrap",
                        background: "rgba(0,0,0,0.6)",
                        padding: "1px 3px",
                        borderRadius: 2,
                      }}
                    >
                      R end
                    </div>
                  </div>
                )}
              </>
            );
          })()}

          {/* Overlay layers */}
          <div
            className="absolute inset-0"
            style={{ pointerEvents: "none" }}
            data-testid="spectrogram-overlay"
          >
            <OverlayContext.Provider value={overlayContext}>
              {/* Region bands: shown at wide zoom (1m, 5m) when regions provided */}
              {allRegions &&
                allRegions.length > 0 &&
                activeRegionId &&
                onSelectRegion &&
                (zoomLevel === "1m" || zoomLevel === "5m") && (
                  <RegionBandOverlay
                    regions={allRegions}
                    activeRegionId={activeRegionId}
                    onSelectRegion={onSelectRegion}
                  />
                )}
              {children}
            </OverlayContext.Provider>
          </div>

          {/* Playhead */}
          {isPlaying && (
            <div
              ref={playheadRef}
              data-testid="playhead"
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: 0,
                height: canvasHeight,
                borderLeft: "1.5px solid #70e0c0",
                pointerEvents: "none",
                zIndex: 6,
              }}
            >
              <div
                style={{
                  position: "absolute",
                  top: -1,
                  left: -5,
                  width: 0,
                  height: 0,
                  borderLeft: "5px solid transparent",
                  borderRight: "5px solid transparent",
                  borderTop: "6px solid #70e0c0",
                }}
              />
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

      {/* Zoom bar */}
      <div
        className="flex items-center gap-1 border-t border-border px-2 py-1.5"
        style={{ marginLeft: FREQ_AXIS_WIDTH }}
      >
        <span className="mr-1 text-[10px] text-muted-foreground">Zoom</span>
        {ZOOM_LEVELS.map((level) => (
          <button
            key={level}
            className={
              level === zoomLevel
                ? "rounded border border-blue-500 bg-blue-500/20 px-2.5 py-0.5 text-[11px] font-semibold text-blue-400"
                : "rounded border border-border px-2.5 py-0.5 text-[11px] text-muted-foreground hover:bg-accent"
            }
            onClick={() => setZoomLevel(level)}
          >
            {level}
          </button>
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
