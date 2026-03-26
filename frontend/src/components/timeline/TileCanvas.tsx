// frontend/src/components/timeline/TileCanvas.tsx
import { useEffect, useRef, useCallback } from "react";
import type { ZoomLevel } from "@/api/types";
import { timelineTileUrl } from "@/api/client";
import {
  TILE_DURATION,
  VIEWPORT_SPAN,
  COLORS,
  CROSSFADE_DURATION_MS,
} from "./constants";

// ---------------------------------------------------------------------------
// Module-level LRU tile image cache
// ---------------------------------------------------------------------------
const MAX_CACHE_SIZE = 200;
const tileCache = new Map<string, HTMLImageElement>();

function getCachedTile(url: string): HTMLImageElement | undefined {
  const img = tileCache.get(url);
  if (img) {
    // Move to end (most recently used)
    tileCache.delete(url);
    tileCache.set(url, img);
  }
  return img;
}

function putCachedTile(url: string, img: HTMLImageElement) {
  if (tileCache.has(url)) tileCache.delete(url);
  tileCache.set(url, img);
  // Evict oldest entries
  while (tileCache.size > MAX_CACHE_SIZE) {
    const oldest = tileCache.keys().next().value;
    if (oldest !== undefined) tileCache.delete(oldest);
  }
}

// ---------------------------------------------------------------------------
// Tile loading tracker (prevents duplicate loads)
// ---------------------------------------------------------------------------
const loadingTiles = new Set<string>();

function loadTile(url: string): Promise<HTMLImageElement> {
  const cached = getCachedTile(url);
  if (cached) return Promise.resolve(cached);

  if (loadingTiles.has(url)) {
    // Already loading — poll until it appears in cache
    return new Promise((resolve) => {
      const check = () => {
        const img = getCachedTile(url);
        if (img) {
          resolve(img);
        } else if (loadingTiles.has(url)) {
          setTimeout(check, 16);
        } else {
          // Load failed and was removed — try again
          resolve(loadTile(url));
        }
      };
      setTimeout(check, 16);
    });
  }

  loadingTiles.add(url);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      loadingTiles.delete(url);
      putCachedTile(url, img);
      resolve(img);
    };
    img.onerror = () => {
      loadingTiles.delete(url);
      reject(new Error(`Failed to load tile: ${url}`));
    };
    img.src = url;
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function tileIndexForTimestamp(
  epoch: number,
  jobStart: number,
  tileDuration: number,
): number {
  return Math.floor((epoch - jobStart) / tileDuration);
}

function tileStartEpoch(
  tileIndex: number,
  jobStart: number,
  tileDuration: number,
): number {
  return jobStart + tileIndex * tileDuration;
}

// ---------------------------------------------------------------------------
// Component props
// ---------------------------------------------------------------------------
export interface TileCanvasProps {
  jobId: string;
  jobStart: number;
  jobEnd: number;
  centerTimestamp: number;
  zoomLevel: ZoomLevel;
  freqRange: [number, number];
  width: number;
  height: number;
}

// ---------------------------------------------------------------------------
// TileCanvas
// ---------------------------------------------------------------------------
export function TileCanvas({
  jobId,
  jobStart,
  jobEnd,
  centerTimestamp,
  zoomLevel,
  freqRange,
  width,
  height,
}: TileCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Track zoom transitions
  const prevZoomRef = useRef<ZoomLevel>(zoomLevel);
  const transitionRef = useRef<{
    prevZoom: ZoomLevel;
    startTime: number;
  } | null>(null);

  // Detect zoom level changes and start crossfade
  useEffect(() => {
    if (prevZoomRef.current !== zoomLevel) {
      transitionRef.current = {
        prevZoom: prevZoomRef.current,
        startTime: performance.now(),
      };
      prevZoomRef.current = zoomLevel;
    }
  }, [zoomLevel]);

  // Calculate visible tile indices for a given zoom level
  const getVisibleTiles = useCallback(
    (zoom: ZoomLevel) => {
      const tileDuration = TILE_DURATION[zoom];
      const viewportSpan = VIEWPORT_SPAN[zoom];
      const halfWidthSec = viewportSpan / 2;

      const viewStart = centerTimestamp - halfWidthSec;
      const viewEnd = centerTimestamp + halfWidthSec;

      const firstTile = Math.max(
        0,
        tileIndexForTimestamp(viewStart, jobStart, tileDuration) - 1,
      );
      const lastTileByView =
        tileIndexForTimestamp(viewEnd, jobStart, tileDuration) + 1;
      const maxTile = Math.ceil((jobEnd - jobStart) / tileDuration) - 1;
      const lastTile = Math.min(lastTileByView, maxTile);

      const tiles: number[] = [];
      for (let i = firstTile; i <= lastTile; i++) {
        tiles.push(i);
      }
      return tiles;
    },
    [centerTimestamp, width, jobStart, jobEnd],
  );

  // Preload visible tiles (current + transitioning zoom)
  useEffect(() => {
    const tiles = getVisibleTiles(zoomLevel);
    for (const idx of tiles) {
      const url = timelineTileUrl(
        jobId,
        zoomLevel,
        idx,
        freqRange[0],
        freqRange[1],
      );
      loadTile(url).catch(() => {
        /* ignore preload errors */
      });
    }

    // Also preload transition layer tiles
    if (transitionRef.current) {
      const prevTiles = getVisibleTiles(transitionRef.current.prevZoom);
      for (const idx of prevTiles) {
        const url = timelineTileUrl(
          jobId,
          transitionRef.current.prevZoom,
          idx,
          freqRange[0],
          freqRange[1],
        );
        loadTile(url).catch(() => {
          /* ignore preload errors */
        });
      }
    }
  }, [jobId, zoomLevel, freqRange, getVisibleTiles]);

  // Draw loop
  const drawRef = useRef<number>(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear with dark background
    ctx.fillStyle = COLORS.bgDark;
    ctx.fillRect(0, 0, width, height);

    const now = performance.now();
    const transition = transitionRef.current;
    let newAlpha = 1;
    let oldAlpha = 0;
    let drawOldLayer = false;

    if (transition) {
      const elapsed = now - transition.startTime;
      if (elapsed < CROSSFADE_DURATION_MS) {
        const t = elapsed / CROSSFADE_DURATION_MS;
        newAlpha = t;
        oldAlpha = 1 - t;
        drawOldLayer = true;
      } else {
        // Transition complete
        transitionRef.current = null;
      }
    }

    // Draw a tile layer
    const drawLayer = (zoom: ZoomLevel, alpha: number) => {
      if (alpha <= 0) return;
      const tileDuration = TILE_DURATION[zoom];
      const viewportSpan = VIEWPORT_SPAN[zoom];
      const pxPerSec = width / viewportSpan;
      const tiles = getVisibleTiles(zoom);

      ctx.globalAlpha = alpha;
      for (const idx of tiles) {
        const url = timelineTileUrl(
          jobId,
          zoom,
          idx,
          freqRange[0],
          freqRange[1],
        );
        const img = getCachedTile(url);
        const tStart = tileStartEpoch(idx, jobStart, tileDuration);
        const x =
          (tStart - centerTimestamp) * pxPerSec + width / 2;
        const tileWidthOnScreen = tileDuration * pxPerSec;

        if (img) {
          ctx.drawImage(img, x, 0, tileWidthOnScreen, height);
        } else {
          // Dark placeholder for loading tile
          ctx.fillStyle = "#060d14";
          ctx.fillRect(x, 0, tileWidthOnScreen, height);
        }
      }
    };

    // Draw old layer first (underneath) during transition
    if (drawOldLayer && transition) {
      drawLayer(transition.prevZoom, oldAlpha);
    }

    // Draw current zoom layer
    drawLayer(zoomLevel, newAlpha);

    // Reset alpha
    ctx.globalAlpha = 1;

    // Continue animation if transitioning
    if (drawOldLayer) {
      drawRef.current = requestAnimationFrame(draw);
    }
  }, [
    width,
    height,
    centerTimestamp,
    zoomLevel,
    jobId,
    jobStart,
    freqRange,
    getVisibleTiles,
  ]);

  // Re-draw on state changes
  useEffect(() => {
    cancelAnimationFrame(drawRef.current);
    draw();

    // If in transition, the draw loop self-schedules via rAF
    return () => cancelAnimationFrame(drawRef.current);
  }, [draw]);

  // Also re-draw periodically while tiles are loading
  useEffect(() => {
    let running = true;
    let handle = 0;

    const poll = () => {
      if (!running) return;
      if (loadingTiles.size > 0) {
        draw();
      }
      handle = requestAnimationFrame(poll);
    };
    handle = requestAnimationFrame(poll);

    return () => {
      running = false;
      cancelAnimationFrame(handle);
    };
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        width,
        height,
        display: "block",
        imageRendering: "auto",
      }}
    />
  );
}
