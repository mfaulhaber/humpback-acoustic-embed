import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTimelineContext } from "../provider/useTimelineContext";
import { OverlayContext } from "../overlays/OverlayContext";
import type { OverlayContextValue } from "../overlays/OverlayContext";
import { TileCanvas } from "../TileCanvas";
import { FrequencyAxis } from "./FrequencyAxis";
import { TimeAxis } from "./TimeAxis";
import { ConfidenceStrip } from "./ConfidenceStrip";
import { Playhead } from "./Playhead";
import { FREQ_AXIS_WIDTH_PX } from "../constants";

const CONFIDENCE_STRIP_HEIGHT = 20;
const TIME_AXIS_HEIGHT = 20;

interface SpectrogramProps {
  tileUrlBuilder: (jobId: string, zoomLevel: string, tileIndex: number, freqMin: number, freqMax: number) => string;
  jobId: string;
  freqRange?: [number, number];
  scores?: (number | null)[];
  windowSec?: number;
  children?: React.ReactNode;
}

export function Spectrogram({
  tileUrlBuilder,
  jobId,
  freqRange = [0, 3000],
  scores,
  windowSec,
  children,
}: SpectrogramProps) {
  const ctx = useTimelineContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.floor(entry.contentRect.width);
        const h = Math.floor(entry.contentRect.height);
        setContainerWidth(w);
        setContainerHeight(h);
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const canvasWidth = Math.max(0, containerWidth - FREQ_AXIS_WIDTH_PX);
  const canvasHeight = Math.max(
    0,
    containerHeight - (scores ? CONFIDENCE_STRIP_HEIGHT : 0) - TIME_AXIS_HEIGHT,
  );

  useEffect(() => {
    if (canvasWidth > 0 && canvasHeight > 0) {
      ctx.setViewportDimensions(canvasWidth, canvasHeight);
    }
  }, [canvasWidth, canvasHeight, ctx.setViewportDimensions]);

  // Drag-to-pan
  const dragRef = useRef<{ startX: number; startCenter: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (ctx.isPlaying) return;
      dragRef.current = { startX: e.clientX, startCenter: ctx.centerTimestamp };
      setIsDragging(true);
    },
    [ctx.isPlaying, ctx.centerTimestamp],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragRef.current || ctx.isPlaying) return;
      const dx = e.clientX - dragRef.current.startX;
      const dt = dx / ctx.pxPerSec;
      const newCenter = dragRef.current.startCenter - dt;
      ctx.pan(newCenter);
    },
    [ctx.isPlaying, ctx.pxPerSec, ctx.pan],
  );

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
    setIsDragging(false);
  }, []);

  useEffect(() => {
    const handleGlobalUp = () => {
      dragRef.current = null;
      setIsDragging(false);
    };
    window.addEventListener("mouseup", handleGlobalUp);
    return () => window.removeEventListener("mouseup", handleGlobalUp);
  }, []);

  const overlayValue: OverlayContextValue = useMemo(
    () => ({
      viewStart: ctx.viewStart,
      viewEnd: ctx.viewEnd,
      pxPerSec: ctx.pxPerSec,
      canvasWidth,
      canvasHeight,
      epochToX: (epoch: number) => (epoch - ctx.centerTimestamp) * ctx.pxPerSec + canvasWidth / 2,
      xToEpoch: (x: number) => ctx.centerTimestamp + (x - canvasWidth / 2) / ctx.pxPerSec,
    }),
    [ctx.viewStart, ctx.viewEnd, ctx.pxPerSec, ctx.centerTimestamp, canvasWidth, canvasHeight],
  );

  const cursor = ctx.isPlaying ? "default" : isDragging ? "grabbing" : "grab";

  return (
    <div ref={containerRef} className="flex-1 flex flex-col relative select-none min-h-0">
      <div className="flex flex-1 min-h-0">
        <FrequencyAxis freqRange={freqRange} height={canvasHeight} />

        <div
          className="relative flex-1 min-w-0 flex flex-col"
          style={{ cursor }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          {canvasWidth > 0 && canvasHeight > 0 && (
            <TileCanvas
              jobId={jobId}
              jobStart={ctx.jobStart}
              jobEnd={ctx.jobEnd}
              centerTimestamp={ctx.centerTimestamp}
              zoomLevel={ctx.activePreset.key}
              freqRange={freqRange}
              width={canvasWidth}
              height={canvasHeight}
              tileDurationOverride={ctx.activePreset.tileDuration}
              viewportSpanOverride={ctx.activePreset.span}
              tileUrlBuilder={tileUrlBuilder}
            />
          )}

          {/* Overlay container */}
          <div className="absolute inset-0" style={{ width: canvasWidth, height: canvasHeight }}>
            <OverlayContext.Provider value={overlayValue}>
              {children}
            </OverlayContext.Provider>
          </div>

          <Playhead canvasWidth={canvasWidth} canvasHeight={canvasHeight + (scores ? CONFIDENCE_STRIP_HEIGHT : 0)} />

          {scores && (
            <ConfidenceStrip
              scores={scores}
              windowSec={windowSec}
              jobStart={ctx.jobStart}
              jobEnd={ctx.jobEnd}
              viewStart={ctx.viewStart}
              viewEnd={ctx.viewEnd}
              pxPerSec={ctx.pxPerSec}
              canvasWidth={canvasWidth}
              centerTimestamp={ctx.centerTimestamp}
            />
          )}

          <TimeAxis
            viewStart={ctx.viewStart}
            viewEnd={ctx.viewEnd}
            viewportSpan={ctx.viewportSpan}
            pxPerSec={ctx.pxPerSec}
            canvasWidth={canvasWidth}
            centerTimestamp={ctx.centerTimestamp}
          />
        </div>
      </div>
    </div>
  );
}
