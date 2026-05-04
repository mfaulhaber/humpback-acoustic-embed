import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTimelineContext } from "../provider/useTimelineContext";
import { OverlayContext } from "../overlays/OverlayContext";
import type { OverlayContextValue } from "../overlays/OverlayContext";
import { TileCanvas } from "../TileCanvas";
import { FrequencyAxis } from "./FrequencyAxis";
import { TimeAxis } from "./TimeAxis";
import { ConfidenceStrip, DEFAULT_STRIP_HEIGHT } from "./ConfidenceStrip";
import type { GradientStops } from "./ConfidenceStrip";
import { Playhead } from "./Playhead";
import { FREQ_AXIS_WIDTH_PX } from "../constants";

const TIME_AXIS_HEIGHT = 20;

interface SpectrogramProps {
  tileUrlBuilder: (jobId: string, zoomLevel: string, tileIndex: number, freqMin: number, freqMax: number) => string;
  jobId: string;
  freqRange?: [number, number];
  scores?: (number | null)[];
  windowSec?: number;
  children?: React.ReactNode;
  stripHeight?: number;
  stripGradient?: GradientStops;
  stripThreshold?: number;
  stripBarMode?: boolean;
}

export function Spectrogram({
  tileUrlBuilder,
  jobId,
  freqRange = [0, 3000],
  scores,
  windowSec,
  children,
  stripHeight,
  stripGradient,
  stripThreshold,
  stripBarMode,
}: SpectrogramProps) {
  const ctx = useTimelineContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltipLayer, setTooltipLayer] = useState<HTMLDivElement | null>(null);
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

  const effectiveStripHeight = stripHeight ?? DEFAULT_STRIP_HEIGHT;
  const canvasWidth = Math.max(0, containerWidth - FREQ_AXIS_WIDTH_PX);
  const canvasHeight = Math.max(
    0,
    containerHeight - (scores ? effectiveStripHeight : 0) - TIME_AXIS_HEIGHT,
  );

  useEffect(() => {
    if (canvasWidth > 0 && canvasHeight > 0) {
      ctx.setViewportDimensions(canvasWidth, canvasHeight);
    }
  }, [canvasWidth, canvasHeight, ctx.setViewportDimensions]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      ctx.beginDragPan(e.clientX);
    },
    [ctx],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      ctx.updateDragPan(e.clientX);
    },
    [ctx],
  );

  const handleMouseUp = useCallback(() => {
    ctx.endDragPan();
  }, [ctx]);

  const overlayValue: OverlayContextValue = useMemo(
    () => ({
      viewStart: ctx.viewStart,
      viewEnd: ctx.viewEnd,
      pxPerSec: ctx.pxPerSec,
      canvasWidth,
      canvasHeight,
      epochToX: (epoch: number) => (epoch - ctx.centerTimestamp) * ctx.pxPerSec + canvasWidth / 2,
      xToEpoch: (x: number) => ctx.centerTimestamp + (x - canvasWidth / 2) / ctx.pxPerSec,
      tooltipPortalTarget: tooltipLayer,
    }),
    [ctx.viewStart, ctx.viewEnd, ctx.pxPerSec, ctx.centerTimestamp, canvasWidth, canvasHeight, tooltipLayer],
  );

  const cursor = ctx.isPlaying ? "default" : ctx.isDraggingTimeline ? "grabbing" : "grab";

  return (
    <div ref={containerRef} className="flex-1 flex flex-col relative select-none min-h-0" data-testid="spectrogram-viewport">
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

          {/*
            Overlay container — split into two siblings:
              1. Clipped band layer: clamps every overlay child to the
                 canvas rectangle so highlights/regions cannot bleed
                 into the FrequencyAxis or right gutter.
              2. Unclipped tooltip layer: portal target for overlays
                 (DetectionOverlay, VocalizationOverlay) whose tooltips
                 must remain readable past the canvas edge.
          */}
          <div
            data-testid="overlay-band-layer"
            style={{
              position: "absolute",
              inset: 0,
              width: canvasWidth,
              height: canvasHeight,
              overflow: "hidden",
            }}
          >
            <OverlayContext.Provider value={overlayValue}>
              {children}
            </OverlayContext.Provider>
          </div>
          {/*
            Tooltip layer must NOT set its own z-index. A positioned element
            without z-index does not create a new stacking context, so the
            portaled tooltip's internal `zIndex: 20` is evaluated in the
            canvas-wrapper's stacking context and paints above the
            Playhead's `zIndex: 10`. Setting a z-index here would scope the
            tooltip's z-index locally and cause Playhead to paint over it.
          */}
          <div
            ref={setTooltipLayer}
            data-testid="overlay-tooltip-layer"
            style={{
              position: "absolute",
              inset: 0,
              pointerEvents: "none",
            }}
          />

          <Playhead canvasWidth={canvasWidth} canvasHeight={canvasHeight + (scores ? effectiveStripHeight : 0)} />

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
              height={stripHeight}
              gradient={stripGradient}
              thresholdValue={stripThreshold}
              barMode={stripBarMode}
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
