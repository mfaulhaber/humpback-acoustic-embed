import { useEffect, useMemo, useRef, useState } from "react";
import type * as React from "react";

import { regionTileUrl } from "@/api/client";
import type { EventEncoderTimelineResponse } from "@/api/sequenceModels";
import { TileCanvas } from "@/components/timeline/TileCanvas";
import { cn } from "@/lib/utils";

import {
  chooseSpectrogramLod,
  EVENT_ENCODER_SPECTROGRAM_LODS,
} from "./eventEncoderSpectrogramLod";

interface TimeRange {
  start: number;
  end: number;
}

interface FrequencyRange {
  min: number;
  max: number;
}

interface EventEncoderSpectrogramStripProps {
  timeline: EventEncoderTimelineResponse;
  timeRange: TimeRange;
  frequencyRange: FrequencyRange;
  playheadTime: number;
  plotLeftPx: number;
  plotRightPx: number;
  onTimeRangeChange: (range: TimeRange) => void;
  onZoomTime: (centerTime: number, factor: number) => void;
  onZoomFrequency: (centerFrequency: number, factor: number) => void;
}

interface Size {
  width: number;
  height: number;
}

interface DragState {
  x: number;
  range: TimeRange;
  moved: boolean;
}

const TIME_AXIS_HEIGHT = 18;
const MIN_TILE_HEIGHT = 64;

export function EventEncoderSpectrogramStrip({
  timeline,
  timeRange,
  frequencyRange,
  playheadTime,
  plotLeftPx,
  plotRightPx,
  onTimeRangeChange,
  onZoomTime,
  onZoomFrequency,
}: EventEncoderSpectrogramStripProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });
  const [currentLodKey, setCurrentLodKey] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const resize = () => {
      const next = node.getBoundingClientRect();
      setSize({
        width: Math.max(0, Math.floor(next.width)),
        height: Math.max(0, Math.floor(next.height)),
      });
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const tileHeight = Math.max(MIN_TILE_HEIGHT, size.height - TIME_AXIS_HEIGHT);
  const plotWidth = Math.max(1, size.width - plotLeftPx - plotRightPx);
  const viewportSpan = Math.max(0.001, timeRange.end - timeRange.start);
  const lod = useMemo(
    () =>
      chooseSpectrogramLod({
        viewportSpan,
        viewportWidth: plotWidth,
        currentKey: currentLodKey,
        lods: EVENT_ENCODER_SPECTROGRAM_LODS,
      }),
    [currentLodKey, plotWidth, viewportSpan],
  );
  useEffect(() => {
    if (lod.key !== currentLodKey) {
      setCurrentLodKey(lod.key);
    }
  }, [currentLodKey, lod.key]);

  const centerTimestamp = (timeRange.start + timeRange.end) / 2;
  const playheadX =
    plotLeftPx + ((playheadTime - timeRange.start) / viewportSpan) * plotWidth;
  const showPlayhead = playheadX >= plotLeftPx && playheadX <= plotLeftPx + plotWidth;
  const ticks = useMemo(
    () => buildTimeTicks(timeRange, timeline.job_start_timestamp, plotWidth),
    [plotWidth, timeRange, timeline.job_start_timestamp],
  );
  const frequencyLabels = useMemo(
    () => [
      frequencyRange.max,
      (frequencyRange.min + frequencyRange.max) / 2,
      frequencyRange.min,
    ],
    [frequencyRange.max, frequencyRange.min],
  );

  const xToTime = (x: number) => {
    const ratio = clamp((x - plotLeftPx) / plotWidth, 0, 1);
    return timeRange.start + ratio * viewportSpan;
  };
  const yToFrequency = (y: number) => {
    const ratio = clamp(y / Math.max(1, tileHeight), 0, 1);
    return frequencyRange.min + (1 - ratio) * (frequencyRange.max - frequencyRange.min);
  };

  const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const point = localPoint(event);
    const factor = event.deltaY < 0 ? 0.82 : 1.22;
    if (event.shiftKey) {
      onZoomFrequency(yToFrequency(point.y), factor);
      return;
    }
    onZoomTime(xToTime(point.x), factor);
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    const point = localPoint(event);
    setDragging(true);
    dragRef.current = {
      x: point.x,
      range: timeRange,
      moved: false,
    };
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const point = localPoint(event);
    const deltaX = point.x - drag.x;
    if (Math.abs(deltaX) > 2) {
      drag.moved = true;
    }
    const span = drag.range.end - drag.range.start;
    const deltaSeconds = (-deltaX / plotWidth) * span;
    onTimeRangeChange({
      start: drag.range.start + deltaSeconds,
      end: drag.range.end + deltaSeconds,
    });
  };

  const endDrag = () => {
    dragRef.current = null;
    setDragging(false);
  };

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative h-40 flex-shrink-0 select-none overflow-hidden border-y border-zinc-800 bg-zinc-950",
        dragging ? "cursor-grabbing" : "cursor-grab",
      )}
      data-testid="eej-piano-roll-spectrogram-strip"
      data-zoom-level={lod.key}
      data-view-start={timeRange.start.toFixed(3)}
      data-view-end={timeRange.end.toFixed(3)}
      onMouseDown={handleMouseDown}
      onMouseLeave={endDrag}
      onMouseMove={handleMouseMove}
      onMouseUp={endDrag}
      onWheel={handleWheel}
    >
      <div
        className="absolute left-0 top-0 border-r border-zinc-800 bg-zinc-950"
        style={{ width: plotLeftPx, height: tileHeight }}
      >
        {frequencyLabels.map((frequency, index) => {
          const top =
            index === 0
              ? 8
              : index === frequencyLabels.length - 1
                ? tileHeight - 10
                : tileHeight / 2;
          return (
            <span
              key={`${frequency}-${index}`}
              className="absolute right-2 -translate-y-1/2 font-mono text-[10px] text-zinc-500"
              style={{ top }}
            >
              {Math.round(frequency)}
            </span>
          );
        })}
      </div>
      <div
        className="absolute top-0 overflow-hidden bg-zinc-950"
        style={{
          left: plotLeftPx,
          width: plotWidth,
          height: tileHeight,
        }}
        data-testid="eej-piano-roll-spectrogram-tiles"
      >
        <TileCanvas
          jobId={timeline.region_detection_job_id}
          jobStart={timeline.job_start_timestamp}
          jobEnd={timeline.job_end_timestamp}
          centerTimestamp={centerTimestamp}
          zoomLevel={lod.key}
          freqRange={[frequencyRange.min, frequencyRange.max]}
          width={plotWidth}
          height={tileHeight}
          tileDurationOverride={lod.tileDuration}
          viewportSpanOverride={viewportSpan}
          tileUrlBuilder={regionTileUrl}
        />
      </div>
      {showPlayhead ? (
        <div
          className="pointer-events-none absolute top-0 z-10 w-px bg-yellow-300"
          style={{ left: playheadX, height: tileHeight }}
          data-testid="eej-piano-roll-spectrogram-playhead"
        />
      ) : null}
      <div
        className="absolute bottom-0 border-t border-zinc-800 bg-zinc-950"
        style={{
          left: plotLeftPx,
          width: plotWidth,
          height: TIME_AXIS_HEIGHT,
        }}
      >
        {ticks.map((tick) => (
          <span
            key={`${tick.time}-${tick.label}`}
            className="absolute top-0 -translate-x-1/2 font-mono text-[10px] leading-[18px] text-zinc-500"
            style={{ left: tick.x }}
          >
            {tick.label}
          </span>
        ))}
      </div>
      <div
        className="pointer-events-none absolute right-2 top-2 rounded border border-zinc-700 bg-zinc-950/85 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400"
        data-testid="eej-piano-roll-spectrogram-lod"
      >
        {lod.key}
      </div>
    </div>
  );
}

function buildTimeTicks(
  timeRange: TimeRange,
  jobStart: number,
  plotWidth: number,
) {
  const span = Math.max(0.001, timeRange.end - timeRange.start);
  const step = chooseTimeStep(span);
  const startOffset = Math.max(
    0,
    Math.ceil((timeRange.start - jobStart) / step) * step,
  );
  const ticks: Array<{ time: number; x: number; label: string }> = [];
  for (
    let offset = startOffset;
    jobStart + offset <= timeRange.end && ticks.length < 12;
    offset += step
  ) {
    const time = jobStart + offset;
    if (time < timeRange.start) continue;
    ticks.push({
      time,
      x: ((time - timeRange.start) / span) * plotWidth,
      label: formatDuration(Math.max(0, time - jobStart)),
    });
  }
  return ticks;
}

function chooseTimeStep(span: number) {
  if (span <= 15) return 2;
  if (span <= 60) return 10;
  if (span <= 180) return 30;
  if (span <= 600) return 60;
  if (span <= 1800) return 300;
  return 600;
}

function localPoint(event: React.MouseEvent | React.WheelEvent) {
  const rect = event.currentTarget.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds)) return "-";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}:${String(remainder).padStart(2, "0")}`;
  const hours = Math.floor(minutes / 60);
  const minuteRemainder = minutes % 60;
  return `${hours}:${String(minuteRemainder).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
