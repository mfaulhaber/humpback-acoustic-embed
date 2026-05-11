import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type * as React from "react";
import { ArrowLeft, Pause, Play } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import {
  type EventEncoderTimelineEvent,
  type EventEncoderTimelineResponse,
  useEventEncoderJob,
  useEventEncoderTimeline,
} from "@/api/sequenceModels";
import { regionAudioSliceUrl } from "@/api/client";
import { usePlayback } from "@/components/timeline/provider/usePlayback";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

import { labelColor } from "./constants";

type YMode = "f0" | "peak";
type UnvoicedMode = "peak" | "bottom" | "hide";

interface Size {
  width: number;
  height: number;
}

interface TimeRange {
  start: number;
  end: number;
}

interface FrequencyRange {
  min: number;
  max: number;
}

interface CursorInfo {
  x: number;
  y: number;
  time: number;
  frequency: number;
}

interface DrawRect {
  event: EventEncoderTimelineEvent;
  x: number;
  y: number;
  width: number;
  height: number;
  centerFrequency: number;
  voiced: boolean;
  bottomLane: boolean;
}

interface DragState {
  x: number;
  y: number;
  range: TimeRange;
  moved: boolean;
}

type PianoRollPlaybackMode = "selected" | "continuous";

interface TokenSummary {
  tokenId: number;
  label: string;
  count: number;
  meanF0: number | null;
}

const LEFT_MARGIN = 62;
const RIGHT_MARGIN = 10;
const TOP_MARGIN = 8;
const BOTTOM_MARGIN = 24;
const TOOLTIP_WIDTH = 256;
const TOOLTIP_ESTIMATED_HEIGHT = 222;
const TOOLTIP_OFFSET = 14;
const TOOLTIP_MARGIN = 8;
const UNVOICED_LANE_HEIGHT = 34;
const MIN_TIME_SPAN_SECONDS = 2;
const EVENT_TIME_BUFFER_SECONDS = 30;
const CONTINUOUS_PLAYBACK_WINDOW_SECONDS = 300;
const MAX_FREQUENCY_HZ = 5000;
const MIN_FREQUENCY_SPAN_HZ = 100;
const DEFAULT_FREQUENCY_MAX = 2000;
const VOICED_THRESHOLD = 0.3;
const FREQUENCY_OPTIONS = [1500, 2000, 3000, 4000, 5000];

export function EventEncoderPianoRollPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { data: detail, isLoading, error } = useEventEncoderJob(jobId ?? null);
  const job = detail?.job;
  const isComplete = job?.status === "complete";
  const [selectedK, setSelectedK] = useState<number | null>(null);
  const {
    data: timeline,
    isLoading: timelineLoading,
    error: timelineError,
  } = useEventEncoderTimeline(job?.id ?? null, selectedK, Boolean(isComplete));

  useEffect(() => {
    if (!timeline || selectedK !== null) return;
    setSelectedK(timeline.selected_k);
  }, [selectedK, timeline]);

  if (isLoading) {
    return (
      <PianoRollFrame>
        <CenteredMessage testId="eej-piano-roll-loading">
          Loading Event Encoder job...
        </CenteredMessage>
      </PianoRollFrame>
    );
  }

  if (error || !job) {
    return (
      <PianoRollFrame>
        <CenteredMessage testId="eej-piano-roll-error">
          Job not found.
        </CenteredMessage>
      </PianoRollFrame>
    );
  }

  if (!isComplete) {
    return (
      <PianoRollFrame>
        <CenteredMessage testId="eej-piano-roll-unavailable">
          Piano roll available after tokenization completes.
        </CenteredMessage>
      </PianoRollFrame>
    );
  }

  if (timelineLoading) {
    return (
      <PianoRollFrame>
        <CenteredMessage testId="eej-piano-roll-timeline-loading">
          Loading piano roll...
        </CenteredMessage>
      </PianoRollFrame>
    );
  }

  if (timelineError || !timeline) {
    return (
      <PianoRollFrame>
        <CenteredMessage testId="eej-piano-roll-timeline-error">
          Timeline artifact is unavailable.
        </CenteredMessage>
      </PianoRollFrame>
    );
  }

  if (timeline.events.length === 0) {
    return (
      <PianoRollFrame>
        <CenteredMessage testId="eej-piano-roll-empty">
          No tokenized events are available.
        </CenteredMessage>
      </PianoRollFrame>
    );
  }

  if (timeline.job_end_timestamp <= timeline.job_start_timestamp) {
    return (
      <PianoRollFrame>
        <CenteredMessage testId="eej-piano-roll-timeline-error">
          Timeline source bounds are unavailable.
        </CenteredMessage>
      </PianoRollFrame>
    );
  }

  return (
    <PianoRollFrame>
      <EventEncoderPianoRollViewer
        timeline={timeline}
        selectedK={timeline.selected_k}
        onSelectedKChange={setSelectedK}
      />
    </PianoRollFrame>
  );
}

function PianoRollFrame({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="fixed bottom-0 left-60 right-0 top-12 z-10 flex min-w-0 flex-col overflow-hidden bg-zinc-950 text-zinc-100"
      data-testid="eej-piano-roll-page"
    >
      {children}
    </div>
  );
}

function CenteredMessage({
  children,
  testId,
}: {
  children: React.ReactNode;
  testId: string;
}) {
  return (
    <div className="flex h-full items-center justify-center">
      <div
        className="rounded-md border border-zinc-700 bg-zinc-900 px-4 py-3 text-sm text-zinc-300"
        data-testid={testId}
      >
        {children}
      </div>
    </div>
  );
}

function EventEncoderPianoRollViewer({
  timeline,
  selectedK,
  onSelectedKChange,
}: {
  timeline: EventEncoderTimelineResponse;
  selectedK: number;
  onSelectedKChange: (k: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const minimapRef = useRef<HTMLCanvasElement | null>(null);
  const canvasWrapRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const playbackModeRef = useRef<PianoRollPlaybackMode | null>(null);

  const [size, setSize] = useState<Size>({ width: 0, height: 0 });
  const [timeRange, setTimeRange] = useState<TimeRange>(() =>
    buildEventBufferedTimeRange(timeline),
  );
  const [frequencyRange, setFrequencyRange] = useState<FrequencyRange>({
    min: 0,
    max: DEFAULT_FREQUENCY_MAX,
  });
  const [yMode, setYMode] = useState<YMode>("f0");
  const [unvoicedMode, setUnvoicedMode] = useState<UnvoicedMode>("peak");
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [hoveredEventId, setHoveredEventId] = useState<string | null>(null);
  const [tokenFilter, setTokenFilter] = useState<number | null>(null);
  const [legendCollapsed, setLegendCollapsed] = useState(false);
  const [cursor, setCursor] = useState<CursorInfo | null>(null);
  const [isDraggingTimeline, setIsDraggingTimeline] = useState(false);
  const [playbackMode, setPlaybackMode] =
    useState<PianoRollPlaybackMode | null>(null);
  const [playheadTime, setPlayheadTime] = useState<number | null>(null);

  const recordingTimeRange = useMemo(
    () => ({
      start: timeline.job_start_timestamp,
      end: timeline.job_end_timestamp,
    }),
    [timeline.job_end_timestamp, timeline.job_start_timestamp],
  );
  const fullTimeRange = useMemo(
    () => buildEventBufferedTimeRange(timeline),
    [timeline],
  );
  const tokenSummaries = useMemo(
    () => buildTokenSummaries(timeline.events),
    [timeline.events],
  );
  const selectedEvent = useMemo(
    () =>
      selectedEventId == null
        ? null
        : timeline.events.find((event) => event.event_id === selectedEventId) ??
          null,
    [selectedEventId, timeline.events],
  );
  const selectedEventIndex = useMemo(
    () =>
      selectedEventId == null
        ? -1
        : timeline.events.findIndex(
            (event) => event.event_id === selectedEventId,
          ),
    [selectedEventId, timeline.events],
  );
  const hoveredEvent = useMemo(
    () =>
      hoveredEventId == null
        ? null
        : timeline.events.find((event) => event.event_id === hoveredEventId) ??
          null,
    [hoveredEventId, timeline.events],
  );
  const tooltipEvent = hoveredEvent ?? selectedEvent;
  const tokenCount = tokenSummaries.length;
  const viewportSpan = Math.max(
    MIN_TIME_SPAN_SECONDS,
    timeRange.end - timeRange.start,
  );
  const viewportCenterTime = (timeRange.start + timeRange.end) / 2;
  const selectedFrequencyOption =
    frequencyRange.min === 0 && FREQUENCY_OPTIONS.includes(frequencyRange.max)
      ? String(frequencyRange.max)
      : "custom";

  const audioUrlBuilder = useCallback(
    (startEpoch: number, durationSec: number) =>
      regionAudioSliceUrl(
        timeline.region_detection_job_id,
        startEpoch,
        durationSec,
      ),
    [timeline.region_detection_job_id],
  );

  const clearPlaybackState = useCallback(() => {
    playbackModeRef.current = null;
    setPlaybackMode(null);
  }, []);

  const syncPlaybackEpoch = useCallback(
    (epoch: number) => {
      setPlayheadTime(epoch);
      setTimeRange((current) =>
        centerRangeOnPlayhead(current, fullTimeRange, epoch),
      );
    },
    [fullTimeRange],
  );

  const {
    play: playSelectedAudio,
    pause: pauseSelectedAudio,
    isPlaying: isSelectedAudioPlaying,
  } = usePlayback({
    mode: "slice",
    audioUrlBuilder,
    speed: 1,
    onTimeUpdate: syncPlaybackEpoch,
    onEnded: () => {
      if (playbackModeRef.current === "selected") {
        clearPlaybackState();
      }
    },
  });
  const {
    play: playContinuousAudio,
    pause: pauseContinuousAudio,
    isPlaying: isContinuousAudioPlaying,
  } = usePlayback({
    mode: "gapless",
    audioUrlBuilder,
    speed: 1,
    onTimeUpdate: syncPlaybackEpoch,
  });
  const isPlaying = isSelectedAudioPlaying || isContinuousAudioPlaying;
  const visiblePlayheadTime = isPlaying
    ? playheadTime ?? viewportCenterTime
    : viewportCenterTime;

  const setActivePlaybackMode = useCallback(
    (mode: PianoRollPlaybackMode) => {
      playbackModeRef.current = mode;
      setPlaybackMode(mode);
    },
    [],
  );

  const resetPlayback = useCallback((preservePlayhead = true) => {
    pauseSelectedAudio();
    pauseContinuousAudio();
    clearPlaybackState();
    if (!preservePlayhead) {
      setPlayheadTime(null);
    }
  }, [clearPlaybackState, pauseContinuousAudio, pauseSelectedAudio]);

  useEffect(() => {
    setTimeRange(buildEventBufferedTimeRange(timeline));
    setFrequencyRange({ min: 0, max: DEFAULT_FREQUENCY_MAX });
    setSelectedEventId(null);
    setHoveredEventId(null);
    setTokenFilter(null);
    resetPlayback(false);
  }, [resetPlayback, timeline]);

  useEffect(() => {
    const node = canvasWrapRef.current;
    if (!node) return;
    const resize = () => {
      const next = node.getBoundingClientRect();
      setSize({
        width: Math.max(0, next.width),
        height: Math.max(0, next.height),
      });
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const makeTransform = useCallback(
    (targetSize: Size = size) =>
      createTransform({
        size: targetSize,
        timeRange,
        frequencyRange,
        unvoicedMode,
      }),
    [frequencyRange, size, timeRange, unvoicedMode],
  );

  const findEventAtPoint = useCallback(
    (x: number, y: number) => {
      const transform = makeTransform();
      for (let index = timeline.events.length - 1; index >= 0; index -= 1) {
        const rect = eventRect(
          timeline.events[index],
          transform,
          yMode,
          unvoicedMode,
        );
        if (!rect) continue;
        if (
          x >= rect.x &&
          x <= rect.x + rect.width &&
          y >= rect.y &&
          y <= rect.y + rect.height
        ) {
          return rect.event;
        }
      }
      return null;
    },
    [makeTransform, timeline.events, unvoicedMode, yMode],
  );

  const centerOnEvent = useCallback(
    (event: EventEncoderTimelineEvent) => {
      const eventCenter = (event.start_timestamp + event.end_timestamp) / 2;
      setTimeRange((current) =>
        centerRangeOnPlayhead(current, fullTimeRange, eventCenter),
      );
    },
    [fullTimeRange],
  );

  const selectEventByIndex = useCallback(
    (index: number) => {
      const next = timeline.events[index];
      if (!next) return;
      setSelectedEventId(next.event_id);
      centerOnEvent(next);
    },
    [centerOnEvent, timeline.events],
  );

  const selectPreviousEvent = useCallback(() => {
    if (!timeline.events.length) return;
    const nextIndex =
      selectedEventIndex <= 0
        ? timeline.events.length - 1
        : selectedEventIndex - 1;
    selectEventByIndex(nextIndex);
  }, [selectEventByIndex, selectedEventIndex, timeline.events.length]);

  const selectNextEvent = useCallback(() => {
    if (!timeline.events.length) return;
    const nextIndex =
      selectedEventIndex < 0 || selectedEventIndex >= timeline.events.length - 1
        ? 0
        : selectedEventIndex + 1;
    selectEventByIndex(nextIndex);
  }, [selectEventByIndex, selectedEventIndex, timeline.events.length]);

  const fitAll = useCallback(() => {
    setTimeRange(fullTimeRange);
  }, [fullTimeRange]);

  const panTimeBy = useCallback(
    (deltaSeconds: number) => {
      setTimeRange((current) =>
        clampTimeRange(
          {
            start: current.start + deltaSeconds,
            end: current.end + deltaSeconds,
          },
          fullTimeRange,
        ),
      );
    },
    [fullTimeRange],
  );

  const zoomTime = useCallback(
    (centerTime: number, factor: number) => {
      setTimeRange((current) =>
        zoomTimeRange(current, fullTimeRange, centerTime, factor),
      );
    },
    [fullTimeRange],
  );

  const zoomFrequency = useCallback((centerFrequency: number, factor: number) => {
    setFrequencyRange((current) =>
      zoomFrequencyRange(current, centerFrequency, factor),
    );
  }, []);

  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      resetPlayback();
      return;
    }
    const sourceEvent = selectedEvent;
    const start =
      sourceEvent?.start_timestamp ??
      clamp(
        visiblePlayheadTime,
        recordingTimeRange.start,
        recordingTimeRange.end,
      );
    setPlayheadTime(start);
    setTimeRange((current) =>
      centerRangeOnPlayhead(current, fullTimeRange, start),
    );
    if (sourceEvent) {
      const duration = Math.max(
        0.1,
        sourceEvent.end_timestamp - sourceEvent.start_timestamp,
      );
      pauseContinuousAudio();
      setActivePlaybackMode("selected");
      playSelectedAudio(start, duration);
      return;
    }
    pauseSelectedAudio();
    setActivePlaybackMode("continuous");
    playContinuousAudio(start, CONTINUOUS_PLAYBACK_WINDOW_SECONDS);
  }, [
    fullTimeRange,
    isPlaying,
    pauseContinuousAudio,
    pauseSelectedAudio,
    playContinuousAudio,
    playSelectedAudio,
    recordingTimeRange.end,
    recordingTimeRange.start,
    resetPlayback,
    selectedEvent,
    setActivePlaybackMode,
    timeRange.end,
    timeRange.start,
    visiblePlayheadTime,
  ]);

  useEffect(() => {
    resetPlayback();
  }, [resetPlayback, selectedEventId, timeline.selected_k]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (isTextEntryTarget(event.target)) return;
      if (isNativeButtonActivation(event)) return;

      switch (event.code) {
        case "Space":
          event.preventDefault();
          togglePlayback();
          break;
        case "KeyA":
          event.preventDefault();
          selectPreviousEvent();
          break;
        case "KeyD":
          event.preventDefault();
          selectNextEvent();
          break;
        case "Escape":
          event.preventDefault();
          setSelectedEventId(null);
          setTokenFilter(null);
          break;
        case "KeyF":
          event.preventDefault();
          fitAll();
          break;
        case "ArrowLeft":
          event.preventDefault();
          panTimeBy(-viewportSpan * 0.1);
          break;
        case "ArrowRight":
          event.preventDefault();
          panTimeBy(viewportSpan * 0.1);
          break;
      }

      if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        zoomTime((timeRange.start + timeRange.end) / 2, 0.7);
      } else if (event.key === "-") {
        event.preventDefault();
        zoomTime((timeRange.start + timeRange.end) / 2, 1.35);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [
    fitAll,
    panTimeBy,
    selectNextEvent,
    selectPreviousEvent,
    timeRange.end,
    timeRange.start,
    togglePlayback,
    viewportSpan,
    zoomTime,
  ]);

  useEffect(() => {
    drawMainCanvas({
      canvas: canvasRef.current,
      size,
      timeline,
      timeRange,
      frequencyRange,
      yMode,
      unvoicedMode,
      selectedEventId,
      hoveredEventId,
      tokenFilter,
      playheadTime: visiblePlayheadTime,
    });
  }, [
    frequencyRange,
    hoveredEventId,
    selectedEventId,
    size,
    timeRange,
    timeline,
    tokenFilter,
    unvoicedMode,
    visiblePlayheadTime,
    yMode,
  ]);

  useEffect(() => {
    drawMinimap({
      canvas: minimapRef.current,
      timeline,
      timeRange,
      frequencyRange,
      yMode,
      selectedK,
      playheadTime: visiblePlayheadTime,
      fullTimeRange,
    });
  }, [
    frequencyRange,
    fullTimeRange,
    selectedK,
    timeRange,
    timeline,
    visiblePlayheadTime,
    yMode,
  ]);

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const point = canvasPoint(event);
    const transform = makeTransform();
    const time = transform.xToTime(point.x);
    const frequency = transform.yToFrequency(point.y);
    setCursor({
      x: point.x,
      y: point.y,
      time,
      frequency,
    });

    const drag = dragRef.current;
    if (drag) {
      const deltaX = point.x - drag.x;
      if (Math.abs(deltaX) > 2 || Math.abs(point.y - drag.y) > 2) {
        drag.moved = true;
      }
      const span = drag.range.end - drag.range.start;
      const plotWidth = Math.max(1, transform.plotRight - transform.plotLeft);
      const deltaSeconds = (-deltaX / plotWidth) * span;
      setTimeRange(
        clampTimeRange(
          {
            start: drag.range.start + deltaSeconds,
            end: drag.range.end + deltaSeconds,
          },
          fullTimeRange,
        ),
      );
      return;
    }

    const hovered = findEventAtPoint(point.x, point.y);
    setHoveredEventId(hovered?.event_id ?? null);
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (event.button !== 0) return;
    const point = canvasPoint(event);
    setIsDraggingTimeline(true);
    dragRef.current = {
      x: point.x,
      y: point.y,
      range: timeRange,
      moved: false,
    };
  };

  const handleMouseUp = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    dragRef.current = null;
    setIsDraggingTimeline(false);
    const point = canvasPoint(event);
    if (drag?.moved) return;
    const clicked = findEventAtPoint(point.x, point.y);
    setSelectedEventId(clicked?.event_id ?? null);
  };

  const handleMouseLeave = () => {
    dragRef.current = null;
    setIsDraggingTimeline(false);
    setCursor(null);
    setHoveredEventId(null);
  };

  const handleDoubleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const point = canvasPoint(event);
    const clicked = findEventAtPoint(point.x, point.y);
    if (!clicked) return;
    setSelectedEventId(clicked.event_id);
    const duration = Math.max(
      MIN_TIME_SPAN_SECONDS,
      clicked.end_timestamp - clicked.start_timestamp,
    );
    const padding = Math.max(1, duration * 2);
    setTimeRange(
      clampTimeRange(
        {
          start: clicked.start_timestamp - padding,
          end: clicked.end_timestamp + padding,
        },
        fullTimeRange,
      ),
    );
  };

  const handleWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const point = canvasPoint(event);
    const transform = makeTransform();
    const factor = event.deltaY < 0 ? 0.82 : 1.22;
    if (event.shiftKey) {
      zoomFrequency(transform.yToFrequency(point.y), factor);
      return;
    }
    zoomTime(transform.xToTime(point.x), factor);
  };

  const handleMinimapClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    const fullSpan = fullTimeRange.end - fullTimeRange.start;
    const center = fullTimeRange.start + fullSpan * ratio;
    const span = timeRange.end - timeRange.start;
    setTimeRange(
      clampTimeRange(
        { start: center - span / 2, end: center + span / 2 },
        fullTimeRange,
      ),
    );
  };

  const canvasCursorClass = isDraggingTimeline
    ? "cursor-grabbing"
    : hoveredEventId
      ? "cursor-pointer"
      : "cursor-grab";

  return (
    <>
      <div className="flex flex-shrink-0 flex-wrap items-center gap-2 border-b border-zinc-800 bg-zinc-950 px-3 py-2 text-xs">
        <Button
          asChild
          variant="outline"
          size="sm"
          className="h-8 border-zinc-700 bg-zinc-900 text-zinc-100 hover:bg-zinc-800"
        >
          <Link
            to={`/app/sequence-models/event-encoder/${timeline.job_id}`}
            data-testid="eej-piano-roll-back"
          >
            <ArrowLeft className="mr-1 h-4 w-4" />
            Detail
          </Link>
        </Button>
        <div className="mr-2 flex min-w-0 items-center gap-2">
          <span className="font-semibold text-zinc-100">Piano Roll</span>
          <span
            className="max-w-56 truncate font-mono text-[11px] text-zinc-400"
            data-testid="eej-piano-roll-job-id"
            title={timeline.job_id}
          >
            {timeline.job_id}
          </span>
        </div>
        <ToolbarStat
          label="events"
          value={String(timeline.events.length)}
          testId="eej-piano-roll-event-count"
        />
        <ToolbarStat
          label="duration"
          value={formatDuration(fullTimeRange.end - fullTimeRange.start)}
          testId="eej-piano-roll-duration"
        />
        <ToolbarStat
          label="tokens"
          value={String(tokenCount)}
          testId="eej-piano-roll-token-count"
        />
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <ToolbarSelect
            label="Y"
            value={yMode}
            testId="eej-piano-roll-y-mode"
            onChange={(value) => setYMode(value as YMode)}
          >
            <option value="f0">Median F0</option>
            <option value="peak">Peak Frequency</option>
          </ToolbarSelect>
          <ToolbarSelect
            label="Hz"
            value={selectedFrequencyOption}
            testId="eej-piano-roll-frequency-max"
            onChange={(value) => {
              const max = Number(value);
              if (Number.isFinite(max)) {
                setFrequencyRange({ min: 0, max });
              }
            }}
          >
            {selectedFrequencyOption === "custom" ? (
              <option value="custom">Custom</option>
            ) : null}
            {FREQUENCY_OPTIONS.map((value) => (
              <option key={value} value={value}>
                {value} Hz
              </option>
            ))}
          </ToolbarSelect>
          <ToolbarSelect
            label="Unvoiced"
            value={unvoicedMode}
            testId="eej-piano-roll-unvoiced-mode"
            onChange={(value) => setUnvoicedMode(value as UnvoicedMode)}
          >
            <option value="peak">Peak</option>
            <option value="bottom">Lane</option>
            <option value="hide">Hide</option>
          </ToolbarSelect>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 border-zinc-700 bg-zinc-900 px-2 text-zinc-100 hover:bg-zinc-800"
            onClick={togglePlayback}
            title={isPlaying ? "Pause" : "Play"}
            data-testid="eej-piano-roll-play"
            data-state={isPlaying ? "playing" : "paused"}
          >
            {isPlaying ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            <span className="ml-1">{isPlaying ? "Pause" : "Play"}</span>
          </Button>
          <ToolbarSelect
            label="k"
            value={String(selectedK)}
            testId="eej-piano-roll-k-select"
            onChange={(value) => {
              const next = Number(value);
              if (Number.isFinite(next)) onSelectedKChange(next);
            }}
          >
            {timeline.valid_k_values.map((k) => (
              <option key={k} value={k}>
                k={k}
              </option>
            ))}
          </ToolbarSelect>
        </div>
      </div>

      <div
        ref={canvasWrapRef}
        className="relative min-h-0 flex-1 overflow-hidden bg-zinc-950"
      >
        <canvas
          ref={canvasRef}
          className={cn("block h-full w-full", canvasCursorClass)}
          data-testid="eej-piano-roll-canvas"
          data-cursor-state={
            isDraggingTimeline ? "dragging" : hoveredEventId ? "hover-token" : "idle"
          }
          data-playback-mode={playbackMode ?? ""}
          data-selected-event={selectedEventId ?? ""}
          data-playhead-time={visiblePlayheadTime.toFixed(3)}
          data-token-filter={tokenFilter ?? ""}
          data-view-end={timeRange.end.toFixed(3)}
          data-view-start={timeRange.start.toFixed(3)}
          onDoubleClick={handleDoubleClick}
          onMouseDown={handleMouseDown}
          onMouseLeave={handleMouseLeave}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onWheel={handleWheel}
        />
        <canvas
          ref={minimapRef}
          className="absolute bottom-4 right-4 h-10 w-60 cursor-pointer rounded border border-zinc-700 bg-zinc-950/90"
          data-testid="eej-piano-roll-minimap"
          width={240}
          height={40}
          onClick={handleMinimapClick}
        />
        <TokenLegend
          collapsed={legendCollapsed}
          selectedToken={tokenFilter}
          selectedK={selectedK}
          summaries={tokenSummaries}
          onToggleCollapsed={() => setLegendCollapsed((value) => !value)}
          onToggleToken={(tokenId) =>
            setTokenFilter((current) => (current === tokenId ? null : tokenId))
          }
        />
        {tooltipEvent && cursor ? (
          <EventTooltip
            canvasSize={size}
            cursor={cursor}
            event={tooltipEvent}
            position={timeline.events.findIndex(
              (event) => event.event_id === tooltipEvent.event_id,
            )}
            selectedK={selectedK}
          />
        ) : null}
      </div>

      <div
        className="flex h-7 flex-shrink-0 items-center gap-4 border-t border-zinc-800 bg-zinc-950 px-3 font-mono text-[11px] text-zinc-400"
        data-testid="eej-piano-roll-status"
      >
        <span data-testid="eej-piano-roll-cursor-time">
          {cursor ? formatRelativeTime(cursor.time, timeline.job_start_timestamp) : "time -"}
        </span>
        <span data-testid="eej-piano-roll-cursor-frequency">
          {cursor ? `${Math.round(cursor.frequency)} Hz` : "freq -"}
        </span>
        <span data-testid="eej-piano-roll-zoom">
          span {formatDuration(viewportSpan)}
        </span>
        <span data-testid="eej-piano-roll-playhead-time">
          playhead{" "}
          {formatRelativeTime(visiblePlayheadTime, timeline.job_start_timestamp)}
        </span>
        <span className="hidden min-w-0 truncate text-zinc-500 md:inline">
          Scroll zooms time | Shift+Scroll zooms Hz | Drag pans | F fits |
          A/D selects events | Space plays
        </span>
      </div>
    </>
  );
}

function ToolbarStat({
  label,
  value,
  testId,
}: {
  label: string;
  value: string;
  testId: string;
}) {
  return (
    <span className="whitespace-nowrap text-zinc-500">
      {label}{" "}
      <span className="font-mono text-zinc-200" data-testid={testId}>
        {value}
      </span>
    </span>
  );
}

function ToolbarSelect({
  children,
  label,
  value,
  testId,
  onChange,
}: {
  children: React.ReactNode;
  label: string;
  value: string;
  testId: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex items-center gap-1 whitespace-nowrap text-zinc-500">
      <span>{label}</span>
      <select
        className="h-8 rounded border border-zinc-700 bg-zinc-900 px-2 text-xs text-zinc-100"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        data-testid={testId}
      >
        {children}
      </select>
    </label>
  );
}

function TokenLegend({
  collapsed,
  selectedToken,
  selectedK,
  summaries,
  onToggleCollapsed,
  onToggleToken,
}: {
  collapsed: boolean;
  selectedToken: number | null;
  selectedK: number;
  summaries: TokenSummary[];
  onToggleCollapsed: () => void;
  onToggleToken: (tokenId: number) => void;
}) {
  return (
    <div
      className="absolute right-4 top-4 max-h-[calc(100%-5rem)] min-w-44 overflow-hidden rounded border border-zinc-700 bg-zinc-950/95 text-[11px] shadow-lg"
      data-testid="eej-piano-roll-legend"
    >
      <button
        type="button"
        className="flex w-full items-center justify-between gap-3 px-2 py-1.5 text-left text-zinc-300 hover:bg-zinc-900"
        onClick={onToggleCollapsed}
        data-testid="eej-piano-roll-legend-toggle"
      >
        <span>Tokens</span>
        <span className="text-zinc-500">{collapsed ? "show" : "hide"}</span>
      </button>
      {!collapsed ? (
        <div
          className="max-h-[min(28rem,calc(100vh-11rem))] overflow-y-auto border-t border-zinc-800 p-1"
          data-testid="eej-piano-roll-legend-body"
        >
          {summaries.map((summary) => {
            const color = labelColor(summary.tokenId, Math.max(selectedK, 1));
            const active = summary.tokenId === selectedToken;
            return (
              <button
                key={summary.tokenId}
                type="button"
                className={cn(
                  "flex w-full items-center gap-2 rounded px-1.5 py-1 text-left hover:bg-zinc-900",
                  active ? "bg-zinc-800 text-zinc-50" : "text-zinc-300",
                )}
                onClick={() => onToggleToken(summary.tokenId)}
                data-testid={`eej-piano-roll-token-${summary.tokenId}`}
                aria-pressed={active}
              >
                <span
                  className="h-2.5 w-2.5 flex-shrink-0 rounded-sm"
                  style={{ backgroundColor: color }}
                />
                <span className="font-mono">{summary.label}</span>
                <span className="text-zinc-500">{summary.count}</span>
                <span className="ml-auto text-zinc-500">
                  {summary.meanF0 == null ? "noise" : `${Math.round(summary.meanF0)} Hz`}
                </span>
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function EventTooltip({
  canvasSize,
  cursor,
  event,
  position,
  selectedK,
}: {
  canvasSize: Size;
  cursor: CursorInfo;
  event: EventEncoderTimelineEvent;
  position: number;
  selectedK: number;
}) {
  const values = event.descriptor_values;
  const color = labelColor(event.token_id, Math.max(1, selectedK));
  const slope = numeric(values.ridge_log_frequency_slope);
  const slopeLabel =
    slope == null || Math.abs(slope) < 0.05
      ? "flat"
      : slope > 0
        ? "rising"
        : "falling";
  const tooltipPosition = placeTooltip(cursor, canvasSize);

  return (
    <div
      className="pointer-events-none absolute z-20 w-64 rounded border border-zinc-700 bg-zinc-950/95 p-2 text-[11px] leading-5 text-zinc-300 shadow-xl"
      style={{
        left: tooltipPosition.left,
        maxHeight: tooltipPosition.maxHeight,
        overflowY: "auto",
        top: tooltipPosition.top,
      }}
      data-testid="eej-piano-roll-tooltip"
    >
      <div className="mb-1 flex items-center gap-2">
        <span
          className="rounded px-1.5 py-0.5 font-mono font-bold text-white"
          style={{ backgroundColor: color }}
        >
          {event.token_label}
        </span>
        <span className="font-mono text-zinc-500">{event.event_id}</span>
      </div>
      <TooltipRow label="duration" value={formatSeconds(event.end_timestamp - event.start_timestamp)} />
      <TooltipRow label="median_f0" value={formatHz(values.median_f0)} />
      <TooltipRow label="f0_range" value={formatHz(values.f0_range)} />
      <TooltipRow label="peak" value={formatHz(values.peak_frequency)} />
      <TooltipRow label="voicing" value={formatRatio(values.voicing_fraction)} />
      <TooltipRow
        label="ridge"
        value={`${formatNumber(values.ridge_log_frequency_slope)} ${slopeLabel}`}
      />
      <TooltipRow label="pulse" value={formatNumber(values.pulse_rate)} />
      <TooltipRow label="gap" value={formatSeconds(values.gap_to_previous)} />
      <TooltipRow label="position" value={position >= 0 ? String(position + 1) : "-"} />
    </div>
  );
}

function TooltipRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-zinc-500">{label}</span>
      <span className="text-right font-mono text-zinc-200">{value}</span>
    </div>
  );
}

function placeTooltip(cursor: CursorInfo, canvasSize: Size) {
  const maxLeft = Math.max(
    TOOLTIP_MARGIN,
    canvasSize.width - TOOLTIP_WIDTH - TOOLTIP_MARGIN,
  );
  const maxTop = Math.max(
    TOOLTIP_MARGIN,
    canvasSize.height - TOOLTIP_ESTIMATED_HEIGHT - TOOLTIP_MARGIN,
  );
  const left =
    cursor.x + TOOLTIP_OFFSET + TOOLTIP_WIDTH <= canvasSize.width - TOOLTIP_MARGIN
      ? cursor.x + TOOLTIP_OFFSET
      : cursor.x - TOOLTIP_WIDTH - TOOLTIP_OFFSET;
  const top =
    cursor.y + TOOLTIP_OFFSET + TOOLTIP_ESTIMATED_HEIGHT <=
    canvasSize.height - TOOLTIP_MARGIN
      ? cursor.y + TOOLTIP_OFFSET
      : cursor.y - TOOLTIP_ESTIMATED_HEIGHT - TOOLTIP_OFFSET;

  return {
    left: clamp(left, TOOLTIP_MARGIN, maxLeft),
    maxHeight: Math.max(120, canvasSize.height - TOOLTIP_MARGIN * 2),
    top: clamp(top, TOOLTIP_MARGIN, maxTop),
  };
}

function drawMainCanvas({
  canvas,
  size,
  timeline,
  timeRange,
  frequencyRange,
  yMode,
  unvoicedMode,
  selectedEventId,
  hoveredEventId,
  tokenFilter,
  playheadTime,
}: {
  canvas: HTMLCanvasElement | null;
  size: Size;
  timeline: EventEncoderTimelineResponse;
  timeRange: TimeRange;
  frequencyRange: FrequencyRange;
  yMode: YMode;
  unvoicedMode: UnvoicedMode;
  selectedEventId: string | null;
  hoveredEventId: string | null;
  tokenFilter: number | null;
  playheadTime: number | null;
}) {
  if (!canvas || size.width <= 0 || size.height <= 0) return;
  const ctx = prepareCanvas(canvas, size);
  const transform = createTransform({
    size,
    timeRange,
    frequencyRange,
    unvoicedMode,
  });

  ctx.clearRect(0, 0, size.width, size.height);
  ctx.fillStyle = "#09090b";
  ctx.fillRect(0, 0, size.width, size.height);

  drawGrid(ctx, transform, timeline.job_start_timestamp, unvoicedMode);

  const rects: DrawRect[] = [];
  for (const event of timeline.events) {
    const rect = eventRect(event, transform, yMode, unvoicedMode);
    if (!rect) continue;
    if (rect.x + rect.width < transform.plotLeft || rect.x > transform.plotRight) {
      continue;
    }
    rects.push(rect);
  }

  for (const rect of rects) {
    drawEventRect(ctx, rect, {
      selectedK: timeline.selected_k,
      selected: rect.event.event_id === selectedEventId,
      hovered: rect.event.event_id === hoveredEventId,
      dimmed: tokenFilter != null && rect.event.token_id !== tokenFilter,
    });
  }

  if (playheadTime != null) {
    drawPlayhead(ctx, transform, playheadTime, timeline.job_start_timestamp);
  }
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  transform: ReturnType<typeof createTransform>,
  jobStart: number,
  unvoicedMode: UnvoicedMode,
) {
  ctx.save();
  ctx.strokeStyle = "#27272a";
  ctx.lineWidth = 1;
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.fillStyle = "#71717a";
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";

  const freqStep =
    transform.frequencyRange.max - transform.frequencyRange.min <= 800
      ? 100
      : transform.frequencyRange.max <= 2200
        ? 200
        : 500;
  const startFrequency =
    Math.ceil(transform.frequencyRange.min / freqStep) * freqStep;
  for (
    let frequency = startFrequency;
    frequency <= transform.frequencyRange.max;
    frequency += freqStep
  ) {
    const y = transform.frequencyToY(frequency);
    if (y < transform.plotTop || y > transform.plotBottom) continue;
    ctx.globalAlpha = 0.75;
    ctx.beginPath();
    ctx.moveTo(transform.plotLeft, y);
    ctx.lineTo(transform.plotRight, y);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillText(`${Math.round(frequency)} Hz`, transform.plotLeft - 8, y);
  }

  const timeStep = chooseTimeStep(transform.timeRange.end - transform.timeRange.start);
  const startOffset =
    Math.max(
      0,
      Math.ceil((transform.timeRange.start - jobStart) / timeStep) * timeStep,
    );
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  for (
    let offset = startOffset;
    jobStart + offset <= transform.timeRange.end;
    offset += timeStep
  ) {
    const time = jobStart + offset;
    const x = transform.timeToX(time);
    if (x < transform.plotLeft || x > transform.plotRight) continue;
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    ctx.moveTo(x, transform.plotTop);
    ctx.lineTo(x, transform.plotBottom);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillText(
      formatRelativeTime(time, jobStart),
      x,
      transform.size.height - 7,
    );
  }

  if (unvoicedMode === "bottom") {
    const laneTop = transform.plotBottom - UNVOICED_LANE_HEIGHT;
    ctx.save();
    ctx.strokeStyle = "#52525b";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(transform.plotLeft, laneTop);
    ctx.lineTo(transform.plotRight, laneTop);
    ctx.stroke();
    ctx.restore();
    ctx.textAlign = "right";
    ctx.fillText("noise", transform.plotLeft - 8, laneTop + 14);
  }

  ctx.restore();
}

function drawEventRect(
  ctx: CanvasRenderingContext2D,
  rect: DrawRect,
  options: {
    selectedK: number;
    selected: boolean;
    hovered: boolean;
    dimmed: boolean;
  },
) {
  const color = labelColor(rect.event.token_id, Math.max(options.selectedK, 1));
  const voicing = clamp(numeric(rect.event.descriptor_values.voicing_fraction) ?? 0, 0, 1);
  const baseAlpha = rect.voiced ? 0.55 + voicing * 0.35 : 0.18;
  const alpha = options.dimmed ? 0.12 : baseAlpha;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
  ctx.globalAlpha = options.dimmed ? 0.2 : 0.95;
  ctx.strokeStyle = color;
  ctx.lineWidth = rect.voiced ? 1 : 2;
  ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
  ctx.restore();

  if (!options.dimmed) {
    drawSlopeLine(ctx, rect);
    drawTokenLabel(ctx, rect, color);
  }

  if (options.selected || options.hovered) {
    ctx.save();
    ctx.strokeStyle = options.selected ? "#fafafa" : "#d4d4d8";
    ctx.lineWidth = options.selected ? 2 : 1;
    ctx.strokeRect(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2);
    ctx.restore();
  }
}

function drawPlayhead(
  ctx: CanvasRenderingContext2D,
  transform: ReturnType<typeof createTransform>,
  playheadTime: number,
  jobStart: number,
) {
  const x = transform.timeToX(playheadTime);
  if (x < transform.plotLeft || x > transform.plotRight) return;

  ctx.save();
  ctx.strokeStyle = "#facc15";
  ctx.fillStyle = "#facc15";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, transform.plotTop);
  ctx.lineTo(x, transform.plotBottom);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x, transform.plotTop + 1);
  ctx.lineTo(x - 4, transform.plotTop + 8);
  ctx.lineTo(x + 4, transform.plotTop + 8);
  ctx.closePath();
  ctx.fill();

  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(formatRelativeTime(playheadTime, jobStart), x + 7, transform.plotTop + 3);
  ctx.restore();
}

function drawSlopeLine(ctx: CanvasRenderingContext2D, rect: DrawRect) {
  if (rect.width < 12 || rect.height < 6) return;
  const slope = numeric(rect.event.descriptor_values.ridge_log_frequency_slope) ?? 0;
  const deflection = clamp(slope / 4, -1, 1) * rect.height * 0.4;
  const yMid = rect.y + rect.height / 2;
  const x1 = rect.x + 3;
  const x2 = rect.x + rect.width - 3;
  const y1 = yMid + deflection;
  const y2 = yMid - deflection;

  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.88)";
  ctx.lineWidth = 1.5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  if (rect.width >= 30 && Math.abs(slope) > 0.3) {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const wing = 5;
    ctx.lineTo(
      x2 - Math.cos(angle - Math.PI / 5) * wing,
      y2 - Math.sin(angle - Math.PI / 5) * wing,
    );
    ctx.moveTo(x2, y2);
    ctx.lineTo(
      x2 - Math.cos(angle + Math.PI / 5) * wing,
      y2 - Math.sin(angle + Math.PI / 5) * wing,
    );
  }
  ctx.stroke();
  ctx.restore();
}

function drawTokenLabel(
  ctx: CanvasRenderingContext2D,
  rect: DrawRect,
  color: string,
) {
  if (rect.width < 20 || rect.height < 10) return;
  const label = rect.event.token_label;
  ctx.save();
  ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
  const textWidth = ctx.measureText(label).width;
  const pillWidth = Math.max(20, textWidth + 8);
  const pillHeight = 12;
  const x = rect.x + rect.width / 2 - pillWidth / 2;
  const y = rect.y + rect.height / 2 - pillHeight / 2;
  ctx.fillStyle = color;
  roundedRect(ctx, x, y, pillWidth, pillHeight, 3);
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, rect.x + rect.width / 2, rect.y + rect.height / 2 + 0.5);
  ctx.restore();
}

function drawMinimap({
  canvas,
  timeline,
  timeRange,
  fullTimeRange,
  frequencyRange,
  yMode,
  selectedK,
  playheadTime,
}: {
  canvas: HTMLCanvasElement | null;
  timeline: EventEncoderTimelineResponse;
  timeRange: TimeRange;
  fullTimeRange: TimeRange;
  frequencyRange: FrequencyRange;
  yMode: YMode;
  selectedK: number;
  playheadTime: number | null;
}) {
  if (!canvas) return;
  const size = { width: 240, height: 40 };
  const ctx = prepareCanvas(canvas, size);
  ctx.clearRect(0, 0, size.width, size.height);
  ctx.fillStyle = "rgba(9, 9, 11, 0.95)";
  ctx.fillRect(0, 0, size.width, size.height);

  const fullStart = fullTimeRange.start;
  const fullSpan = Math.max(1, fullTimeRange.end - fullStart);
  for (const event of timeline.events) {
    const x = ((event.start_timestamp - fullStart) / fullSpan) * size.width;
    const frequency = eventCenterFrequency(event, yMode);
    const y = size.height * (1 - clamp(frequency / MAX_FREQUENCY_HZ, 0, 1));
    ctx.fillStyle = labelColor(event.token_id, Math.max(selectedK, 1));
    ctx.globalAlpha = 0.85;
    ctx.fillRect(Math.round(x), Math.round(y), 2, 2);
  }
  ctx.globalAlpha = 1;
  const x1 = ((timeRange.start - fullStart) / fullSpan) * size.width;
  const x2 = ((timeRange.end - fullStart) / fullSpan) * size.width;
  const y1 = size.height * (1 - clamp(frequencyRange.max / MAX_FREQUENCY_HZ, 0, 1));
  const y2 = size.height * (1 - clamp(frequencyRange.min / MAX_FREQUENCY_HZ, 0, 1));
  ctx.strokeStyle = "#f8fafc";
  ctx.lineWidth = 1;
  ctx.strokeRect(x1, y1, Math.max(2, x2 - x1), Math.max(4, y2 - y1));

  if (playheadTime != null) {
    const playheadX = ((playheadTime - fullStart) / fullSpan) * size.width;
    ctx.strokeStyle = "#facc15";
    ctx.beginPath();
    ctx.moveTo(playheadX, 0);
    ctx.lineTo(playheadX, size.height);
    ctx.stroke();
  }
}

function createTransform({
  size,
  timeRange,
  frequencyRange,
  unvoicedMode,
}: {
  size: Size;
  timeRange: TimeRange;
  frequencyRange: FrequencyRange;
  unvoicedMode: UnvoicedMode;
}) {
  const plotLeft = LEFT_MARGIN;
  const plotRight = Math.max(plotLeft + 1, size.width - RIGHT_MARGIN);
  const plotTop = TOP_MARGIN;
  const plotBottom = Math.max(plotTop + 1, size.height - BOTTOM_MARGIN);
  const frequencyBottom =
    unvoicedMode === "bottom"
      ? Math.max(plotTop + 1, plotBottom - UNVOICED_LANE_HEIGHT)
      : plotBottom;
  const plotWidth = plotRight - plotLeft;
  const frequencyHeight = frequencyBottom - plotTop;
  const timeSpan = Math.max(0.001, timeRange.end - timeRange.start);
  const frequencySpan = Math.max(0.001, frequencyRange.max - frequencyRange.min);

  return {
    size,
    timeRange,
    frequencyRange,
    plotLeft,
    plotRight,
    plotTop,
    plotBottom,
    frequencyBottom,
    timeToX(time: number) {
      return plotLeft + ((time - timeRange.start) / timeSpan) * plotWidth;
    },
    xToTime(x: number) {
      return timeRange.start + ((x - plotLeft) / plotWidth) * timeSpan;
    },
    frequencyToY(frequency: number) {
      return plotTop + (1 - (frequency - frequencyRange.min) / frequencySpan) * frequencyHeight;
    },
    yToFrequency(y: number) {
      return frequencyRange.min + (1 - (y - plotTop) / frequencyHeight) * frequencySpan;
    },
  };
}

function eventRect(
  event: EventEncoderTimelineEvent,
  transform: ReturnType<typeof createTransform>,
  yMode: YMode,
  unvoicedMode: UnvoicedMode,
): DrawRect | null {
  const voiced = isVoiced(event);
  if (!voiced && unvoicedMode === "hide") return null;

  const x1 = transform.timeToX(event.start_timestamp);
  const x2 = transform.timeToX(event.end_timestamp);
  const width = Math.max(2, x2 - x1);

  if (!voiced && unvoicedMode === "bottom") {
    const laneTop = transform.plotBottom - UNVOICED_LANE_HEIGHT;
    return {
      event,
      x: x1,
      y: laneTop + 5,
      width,
      height: UNVOICED_LANE_HEIGHT - 10,
      centerFrequency: 0,
      voiced,
      bottomLane: true,
    };
  }

  const centerFrequency = eventCenterFrequency(event, yMode);
  const f0Range = Math.max(0, numeric(event.descriptor_values.f0_range) ?? 0);
  const yTop = transform.frequencyToY(centerFrequency + f0Range / 2);
  const yBottom = transform.frequencyToY(centerFrequency - f0Range / 2);
  const height = Math.max(4, yBottom - yTop);
  const y = height === 4
    ? transform.frequencyToY(centerFrequency) - 2
    : yTop;

  return {
    event,
    x: x1,
    y: clamp(y, transform.plotTop, transform.plotBottom - height),
    width,
    height,
    centerFrequency,
    voiced,
    bottomLane: false,
  };
}

function eventCenterFrequency(event: EventEncoderTimelineEvent, yMode: YMode) {
  const values = event.descriptor_values;
  const peak = numeric(values.peak_frequency) ?? 0;
  if (yMode === "peak" || !isVoiced(event)) return peak;
  return numeric(values.median_f0) ?? peak;
}

function isVoiced(event: EventEncoderTimelineEvent) {
  return (numeric(event.descriptor_values.voicing_fraction) ?? 0) > VOICED_THRESHOLD;
}

function prepareCanvas(canvas: HTMLCanvasElement, size: Size) {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(size.width * dpr));
  const height = Math.max(1, Math.floor(size.height * dpr));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  canvas.style.width = `${size.width}px`;
  canvas.style.height = `${size.height}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function buildTokenSummaries(events: EventEncoderTimelineEvent[]): TokenSummary[] {
  const byToken = new Map<
    number,
    { label: string; count: number; f0Sum: number; f0Count: number }
  >();
  for (const event of events) {
    const current = byToken.get(event.token_id) ?? {
      label: event.token_label,
      count: 0,
      f0Sum: 0,
      f0Count: 0,
    };
    current.count += 1;
    const medianF0 = numeric(event.descriptor_values.median_f0);
    if (isVoiced(event) && medianF0 != null) {
      current.f0Sum += medianF0;
      current.f0Count += 1;
    }
    byToken.set(event.token_id, current);
  }
  return Array.from(byToken.entries())
    .map(([tokenId, summary]) => ({
      tokenId,
      label: summary.label,
      count: summary.count,
      meanF0: summary.f0Count ? summary.f0Sum / summary.f0Count : null,
    }))
    .sort((a, b) => a.tokenId - b.tokenId);
}

function buildEventBufferedTimeRange(
  timeline: EventEncoderTimelineResponse,
): TimeRange {
  if (!timeline.events.length) {
    return {
      start: timeline.job_start_timestamp,
      end: Math.max(
        timeline.job_start_timestamp + MIN_TIME_SPAN_SECONDS,
        timeline.job_end_timestamp,
      ),
    };
  }

  const firstStart = Math.min(
    ...timeline.events.map((event) => event.start_timestamp),
  );
  const lastEnd = Math.max(
    ...timeline.events.map((event) => event.end_timestamp),
  );
  const start = firstStart - EVENT_TIME_BUFFER_SECONDS;
  const end = lastEnd + EVENT_TIME_BUFFER_SECONDS;
  return {
    start,
    end: Math.max(start + MIN_TIME_SPAN_SECONDS, end),
  };
}

function zoomTimeRange(
  range: TimeRange,
  fullRange: TimeRange,
  center: number,
  factor: number,
) {
  const fullSpan = fullRange.end - fullRange.start;
  const currentSpan = range.end - range.start;
  const nextSpan = clamp(
    currentSpan * factor,
    MIN_TIME_SPAN_SECONDS,
    Math.max(MIN_TIME_SPAN_SECONDS, fullSpan),
  );
  const ratio = clamp((center - range.start) / currentSpan, 0, 1);
  return clampTimeRange(
    {
      start: center - nextSpan * ratio,
      end: center + nextSpan * (1 - ratio),
    },
    fullRange,
  );
}

function clampTimeRange(range: TimeRange, fullRange: TimeRange): TimeRange {
  const fullSpan = fullRange.end - fullRange.start;
  const requestedSpan = range.end - range.start;
  const span = clamp(
    requestedSpan,
    MIN_TIME_SPAN_SECONDS,
    Math.max(MIN_TIME_SPAN_SECONDS, fullSpan),
  );
  let start = range.start;
  let end = start + span;
  if (start < fullRange.start) {
    start = fullRange.start;
    end = start + span;
  }
  if (end > fullRange.end) {
    end = fullRange.end;
    start = end - span;
  }
  return { start, end };
}

function centerRangeOnPlayhead(
  range: TimeRange,
  fullRange: TimeRange,
  playheadTime: number,
): TimeRange {
  const span = range.end - range.start;
  const center = clamp(playheadTime, fullRange.start, fullRange.end);
  const next = clampTimeRange(
    {
      start: center - span / 2,
      end: center + span / 2,
    },
    {
      start: fullRange.start - span / 2,
      end: fullRange.end + span / 2,
    },
  );
  if (
    Math.abs(next.start - range.start) < 0.001 &&
    Math.abs(next.end - range.end) < 0.001
  ) {
    return range;
  }
  return next;
}

function zoomFrequencyRange(
  range: FrequencyRange,
  center: number,
  factor: number,
) {
  const currentSpan = range.max - range.min;
  const nextSpan = clamp(
    currentSpan * factor,
    MIN_FREQUENCY_SPAN_HZ,
    MAX_FREQUENCY_HZ,
  );
  const ratio = clamp((center - range.min) / currentSpan, 0, 1);
  let min = center - nextSpan * ratio;
  let max = center + nextSpan * (1 - ratio);
  if (min < 0) {
    max -= min;
    min = 0;
  }
  if (max > MAX_FREQUENCY_HZ) {
    min -= max - MAX_FREQUENCY_HZ;
    max = MAX_FREQUENCY_HZ;
  }
  return {
    min: Math.max(0, min),
    max: Math.min(MAX_FREQUENCY_HZ, Math.max(max, min + MIN_FREQUENCY_SPAN_HZ)),
  };
}

function chooseTimeStep(span: number) {
  if (span <= 15) return 2;
  if (span <= 60) return 10;
  if (span <= 180) return 30;
  if (span <= 600) return 60;
  if (span <= 1800) return 300;
  return 600;
}

function canvasPoint(event: React.MouseEvent<HTMLCanvasElement>) {
  const rect = event.currentTarget.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
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

function formatRelativeTime(time: number, start: number) {
  return formatDuration(Math.max(0, time - start));
}

function formatSeconds(value: unknown) {
  const number = numeric(value);
  return number == null ? "-" : `${number.toFixed(2)} s`;
}

function formatHz(value: unknown) {
  const number = numeric(value);
  return number == null ? "-" : `${Math.round(number)} Hz`;
}

function formatRatio(value: unknown) {
  const number = numeric(value);
  return number == null ? "-" : number.toFixed(2);
}

function formatNumber(value: unknown) {
  const number = numeric(value);
  return number == null ? "-" : number.toFixed(3);
}

function isTextEntryTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    target.isContentEditable
  );
}

function isNativeButtonActivation(event: KeyboardEvent): boolean {
  if (event.code !== "Space" && event.code !== "Enter") return false;
  if (!(event.target instanceof HTMLElement)) return false;
  return event.target.closest("button") != null;
}
