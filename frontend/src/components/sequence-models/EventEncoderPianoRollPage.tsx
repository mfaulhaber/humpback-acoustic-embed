import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type * as React from "react";
import { ArrowLeft, ChevronDown, ChevronUp, Pause, Play } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import {
  ApiError,
  type EventEncoderTimelineEvent,
  type EventEncoderTimelineResponse,
  type PianoRollNote,
  type PianoRollNoteContourFrame,
  useCreatePianoRollNotesJob,
  useEventEncoderJob,
  useEventEncoderTimeline,
  usePianoRollNoteContours,
  usePianoRollNotes,
  usePianoRollNotesStatus,
} from "@/api/sequenceModels";
import { regionAudioSliceUrl } from "@/api/client";
import { usePlayback } from "@/components/timeline/provider/usePlayback";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/use-toast";
import { cn } from "@/lib/utils";

import { labelColor } from "./constants";
import {
  hasRidgeFrequencyDescriptors,
  isEventVoiced,
  resolveEventDisplayBand,
  type EventEncoderYMode,
} from "./eventEncoderDisplayBand";
import { EventEncoderSpectrogramStrip } from "./EventEncoderSpectrogramStrip";
import {
  MIDI_MAX_PITCH,
  MIDI_MIN_PITCH,
  MIDI_PITCH_COUNT,
  isBlackKey,
  isInPianoBand,
  midiNoteName,
  midiPitchAtY,
  midiPitchToY,
  partialIndexLabel,
} from "./pianoRollAxis";
import { MidiExportButton } from "./MidiExportButton";
import { PianoRollNotesStatusPill } from "./PianoRollNotesStatusPill";

type YMode = EventEncoderYMode;
type UnvoicedMode = "peak" | "bottom" | "hide";
type ViewMode = "notes" | YMode;

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
const BOTTOM_MARGIN = 8;
const TOOLTIP_WIDTH = 256;
const TOOLTIP_ESTIMATED_HEIGHT = 260;
const TOOLTIP_OFFSET = 14;
const TOOLTIP_MARGIN = 8;
const UNVOICED_LANE_HEIGHT = 34;
const MIN_TIME_SPAN_SECONDS = 2;
const EVENT_TIME_BUFFER_SECONDS = 30;
const CONTINUOUS_PLAYBACK_WINDOW_SECONDS = 300;
const MAX_FREQUENCY_HZ = 6000;
const MIN_FREQUENCY_SPAN_HZ = 100;
const DEFAULT_FREQUENCY_MAX = 2000;
const DEFAULT_RIDGE_FREQUENCY_MAX = 6000;
const FREQUENCY_OPTIONS = [1500, 2000, 3000, 4000, 5000, 6000];
const CONTOUR_BATCH_LIMIT = 2000;
// Bound the per-page contour cache so a long pan across a job with
// millions of v3 notes does not retain every contour for the page
// lifetime. 50_000 covers ~25 full viewports at the batch cap and is
// dwarfed by typical browser tab memory budgets; FIFO eviction is fine
// because the access pattern is dominated by the current viewport, not
// long-tail re-hits.
const CONTOUR_CACHE_CAP = 50000;

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
  const canvasWrapRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const playbackModeRef = useRef<PianoRollPlaybackMode | null>(null);

  const defaultRectangleMode: YMode = hasRidgeFrequencyDescriptors(timeline.events)
    ? "ridge"
    : "f0";

  const [size, setSize] = useState<Size>({ width: 0, height: 0 });
  const [timeRange, setTimeRange] = useState<TimeRange>(() =>
    buildEventBufferedTimeRange(timeline),
  );
  const [frequencyRange, setFrequencyRange] = useState<FrequencyRange>(() =>
    defaultFrequencyRange(timeline),
  );
  const [viewMode, setViewMode] = useState<ViewMode>(defaultRectangleMode);
  const [lastRectangleMode, setLastRectangleMode] =
    useState<YMode>(defaultRectangleMode);
  const userViewModeOverrideRef = useRef(false);
  const notesFallbackToastedRef = useRef(false);
  const [unvoicedMode, setUnvoicedMode] = useState<UnvoicedMode>("peak");
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [hoveredEventId, setHoveredEventId] = useState<string | null>(null);
  const [hoveredNoteIndex, setHoveredNoteIndex] = useState<number | null>(null);
  const [tokenFilter, setTokenFilter] = useState<number | null>(null);
  const [legendCollapsed, setLegendCollapsed] = useState(false);
  const [cursor, setCursor] = useState<CursorInfo | null>(null);
  const [isDraggingTimeline, setIsDraggingTimeline] = useState(false);
  const [playbackMode, setPlaybackMode] =
    useState<PianoRollPlaybackMode | null>(null);
  const [playheadTime, setPlayheadTime] = useState<number | null>(null);
  const [spectrogramCollapsed, setSpectrogramCollapsed] = useState(false);

  const { data: notesStatus } = usePianoRollNotesStatus(timeline.job_id);
  const notesAvailable = notesStatus?.status === "complete";
  const yMode: YMode = viewMode === "notes" ? lastRectangleMode : viewMode;
  const isNotesMode = viewMode === "notes";

  const {
    data: notesData,
    isError: notesQueryError,
    isFetched: notesQueryFetched,
  } = usePianoRollNotes(timeline.job_id, {}, isNotesMode && notesAvailable);

  useEffect(() => {
    if (userViewModeOverrideRef.current) return;
    if (notesAvailable && viewMode !== "notes") {
      setViewMode("notes");
    }
  }, [notesAvailable, viewMode]);

  useEffect(() => {
    if (!isNotesMode) {
      notesFallbackToastedRef.current = false;
      return;
    }
    if (notesQueryError && !notesFallbackToastedRef.current) {
      notesFallbackToastedRef.current = true;
      userViewModeOverrideRef.current = true;
      toast({
        title: "Notes unavailable",
        description: "Falling back to rectangle view.",
        variant: "destructive",
      });
      setViewMode(lastRectangleMode);
    }
  }, [isNotesMode, lastRectangleMode, notesQueryError]);

  useEffect(() => {
    // Re-enable auto-switch only after recovering from a /notes fetch failure;
    // a deliberate user mode pick stays put even if notesAvailable flips.
    if (notesAvailable && notesFallbackToastedRef.current) {
      notesFallbackToastedRef.current = false;
      userViewModeOverrideRef.current = false;
    }
  }, [notesAvailable]);

  useEffect(() => {
    setHoveredNoteIndex(null);
  }, [tokenFilter]);

  const setViewModeFromUser = useCallback((next: ViewMode) => {
    userViewModeOverrideRef.current = true;
    if (next !== "notes") {
      setLastRectangleMode(next);
    }
    setViewMode(next);
  }, []);

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
    setFrequencyRange(defaultFrequencyRange(timeline));
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

  const setClampedTimeRange = useCallback(
    (range: TimeRange) => {
      setTimeRange(clampTimeRange(range, fullTimeRange));
    },
    [fullTimeRange],
  );

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

  const visibleNotes = useMemo(() => {
    if (!notesData) return [] as PianoRollNote[];
    return notesData.notes.filter(
      (note) =>
        note.start_utc + note.duration_s >= timeRange.start &&
        note.start_utc <= timeRange.end,
    );
  }, [notesData, timeRange.end, timeRange.start]);

  // Accumulated contour cache keyed by note_uid. Panning across the
  // timeline keeps previously-fetched ribbons available without
  // re-requesting them. Cache survives so long as the page stays
  // mounted; switching jobs replaces the cache by remount.
  const contourCacheRef = useRef<Map<string, PianoRollNoteContourFrame[]>>(
    new Map(),
  );
  const [contourVersion, setContourVersion] = useState(0);

  // Server caps a single /notes/contours request at 2000 note_uids
  // (ADR-069 §7). On a dense viewport the uncached set easily exceeds
  // that, so request one batch per render: each successful batch bumps
  // `contourVersion`, which re-runs this memo, drops the now-cached uids,
  // and exposes the next chunk for the hook to fetch.
  const uncachedContourUids = useMemo(() => {
    if (!isNotesMode) return [] as string[];
    const out: string[] = [];
    for (const note of visibleNotes) {
      if (!note.note_uid) continue;
      if (contourCacheRef.current.has(note.note_uid)) continue;
      out.push(note.note_uid);
      if (out.length >= CONTOUR_BATCH_LIMIT) break;
    }
    return out;
    // contourVersion is intentionally a dep so successive batches fire
    // as previous ones land in the cache.
  }, [isNotesMode, visibleNotes, contourVersion]);

  const contourQuery = usePianoRollNoteContours(
    timeline.job_id,
    uncachedContourUids,
    isNotesMode && uncachedContourUids.length > 0,
    notesData?.extractor_version,
  );
  const contourFetchError = contourQuery.isError;
  const contourFetched = contourQuery.data;

  useEffect(() => {
    if (!contourFetched) return;
    let changed = false;
    const cache = contourCacheRef.current;
    for (const [uid, rows] of Object.entries(contourFetched.contours)) {
      if (!cache.has(uid)) {
        cache.set(uid, rows);
        changed = true;
      }
    }
    // FIFO eviction: Map preserves insertion order, so the oldest entry
    // is the first key. Drop in a loop because a single batch can push
    // the cache up to CONTOUR_BATCH_LIMIT entries past the cap.
    while (cache.size > CONTOUR_CACHE_CAP) {
      const oldest = cache.keys().next().value;
      if (oldest === undefined) break;
      cache.delete(oldest);
      changed = true;
    }
    if (changed) {
      setContourVersion((v) => v + 1);
    }
  }, [contourFetched]);

  const contourToastedRef = useRef(false);
  useEffect(() => {
    if (!isNotesMode) {
      contourToastedRef.current = false;
      return;
    }
    if (contourFetchError && !contourToastedRef.current) {
      contourToastedRef.current = true;
      toast({
        title: "Contours unavailable",
        description: "Some notes are showing flat — contour fetch failed.",
        variant: "destructive",
      });
    }
  }, [contourFetchError, isNotesMode]);

  const visibleContours = useMemo(() => {
    const out = new Map<string, PianoRollNoteContourFrame[]>();
    for (const note of visibleNotes) {
      if (!note.note_uid) continue;
      const rows = contourCacheRef.current.get(note.note_uid);
      if (rows) {
        out.set(note.note_uid, rows);
      }
    }
    // Tie the memo to ``contourVersion`` so newly-arrived rows
    // re-render even though the Map identity is stable.
    void contourVersion;
    return out;
  }, [contourVersion, visibleNotes]);

  const hoveredNote =
    hoveredNoteIndex == null ? null : visibleNotes[hoveredNoteIndex] ?? null;

  const findNoteAtPoint = useCallback(
    (x: number, y: number) => {
      const transform = makeTransform();
      const rowHeight =
        (transform.plotBottom - transform.plotTop) / MIDI_PITCH_COUNT;
      // Polyline ribbons can swing beyond the row's vertical extent, so
      // ribbon notes use distance-to-polyline; flat-bar notes (no
      // contour fetched) keep the row-height bounding box. ADR-069 §9.3
      // specifies ≤ 6 px hit-test radius in canvas space.
      const hitRadius = 6;
      const hitRadiusSq = hitRadius * hitRadius;
      for (let i = visibleNotes.length - 1; i >= 0; i -= 1) {
        const note = visibleNotes[i];
        const left = transform.timeToX(note.start_utc);
        const right = transform.timeToX(note.start_utc + note.duration_s);
        if (x < left - hitRadius || x > right + hitRadius) continue;
        const contour = note.note_uid
          ? visibleContours.get(note.note_uid)
          : undefined;
        if (contour && contour.length >= 2) {
          let minDistSq = Number.POSITIVE_INFINITY;
          let prevX = transform.timeToX(note.start_utc + contour[0].time_offset_s);
          let prevY = midiPitchToY(
            note.midi_pitch + contour[0].cents_from_pitch / 100,
            transform.plotTop,
            transform.plotBottom,
          );
          for (let k = 1; k < contour.length; k += 1) {
            const px = transform.timeToX(note.start_utc + contour[k].time_offset_s);
            const py = midiPitchToY(
              note.midi_pitch + contour[k].cents_from_pitch / 100,
              transform.plotTop,
              transform.plotBottom,
            );
            const dx = px - prevX;
            const dy = py - prevY;
            const lenSq = dx * dx + dy * dy;
            let t = 0;
            if (lenSq > 0) {
              t = Math.max(
                0,
                Math.min(1, ((x - prevX) * dx + (y - prevY) * dy) / lenSq),
              );
            }
            const cx = prevX + t * dx;
            const cy = prevY + t * dy;
            const ex = x - cx;
            const ey = y - cy;
            const distSq = ex * ex + ey * ey;
            if (distSq < minDistSq) minDistSq = distSq;
            prevX = px;
            prevY = py;
          }
          if (minDistSq <= hitRadiusSq) {
            return { note, index: i };
          }
          continue;
        }
        const yCenter = midiPitchToY(
          note.midi_pitch,
          transform.plotTop,
          transform.plotBottom,
        );
        if (
          x >= left &&
          x <= right &&
          y >= yCenter - rowHeight / 2 &&
          y <= yCenter + rowHeight / 2
        ) {
          return { note, index: i };
        }
      }
      return null;
    },
    [makeTransform, visibleContours, visibleNotes],
  );

  useEffect(() => {
    if (isNotesMode) {
      drawNotesCanvas({
        canvas: canvasRef.current,
        size,
        notes: visibleNotes,
        contours: visibleContours,
        timeRange,
        selectedK: timeline.selected_k,
        playheadTime: visiblePlayheadTime,
        jobStart: timeline.job_start_timestamp,
        hoveredNoteIndex,
        tokenFilter,
      });
      return;
    }
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
    hoveredNoteIndex,
    isNotesMode,
    selectedEventId,
    size,
    timeRange,
    timeline,
    tokenFilter,
    unvoicedMode,
    visibleContours,
    visibleNotes,
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

    if (isNotesMode) {
      const hit = findNoteAtPoint(point.x, point.y);
      setHoveredNoteIndex(hit?.index ?? null);
      setHoveredEventId(null);
      return;
    }

    const hovered = findEventAtPoint(point.x, point.y);
    setHoveredEventId(hovered?.event_id ?? null);
    setHoveredNoteIndex(null);
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
    if (isNotesMode) {
      const hit = findNoteAtPoint(point.x, point.y);
      setSelectedEventId(hit?.note.event_id ?? null);
      return;
    }
    const clicked = findEventAtPoint(point.x, point.y);
    setSelectedEventId(clicked?.event_id ?? null);
  };

  const handleMouseLeave = () => {
    dragRef.current = null;
    setIsDraggingTimeline(false);
    setCursor(null);
    setHoveredEventId(null);
    setHoveredNoteIndex(null);
  };

  const handleDoubleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const point = canvasPoint(event);
    if (isNotesMode) {
      const hit = findNoteAtPoint(point.x, point.y);
      if (!hit) return;
      const note = hit.note;
      setSelectedEventId(note.event_id);
      const duration = Math.max(MIN_TIME_SPAN_SECONDS, note.duration_s);
      const padding = Math.max(1, duration * 2);
      setTimeRange(
        clampTimeRange(
          {
            start: note.start_utc - padding,
            end: note.start_utc + note.duration_s + padding,
          },
          fullTimeRange,
        ),
      );
      return;
    }
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
          <PianoRollMidiExportControls
            jobId={timeline.job_id}
            windowStartUtc={timeRange.start}
            windowEndUtc={timeRange.end}
          />
          <NotesStatusControls jobId={timeline.job_id} />
          <ToolbarSelect
            label="Y"
            value={viewMode}
            testId="eej-piano-roll-y-mode"
            onChange={(value) => setViewModeFromUser(value as ViewMode)}
          >
            <option
              value="notes"
              disabled={!notesAvailable}
              title={
                notesAvailable
                  ? "MIDI notes from the piano roll notes sidecar"
                  : "Notes mode requires a completed piano roll notes job"
              }
            >
              Notes{notesAvailable ? "" : " (unavailable)"}
            </option>
            <option value="ridge">Ridge</option>
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
            size="icon"
            className="h-8 w-8 border-zinc-700 bg-zinc-900 text-zinc-100 hover:bg-zinc-800"
            onClick={() => setSpectrogramCollapsed((value) => !value)}
            title={
              spectrogramCollapsed
                ? "Show spectrogram strip"
                : "Hide spectrogram strip"
            }
            aria-label={
              spectrogramCollapsed
                ? "Show spectrogram strip"
                : "Hide spectrogram strip"
            }
            aria-pressed={!spectrogramCollapsed}
            data-testid="eej-piano-roll-spectrogram-toggle"
          >
            {spectrogramCollapsed ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronUp className="h-4 w-4" />
            )}
          </Button>
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
            isDraggingTimeline
              ? "dragging"
              : isNotesMode
                ? hoveredNoteIndex != null
                  ? "hover-note"
                  : "idle"
                : hoveredEventId
                  ? "hover-token"
                  : "idle"
          }
          data-view-mode={viewMode}
          data-notes-mode={isNotesMode ? "true" : "false"}
          data-notes-count={isNotesMode ? String(visibleNotes.length) : ""}
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
        {isNotesMode ? (
          hoveredNote && cursor ? (
            <NoteTooltip
              canvasSize={size}
              contour={
                hoveredNote.note_uid
                  ? visibleContours.get(hoveredNote.note_uid)
                  : undefined
              }
              cursor={cursor}
              note={hoveredNote}
              selectedK={selectedK}
            />
          ) : null
        ) : tooltipEvent && cursor ? (
          <EventTooltip
            canvasSize={size}
            cursor={cursor}
            event={tooltipEvent}
            position={timeline.events.findIndex(
              (event) => event.event_id === tooltipEvent.event_id,
            )}
            selectedK={selectedK}
            yMode={yMode}
          />
        ) : null}
        {isNotesMode &&
        notesQueryFetched &&
        notesData &&
        visibleNotes.length === 0 ? (
          <div
            className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded border border-zinc-700 bg-zinc-900/80 px-3 py-1 text-xs text-zinc-300"
            data-testid="eej-piano-roll-notes-empty"
          >
            No notes in this viewport
          </div>
        ) : null}
      </div>

      {!spectrogramCollapsed ? (
        <EventEncoderSpectrogramStrip
          timeline={timeline}
          timeRange={timeRange}
          frequencyRange={frequencyRange}
          playheadTime={visiblePlayheadTime}
          plotLeftPx={LEFT_MARGIN}
          plotRightPx={RIGHT_MARGIN}
          onTimeRangeChange={setClampedTimeRange}
          onZoomTime={zoomTime}
          onZoomFrequency={zoomFrequency}
        />
      ) : null}

      <div
        className="flex h-7 flex-shrink-0 items-center gap-4 border-t border-zinc-800 bg-zinc-950 px-3 font-mono text-[11px] text-zinc-400"
        data-testid="eej-piano-roll-status"
      >
        <span data-testid="eej-piano-roll-cursor-time">
          {cursor ? formatRelativeTime(cursor.time, timeline.job_start_timestamp) : "time -"}
        </span>
        <span data-testid="eej-piano-roll-cursor-frequency">
          {cursor
            ? isNotesMode
              ? formatCursorMidi(cursor.y, size.height)
              : `${Math.round(cursor.frequency)} Hz`
            : "freq -"}
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

function PianoRollMidiExportControls({
  jobId,
  windowStartUtc,
  windowEndUtc,
}: {
  jobId: string;
  windowStartUtc: number;
  windowEndUtc: number;
}) {
  const { data: notesStatus } = usePianoRollNotesStatus(jobId);
  return (
    <MidiExportButton
      jobId={jobId}
      notesStatus={notesStatus}
      windowStartUtc={windowStartUtc}
      windowEndUtc={windowEndUtc}
    />
  );
}

function NotesStatusControls({ jobId }: { jobId: string }) {
  const { data: status } = usePianoRollNotesStatus(jobId);
  const mutation = useCreatePianoRollNotesJob(jobId);
  const [errorOpen, setErrorOpen] = useState(false);
  if (!status) return null;
  const value = status.status;
  const isActive = value === "queued" || value === "running";
  const showGenerate = value === "absent" || value === "failed";
  const buttonDisabled = isActive || mutation.isPending;
  const buttonLabel =
    value === "failed" ? "Re-run" : isActive ? "Generating…" : "Generate notes";
  const errorMessage =
    status.status === "failed" ? status.error_message : null;

  return (
    <div className="flex items-center gap-2" data-testid="eej-piano-roll-notes-controls">
      <button
        type="button"
        className="cursor-pointer border-none bg-transparent p-0"
        onClick={() => {
          if (value === "failed") setErrorOpen((open) => !open);
        }}
        aria-label={
          value === "failed"
            ? "Show piano roll notes error details"
            : "Piano roll notes status"
        }
        data-testid="eej-piano-roll-notes-pill-button"
      >
        <PianoRollNotesStatusPill
          status={status}
          onRequestV3Upgrade={async () => {
            try {
              await mutation.mutateAsync({ extractor_version: "v3" });
            } catch (err) {
              // The displayed status is the latest *complete* row, so the
              // v3-available badge can render even when a v3 row is
              // already queued or running. The backend then returns 409;
              // surface the in-flight state with a non-blocking toast
              // instead of letting the global error path raise.
              if (err instanceof ApiError && err.status === 409) {
                toast({
                  title: "Already enqueued",
                  description: "A v3 notes job is already queued or running for this encoder.",
                });
                return;
              }
              throw err;
            }
          }}
        />
      </button>
      {showGenerate ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="h-8 border-zinc-700 bg-zinc-900 px-2 text-zinc-100 hover:bg-zinc-800"
          disabled={buttonDisabled}
          onClick={() => mutation.mutate({})}
          data-testid="eej-piano-roll-notes-generate"
        >
          {buttonLabel}
        </Button>
      ) : isActive ? (
        <span
          className="inline-flex h-8 items-center rounded border border-zinc-700 bg-zinc-900 px-2 text-xs text-zinc-300"
          data-testid="eej-piano-roll-notes-progress"
        >
          {buttonLabel}
        </span>
      ) : null}
      {errorMessage && errorOpen ? (
        <span
          className="max-w-xs truncate rounded border border-red-700 bg-red-950 px-2 py-1 text-[11px] text-red-200"
          title={errorMessage}
          data-testid="eej-piano-roll-notes-error"
        >
          {errorMessage}
        </span>
      ) : null}
    </div>
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
  yMode,
}: {
  canvasSize: Size;
  cursor: CursorInfo;
  event: EventEncoderTimelineEvent;
  position: number;
  selectedK: number;
  yMode: YMode;
}) {
  const values = event.descriptor_values;
  const color = labelColor(event.token_id, Math.max(1, selectedK));
  const displayBand = resolveEventDisplayBand(event, yMode);
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
      <TooltipRow
        label="display_band"
        value={`${formatHz(displayBand.lowFrequency)} - ${formatHz(displayBand.highFrequency)}`}
      />
      {numeric(values.ridge_median_frequency) != null ? (
        <>
          <TooltipRow label="ridge_mid" value={formatHz(values.ridge_median_frequency)} />
          <TooltipRow
            label="ridge_band"
            value={`${formatHz(values.ridge_low_frequency)} - ${formatHz(values.ridge_high_frequency)}`}
          />
          <TooltipRow label="ridge_cov" value={formatRatio(values.ridge_coverage)} />
          <TooltipRow label="ridge_energy" value={formatRatio(values.ridge_energy_ratio)} />
          <TooltipRow label="band_peak" value={formatHz(values.band_limited_peak_frequency)} />
          <TooltipRow label="centroid" value={formatHz(values.spectral_centroid)} />
          <TooltipRow label="bandwidth" value={formatHz(values.bandwidth)} />
          <TooltipRow label="high_band" value={formatRatio(values.high_band_energy_ratio)} />
        </>
      ) : null}
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

function drawNotesCanvas({
  canvas,
  size,
  notes,
  contours,
  timeRange,
  selectedK,
  playheadTime,
  jobStart,
  hoveredNoteIndex,
  tokenFilter,
}: {
  canvas: HTMLCanvasElement | null;
  size: Size;
  notes: PianoRollNote[];
  contours: Map<string, PianoRollNoteContourFrame[]>;
  timeRange: TimeRange;
  selectedK: number;
  playheadTime: number | null;
  jobStart: number;
  hoveredNoteIndex: number | null;
  tokenFilter: number | null;
}) {
  if (!canvas || size.width <= 0 || size.height <= 0) return;
  const ctx = prepareCanvas(canvas, size);
  const transform = createTransform({
    size,
    timeRange,
    frequencyRange: { min: 0, max: 1 },
    unvoicedMode: "peak",
  });

  ctx.clearRect(0, 0, size.width, size.height);
  ctx.fillStyle = "#09090b";
  ctx.fillRect(0, 0, size.width, size.height);

  drawNotesGrid(ctx, transform, jobStart);

  const rowHeight =
    (transform.plotBottom - transform.plotTop) / MIDI_PITCH_COUNT;
  for (let i = 0; i < notes.length; i += 1) {
    const note = notes[i];
    if (tokenFilter != null && note.event_token !== tokenFilter) continue;
    const x1 = transform.timeToX(note.start_utc);
    const x2 = transform.timeToX(note.start_utc + note.duration_s);
    if (x2 < transform.plotLeft || x1 > transform.plotRight) continue;
    const color = labelColor(
      Math.max(0, note.event_token),
      Math.max(1, selectedK),
    );
    const contour = note.note_uid ? contours.get(note.note_uid) : undefined;
    const isHovered = i === hoveredNoteIndex;
    if (contour && contour.length > 1) {
      drawNoteRibbon({
        ctx,
        note,
        contour,
        transform,
        tokenColor: color,
        isHovered,
      });
      continue;
    }

    const width = Math.max(2, x2 - x1);
    const yCenter = midiPitchToY(
      note.midi_pitch,
      transform.plotTop,
      transform.plotBottom,
    );
    const height = Math.max(2, rowHeight - 1);
    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(x1, yCenter - height / 2, width, height);
    ctx.globalAlpha = Math.max(0.2, Math.min(1, note.velocity / 127));
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(x1, yCenter - height / 2, width, height);
    ctx.restore();

    if (isHovered) {
      ctx.save();
      ctx.strokeStyle = "#fafafa";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(
        x1 - 1,
        yCenter - height / 2 - 1,
        width + 2,
        height + 2,
      );
      ctx.restore();
    }
  }

  if (playheadTime != null) {
    drawPlayhead(ctx, transform, playheadTime, jobStart);
  }
}


function drawNoteRibbon({
  ctx,
  note,
  contour,
  transform,
  tokenColor,
  isHovered,
}: {
  ctx: CanvasRenderingContext2D;
  note: PianoRollNote;
  contour: PianoRollNoteContourFrame[];
  transform: ReturnType<typeof createTransform>;
  tokenColor: string;
  isHovered: boolean;
}) {
  if (contour.length < 2) return;
  const baseY = midiPitchToY(
    note.midi_pitch,
    transform.plotTop,
    transform.plotBottom,
  );
  const semitoneHeight =
    (transform.plotBottom - transform.plotTop) / MIDI_PITCH_COUNT;
  const points: Array<[number, number]> = [];
  for (const frame of contour) {
    const t = note.start_utc + frame.time_offset_s;
    const x = transform.timeToX(t);
    const semitoneOffset = frame.cents_from_pitch / 100;
    const y = baseY - semitoneOffset * semitoneHeight;
    points.push([x, y]);
  }

  ctx.save();
  // Filled ribbon body — 4 px vertical band centred on the polyline.
  ctx.fillStyle = tokenColor;
  ctx.globalAlpha = 0.3;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1] - 2);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i][0], points[i][1] - 2);
  }
  for (let i = points.length - 1; i >= 0; i -= 1) {
    ctx.lineTo(points[i][0], points[i][1] + 2);
  }
  ctx.closePath();
  ctx.fill();

  // Stroked polyline — opacity tracks velocity for visual emphasis.
  ctx.globalAlpha = Math.max(0.25, Math.min(1, note.velocity / 127));
  ctx.strokeStyle = tokenColor;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.stroke();

  if (isHovered) {
    ctx.strokeStyle = "#fafafa";
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawNotesGrid(
  ctx: CanvasRenderingContext2D,
  transform: ReturnType<typeof createTransform>,
  jobStart: number,
) {
  ctx.save();
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";

  const rowHeight =
    (transform.plotBottom - transform.plotTop) / MIDI_PITCH_COUNT;

  // Tint the extended bands outside the 88-key piano range so the
  // out-of-piano content is visible without overpowering the
  // black/white-key shading that anchors callers used to MIDI 21..108.
  for (let midi = MIDI_MIN_PITCH; midi <= MIDI_MAX_PITCH; midi += 1) {
    const yCenter = midiPitchToY(midi, transform.plotTop, transform.plotBottom);
    const yTop = yCenter - rowHeight / 2;
    if (!isInPianoBand(midi)) {
      ctx.fillStyle = "#1f1d24";
      ctx.fillRect(
        transform.plotLeft,
        yTop,
        transform.plotRight - transform.plotLeft,
        rowHeight,
      );
    } else if (isBlackKey(midi)) {
      ctx.fillStyle = "#18181b";
      ctx.fillRect(transform.plotLeft, yTop, transform.plotRight - transform.plotLeft, rowHeight);
    }
  }

  ctx.strokeStyle = "#27272a";
  ctx.lineWidth = 1;
  for (let midi = MIDI_MIN_PITCH; midi <= MIDI_MAX_PITCH; midi += 1) {
    const yCenter = midiPitchToY(midi, transform.plotTop, transform.plotBottom);
    const yTop = yCenter - rowHeight / 2;
    ctx.globalAlpha = midi % 12 === 0 ? 0.85 : 0.35;
    ctx.beginPath();
    ctx.moveTo(transform.plotLeft, yTop);
    ctx.lineTo(transform.plotRight, yTop);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  ctx.fillStyle = "#a1a1aa";
  // Label every C from C0 (MIDI 12) up through C9 (MIDI 120). Add G9
  // explicitly so the topmost extended band remains anchored.
  const cMidis: number[] = [];
  for (let midi = 12; midi <= MIDI_MAX_PITCH; midi += 12) {
    cMidis.push(midi);
  }
  if (!cMidis.includes(127) && MIDI_MAX_PITCH >= 127) {
    cMidis.push(127);
  }
  for (const midi of cMidis) {
    if (midi < MIDI_MIN_PITCH || midi > MIDI_MAX_PITCH) continue;
    const yCenter = midiPitchToY(midi, transform.plotTop, transform.plotBottom);
    ctx.fillText(midiNoteName(midi), transform.plotLeft - 8, yCenter);
  }
  // G9 (MIDI 127) is the conventional MIDI top — label it for the spec.
  if (MIDI_MAX_PITCH >= 119) {
    const yG9 = midiPitchToY(119, transform.plotTop, transform.plotBottom);
    ctx.fillText("G9", transform.plotLeft - 8, yG9);
  }

  const timeStep = chooseTimeStep(transform.timeRange.end - transform.timeRange.start);
  const startOffset =
    Math.max(
      0,
      Math.ceil((transform.timeRange.start - jobStart) / timeStep) * timeStep,
    );
  for (
    let offset = startOffset;
    jobStart + offset <= transform.timeRange.end;
    offset += timeStep
  ) {
    const time = jobStart + offset;
    const x = transform.timeToX(time);
    if (x < transform.plotLeft || x > transform.plotRight) continue;
    ctx.globalAlpha = 0.4;
    ctx.beginPath();
    ctx.moveTo(x, transform.plotTop);
    ctx.lineTo(x, transform.plotBottom);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  ctx.restore();
}

function formatCursorMidi(y: number, canvasHeight: number) {
  const plotTop = TOP_MARGIN;
  const plotBottom = Math.max(plotTop + 1, canvasHeight - BOTTOM_MARGIN);
  const midi = midiPitchAtY(y, plotTop, plotBottom);
  if (!Number.isFinite(midi)) return "midi -";
  const rounded = Math.round(midi);
  if (rounded < MIDI_MIN_PITCH || rounded > MIDI_MAX_PITCH) return "midi -";
  return `MIDI ${rounded} (${midiNoteName(rounded)})`;
}

function NoteTooltip({
  canvasSize,
  contour,
  cursor,
  note,
  selectedK,
}: {
  canvasSize: Size;
  contour: PianoRollNoteContourFrame[] | undefined;
  cursor: CursorInfo;
  note: PianoRollNote;
  selectedK: number;
}) {
  const color = labelColor(
    Math.max(0, note.event_token),
    Math.max(1, selectedK),
  );
  const position = placeTooltip(cursor, canvasSize);
  return (
    <div
      className="pointer-events-none absolute z-20 w-64 rounded border border-zinc-700 bg-zinc-950/95 p-2 text-[11px] leading-5 text-zinc-300 shadow-xl"
      style={{
        left: position.left,
        maxHeight: position.maxHeight,
        overflowY: "auto",
        top: position.top,
      }}
      data-testid="eej-piano-roll-note-tooltip"
    >
      <div className="mb-1 flex items-center gap-2">
        <span
          className="rounded px-1.5 py-0.5 font-mono font-bold text-white"
          style={{ backgroundColor: color }}
          data-testid="eej-piano-roll-note-tooltip-color"
        >
          MIDI {note.midi_pitch} ({midiNoteName(note.midi_pitch)})
        </span>
      </div>
      <TooltipRow label="velocity" value={String(note.velocity)} />
      <TooltipRow
        label="duration"
        value={`${note.duration_s.toFixed(2)} s`}
      />
      <TooltipRow label="event_id" value={note.event_id} />
      <TooltipRow label="token" value={String(note.event_token)} />
      <TooltipRow label="partial" value={partialIndexLabel(note.partial_index)} />
      {contour && contour.length > 0 ? (
        <TooltipRow
          label="Δpitch"
          value={formatDeltaPitchCents(contour)}
        />
      ) : null}
    </div>
  );
}

function formatDeltaPitchCents(
  contour: PianoRollNoteContourFrame[],
): string {
  let maxAbs = 0;
  for (const row of contour) {
    const abs = Math.abs(row.cents_from_pitch);
    if (abs > maxAbs) maxAbs = abs;
  }
  return `±${maxAbs.toFixed(0)}¢`;
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
  const displayBand = resolveEventDisplayBand(event, yMode);
  const voiced = displayBand.ridgeTrusted || displayBand.voiced;
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

  const centerFrequency = displayBand.centerFrequency;
  const yTop = transform.frequencyToY(displayBand.highFrequency);
  const yBottom = transform.frequencyToY(displayBand.lowFrequency);
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
    if (isEventVoiced(event) && medianF0 != null) {
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

function defaultFrequencyRange(
  timeline: EventEncoderTimelineResponse,
): FrequencyRange {
  return {
    min: 0,
    max: hasRidgeFrequencyDescriptors(timeline.events)
      ? DEFAULT_RIDGE_FREQUENCY_MAX
      : DEFAULT_FREQUENCY_MAX,
  };
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
