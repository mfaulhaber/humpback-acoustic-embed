import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";

import {
  type EventEncoderJob,
  type EventEncoderTimelineEvent,
  type EventEncoderTimelineResponse,
  useEventEncoderTimeline,
} from "@/api/sequenceModels";
import { regionAudioSliceUrl, regionTileUrl } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TimelineProvider } from "@/components/timeline/provider/TimelineProvider";
import { REVIEW_ZOOM } from "@/components/timeline/provider/types";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import { Spectrogram } from "@/components/timeline/spectrogram/Spectrogram";
import { ZoomSelector } from "@/components/timeline/controls/ZoomSelector";

import { labelColor } from "./constants";
import { EventEncoderTokenOverlay } from "./EventEncoderTokenOverlay";

interface EventEncoderTimelinePanelProps {
  job: EventEncoderJob;
}

export function EventEncoderTimelinePanel({ job }: EventEncoderTimelinePanelProps) {
  const isComplete = job.status === "complete";
  const [selectedK, setSelectedK] = useState<number | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [selectedListIndex, setSelectedListIndex] = useState(0);
  const {
    data,
    isLoading,
    error,
  } = useEventEncoderTimeline(job.id, selectedK, isComplete);

  const events = data?.events ?? [];

  useEffect(() => {
    if (!data || selectedK !== null) return;
    setSelectedK(data.selected_k);
  }, [data, selectedK]);

  useEffect(() => {
    if (!data) return;
    if (!events.length) {
      setSelectedEventId(null);
      setSelectedListIndex(0);
      return;
    }
    const currentIndex = events.findIndex((event) => event.event_id === selectedEventId);
    if (currentIndex >= 0) {
      setSelectedListIndex(currentIndex);
      return;
    }
    const nextIndex = Math.min(selectedListIndex, events.length - 1);
    setSelectedEventId(events[nextIndex].event_id);
    setSelectedListIndex(nextIndex);
  }, [data, events, selectedEventId, selectedListIndex]);

  const handleKChange = (value: string) => {
    const nextK = Number(value);
    if (Number.isFinite(nextK)) {
      setSelectedK(nextK);
    }
  };

  return (
    <Card data-testid="eej-timeline-panel">
      <CardHeader>
        <CardTitle>Timeline</CardTitle>
      </CardHeader>
      <CardContent>
        {!isComplete ? (
          <TimelineMessage
            testId="eej-timeline-unavailable"
            message="Timeline available after tokenization completes."
          />
        ) : isLoading ? (
          <TimelineMessage testId="eej-timeline-loading" message="Loading timeline..." />
        ) : error ? (
          <TimelineMessage
            testId="eej-timeline-error"
            message="Timeline artifact is unavailable."
          />
        ) : data == null || data.events.length === 0 ? (
          <TimelineMessage
            testId="eej-timeline-empty"
            message="No tokenized events are available."
          />
        ) : data.job_end_timestamp <= data.job_start_timestamp ? (
          <TimelineMessage
            testId="eej-timeline-error"
            message="Timeline source bounds are unavailable."
          />
        ) : (
          <TimelineProvider
            key={`${job.id}-${data.region_detection_job_id}`}
            jobStart={data.job_start_timestamp}
            jobEnd={data.job_end_timestamp}
            zoomLevels={REVIEW_ZOOM}
            defaultZoom="30s"
            playback="slice"
            audioUrlBuilder={(startEpoch, durationSec) =>
              regionAudioSliceUrl(
                data.region_detection_job_id,
                startEpoch,
                durationSec,
              )
            }
            disableKeyboardShortcuts
            scrollOnPlayback={false}
          >
            <EventEncoderTimelineBody
              timeline={data}
              selectedEventId={selectedEventId}
              selectedListIndex={selectedListIndex}
              onSelectEvent={(eventId, index) => {
                setSelectedEventId(eventId);
                setSelectedListIndex(index);
              }}
              onSelectIndex={(index) => {
                const next = data.events[index];
                if (!next) return;
                setSelectedEventId(next.event_id);
                setSelectedListIndex(index);
              }}
              onKChange={handleKChange}
            />
          </TimelineProvider>
        )}
      </CardContent>
    </Card>
  );
}

function TimelineMessage({
  message,
  testId,
}: {
  message: string;
  testId: string;
}) {
  return (
    <div
      className="rounded-md border border-dashed p-4 text-sm text-muted-foreground"
      data-testid={testId}
    >
      {message}
    </div>
  );
}

function EventEncoderTimelineBody({
  timeline,
  selectedEventId,
  selectedListIndex,
  onSelectEvent,
  onSelectIndex,
  onKChange,
}: {
  timeline: EventEncoderTimelineResponse;
  selectedEventId: string | null;
  selectedListIndex: number;
  onSelectEvent: (eventId: string, index: number) => void;
  onSelectIndex: (index: number) => void;
  onKChange: (value: string) => void;
}) {
  const ctx = useTimelineContext();
  const events = timeline.events;
  const selectedIndex = events.findIndex(
    (event) => event.event_id === selectedEventId,
  );
  const selectedEvent = selectedIndex >= 0 ? events[selectedIndex] : null;
  const effectiveIndex = selectedIndex >= 0 ? selectedIndex : selectedListIndex;
  const selectedTokenColor = selectedEvent
    ? labelColor(selectedEvent.token_id, Math.max(timeline.selected_k, 1))
    : undefined;
  const lastCenteredRef = useRef<string | null>(null);

  const centerEvent = useCallback(
    (event: EventEncoderTimelineEvent) => {
      ctx.seekTo((event.start_timestamp + event.end_timestamp) / 2);
    },
    [ctx],
  );

  const selectIndex = useCallback(
    (index: number) => {
      const bounded = Math.max(0, Math.min(events.length - 1, index));
      const event = events[bounded];
      if (!event) return;
      onSelectIndex(bounded);
      centerEvent(event);
    },
    [centerEvent, events, onSelectIndex],
  );

  const toggleSelectedPlayback = useCallback(() => {
    if (ctx.isPlaying) {
      ctx.pause();
      return;
    }
    if (selectedEvent) {
      ctx.play(
        selectedEvent.start_timestamp,
        Math.max(0.1, selectedEvent.end_timestamp - selectedEvent.start_timestamp),
      );
      return;
    }
    ctx.play(ctx.viewStart, Math.min(ctx.viewportSpan, 30));
  }, [ctx, selectedEvent]);

  useEffect(() => {
    if (!selectedEvent || lastCenteredRef.current === selectedEvent.event_id) return;
    lastCenteredRef.current = selectedEvent.event_id;
    centerEvent(selectedEvent);
  }, [centerEvent, selectedEvent]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isTextEntryTarget(e.target)) return;

      switch (e.code) {
        case "KeyA":
          e.preventDefault();
          selectIndex(effectiveIndex - 1);
          break;
        case "KeyD":
          e.preventDefault();
          selectIndex(effectiveIndex + 1);
          break;
        case "Space":
          e.preventDefault();
          toggleSelectedPlayback();
          break;
        case "ArrowLeft":
          e.preventDefault();
          ctx.pan(ctx.centerTimestamp - ctx.viewportSpan * 0.1);
          break;
        case "ArrowRight":
          e.preventDefault();
          ctx.pan(ctx.centerTimestamp + ctx.viewportSpan * 0.1);
          break;
      }

      if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        ctx.zoomIn();
      } else if (e.key === "-") {
        e.preventDefault();
        ctx.zoomOut();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [ctx, effectiveIndex, selectIndex, toggleSelectedPlayback]);

  const tileUrlBuilder = useCallback(
    (_jobId: string, zoomLevel: string, tileIndex: number) =>
      regionTileUrl(timeline.region_detection_job_id, zoomLevel, tileIndex),
    [timeline.region_detection_job_id],
  );

  return (
    <div className="rounded-md border" data-testid="eej-timeline-viewer">
      <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2 text-sm">
        <Button
          type="button"
          variant="outline"
          size="icon"
          className="h-8 w-8"
          disabled={effectiveIndex <= 0}
          onClick={() => selectIndex(effectiveIndex - 1)}
          title="Previous event"
          data-testid="eej-event-prev"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <Button
          type="button"
          variant="outline"
          size="icon"
          className="h-8 w-8"
          disabled={effectiveIndex >= events.length - 1}
          onClick={() => selectIndex(effectiveIndex + 1)}
          title="Next event"
          data-testid="eej-event-next"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
        <span className="min-w-[6.5rem] text-xs text-muted-foreground" data-testid="eej-event-counter">
          Event {events.length ? effectiveIndex + 1 : 0} / {events.length}
        </span>
        {selectedEvent ? (
          <Badge
            variant="outline"
            className="font-mono"
            style={{
              borderColor: selectedTokenColor,
              color: selectedTokenColor,
            }}
            data-testid="eej-selected-token"
          >
            {selectedEvent.token_label}
          </Badge>
        ) : null}
        {selectedEvent ? (
          <span className="text-xs text-muted-foreground" data-testid="eej-selected-confidence">
            confidence {selectedEvent.token_confidence.toFixed(3)}
          </span>
        ) : null}
        {timeline.valid_k_values.length > 1 ? (
          <select
            className="ml-auto h-8 rounded-md border bg-background px-2 text-xs"
            value={String(timeline.selected_k)}
            onChange={(e) => onKChange(e.target.value)}
            data-testid="eej-k-select"
          >
            {timeline.valid_k_values.map((k) => (
              <option key={k} value={k}>
                k={k}
              </option>
            ))}
          </select>
        ) : null}
        <Button
          type="button"
          variant="outline"
          size="icon"
          className={timeline.valid_k_values.length > 1 ? "h-8 w-8" : "ml-auto h-8 w-8"}
          onClick={toggleSelectedPlayback}
          title={ctx.isPlaying ? "Pause" : "Play event"}
          data-testid="eej-event-play"
        >
          {ctx.isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </Button>
      </div>

      <div className="flex flex-col" style={{ height: 220 }}>
        <Spectrogram
          jobId={timeline.region_detection_job_id}
          tileUrlBuilder={tileUrlBuilder}
          freqRange={[0, 3000]}
        >
          <EventEncoderTokenOverlay
            events={events}
            selectedEventId={selectedEventId}
            selectedK={timeline.selected_k}
            onSelectEvent={(eventId) => {
              const index = events.findIndex((event) => event.event_id === eventId);
              onSelectEvent(eventId, Math.max(0, index));
            }}
          />
        </Spectrogram>
      </div>
      <div className="border-t border-border px-2 py-1">
        <ZoomSelector />
      </div>
    </div>
  );
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
