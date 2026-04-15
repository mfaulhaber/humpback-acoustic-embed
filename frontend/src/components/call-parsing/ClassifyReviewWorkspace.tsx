import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useClassificationJobs,
  useSegmentationJobs,
  useRegionDetectionJobs,
  useRegionJobRegions,
  useTypedEvents,
  useTypeCorrections,
  useUpsertTypeCorrections,
  useEventClassifierModels,
  useCreateClassifierTrainingJob,
  useCreateClassificationJob,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { regionAudioSliceUrl } from "@/api/client";
import type {
  EventClassificationJob,
  TypedEventRow,
} from "@/api/types";
import { toast } from "@/components/ui/use-toast";
import { Button } from "@/components/ui/button";
import {
  ChevronLeft,
  ChevronRight,
  Play,
  Square,
  Save,
  X,
  RotateCcw,
} from "lucide-react";
import { RegionSpectrogramViewer } from "./RegionSpectrogramViewer";
import { EventBarOverlay, type EffectiveEvent } from "./EventBarOverlay";
import { TypePalette } from "./TypePalette";
import {
  ClassifyDetailPanel,
  type AggregatedEvent,
} from "./ClassifyDetailPanel";

/** Aggregate typed event rows by event_id */
function aggregateEvents(
  rows: TypedEventRow[],
  corrections: Map<string, string | null>,
): AggregatedEvent[] {
  const byEvent = new Map<
    string,
    {
      eventId: string;
      regionId: string;
      startSec: number;
      endSec: number;
      scores: { type_name: string; score: number; above_threshold: boolean }[];
    }
  >();

  for (const r of rows) {
    let entry = byEvent.get(r.event_id);
    if (!entry) {
      entry = {
        eventId: r.event_id,
        regionId: r.region_id,
        startSec: r.start_sec,
        endSec: r.end_sec,
        scores: [],
      };
      byEvent.set(r.event_id, entry);
    }
    entry.scores.push({
      type_name: r.type_name,
      score: r.score,
      above_threshold: r.above_threshold,
    });
  }

  const result: AggregatedEvent[] = [];
  for (const entry of byEvent.values()) {
    const sortedScores = [...entry.scores].sort((a, b) => b.score - a.score);
    const best = sortedScores.find((s) => s.above_threshold);
    const correctedType = corrections.has(entry.eventId)
      ? corrections.get(entry.eventId) ?? null
      : undefined;

    result.push({
      eventId: entry.eventId,
      regionId: entry.regionId,
      startSec: entry.startSec,
      endSec: entry.endSec,
      predictedType: best?.type_name ?? null,
      predictedScore: best?.score ?? null,
      correctedType,
      allScores: sortedScores,
    });
  }

  return result.sort((a, b) => a.startSec - b.startSec);
}

export function ClassifyReviewWorkspace({
  initialJobId,
}: {
  initialJobId?: string;
}) {
  const { data: classifyJobs = [] } = useClassificationJobs(0);
  const { data: segJobs = [] } = useSegmentationJobs();
  const { data: regionJobs = [] } = useRegionDetectionJobs();
  const { data: hydrophones = [] } = useHydrophones();
  const { data: classifierModels = [], refetch: refetchModels } =
    useEventClassifierModels();

  const completeJobs = useMemo(
    () => classifyJobs.filter((j) => j.status === "complete"),
    [classifyJobs],
  );

  const [selectedJobId, setSelectedJobId] = useState<string | null>(
    initialJobId ?? null,
  );

  useEffect(() => {
    if (initialJobId && completeJobs.some((j) => j.id === initialJobId)) {
      setSelectedJobId(initialJobId);
    }
  }, [initialJobId, completeJobs]);

  const selectedJob =
    completeJobs.find((j) => j.id === selectedJobId) ?? null;

  // Trace job chain: classify → segment → region detection
  const segJob = useMemo(
    () =>
      selectedJob
        ? segJobs.find(
            (s) => s.id === selectedJob.event_segmentation_job_id,
          ) ?? null
        : null,
    [selectedJob, segJobs],
  );
  const regionDetectionJobId = segJob?.region_detection_job_id ?? null;

  const { data: regions = [] } = useRegionJobRegions(regionDetectionJobId);
  const { data: typedEventRows = [] } = useTypedEvents(selectedJobId);
  const { data: savedCorrections = [] } = useTypeCorrections(selectedJobId);

  // Pending corrections: Map<eventId, typeName | null>
  const [pendingCorrections, setPendingCorrections] = useState<
    Map<string, string | null>
  >(new Map());
  const [currentEventIndex, setCurrentEventIndex] = useState(0);
  const [viewStart, setViewStart] = useState<number | undefined>(undefined);
  const [viewSpan, setViewSpan] = useState(30);
  const [scrollToCenter, setScrollToCenter] = useState<number | undefined>(
    undefined,
  );

  // Audio playback
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackOriginSec, setPlaybackOriginSec] = useState(0);

  const isDirty = pendingCorrections.size > 0;

  // Reset state when job changes
  useEffect(() => {
    setPendingCorrections(new Map());
    setCurrentEventIndex(0);
    setActiveTrainingJobId(null);
    setTrainingStartedAt(null);
    setRetrainError(null);
  }, [selectedJobId]);

  // Build merged correction map: saved + pending (pending overrides saved)
  const mergedCorrections = useMemo(() => {
    const map = new Map<string, string | null>();
    for (const c of savedCorrections) {
      map.set(c.event_id, c.type_name);
    }
    for (const [eventId, typeName] of pendingCorrections) {
      map.set(eventId, typeName);
    }
    return map;
  }, [savedCorrections, pendingCorrections]);

  // Aggregated events
  const events = useMemo(
    () => aggregateEvents(typedEventRows, mergedCorrections),
    [typedEventRows, mergedCorrections],
  );

  const currentEvent = events[currentEventIndex] ?? null;

  // Derive the palette highlight from the current event's effective type
  const currentEventType: string | null = useMemo(() => {
    if (!currentEvent) return null;
    if (currentEvent.correctedType !== undefined) {
      // null correction = negative → "" in palette convention
      return currentEvent.correctedType === null ? "" : currentEvent.correctedType;
    }
    return currentEvent.predictedType;
  }, [currentEvent]);

  // Clicking a type in the palette applies it to the current event
  const handleSelectType = useCallback(
    (typeName: string | null) => {
      if (!currentEvent || typeName === null) return;
      const correctionValue = typeName === "" ? null : typeName;
      setPendingCorrections((prev) => {
        const next = new Map(prev);
        next.set(currentEvent.eventId, correctionValue);
        return next;
      });
    },
    [currentEvent],
  );

  // Current region for spectrogram
  const currentRegion = useMemo(
    () =>
      currentEvent
        ? regions.find((r) => r.region_id === currentEvent.regionId) ?? null
        : regions[0] ?? null,
    [currentEvent, regions],
  );

  // Build effective events for the spectrogram overlay (current region only)
  const regionEffectiveEvents: EffectiveEvent[] = useMemo(() => {
    if (!currentRegion) return [];
    return events
      .filter((e) => e.regionId === currentRegion.region_id)
      .map((e) => ({
        eventId: e.eventId,
        regionId: e.regionId,
        startSec: e.startSec,
        endSec: e.endSec,
        originalStartSec: e.startSec,
        originalEndSec: e.endSec,
        confidence: e.predictedScore ?? 0,
        correctionType:
          e.correctedType !== undefined ? ("adjust" as const) : null,
      }));
  }, [events, currentRegion]);

  // Scroll spectrogram only when the current event is not fully visible
  useEffect(() => {
    if (!currentEvent || viewStart === undefined) return;
    const viewEnd = viewStart + viewSpan;
    const pad = viewSpan * 0.1; // 10% padding
    const fullyVisible =
      currentEvent.startSec >= viewStart + pad &&
      currentEvent.endSec <= viewEnd - pad;
    if (!fullyVisible) {
      // Place event end near the right edge with padding
      setScrollToCenter(currentEvent.endSec + pad - viewSpan / 2);
    }
  }, [currentEvent, viewStart, viewSpan]);

  // Navigation
  const goPrev = useCallback(() => {
    setCurrentEventIndex((i) => Math.max(0, i - 1));
  }, []);
  const goNext = useCallback(() => {
    setCurrentEventIndex((i) => Math.min(events.length - 1, i + 1));
  }, [events.length]);

  // Mark as negative
  const markNegative = useCallback(() => {
    if (!currentEvent) return;
    setPendingCorrections((prev) => {
      const next = new Map(prev);
      next.set(currentEvent.eventId, null);
      return next;
    });
  }, [currentEvent]);

  // Playback
  const startPlayback = useCallback(
    (startSec: number, duration: number) => {
      const audio = audioRef.current;
      if (!audio || !regionDetectionJobId) return;
      audio.pause();
      audio.currentTime = 0;
      setPlaybackOriginSec(startSec);
      audio.src = regionAudioSliceUrl(regionDetectionJobId, startSec, duration);
      audio.play().catch(() => setIsPlaying(false));
      setIsPlaying(true);
    },
    [regionDetectionJobId],
  );

  const stopPlayback = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.pause();
    audio.currentTime = 0;
    setIsPlaying(false);
  }, []);

  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      stopPlayback();
    } else if (currentEvent) {
      const duration = currentEvent.endSec - currentEvent.startSec;
      startPlayback(currentEvent.startSec, duration);
    }
  }, [isPlaying, currentEvent, startPlayback, stopPlayback]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      const tag = el.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      switch (e.code) {
        case "ArrowLeft":
        case "BracketLeft":
        case "KeyA":
          e.preventDefault();
          goPrev();
          break;
        case "ArrowRight":
        case "BracketRight":
        case "KeyD":
          e.preventDefault();
          goNext();
          break;
        case "Space":
          e.preventDefault();
          togglePlayback();
          break;
        case "Backspace":
        case "Delete":
          e.preventDefault();
          markNegative();
          break;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [goPrev, goNext, togglePlayback, markNegative]);

  // Save
  const saveMutation = useUpsertTypeCorrections();

  const handleSave = useCallback(() => {
    if (!selectedJobId || pendingCorrections.size === 0) return;
    const corrections = Array.from(pendingCorrections.entries()).map(
      ([event_id, type_name]) => ({ event_id, type_name }),
    );
    saveMutation.mutate(
      { jobId: selectedJobId, corrections },
      {
        onSuccess: () => {
          setPendingCorrections(new Map());
          toast({
            title: "Corrections saved",
            description: `${corrections.length} correction${corrections.length !== 1 ? "s" : ""} saved.`,
          });
        },
      },
    );
  }, [selectedJobId, pendingCorrections, saveMutation]);

  const handleCancel = useCallback(() => {
    if (isDirty && !window.confirm("Discard unsaved corrections?")) return;
    setPendingCorrections(new Map());
  }, [isDirty]);

  // Warn before navigating away with unsaved corrections
  useEffect(() => {
    if (!isDirty) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isDirty]);

  // ---- Retrain ----
  const [activeTrainingJobId, setActiveTrainingJobId] = useState<string | null>(
    null,
  );
  const [trainingStartedAt, setTrainingStartedAt] = useState<Date | null>(null);
  const [retrainError, setRetrainError] = useState<string | null>(null);

  const createTraining = useCreateClassifierTrainingJob();
  const createClassifyJob = useCreateClassificationJob();

  const handleRetrain = useCallback(() => {
    if (!selectedJobId) return;
    const ok = window.confirm(
      "Train a new event classifier model from corrections on this job?",
    );
    if (!ok) return;
    setRetrainError(null);
    createTraining.mutate(
      { source_job_ids: [selectedJobId] },
      {
        onSuccess: (data) => {
          setActiveTrainingJobId(data.id);
          setTrainingStartedAt(new Date());
          toast({
            title: "Training job started",
            description: "The model will train in the background.",
          });
        },
        onError: (err) => {
          setRetrainError((err as Error).message);
          toast({
            title: "Failed to start training",
            description: (err as Error).message,
            variant: "destructive",
          });
        },
      },
    );
  }, [selectedJobId, createTraining]);

  // Poll models while training
  const isPolling = activeTrainingJobId !== null && retrainError === null;
  useEffect(() => {
    if (!isPolling) return;
    const interval = setInterval(() => void refetchModels(), 3000);
    return () => clearInterval(interval);
  }, [isPolling, refetchModels]);

  const newModelReady = useMemo(() => {
    if (!activeTrainingJobId || !trainingStartedAt) return null;
    // Find a model created after we started training
    return (
      classifierModels.find((m) => {
        return new Date(m.created_at) > trainingStartedAt;
      }) ?? null
    );
  }, [activeTrainingJobId, trainingStartedAt, classifierModels]);

  const handleReclassify = useCallback(() => {
    if (!segJob || !newModelReady) return;
    createClassifyJob.mutate(
      {
        event_segmentation_job_id: segJob.id,
        vocalization_model_id: newModelReady.id,
      },
      {
        onSuccess: () => {
          toast({
            title: "Classification job created",
            description:
              "Select it from the job list when complete.",
          });
        },
      },
    );
  }, [segJob, newModelReady, createClassifyJob]);

  // Label for job selector
  const jobLabel = useCallback(
    (job: EventClassificationJob) => {
      const shortId = job.id.slice(0, 8);
      const sj = segJobs.find(
        (s) => s.id === job.event_segmentation_job_id,
      );
      if (!sj) return `${shortId} — ${job.typed_event_count ?? 0} events`;
      const rj = regionJobs.find(
        (r) => r.id === sj.region_detection_job_id,
      );
      if (!rj) return `${shortId} — ${job.typed_event_count ?? 0} events`;
      const hName = rj.hydrophone_id
        ? (hydrophones.find((hp) => hp.id === rj.hydrophone_id)?.name ??
            rj.hydrophone_id)
        : "file";
      return `${hName} — ${shortId} — ${job.typed_event_count ?? 0} events`;
    },
    [segJobs, regionJobs, hydrophones],
  );

  const hasCorrections = savedCorrections.length > 0;

  return (
    <div className="space-y-4">
      {/* Job selector */}
      <div className="flex items-center gap-4">
        <label htmlFor="classify-review-job" className="text-sm font-medium">
          Job
        </label>
        <select
          id="classify-review-job"
          className="rounded-md border bg-background px-3 py-2 text-sm"
          value={selectedJobId ?? ""}
          onChange={(e) => setSelectedJobId(e.target.value || null)}
        >
          <option value="">Select a completed classification job</option>
          {completeJobs.map((j) => (
            <option key={j.id} value={j.id}>
              {jobLabel(j)}
            </option>
          ))}
        </select>
        {isDirty && (
          <span className="text-xs text-yellow-500">
            {pendingCorrections.size} unsaved change
            {pendingCorrections.size !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Workspace */}
      {selectedJob ? (
        <div className="rounded-md border">
          {/* Toolbar */}
          <div className="flex items-center justify-between px-4 py-2 border-b">
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={goPrev}
                disabled={currentEventIndex === 0}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-xs text-muted-foreground tabular-nums min-w-[80px] text-center">
                Event {events.length > 0 ? currentEventIndex + 1 : 0} of{" "}
                {events.length}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={goNext}
                disabled={currentEventIndex >= events.length - 1}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>

              <div className="w-px h-5 bg-border mx-1" />

              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={togglePlayback}
              >
                {isPlaying ? (
                  <Square className="h-3.5 w-3.5" />
                ) : (
                  <Play className="h-3.5 w-3.5" />
                )}
              </Button>
            </div>

            <div className="flex items-center gap-2">
              {hasCorrections && (
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={handleRetrain}
                  disabled={createTraining.isPending || isPolling}
                >
                  <RotateCcw className="h-3 w-3 mr-1" />
                  {isPolling ? "Training…" : "Retrain"}
                </Button>
              )}

              <div className="w-px h-5 bg-border" />

              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={handleCancel}
                disabled={!isDirty}
              >
                <X className="h-3 w-3 mr-1" />
                Cancel
              </Button>
              <Button
                size="sm"
                className="h-7 text-xs"
                onClick={handleSave}
                disabled={!isDirty || saveMutation.isPending}
              >
                <Save className="h-3 w-3 mr-1" />
                {saveMutation.isPending ? "Saving…" : "Save"}
              </Button>
            </div>
          </div>

          {/* Spectrogram */}
          {currentRegion && regionDetectionJobId ? (
            <RegionSpectrogramViewer
              regionJobId={regionDetectionJobId}
              region={currentRegion}
              onViewStartChange={setViewStart}
              onViewSpanChange={setViewSpan}
              scrollToCenter={scrollToCenter}
              audioRef={audioRef}
              isPlaying={isPlaying}
              playbackOriginSec={playbackOriginSec}
            >
              <EventBarOverlay
                events={regionEffectiveEvents}
                selectedEventId={currentEvent?.eventId ?? null}
                onSelectEvent={(eventId) => {
                  if (!eventId) return;
                  const idx = events.findIndex((e) => e.eventId === eventId);
                  if (idx >= 0) setCurrentEventIndex(idx);
                }}
                onAdjust={() => {}}
                onAdd={() => {}}
                addMode={false}
                activeRegionId={currentRegion.region_id}
              />
            </RegionSpectrogramViewer>
          ) : (
            <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground">
              No events to display
            </div>
          )}

          {/* Type palette */}
          <TypePalette activeType={currentEventType} onSelectType={handleSelectType} />

          {/* Detail panel */}
          <ClassifyDetailPanel event={currentEvent} />
        </div>
      ) : (
        <div className="py-8 text-center text-muted-foreground">
          Select a completed classification job to begin reviewing event types.
        </div>
      )}

      <audio
        ref={audioRef}
        onEnded={() => setIsPlaying(false)}
        style={{ display: "none" }}
      />
    </div>
  );
}
