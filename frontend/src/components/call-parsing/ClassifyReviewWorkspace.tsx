import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useClassificationJobs,
  useSegmentationJobs,
  useRegionDetectionJobs,
  useRegionJobRegions,
  useTypedEvents,
  useTypeCorrections,
  useUpsertTypeCorrections,
  useBoundaryCorrections,
  useSaveBoundaryCorrections,
  useEventClassifierModels,
  useCreateClassifierTrainingJob,
  useCreateClassificationJob,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { regionTileUrl, regionAudioSliceUrl } from "@/api/client";
import type {
  BoundaryCorrection,
  EventClassificationJob,
  Region,
  TypedEventRow,
} from "@/api/types";
import { toast } from "@/components/ui/use-toast";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  ChevronLeft,
  ChevronRight,
  Play,
  Square,
  Save,
  X,
  RotateCcw,
} from "lucide-react";
import { TimelineProvider } from "@/components/timeline/provider/TimelineProvider";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import { REVIEW_ZOOM } from "@/components/timeline/provider/types";
import { Spectrogram } from "@/components/timeline/spectrogram/Spectrogram";
import { RegionBoundaryMarkers } from "@/components/timeline/overlays/RegionBoundaryMarkers";
import { EventBarOverlay, type EffectiveEvent } from "@/components/timeline/overlays/EventBarOverlay";
import { ZoomSelector } from "@/components/timeline/controls/ZoomSelector";
import { TypePalette } from "./TypePalette";
import {
  ClassifyDetailPanel,
  type AggregatedEvent,
} from "./ClassifyDetailPanel";

/** Resolve the effective type + source for a single event given its
 *  predicted (inference) and corrected (human) types. Correction overrides
 *  prediction; a null correction means the human marked the event negative. */
function resolveEventType(
  predictedType: string | null,
  correctedType: string | null | undefined,
): {
  effectiveType: string | null;
  typeSource: "inference" | "correction" | "negative" | null;
} {
  if (correctedType === null) {
    return { effectiveType: null, typeSource: "negative" };
  }
  if (typeof correctedType === "string") {
    return { effectiveType: correctedType, typeSource: "correction" };
  }
  if (typeof predictedType === "string") {
    return { effectiveType: predictedType, typeSource: "inference" };
  }
  return { effectiveType: null, typeSource: null };
}

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

  // Upstream segmentation job ID for boundary corrections
  const segJobId = selectedJob?.event_segmentation_job_id ?? null;
  const { data: savedBoundaryCorrections = [] } =
    useBoundaryCorrections(segJobId);

  // Pending type corrections: Map<eventId, typeName | null>
  const [pendingCorrections, setPendingCorrections] = useState<
    Map<string, string | null>
  >(new Map());
  // Pending boundary corrections: Map<eventId, BoundaryCorrection>
  const [pendingBoundaryCorrections, setPendingBoundaryCorrections] = useState<
    Map<string, BoundaryCorrection>
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

  const isDirty =
    pendingCorrections.size > 0 || pendingBoundaryCorrections.size > 0;
  const unsavedCount =
    pendingCorrections.size + pendingBoundaryCorrections.size;

  // Reset state when job changes
  useEffect(() => {
    setPendingCorrections(new Map());
    setPendingBoundaryCorrections(new Map());
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

  // Aggregated events (full list, including deleted — needed for ghost rendering)
  const events = useMemo(
    () => aggregateEvents(typedEventRows, mergedCorrections),
    [typedEventRows, mergedCorrections],
  );

  // Navigable events: exclude deleted events and include added boundary events
  // so the user can step through, select, and label all active events.
  const navigableEvents = useMemo(() => {
    const deletedIds = new Set(
      savedBoundaryCorrections
        .filter((c) => c.correction_type === "delete")
        .map((c) => c.event_id),
    );
    const filtered =
      deletedIds.size === 0
        ? [...events]
        : events.filter((e) => !deletedIds.has(e.eventId));

    // Include saved "add" boundary corrections as navigable events
    const existingIds = new Set(events.map((e) => e.eventId));
    for (const corr of savedBoundaryCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.start_sec != null &&
        corr.end_sec != null &&
        !existingIds.has(corr.event_id)
      ) {
        const correctedType = mergedCorrections.has(corr.event_id)
          ? mergedCorrections.get(corr.event_id) ?? null
          : undefined;
        filtered.push({
          eventId: corr.event_id,
          regionId: corr.region_id,
          startSec: corr.start_sec,
          endSec: corr.end_sec,
          predictedType: null,
          predictedScore: null,
          correctedType,
          allScores: [],
        });
      }
    }

    // Include pending "add" boundary corrections as navigable events
    for (const [key, corr] of pendingBoundaryCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.start_sec != null &&
        corr.end_sec != null &&
        !existingIds.has(key)
      ) {
        const correctedType = mergedCorrections.has(key)
          ? mergedCorrections.get(key) ?? null
          : undefined;
        filtered.push({
          eventId: key,
          regionId: corr.region_id,
          startSec: corr.start_sec,
          endSec: corr.end_sec,
          predictedType: null,
          predictedScore: null,
          correctedType,
          allScores: [],
        });
      }
    }

    return filtered.sort((a, b) => a.startSec - b.startSec);
  }, [events, savedBoundaryCorrections, pendingBoundaryCorrections, mergedCorrections]);

  // Clamp index when navigable list shrinks (e.g., after saving deletes)
  useEffect(() => {
    if (
      navigableEvents.length > 0 &&
      currentEventIndex >= navigableEvents.length
    ) {
      setCurrentEventIndex(navigableEvents.length - 1);
    }
  }, [navigableEvents.length, currentEventIndex]);

  const currentEvent = navigableEvents[currentEventIndex] ?? null;

  // Derive the palette highlight from the current event's effective type
  const currentEventType: string | null = useMemo(() => {
    if (!currentEvent) return null;
    if (currentEvent.correctedType !== undefined) {
      // null correction = negative → "" in palette convention
      return currentEvent.correctedType === null ? "" : currentEvent.correctedType;
    }
    return currentEvent.predictedType;
  }, [currentEvent]);

  // Clicking a type in the palette applies it to the current event.
  // Clicking the type that already represents the event's effective label is
  // meaningful only when the label came from inference — it promotes the
  // prediction to a human correction. When the event is already corrected to
  // that same value, the click is idempotent and we skip the state update so
  // Save does not light up.
  const handleSelectType = useCallback(
    (typeName: string | null) => {
      if (!currentEvent || typeName === null) return;
      const correctionValue = typeName === "" ? null : typeName;
      if (currentEvent.correctedType === correctionValue) return;
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

  // Boundary editing: adjust handler
  const handleAdjust = useCallback(
    (eventId: string, startSec: number, endSec: number) => {
      setPendingBoundaryCorrections((prev) => {
        const next = new Map(prev);
        const existing = prev.get(eventId);
        const saved = savedBoundaryCorrections.find(
          (c) => c.event_id === eventId,
        );
        // Preserve "add" type when adjusting an added event
        const isAdd =
          existing?.correction_type === "add" ||
          saved?.correction_type === "add";
        const ev = events.find((e) => e.eventId === eventId);
        next.set(eventId, {
          event_id: eventId,
          region_id:
            ev?.regionId ??
            existing?.region_id ??
            saved?.region_id ??
            currentRegion?.region_id ??
            "",
          correction_type: isAdd ? "add" : "adjust",
          start_sec: startSec,
          end_sec: endSec,
        });
        return next;
      });
    },
    [events, savedBoundaryCorrections, currentRegion],
  );

  // Build effective events for the spectrogram overlay (current region only)
  // Incorporates both saved and pending boundary corrections.
  const regionEffectiveEvents: EffectiveEvent[] = useMemo(() => {
    if (!currentRegion) return [];
    const savedBoundaryMap = new Map(
      savedBoundaryCorrections.map((c) => [c.event_id, c]),
    );

    const result: EffectiveEvent[] = events
      .filter((e) => e.regionId === currentRegion.region_id)
      .map((e) => {
        const pending = pendingBoundaryCorrections.get(e.eventId);
        const saved = savedBoundaryMap.get(e.eventId);
        const types = resolveEventType(e.predictedType, e.correctedType);

        if (pending) {
          if (pending.correction_type === "delete") {
            return {
              eventId: e.eventId,
              regionId: e.regionId,
              startSec: e.startSec,
              endSec: e.endSec,
              originalStartSec: e.startSec,
              originalEndSec: e.endSec,
              confidence: e.predictedScore ?? 0,
              correctionType: "delete" as const,
              ...types,
            };
          }
          return {
            eventId: e.eventId,
            regionId: e.regionId,
            startSec: pending.start_sec ?? e.startSec,
            endSec: pending.end_sec ?? e.endSec,
            originalStartSec: e.startSec,
            originalEndSec: e.endSec,
            confidence: e.predictedScore ?? 0,
            correctionType: pending.correction_type,
            ...types,
          };
        }

        if (saved) {
          if (saved.correction_type === "delete") {
            return {
              eventId: e.eventId,
              regionId: e.regionId,
              startSec: e.startSec,
              endSec: e.endSec,
              originalStartSec: e.startSec,
              originalEndSec: e.endSec,
              confidence: e.predictedScore ?? 0,
              correctionType: "delete" as const,
              ...types,
            };
          }
          return {
            eventId: e.eventId,
            regionId: e.regionId,
            startSec: saved.start_sec ?? e.startSec,
            endSec: saved.end_sec ?? e.endSec,
            originalStartSec: e.startSec,
            originalEndSec: e.endSec,
            confidence: e.predictedScore ?? 0,
            correctionType: saved.correction_type as EffectiveEvent["correctionType"],
            ...types,
          };
        }

        return {
          eventId: e.eventId,
          regionId: e.regionId,
          startSec: e.startSec,
          endSec: e.endSec,
          originalStartSec: e.startSec,
          originalEndSec: e.endSec,
          confidence: e.predictedScore ?? 0,
          correctionType:
            e.correctedType !== undefined ? ("adjust" as const) : null,
          ...types,
        };
      });

    // Include saved "add" boundary corrections (no original event). These
    // events aren't in the aggregated list, so their only type signal is the
    // merged human correction (if any) — pure inference has nowhere to come
    // from here.
    const originalEventIds = new Set(events.map((e) => e.eventId));
    for (const corr of savedBoundaryCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.start_sec != null &&
        corr.end_sec != null &&
        corr.region_id === currentRegion.region_id &&
        !originalEventIds.has(corr.event_id) &&
        !pendingBoundaryCorrections.has(corr.event_id)
      ) {
        const correctedType = mergedCorrections.has(corr.event_id)
          ? mergedCorrections.get(corr.event_id) ?? null
          : undefined;
        const types = resolveEventType(null, correctedType);
        result.push({
          eventId: corr.event_id,
          regionId: corr.region_id,
          startSec: corr.start_sec,
          endSec: corr.end_sec,
          originalStartSec: corr.start_sec,
          originalEndSec: corr.end_sec,
          confidence: 0,
          correctionType: "add",
          ...types,
        });
      }
    }

    // Include pending "add" boundary corrections
    for (const [key, corr] of pendingBoundaryCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.start_sec != null &&
        corr.end_sec != null &&
        corr.region_id === currentRegion.region_id &&
        !originalEventIds.has(key)
      ) {
        const correctedType = mergedCorrections.has(key)
          ? mergedCorrections.get(key) ?? null
          : undefined;
        const types = resolveEventType(null, correctedType);
        result.push({
          eventId: key,
          regionId: corr.region_id,
          startSec: corr.start_sec,
          endSec: corr.end_sec,
          originalStartSec: corr.start_sec,
          originalEndSec: corr.end_sec,
          confidence: 0,
          correctionType: "add",
          ...types,
        });
      }
    }

    return result;
  }, [
    events,
    currentRegion,
    savedBoundaryCorrections,
    pendingBoundaryCorrections,
    mergedCorrections,
  ]);

  // Display event: same as currentEvent but with corrected boundaries applied
  // so the detail panel and playback reflect boundary edits.
  const displayEvent: AggregatedEvent | null = useMemo(() => {
    if (!currentEvent) return null;
    const effective = regionEffectiveEvents.find(
      (e) => e.eventId === currentEvent.eventId,
    );
    if (!effective) return currentEvent;
    return {
      ...currentEvent,
      startSec: effective.startSec,
      endSec: effective.endSec,
    };
  }, [currentEvent, regionEffectiveEvents]);

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
    setCurrentEventIndex((i) => Math.min(navigableEvents.length - 1, i + 1));
  }, [navigableEvents.length]);

  // Mark as negative (type correction)
  const markNegative = useCallback(() => {
    if (!currentEvent) return;
    setPendingCorrections((prev) => {
      const next = new Map(prev);
      next.set(currentEvent.eventId, null);
      return next;
    });
  }, [currentEvent]);

  // Delete event (boundary correction)
  const handleDeleteEvent = useCallback(() => {
    if (!currentEvent) return;
    setPendingBoundaryCorrections((prev) => {
      const next = new Map(prev);
      const existing = next.get(currentEvent.eventId);
      if (existing?.correction_type === "add") {
        // Remove the pending add entirely
        next.delete(currentEvent.eventId);
      } else {
        next.set(currentEvent.eventId, {
          event_id: currentEvent.eventId,
          region_id: currentEvent.regionId,
          correction_type: "delete",
          start_sec: null,
          end_sec: null,
        });
      }
      return next;
    });
    // Advance to next event (clamp if already at end)
    setCurrentEventIndex((i) => Math.min(navigableEvents.length - 1, i + 1));
  }, [currentEvent, navigableEvents.length]);

  // Add event via right-click
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    sec: number;
  } | null>(null);

  const handleAddEvent = useCallback(
    (sec: number) => {
      if (!currentRegion) return;
      const tempId = `add-${crypto.randomUUID()}`;
      const start = Math.round((sec - 0.5) * 10) / 10;
      const end = Math.round((sec + 0.5) * 10) / 10;
      setPendingBoundaryCorrections((prev) => {
        const next = new Map(prev);
        next.set(tempId, {
          event_id: tempId,
          region_id: currentRegion.region_id,
          correction_type: "add",
          start_sec: start,
          end_sec: end,
        });
        return next;
      });
      setContextMenu(null);
    },
    [currentRegion],
  );

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
    } else if (displayEvent) {
      const duration = displayEvent.endSec - displayEvent.startSec;
      startPlayback(displayEvent.startSec, duration);
    }
  }, [isPlaying, displayEvent, startPlayback, stopPlayback]);

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
          handleDeleteEvent();
          break;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [goPrev, goNext, togglePlayback, handleDeleteEvent]);

  // Save — persist both type and boundary corrections in parallel
  const saveMutation = useUpsertTypeCorrections();
  const saveBoundaryMutation = useSaveBoundaryCorrections();

  const handleSave = useCallback(() => {
    if (!selectedJobId || !isDirty) return;

    const hasTypeCorrections = pendingCorrections.size > 0;
    const hasBoundaryCorrections = pendingBoundaryCorrections.size > 0;
    let typeOk = !hasTypeCorrections;
    let boundaryOk = !hasBoundaryCorrections;

    const checkDone = () => {
      if (typeOk && boundaryOk) {
        const total =
          (hasTypeCorrections ? pendingCorrections.size : 0) +
          (hasBoundaryCorrections ? pendingBoundaryCorrections.size : 0);
        toast({
          title: "Corrections saved",
          description: `${total} correction${total !== 1 ? "s" : ""} saved.`,
        });
      }
    };

    if (hasTypeCorrections) {
      const corrections = Array.from(pendingCorrections.entries()).map(
        ([event_id, type_name]) => ({ event_id, type_name }),
      );
      saveMutation.mutate(
        { jobId: selectedJobId, corrections },
        {
          onSuccess: () => {
            setPendingCorrections(new Map());
            typeOk = true;
            checkDone();
          },
          onError: (err) => {
            toast({
              title: "Failed to save type corrections",
              description: (err as Error).message,
              variant: "destructive",
            });
          },
        },
      );
    }

    if (hasBoundaryCorrections && segJobId) {
      const corrections = Array.from(pendingBoundaryCorrections.values());
      saveBoundaryMutation.mutate(
        { jobId: segJobId, body: { corrections } },
        {
          onSuccess: () => {
            setPendingBoundaryCorrections(new Map());
            boundaryOk = true;
            checkDone();
          },
          onError: (err) => {
            toast({
              title: "Failed to save boundary corrections",
              description: (err as Error).message,
              variant: "destructive",
            });
          },
        },
      );
    }
  }, [
    selectedJobId,
    segJobId,
    isDirty,
    pendingCorrections,
    pendingBoundaryCorrections,
    saveMutation,
    saveBoundaryMutation,
  ]);

  const handleCancel = useCallback(() => {
    if (isDirty && !window.confirm("Discard unsaved corrections?")) return;
    setPendingCorrections(new Map());
    setPendingBoundaryCorrections(new Map());
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
  const [correctionsOnly, setCorrectionsOnly] = useState(true);

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
      {
        source_job_ids: [selectedJobId],
        config: { corrections_only: correctionsOnly },
      },
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
  }, [selectedJobId, correctionsOnly, createTraining]);

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
            {unsavedCount} unsaved change
            {unsavedCount !== 1 ? "s" : ""}
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
                Event{" "}
                {navigableEvents.length > 0 ? currentEventIndex + 1 : 0} of{" "}
                {navigableEvents.length}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={goNext}
                disabled={currentEventIndex >= navigableEvents.length - 1}
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
                <>
                  <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Checkbox
                      checked={correctionsOnly}
                      onCheckedChange={(v) => setCorrectionsOnly(v === true)}
                    />
                    Corrected only
                  </label>
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
                </>
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
                disabled={
                  !isDirty ||
                  saveMutation.isPending ||
                  saveBoundaryMutation.isPending
                }
              >
                <Save className="h-3 w-3 mr-1" />
                {saveMutation.isPending || saveBoundaryMutation.isPending
                  ? "Saving…"
                  : "Save"}
              </Button>
            </div>
          </div>

          {/* Spectrogram with right-click context menu for adding events */}
          {currentRegion && regionDetectionJobId ? (
            <div
              className="relative"
              onContextMenu={(e) => {
                e.preventDefault();
                const rect = e.currentTarget.getBoundingClientRect();
                setContextMenu({
                  x: e.clientX - rect.left,
                  y: e.clientY - rect.top,
                  sec:
                    viewStart !== undefined
                      ? viewStart +
                        ((e.clientX - rect.left) / rect.width) * viewSpan
                      : 0,
                });
              }}
              onClick={() => setContextMenu(null)}
            >
              <TimelineProvider
                key={`${selectedJobId}-${currentRegion.region_id}`}
                jobStart={0}
                jobEnd={currentRegion.padded_end_sec}
                zoomLevels={REVIEW_ZOOM}
                defaultZoom="30s"
                playback="slice"
                audioUrlBuilder={(startEpoch, durationSec) =>
                  regionAudioSliceUrl(regionDetectionJobId, startEpoch, durationSec)
                }
              >
                <ClassifyViewerBody
                  regionDetectionJobId={regionDetectionJobId}
                  region={currentRegion}
                  regionEffectiveEvents={regionEffectiveEvents}
                  selectedEventId={currentEvent?.eventId ?? null}
                  onSelectEvent={(eventId) => {
                    if (!eventId) return;
                    const idx = navigableEvents.findIndex((e) => e.eventId === eventId);
                    if (idx >= 0) setCurrentEventIndex(idx);
                  }}
                  onAdjust={handleAdjust}
                  scrollToCenter={scrollToCenter}
                  onViewStartChange={setViewStart}
                  onViewSpanChange={setViewSpan}
                />
              </TimelineProvider>

              {/* Context menu */}
              {contextMenu && (
                <div
                  className="absolute z-50 bg-popover border rounded-md shadow-md py-1"
                  style={{ left: contextMenu.x, top: contextMenu.y }}
                >
                  <button
                    className="w-full px-3 py-1.5 text-sm text-left hover:bg-accent"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleAddEvent(contextMenu.sec);
                    }}
                  >
                    Add event
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground">
              No events to display
            </div>
          )}

          {/* Type palette */}
          <TypePalette activeType={currentEventType} onSelectType={handleSelectType} />

          {/* Detail panel */}
          <ClassifyDetailPanel event={displayEvent} />
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

interface ClassifyViewerBodyProps {
  regionDetectionJobId: string;
  region: Region;
  regionEffectiveEvents: EffectiveEvent[];
  selectedEventId: string | null;
  onSelectEvent: (eventId: string | null) => void;
  onAdjust: (eventId: string, startSec: number, endSec: number) => void;
  scrollToCenter: number | undefined;
  onViewStartChange: (v: number) => void;
  onViewSpanChange: (v: number) => void;
}

function ClassifyViewerBody({
  regionDetectionJobId,
  region,
  regionEffectiveEvents,
  selectedEventId,
  onSelectEvent,
  onAdjust,
  scrollToCenter,
  onViewStartChange,
  onViewSpanChange,
}: ClassifyViewerBodyProps) {
  const ctx = useTimelineContext();

  // Re-center when region changes
  const prevRegionRef = useRef<string>(region.region_id);
  useEffect(() => {
    if (region.region_id !== prevRegionRef.current) {
      prevRegionRef.current = region.region_id;
      const dur = region.padded_end_sec - region.padded_start_sec;
      const span = ctx.activePreset.span;
      ctx.seekTo(region.padded_start_sec + Math.min(dur, span) / 2);
    }
  }, [region.region_id, region.padded_start_sec, region.padded_end_sec, ctx]);

  // External scroll-to-center
  const prevScrollRef = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (scrollToCenter !== undefined && scrollToCenter !== prevScrollRef.current) {
      prevScrollRef.current = scrollToCenter;
      ctx.seekTo(scrollToCenter);
    }
  }, [scrollToCenter, ctx]);

  // Report viewStart/viewSpan to parent
  useEffect(() => {
    onViewStartChange(ctx.viewStart);
  }, [ctx.viewStart, onViewStartChange]);
  useEffect(() => {
    onViewSpanChange(ctx.viewportSpan);
  }, [ctx.viewportSpan, onViewSpanChange]);

  const tileUrlBuilder = useCallback(
    (_jobId: string, zoomLevel: string, tileIndex: number) =>
      regionTileUrl(regionDetectionJobId, zoomLevel, tileIndex),
    [regionDetectionJobId],
  );

  return (
    <div className="w-full select-none">
      <div className="flex flex-col" style={{ height: 200 }}>
        <Spectrogram
          jobId={regionDetectionJobId}
          tileUrlBuilder={tileUrlBuilder}
          freqRange={[0, 3000]}
        >
          <RegionBoundaryMarkers startEpoch={region.start_sec} endEpoch={region.end_sec} />
          <EventBarOverlay
            events={regionEffectiveEvents}
            selectedEventId={selectedEventId}
            onSelectEvent={onSelectEvent}
            onAdjust={onAdjust}
            onAdd={() => {}}
            addMode={false}
            activeRegionId={region.region_id}
          />
        </Spectrogram>
      </div>
      <div className="border-t border-border px-2 py-1">
        <ZoomSelector />
      </div>
    </div>
  );
}
