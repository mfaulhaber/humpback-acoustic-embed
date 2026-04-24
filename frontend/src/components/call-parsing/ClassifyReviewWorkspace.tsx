import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useClassificationJobs,
  useSegmentationJobs,
  useRegionDetectionJobs,
  useRegionJobRegions,
  useTypedEvents,
  useVocalizationCorrections,
  useUpsertVocalizationCorrections,
  useEventBoundaryCorrections,
  useUpsertEventBoundaryCorrections,
  useEventClassifierModels,
  useCreateClassifierTrainingJob,
  useCreateClassificationJob,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { regionTileUrl, regionAudioSliceUrl } from "@/api/client";
import type {
  EventBoundaryCorrectionItem,
  EventBoundaryCorrectionResponse,
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
import type { TimelinePlaybackHandle } from "@/components/timeline/provider/types";
import { Spectrogram } from "@/components/timeline/spectrogram/Spectrogram";
import { RegionBoundaryMarkers } from "@/components/timeline/overlays/RegionBoundaryMarkers";
import { EventBarOverlay, type EffectiveEvent } from "@/components/timeline/overlays/EventBarOverlay";
import { ZoomSelector } from "@/components/timeline/controls/ZoomSelector";
import { TypePalette } from "./TypePalette";
import {
  ClassifyDetailPanel,
  type AggregatedEvent,
} from "./ClassifyDetailPanel";

type SavedEventBounds = {
  eventId: string;
  startSec: number;
  endSec: number;
};

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

function collectSavedEventBounds(
  typedEventRows: TypedEventRow[],
  savedBoundaryCorrections: EventBoundaryCorrectionResponse[],
): SavedEventBounds[] {
  const boundsByEvent = new Map<string, SavedEventBounds>();

  for (const row of typedEventRows) {
    if (!boundsByEvent.has(row.event_id)) {
      boundsByEvent.set(row.event_id, {
        eventId: row.event_id,
        startSec: row.start_sec,
        endSec: row.end_sec,
      });
    }
  }

  for (const correction of savedBoundaryCorrections) {
    if (
      correction.correction_type === "add" &&
      correction.corrected_start_sec != null &&
      correction.corrected_end_sec != null
    ) {
      boundsByEvent.set(`saved-add-${correction.id}`, {
        eventId: `saved-add-${correction.id}`,
        startSec: correction.corrected_start_sec,
        endSec: correction.corrected_end_sec,
      });
    }
  }

  return Array.from(boundsByEvent.values());
}

export function buildMergedCorrections(
  typedEventRows: TypedEventRow[],
  savedVocCorrections: Array<{
    start_sec: number;
    end_sec: number;
    type_name: string;
    correction_type: "add" | "remove";
  }>,
  savedBoundaryCorrections: EventBoundaryCorrectionResponse[],
  pendingCorrections: Map<string, string | null>,
): Map<string, string | null> {
  const map = new Map<string, string | null>();
  const savedEventBounds = collectSavedEventBounds(
    typedEventRows,
    savedBoundaryCorrections,
  );

  for (const { eventId, startSec, endSec } of savedEventBounds) {
    for (const correction of savedVocCorrections) {
      if (
        correction.start_sec < endSec &&
        correction.end_sec > startSec
      ) {
        if (correction.correction_type === "add") {
          map.set(eventId, correction.type_name);
        } else if (!map.has(eventId)) {
          map.set(eventId, null);
        }
      }
    }
  }

  for (const [eventId, typeName] of pendingCorrections) {
    map.set(eventId, typeName);
  }

  return map;
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
  const { data: savedVocCorrections = [] } =
    useVocalizationCorrections(regionDetectionJobId);

  const { data: savedBoundaryCorrections = [] } =
    useEventBoundaryCorrections(regionDetectionJobId);

  // Pending type corrections: Map<eventId, typeName | null>
  const [pendingCorrections, setPendingCorrections] = useState<
    Map<string, string | null>
  >(new Map());
  // Pending boundary corrections: Map<eventId, EventBoundaryCorrectionItem>
  const [pendingBoundaryCorrections, setPendingBoundaryCorrections] = useState<
    Map<string, EventBoundaryCorrectionItem>
  >(new Map());
  const [currentEventIndex, setCurrentEventIndex] = useState(0);
  const [viewStart, setViewStart] = useState<number | undefined>(undefined);
  const [viewSpan, setViewSpan] = useState(30);
  const [scrollToCenter, setScrollToCenter] = useState<number | undefined>(
    undefined,
  );

  // Playback via TimelineProvider ref handle
  const playbackRef = useRef<TimelinePlaybackHandle>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [userZoom, setUserZoom] = useState("30s");

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

  // Build merged correction map: resolve saved vocalization corrections to
  // event_id → type_name by matching time range overlap, then overlay pending.
  const mergedCorrections = useMemo(() => {
    return buildMergedCorrections(
      typedEventRows,
      savedVocCorrections,
      savedBoundaryCorrections,
      pendingCorrections,
    );
  }, [
    typedEventRows,
    savedVocCorrections,
    savedBoundaryCorrections,
    pendingCorrections,
  ]);

  // Aggregated events (full list, including deleted — needed for ghost rendering)
  const events = useMemo(
    () => aggregateEvents(typedEventRows, mergedCorrections),
    [typedEventRows, mergedCorrections],
  );

  // Navigable events: exclude deleted events and include added boundary events
  // so the user can step through, select, and label all active events.
  const navigableEvents = useMemo(() => {
    // Build set of deleted event keys: (region_id, original_start, original_end)
    const deletedKeys = new Set(
      savedBoundaryCorrections
        .filter((c) => c.correction_type === "delete")
        .map((c) => `${c.region_id}:${c.original_start_sec}:${c.original_end_sec}`),
    );
    const filtered =
      deletedKeys.size === 0
        ? [...events]
        : events.filter(
            (e) => !deletedKeys.has(`${e.regionId}:${e.startSec}:${e.endSec}`),
          );

    // Include saved "add" boundary corrections as navigable events
    const existingIds = new Set(events.map((e) => e.eventId));
    for (const corr of savedBoundaryCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.corrected_start_sec != null &&
        corr.corrected_end_sec != null
      ) {
        const addId = `saved-add-${corr.id}`;
        if (!existingIds.has(addId)) {
          const correctedType = mergedCorrections.has(addId)
            ? mergedCorrections.get(addId) ?? null
            : undefined;
          filtered.push({
            eventId: addId,
            regionId: corr.region_id,
            startSec: corr.corrected_start_sec,
            endSec: corr.corrected_end_sec,
            predictedType: null,
            predictedScore: null,
            correctedType,
            allScores: [],
          });
        }
      }
    }

    // Include pending "add" boundary corrections as navigable events
    for (const [key, corr] of pendingBoundaryCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.corrected_start_sec != null &&
        corr.corrected_end_sec != null &&
        !existingIds.has(key)
      ) {
        const correctedType = mergedCorrections.has(key)
          ? mergedCorrections.get(key) ?? null
          : undefined;
        filtered.push({
          eventId: key,
          regionId: corr.region_id,
          startSec: corr.corrected_start_sec,
          endSec: corr.corrected_end_sec,
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
        const ev = events.find((e) => e.eventId === eventId);
        // Check if event is an "add" correction: pending add, or saved add (eventId starts with "saved-add-")
        const isAdd = existing?.correction_type === "add" || eventId.startsWith("saved-add-") || eventId.startsWith("add-");
        const regionId =
          ev?.regionId ?? existing?.region_id ?? currentRegion?.region_id ?? "";
        if (isAdd) {
          next.set(eventId, {
            region_id: regionId,
            correction_type: "add",
            original_start_sec: null,
            original_end_sec: null,
            corrected_start_sec: startSec,
            corrected_end_sec: endSec,
          });
        } else {
          next.set(eventId, {
            region_id: regionId,
            correction_type: "adjust",
            original_start_sec: ev?.startSec ?? startSec,
            original_end_sec: ev?.endSec ?? endSec,
            corrected_start_sec: startSec,
            corrected_end_sec: endSec,
          });
        }
        return next;
      });
    },
    [events, currentRegion],
  );

  // Build effective events for the spectrogram overlay (current region only)
  // Incorporates both saved and pending boundary corrections.
  const regionEffectiveEvents: EffectiveEvent[] = useMemo(() => {
    if (!currentRegion) return [];
    // Index saved boundary corrections by (region_id, original_start, original_end) for adjust/delete
    const savedBoundaryByKey = new Map<string, EventBoundaryCorrectionResponse>();
    for (const c of savedBoundaryCorrections) {
      if (c.correction_type !== "add") {
        const key = `${c.region_id}:${c.original_start_sec}:${c.original_end_sec}`;
        savedBoundaryByKey.set(key, c);
      }
    }

    const result: EffectiveEvent[] = events
      .filter((e) => e.regionId === currentRegion.region_id)
      .map((e) => {
        const pending = pendingBoundaryCorrections.get(e.eventId);
        const savedKey = `${e.regionId}:${e.startSec}:${e.endSec}`;
        const saved = savedBoundaryByKey.get(savedKey);
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
            startSec: pending.corrected_start_sec ?? e.startSec,
            endSec: pending.corrected_end_sec ?? e.endSec,
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
            startSec: saved.corrected_start_sec ?? e.startSec,
            endSec: saved.corrected_end_sec ?? e.endSec,
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

    // Include saved "add" boundary corrections
    const originalEventIds = new Set(events.map((e) => e.eventId));
    for (const corr of savedBoundaryCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.corrected_start_sec != null &&
        corr.corrected_end_sec != null &&
        corr.region_id === currentRegion.region_id
      ) {
        const addId = `saved-add-${corr.id}`;
        if (!originalEventIds.has(addId) && !pendingBoundaryCorrections.has(addId)) {
          const correctedType = mergedCorrections.has(addId)
            ? mergedCorrections.get(addId) ?? null
            : undefined;
          const types = resolveEventType(null, correctedType);
          result.push({
            eventId: addId,
            regionId: corr.region_id,
            startSec: corr.corrected_start_sec,
            endSec: corr.corrected_end_sec,
            originalStartSec: corr.corrected_start_sec,
            originalEndSec: corr.corrected_end_sec,
            confidence: 0,
            correctionType: "add",
            ...types,
          });
        }
      }
    }

    // Include pending "add" boundary corrections
    for (const [key, corr] of pendingBoundaryCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.corrected_start_sec != null &&
        corr.corrected_end_sec != null &&
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
          startSec: corr.corrected_start_sec,
          endSec: corr.corrected_end_sec,
          originalStartSec: corr.corrected_start_sec,
          originalEndSec: corr.corrected_end_sec,
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

  // Directional scroll when the current event is not fully visible
  const navDirectionRef = useRef<"forward" | "backward">("forward");

  useEffect(() => {
    if (!currentEvent || viewStart === undefined) return;
    const viewEnd = viewStart + viewSpan;
    const pad = viewSpan * 0.15;
    const fullyVisible =
      currentEvent.startSec >= viewStart + pad &&
      currentEvent.endSec <= viewEnd - pad;
    if (!fullyVisible) {
      let target: number;
      if (navDirectionRef.current === "forward") {
        target = currentEvent.endSec + pad - viewSpan / 2;
      } else {
        target = currentEvent.startSec - pad + viewSpan / 2;
      }
      setScrollToCenter(target);
    }
  }, [currentEvent, viewStart, viewSpan]);

  // Navigation
  const goPrev = useCallback(() => {
    navDirectionRef.current = "backward";
    setCurrentEventIndex((i) => Math.max(0, i - 1));
  }, []);
  const goNext = useCallback(() => {
    navDirectionRef.current = "forward";
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
        next.delete(currentEvent.eventId);
      } else {
        next.set(currentEvent.eventId, {
          region_id: currentEvent.regionId,
          correction_type: "delete",
          original_start_sec: currentEvent.startSec,
          original_end_sec: currentEvent.endSec,
          corrected_start_sec: null,
          corrected_end_sec: null,
        });
      }
      return next;
    });
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
          region_id: currentRegion.region_id,
          correction_type: "add",
          original_start_sec: null,
          original_end_sec: null,
          corrected_start_sec: start,
          corrected_end_sec: end,
        });
        return next;
      });
      setContextMenu(null);
    },
    [currentRegion],
  );

  // Playback via ref handle
  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      playbackRef.current?.pause();
    } else if (displayEvent) {
      const duration = displayEvent.endSec - displayEvent.startSec;
      playbackRef.current?.play(displayEvent.startSec, duration);
    }
  }, [isPlaying, displayEvent]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      const tag = el.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      switch (e.code) {
        case "BracketLeft":
        case "KeyA":
          e.preventDefault();
          goPrev();
          break;
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

  // Save — persist both vocalization and boundary corrections in parallel
  const saveMutation = useUpsertVocalizationCorrections();
  const saveBoundaryMutation = useUpsertEventBoundaryCorrections();

  const handleSave = useCallback(() => {
    if (!regionDetectionJobId || !isDirty) return;

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
      const vocCorrections: Array<{
        start_sec: number;
        end_sec: number;
        type_name: string;
        correction_type: "add" | "remove";
      }> = [];
      for (const [eventId, typeName] of pendingCorrections) {
        const evt = navigableEvents.find((e) => e.eventId === eventId);
        if (!evt) continue;
        if (typeName) {
          vocCorrections.push({
            start_sec: evt.startSec,
            end_sec: evt.endSec,
            type_name: typeName,
            correction_type: "add",
          });
        } else {
          if (evt.predictedType) {
            vocCorrections.push({
              start_sec: evt.startSec,
              end_sec: evt.endSec,
              type_name: evt.predictedType,
              correction_type: "remove",
            });
          }
        }
      }
      if (vocCorrections.length > 0) {
        saveMutation.mutate(
          { regionDetectionJobId, corrections: vocCorrections },
          {
            onSuccess: () => {
              setPendingCorrections(new Map());
              typeOk = true;
              checkDone();
            },
            onError: (err: unknown) => {
              toast({
                title: "Failed to save type corrections",
                description: (err as Error).message,
                variant: "destructive",
              });
            },
          },
        );
      } else {
        setPendingCorrections(new Map());
        typeOk = true;
        checkDone();
      }
    }

    if (hasBoundaryCorrections) {
      const corrections = Array.from(pendingBoundaryCorrections.values());
      saveBoundaryMutation.mutate(
        { regionDetectionJobId, corrections },
        {
          onSuccess: () => {
            setPendingBoundaryCorrections(new Map());
            boundaryOk = true;
            checkDone();
          },
          onError: (err: unknown) => {
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
    regionDetectionJobId,
    isDirty,
    pendingCorrections,
    pendingBoundaryCorrections,
    navigableEvents,
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

  const hasCorrections = savedVocCorrections.length > 0;

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
                ref={playbackRef}
                key={`${selectedJobId}-${currentRegion.region_id}`}
                jobStart={0}
                jobEnd={currentRegion.padded_end_sec}
                zoomLevels={REVIEW_ZOOM}
                defaultZoom={userZoom}
                playback="slice"
                audioUrlBuilder={(startEpoch, durationSec) =>
                  regionAudioSliceUrl(regionDetectionJobId, startEpoch, durationSec)
                }
                disableKeyboardShortcuts
                scrollOnPlayback={false}
                onZoomChange={setUserZoom}
                onPlayStateChange={setIsPlaying}
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

  // Zoom/pan keyboard shortcuts (provider shortcuts are disabled)
  const { zoomIn: ctxZoomIn, zoomOut: ctxZoomOut, pan: ctxPan, centerTimestamp: ctxCenter, viewportSpan: ctxSpan } = ctx;
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT") return;

      switch (e.key) {
        case "+":
        case "=":
          e.preventDefault();
          ctxZoomIn();
          break;
        case "-":
          e.preventDefault();
          ctxZoomOut();
          break;
        case "ArrowLeft":
          e.preventDefault();
          ctxPan(ctxCenter - ctxSpan * 0.1);
          break;
        case "ArrowRight":
          e.preventDefault();
          ctxPan(ctxCenter + ctxSpan * 0.1);
          break;
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [ctxZoomIn, ctxZoomOut, ctxPan, ctxCenter, ctxSpan]);

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
