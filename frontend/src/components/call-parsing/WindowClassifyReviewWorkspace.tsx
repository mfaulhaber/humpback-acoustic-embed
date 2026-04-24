import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useWindowClassificationJobs,
  useWindowScores,
  useSegmentationJobs,
  useSegmentationJobEvents,
  useRegionJobRegions,
  useVocalizationCorrections,
  useUpsertVocalizationCorrections,
  useEventBoundaryCorrections,
  useUpsertEventBoundaryCorrections,
} from "@/hooks/queries/useCallParsing";
import { useVocClassifierModel } from "@/hooks/queries/useVocalization";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { regionTileUrl, regionAudioSliceUrl } from "@/api/client";
import type {
  Region,
  WindowClassificationJob,
  WindowScoreRow,
  SegmentationEvent,
  VocalizationCorrectionItem,
  EventBoundaryCorrectionItem,
  EventBoundaryCorrectionResponse,
} from "@/api/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ChevronLeft,
  ChevronRight,
  Play,
  Square,
  Save,
  X,
  Plus,
  Trash2,
  MousePointerClick,
} from "lucide-react";
import { toast } from "@/components/ui/use-toast";
import { TimelineProvider } from "@/components/timeline/provider/TimelineProvider";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import { REVIEW_ZOOM } from "@/components/timeline/provider/types";
import type { TimelinePlaybackHandle } from "@/components/timeline/provider/types";
import { Spectrogram } from "@/components/timeline/spectrogram/Spectrogram";
import { ZoomSelector } from "@/components/timeline/controls/ZoomSelector";
import { EventBarOverlay, type EffectiveEvent } from "@/components/timeline/overlays/EventBarOverlay";
import { useOverlayContext } from "@/components/timeline/overlays/OverlayContext";
import type { GradientStops } from "@/components/timeline/spectrogram/ConfidenceStrip";
import { typeColor } from "./TypePalette";

const WINDOW_SIZE_SEC = 5.0;
const HOP_SEC = 1.0;
const STRIP_HEIGHT = 26;

const HEATMAP_GRADIENT: GradientStops = [
  [0.0, "#0a2040"],
  [0.15, "#2060a0"],
  [0.3, "#20a060"],
  [0.45, "#40c040"],
  [0.6, "#a0d020"],
  [0.75, "#d0c020"],
  [0.85, "#d06020"],
  [0.95, "#c02020"],
  [1.0, "#801010"],
] as const;

interface PendingCorrection {
  start_sec: number;
  end_sec: number;
  type_name: string;
  correction_type: "add" | "remove";
}

function correctionKey(c: {
  start_sec: number;
  end_sec: number;
  type_name: string;
}) {
  return `${c.start_sec}:${c.end_sec}:${c.type_name}`;
}

interface EventWithRegion extends SegmentationEvent {
  regionIndex: number;
  eventIndexInRegion: number;
}

export function WindowClassifyReviewWorkspace({
  initialJobId,
}: {
  initialJobId?: string;
}) {
  const { data: wcJobs = [] } = useWindowClassificationJobs(0);
  const { data: hydrophones = [] } = useHydrophones();
  const { data: segJobs = [] } = useSegmentationJobs(0);

  const completeJobs = useMemo(
    () => wcJobs.filter((j) => j.status === "complete"),
    [wcJobs],
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

  const regionDetectionJobId = selectedJob?.region_detection_job_id ?? null;
  const { data: regions = [] } = useRegionJobRegions(regionDetectionJobId);
  const { data: allScoreRows = [] } = useWindowScores(
    selectedJobId ?? undefined,
  );

  const { data: vocModel } = useVocClassifierModel(
    selectedJob?.vocalization_model_id ?? null,
  );

  const vocabulary: string[] = useMemo(() => {
    if (selectedJob?.vocabulary_snapshot) {
      try {
        return JSON.parse(selectedJob.vocabulary_snapshot) as string[];
      } catch {
        /* ignore */
      }
    }
    return vocModel?.vocabulary_snapshot ?? [];
  }, [selectedJob?.vocabulary_snapshot, vocModel?.vocabulary_snapshot]);

  const perClassThresholds: Record<string, number> = useMemo(
    () => vocModel?.per_class_thresholds ?? {},
    [vocModel?.per_class_thresholds],
  );

  // Find completed Pass 2 segmentation job for this detection job
  const segmentationJob = useMemo(
    () =>
      regionDetectionJobId
        ? segJobs.find(
            (j) =>
              j.region_detection_job_id === regionDetectionJobId &&
              j.status === "complete",
          ) ?? null
        : null,
    [segJobs, regionDetectionJobId],
  );

  const { data: rawEvents = [] } = useSegmentationJobEvents(
    segmentationJob?.id ?? null,
  );

  // Saved vocalization corrections
  const { data: savedCorrections = [] } =
    useVocalizationCorrections(regionDetectionJobId);

  // Saved boundary corrections
  const { data: savedBoundaryCorrections = [] } =
    useEventBoundaryCorrections(regionDetectionJobId);

  // Group events by region and annotate with indices
  const eventsByRegion = useMemo(() => {
    const grouped: Record<string, EventWithRegion[]> = {};
    for (const evt of rawEvents) {
      if (!grouped[evt.region_id]) grouped[evt.region_id] = [];
    }
    // Sort events within each region by start time
    for (const evt of rawEvents) {
      grouped[evt.region_id].push({
        ...evt,
        regionIndex: 0,
        eventIndexInRegion: 0,
      });
    }
    // Assign indices
    const regionOrder = regions.map((r) => r.region_id);
    for (let ri = 0; ri < regionOrder.length; ri++) {
      const evts = grouped[regionOrder[ri]];
      if (!evts) continue;
      evts.sort((a, b) => a.start_sec - b.start_sec);
      for (let ei = 0; ei < evts.length; ei++) {
        evts[ei].regionIndex = ri;
        evts[ei].eventIndexInRegion = ei;
      }
    }
    return grouped;
  }, [rawEvents, regions]);

  // Flat list of all events in region order
  const allEvents = useMemo(() => {
    const result: EventWithRegion[] = [];
    for (const r of regions) {
      const evts = eventsByRegion[r.region_id];
      if (evts) result.push(...evts);
    }
    return result;
  }, [regions, eventsByRegion]);

  // Pending boundary corrections: Map<eventId, EventBoundaryCorrectionItem>
  const [pendingBoundaryCorrections, setPendingBoundaryCorrections] = useState<
    Map<string, EventBoundaryCorrectionItem>
  >(new Map());

  const [addMode, setAddMode] = useState(false);

  // Build effective events merging boundary corrections with original events
  const effectiveEvents: EffectiveEvent[] = useMemo(() => {
    // Index saved corrections by (region_id, original_start, original_end) for adjust/delete
    const savedByKey = new Map<string, EventBoundaryCorrectionResponse>();
    for (const c of savedBoundaryCorrections) {
      if (c.correction_type !== "add") {
        const key = `${c.region_id}:${c.original_start_sec}:${c.original_end_sec}`;
        savedByKey.set(key, c);
      }
    }

    const result: EffectiveEvent[] = allEvents.map((ev) => {
      const pending = pendingBoundaryCorrections.get(ev.event_id);
      const savedKey = `${ev.region_id}:${ev.start_sec}:${ev.end_sec}`;
      const saved = savedByKey.get(savedKey);

      if (pending) {
        return {
          eventId: ev.event_id,
          regionId: ev.region_id,
          startSec: pending.corrected_start_sec ?? ev.start_sec,
          endSec: pending.corrected_end_sec ?? ev.end_sec,
          originalStartSec: ev.start_sec,
          originalEndSec: ev.end_sec,
          confidence: ev.segmentation_confidence,
          correctionType: pending.correction_type,
          effectiveType: null,
          typeSource: null,
        };
      }

      if (saved) {
        return {
          eventId: ev.event_id,
          regionId: ev.region_id,
          startSec: saved.corrected_start_sec ?? ev.start_sec,
          endSec: saved.corrected_end_sec ?? ev.end_sec,
          originalStartSec: ev.start_sec,
          originalEndSec: ev.end_sec,
          confidence: ev.segmentation_confidence,
          correctionType: saved.correction_type as EffectiveEvent["correctionType"],
          effectiveType: null,
          typeSource: null,
        };
      }

      return {
        eventId: ev.event_id,
        regionId: ev.region_id,
        startSec: ev.start_sec,
        endSec: ev.end_sec,
        originalStartSec: ev.start_sec,
        originalEndSec: ev.end_sec,
        confidence: ev.segmentation_confidence,
        correctionType: null,
        effectiveType: null,
        typeSource: null,
      };
    });

    // Add saved "add" boundary corrections
    for (const corr of savedBoundaryCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.corrected_start_sec != null &&
        corr.corrected_end_sec != null
      ) {
        const addId = `saved-add-${corr.id}`;
        if (!pendingBoundaryCorrections.has(addId)) {
          result.push({
            eventId: addId,
            regionId: corr.region_id,
            startSec: corr.corrected_start_sec,
            endSec: corr.corrected_end_sec,
            originalStartSec: corr.corrected_start_sec,
            originalEndSec: corr.corrected_end_sec,
            confidence: 0,
            correctionType: "add",
            effectiveType: null,
            typeSource: null,
          });
        }
      }
    }

    // Add pending "add" boundary corrections
    for (const [key, corr] of pendingBoundaryCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.corrected_start_sec != null &&
        corr.corrected_end_sec != null
      ) {
        result.push({
          eventId: key,
          regionId: corr.region_id,
          startSec: corr.corrected_start_sec,
          endSec: corr.corrected_end_sec,
          originalStartSec: corr.corrected_start_sec,
          originalEndSec: corr.corrected_end_sec,
          confidence: 0,
          correctionType: "add",
          effectiveType: null,
          typeSource: null,
        });
      }
    }

    return result;
  }, [allEvents, savedBoundaryCorrections, pendingBoundaryCorrections]);

  // Event/region navigation state
  const [selectedEventIdx, setSelectedEventIdx] = useState(0);

  useEffect(() => {
    setSelectedEventIdx(0);
    setPendingCorrections(new Map());
    setPendingBoundaryCorrections(new Map());
    setAddMode(false);
  }, [selectedJobId]);

  const selectedEvent = allEvents[selectedEventIdx] ?? null;
  const currentRegionIndex = selectedEvent?.regionIndex ?? 0;
  const currentRegion = regions[currentRegionIndex] ?? null;

  // Effective events filtered to current region
  const regionEffectiveEvents = useMemo(
    () =>
      currentRegion
        ? effectiveEvents.filter((e) => e.regionId === currentRegion.region_id)
        : [],
    [effectiveEvents, currentRegion],
  );

  // Type selector for confidence strip
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [thresholdOverride, setThresholdOverride] = useState<string>("");

  const effectiveThreshold = useMemo(() => {
    const parsed = parseFloat(thresholdOverride);
    if (!isNaN(parsed)) return parsed;
    if (selectedType && perClassThresholds[selectedType] != null) {
      return perClassThresholds[selectedType];
    }
    const thresholds = Object.values(perClassThresholds);
    if (thresholds.length > 0) {
      return thresholds.reduce((a, b) => a + b, 0) / thresholds.length;
    }
    return 0.5;
  }, [thresholdOverride, selectedType, perClassThresholds]);

  // Timeline extent covers all regions
  const timelineEnd = useMemo(() => {
    if (regions.length === 0) return 0;
    return Math.max(...regions.map((r) => r.padded_end_sec));
  }, [regions]);

  // Build confidence strip scores array
  const stripScores = useMemo(() => {
    if (allScoreRows.length === 0 || timelineEnd === 0) return [];
    const n = Math.ceil(timelineEnd / HOP_SEC);
    const arr: (number | null)[] = new Array(n).fill(null);

    const lookup = new Map<number, WindowScoreRow>();
    for (const row of allScoreRows) {
      lookup.set(Math.round(row.time_sec * 10) / 10, row);
    }

    for (let i = 0; i < n; i++) {
      const t = Math.round(i * HOP_SEC * 10) / 10;
      const row = lookup.get(t);
      if (!row) continue;
      if (selectedType) {
        arr[i] = row.scores[selectedType] ?? null;
      } else {
        const values = Object.values(row.scores);
        arr[i] = values.length > 0 ? Math.max(...values) : null;
      }
    }
    return arr;
  }, [allScoreRows, timelineEnd, selectedType]);

  // Compute max scores for selected event from overlapping windows
  const eventMaxScores = useMemo(() => {
    if (!selectedEvent) return {};
    const scores: Record<string, number> = {};
    for (const row of allScoreRows) {
      const winStart = row.time_sec;
      const winEnd = row.time_sec + WINDOW_SIZE_SEC;
      if (winStart < selectedEvent.end_sec && winEnd > selectedEvent.start_sec) {
        for (const [typeName, score] of Object.entries(row.scores)) {
          if (score > (scores[typeName] ?? 0)) {
            scores[typeName] = score;
          }
        }
      }
    }
    return scores;
  }, [selectedEvent, allScoreRows]);

  // Pending vocalization corrections: Map<correctionKey, PendingCorrection>
  const [pendingCorrections, setPendingCorrections] = useState<
    Map<string, PendingCorrection>
  >(new Map());

  const isDirty =
    pendingCorrections.size > 0 || pendingBoundaryCorrections.size > 0;
  const unsavedCount =
    pendingCorrections.size + pendingBoundaryCorrections.size;

  // Merged corrections (saved + pending, pending overrides)
  const mergedCorrections = useMemo(() => {
    const map = new Map<string, "add" | "remove">();
    for (const c of savedCorrections) {
      map.set(correctionKey(c), c.correction_type);
    }
    for (const [key, c] of pendingCorrections) {
      map.set(key, c.correction_type);
    }
    return map;
  }, [savedCorrections, pendingCorrections]);

  // Playback
  const playbackRef = useRef<TimelinePlaybackHandle>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [userZoom, setUserZoom] = useState("30s");

  // Event navigation
  const goPrevEvent = useCallback(() => {
    setSelectedEventIdx((i) => Math.max(0, i - 1));
  }, []);
  const goNextEvent = useCallback(() => {
    setSelectedEventIdx((i) => Math.min(allEvents.length - 1, i + 1));
  }, [allEvents.length]);
  const goPrevRegion = useCallback(() => {
    if (!selectedEvent || !regions.length) return;
    const targetRi = selectedEvent.regionIndex - 1;
    if (targetRi < 0) return;
    const targetRegion = regions[targetRi];
    const evts = eventsByRegion[targetRegion.region_id];
    if (evts && evts.length > 0) {
      // Last event of previous region
      const lastEvt = evts[evts.length - 1];
      const idx = allEvents.indexOf(lastEvt);
      if (idx >= 0) setSelectedEventIdx(idx);
    }
  }, [selectedEvent, regions, eventsByRegion, allEvents]);
  const goNextRegion = useCallback(() => {
    if (!selectedEvent || !regions.length) return;
    const targetRi = selectedEvent.regionIndex + 1;
    if (targetRi >= regions.length) return;
    const targetRegion = regions[targetRi];
    const evts = eventsByRegion[targetRegion.region_id];
    if (evts && evts.length > 0) {
      // First event of next region
      const firstEvt = evts[0];
      const idx = allEvents.indexOf(firstEvt);
      if (idx >= 0) setSelectedEventIdx(idx);
    }
  }, [selectedEvent, regions, eventsByRegion, allEvents]);

  // Playback bounded to selected event
  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      playbackRef.current?.pause();
    } else if (selectedEvent) {
      const duration = selectedEvent.end_sec - selectedEvent.start_sec;
      playbackRef.current?.play(selectedEvent.start_sec, duration);
    }
  }, [isPlaying, selectedEvent]);

  // Badge click: toggle add/remove correction for event
  const handleBadgeClick = useCallback(
    (typeName: string) => {
      if (!selectedEvent) return;
      const key = correctionKey({
        start_sec: selectedEvent.start_sec,
        end_sec: selectedEvent.end_sec,
        type_name: typeName,
      });
      const existing = mergedCorrections.get(key);
      const score = eventMaxScores[typeName] ?? 0;
      const threshold = perClassThresholds[typeName] ?? effectiveThreshold;
      const isAbove = score >= threshold;

      setPendingCorrections((prev) => {
        const next = new Map(prev);
        if (existing === "remove") {
          next.delete(key);
        } else if (isAbove && existing !== "add") {
          next.set(key, {
            start_sec: selectedEvent.start_sec,
            end_sec: selectedEvent.end_sec,
            correction_type: "remove",
            type_name: typeName,
          });
        } else {
          next.delete(key);
        }
        return next;
      });
    },
    [
      selectedEvent,
      mergedCorrections,
      eventMaxScores,
      perClassThresholds,
      effectiveThreshold,
    ],
  );

  // Add type via popover
  const handleAddType = useCallback(
    (typeName: string) => {
      if (!selectedEvent) return;
      const key = correctionKey({
        start_sec: selectedEvent.start_sec,
        end_sec: selectedEvent.end_sec,
        type_name: typeName,
      });
      setPendingCorrections((prev) => {
        const next = new Map(prev);
        next.set(key, {
          start_sec: selectedEvent.start_sec,
          end_sec: selectedEvent.end_sec,
          correction_type: "add",
          type_name: typeName,
        });
        return next;
      });
    },
    [selectedEvent],
  );

  // Boundary editing: adjust handler
  const handleBoundaryAdjust = useCallback(
    (eventId: string, startSec: number, endSec: number) => {
      setPendingBoundaryCorrections((prev) => {
        const next = new Map(prev);
        const existing = prev.get(eventId);
        const isAdd = existing?.correction_type === "add";
        const ev = allEvents.find((e) => e.event_id === eventId);
        const effectiveEv = effectiveEvents.find((e) => e.eventId === eventId);
        const regionId =
          ev?.region_id ?? existing?.region_id ?? currentRegion?.region_id ?? "";
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
            original_start_sec: ev?.start_sec ?? effectiveEv?.originalStartSec ?? startSec,
            original_end_sec: ev?.end_sec ?? effectiveEv?.originalEndSec ?? endSec,
            corrected_start_sec: startSec,
            corrected_end_sec: endSec,
          });
        }
        return next;
      });
    },
    [allEvents, effectiveEvents, currentRegion],
  );

  // Boundary editing: add handler
  const handleBoundaryAdd = useCallback(
    (regionId: string, startSec: number, endSec: number) => {
      const tempId = `add-${crypto.randomUUID()}`;
      setPendingBoundaryCorrections((prev) => {
        const next = new Map(prev);
        next.set(tempId, {
          region_id: regionId,
          correction_type: "add",
          original_start_sec: null,
          original_end_sec: null,
          corrected_start_sec: startSec,
          corrected_end_sec: endSec,
        });
        return next;
      });
      setAddMode(false);
    },
    [],
  );

  // Boundary editing: delete handler
  const handleBoundaryDelete = useCallback(
    (eventId: string) => {
      setPendingBoundaryCorrections((prev) => {
        const next = new Map(prev);
        const existing = next.get(eventId);
        if (existing?.correction_type === "delete") {
          next.delete(eventId);
        } else if (existing?.correction_type === "add") {
          next.delete(eventId);
        } else {
          const ev = allEvents.find((e) => e.event_id === eventId);
          next.set(eventId, {
            region_id: ev?.region_id ?? currentRegion?.region_id ?? "",
            correction_type: "delete",
            original_start_sec: ev?.start_sec ?? null,
            original_end_sec: ev?.end_sec ?? null,
            corrected_start_sec: null,
            corrected_end_sec: null,
          });
        }
        return next;
      });
    },
    [allEvents, currentRegion],
  );

  // Save
  const upsertMutation = useUpsertVocalizationCorrections();
  const upsertBoundaryMutation = useUpsertEventBoundaryCorrections();

  const handleSave = useCallback(() => {
    if (!regionDetectionJobId || !isDirty) return;

    const hasVocCorrections = pendingCorrections.size > 0;
    const hasBoundaryCorrections = pendingBoundaryCorrections.size > 0;
    let vocOk = !hasVocCorrections;
    let boundaryOk = !hasBoundaryCorrections;

    const checkDone = () => {
      if (vocOk && boundaryOk) {
        const total = unsavedCount;
        toast({
          title: "Corrections saved",
          description: `${total} correction${total !== 1 ? "s" : ""} saved.`,
        });
      }
    };

    if (hasVocCorrections) {
      const corrections: VocalizationCorrectionItem[] = Array.from(
        pendingCorrections.values(),
      );
      upsertMutation.mutate(
        { regionDetectionJobId, corrections },
        {
          onSuccess: () => {
            setPendingCorrections(new Map());
            vocOk = true;
            checkDone();
          },
          onError: (err: unknown) => {
            toast({
              title: "Failed to save vocalization corrections",
              description: (err as Error).message,
              variant: "destructive",
            });
          },
        },
      );
    }

    if (hasBoundaryCorrections) {
      const corrections = Array.from(pendingBoundaryCorrections.values());
      upsertBoundaryMutation.mutate(
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
    unsavedCount,
    pendingCorrections,
    pendingBoundaryCorrections,
    upsertMutation,
    upsertBoundaryMutation,
  ]);

  const handleCancel = useCallback(() => {
    if (isDirty && !window.confirm("Discard unsaved corrections?")) return;
    setPendingCorrections(new Map());
    setPendingBoundaryCorrections(new Map());
    setAddMode(false);
  }, [isDirty]);

  // beforeunload warning
  useEffect(() => {
    if (!isDirty) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isDirty]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (
        el.tagName === "INPUT" ||
        el.tagName === "TEXTAREA" ||
        el.tagName === "SELECT"
      )
        return;

      if (e.shiftKey) {
        switch (e.code) {
          case "ArrowRight":
          case "KeyD":
            e.preventDefault();
            goNextRegion();
            return;
          case "ArrowLeft":
          case "KeyA":
            e.preventDefault();
            goPrevRegion();
            return;
        }
      }

      switch (e.code) {
        case "ArrowRight":
        case "KeyD":
          e.preventDefault();
          goNextEvent();
          break;
        case "ArrowLeft":
        case "KeyA":
          e.preventDefault();
          goPrevEvent();
          break;
        case "Space":
          e.preventDefault();
          togglePlayback();
          break;
        case "Backspace":
        case "Delete":
          e.preventDefault();
          if (selectedEvent) handleBoundaryDelete(selectedEvent.event_id);
          break;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [goPrevEvent, goNextEvent, goPrevRegion, goNextRegion, togglePlayback, selectedEvent, handleBoundaryDelete]);

  // Job label
  const jobLabel = useCallback(
    (job: WindowClassificationJob) => {
      const shortId = job.id.slice(0, 8);
      const wc = job.window_count ?? 0;
      return `${shortId} — ${wc} windows`;
    },
    [],
  );

  // Event labels for the detail panel
  const selectedEventLabels = useMemo(() => {
    if (!selectedEvent) return [];
    return vocabulary.map((typeName) => {
      const score = eventMaxScores[typeName] ?? 0;
      const threshold = perClassThresholds[typeName] ?? effectiveThreshold;
      const aboveThreshold = score >= threshold;
      const key = correctionKey({
        start_sec: selectedEvent.start_sec,
        end_sec: selectedEvent.end_sec,
        type_name: typeName,
      });
      const correction = mergedCorrections.get(key);
      const isPending = pendingCorrections.has(key);
      return {
        typeName,
        score,
        threshold,
        aboveThreshold,
        correction,
        isPending,
      };
    });
  }, [
    selectedEvent,
    vocabulary,
    eventMaxScores,
    perClassThresholds,
    effectiveThreshold,
    mergedCorrections,
    pendingCorrections,
  ]);

  // Count events in current region
  const currentRegionEvents = currentRegion
    ? eventsByRegion[currentRegion.region_id] ?? []
    : [];
  const eventIndexInRegion = selectedEvent?.eventIndexInRegion ?? 0;

  return (
    <div className="space-y-4">
      {/* Job selector */}
      <div className="flex items-center gap-4">
        <label htmlFor="wc-review-job" className="text-sm font-medium">
          Job
        </label>
        <select
          id="wc-review-job"
          className="rounded-md border bg-background px-3 py-2 text-sm"
          value={selectedJobId ?? ""}
          onChange={(e) => setSelectedJobId(e.target.value || null)}
        >
          <option value="">Select a completed window classification job</option>
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
      {selectedJob && !segmentationJob ? (
        <div className="py-8 text-center text-muted-foreground">
          No completed event segmentation (Pass 2) job exists for this
          detection job. Run event segmentation first to enable event-level
          review.
        </div>
      ) : selectedJob && segmentationJob ? (
        <div className="rounded-md border">
          {/* Toolbar */}
          <div className="flex items-center justify-between px-4 py-2 border-b">
            <div className="flex items-center gap-2">
              {/* Region navigator */}
              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={goPrevRegion}
                disabled={currentRegionIndex === 0}
                title="Previous region (Shift+←)"
              >
                <ChevronLeft className="h-4 w-4" />
                <ChevronLeft className="h-4 w-4 -ml-2.5" />
              </Button>
              <span className="text-xs text-muted-foreground tabular-nums min-w-[90px] text-center">
                Region {regions.length > 0 ? currentRegionIndex + 1 : 0} /{" "}
                {regions.length}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={goNextRegion}
                disabled={currentRegionIndex >= regions.length - 1}
                title="Next region (Shift+→)"
              >
                <ChevronRight className="h-4 w-4" />
                <ChevronRight className="h-4 w-4 -ml-2.5" />
              </Button>

              <div className="w-px h-5 bg-border mx-1" />

              {/* Event navigator */}
              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={goPrevEvent}
                disabled={selectedEventIdx === 0}
                title="Previous event (←)"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-xs text-muted-foreground tabular-nums min-w-[80px] text-center">
                Event {currentRegionEvents.length > 0 ? eventIndexInRegion + 1 : 0} /{" "}
                {currentRegionEvents.length}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={goNextEvent}
                disabled={selectedEventIdx >= allEvents.length - 1}
                title="Next event (→)"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>

              <div className="w-px h-5 bg-border mx-1" />

              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={togglePlayback}
                disabled={!selectedEvent}
                title="Play/pause event (Space)"
              >
                {isPlaying ? (
                  <Square className="h-3.5 w-3.5" />
                ) : (
                  <Play className="h-3.5 w-3.5" />
                )}
              </Button>

              <div className="w-px h-5 bg-border mx-1" />

              <Button
                variant={addMode ? "default" : "ghost"}
                size="sm"
                className="h-7 text-xs"
                onClick={() => setAddMode((prev) => !prev)}
                title="Add event mode"
              >
                <MousePointerClick className="h-3 w-3 mr-1" />
                Add
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={() => {
                  if (selectedEvent) handleBoundaryDelete(selectedEvent.event_id);
                }}
                disabled={!selectedEvent}
                title="Delete event (Del)"
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Delete
              </Button>
            </div>

            <div className="flex items-center gap-2">
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
                  upsertMutation.isPending ||
                  upsertBoundaryMutation.isPending
                }
              >
                <Save className="h-3 w-3 mr-1" />
                {upsertMutation.isPending || upsertBoundaryMutation.isPending
                  ? "Saving…"
                  : "Save"}
              </Button>
            </div>
          </div>

          {/* Type selector + threshold controls */}
          <div className="flex items-center gap-3 px-4 py-1.5 border-b bg-muted/30">
            <label className="text-xs text-muted-foreground">Type</label>
            <select
              className="rounded border bg-background px-2 py-1 text-xs"
              value={selectedType ?? ""}
              onChange={(e) => setSelectedType(e.target.value || null)}
            >
              <option value="">All types (max)</option>
              {vocabulary.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>

            <label className="text-xs text-muted-foreground ml-2">
              Threshold
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              className="rounded border bg-background px-2 py-1 text-xs w-20"
              placeholder={effectiveThreshold.toFixed(2)}
              value={thresholdOverride}
              onChange={(e) => setThresholdOverride(e.target.value)}
            />
          </div>

          {/* Spectrogram + strip */}
          {currentRegion && regionDetectionJobId ? (
            <TimelineProvider
              ref={playbackRef}
              key={selectedJobId}
              jobStart={0}
              jobEnd={timelineEnd}
              zoomLevels={REVIEW_ZOOM}
              defaultZoom={userZoom}
              playback="slice"
              audioUrlBuilder={(startEpoch, durationSec) =>
                regionAudioSliceUrl(
                  regionDetectionJobId,
                  startEpoch,
                  durationSec,
                )
              }
              disableKeyboardShortcuts
              scrollOnPlayback={false}
              onZoomChange={setUserZoom}
              onPlayStateChange={setIsPlaying}
            >
              <WindowClassifyViewerBody
                regionDetectionJobId={regionDetectionJobId}
                region={currentRegion}
                allRegions={regions}
                stripScores={stripScores}
                selectedEvent={selectedEvent}
                allEvents={allEvents}
                allScoreRows={allScoreRows}
                onSelectEvent={(idx) => setSelectedEventIdx(idx)}
                regionEffectiveEvents={regionEffectiveEvents}
                onAdjust={handleBoundaryAdjust}
                onAdd={handleBoundaryAdd}
                addMode={addMode}
              />
            </TimelineProvider>
          ) : (
            <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground">
              No regions to display
            </div>
          )}

          {/* Detail panel */}
          <EventDetailPanel
            selectedEvent={selectedEvent}
            labels={selectedEventLabels}
            region={currentRegion}
            onBadgeClick={handleBadgeClick}
            onAddType={handleAddType}
            vocabulary={vocabulary}
          />
        </div>
      ) : (
        <div className="py-8 text-center text-muted-foreground">
          Select a completed window classification job to begin reviewing
          scores.
        </div>
      )}
    </div>
  );
}

// ---- Viewer body (inside TimelineProvider) ----

interface WindowClassifyViewerBodyProps {
  regionDetectionJobId: string;
  region: Region;
  allRegions: Region[];
  stripScores: (number | null)[];
  selectedEvent: EventWithRegion | null;
  allEvents: EventWithRegion[];
  allScoreRows: WindowScoreRow[];
  onSelectEvent: (globalIndex: number) => void;
  regionEffectiveEvents: EffectiveEvent[];
  onAdjust: (eventId: string, startSec: number, endSec: number) => void;
  onAdd: (regionId: string, startSec: number, endSec: number) => void;
  addMode: boolean;
}

function WindowClassifyViewerBody({
  regionDetectionJobId,
  region,
  allRegions,
  stripScores,
  selectedEvent,
  allEvents,
  allScoreRows,
  onSelectEvent,
  regionEffectiveEvents,
  onAdjust,
  onAdd,
  addMode,
}: WindowClassifyViewerBodyProps) {
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

  // Center on selected event
  useEffect(() => {
    if (selectedEvent) {
      const evtCenter =
        (selectedEvent.start_sec + selectedEvent.end_sec) / 2;
      ctx.seekTo(evtCenter);
    }
  }, [selectedEvent?.start_sec, selectedEvent?.end_sec]); // eslint-disable-line react-hooks/exhaustive-deps

  // Zoom/pan keyboard shortcuts (provider shortcuts disabled)
  const {
    zoomIn: ctxZoomIn,
    zoomOut: ctxZoomOut,
    pan: ctxPan,
    centerTimestamp: ctxCenter,
    viewportSpan: ctxSpan,
  } = ctx;
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (
        el.tagName === "INPUT" ||
        el.tagName === "TEXTAREA" ||
        el.tagName === "SELECT"
      )
        return;

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
          scores={stripScores.length > 0 ? stripScores : undefined}
          windowSec={HOP_SEC}
          stripHeight={STRIP_HEIGHT}
          stripGradient={HEATMAP_GRADIENT}
        >
          <AllRegionLines
            regions={allRegions}
            activeRegionId={region.region_id}
          />
          <EventBarOverlay
            events={regionEffectiveEvents}
            selectedEventId={selectedEvent?.event_id ?? null}
            onSelectEvent={(eventId) => {
              if (!eventId) return;
              const idx = allEvents.findIndex((e) => e.event_id === eventId);
              if (idx >= 0) onSelectEvent(idx);
            }}
            onAdjust={onAdjust}
            onAdd={onAdd}
            addMode={addMode}
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

// ---- All-region boundary lines (no shading) ----

function AllRegionLines({
  regions,
  activeRegionId,
}: {
  regions: Region[];
  activeRegionId: string;
}) {
  const { epochToX, canvasWidth, canvasHeight } = useOverlayContext();

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        overflow: "hidden",
      }}
    >
      {regions.map((r) => {
        const isActive = r.region_id === activeRegionId;
        const startX = epochToX(r.start_sec);
        const endX = epochToX(r.end_sec);
        const color = isActive
          ? "rgba(59, 130, 246, 0.8)"
          : "rgba(59, 130, 246, 0.35)";
        const lineWidth = isActive ? "2px" : "1px";

        return (
          <div key={r.region_id}>
            {startX >= 0 && startX <= canvasWidth && (
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: startX,
                  width: 0,
                  height: canvasHeight,
                  borderLeft: `${lineWidth} dashed ${color}`,
                }}
              />
            )}
            {endX >= 0 && endX <= canvasWidth && (
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: endX,
                  width: 0,
                  height: canvasHeight,
                  borderLeft: `${lineWidth} dashed ${color}`,
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---- Event detail panel ----

interface EventLabel {
  typeName: string;
  score: number;
  threshold: number;
  aboveThreshold: boolean;
  correction: "add" | "remove" | undefined;
  isPending: boolean;
}

function EventDetailPanel({
  selectedEvent,
  labels,
  region,
  onBadgeClick,
  onAddType,
  vocabulary,
}: {
  selectedEvent: EventWithRegion | null;
  labels: EventLabel[];
  region: Region | null;
  onBadgeClick: (typeName: string) => void;
  onAddType: (typeName: string) => void;
  vocabulary: string[];
}) {
  const [showAddPopover, setShowAddPopover] = useState(false);

  if (!selectedEvent) {
    return (
      <div className="px-4 py-3 text-sm text-muted-foreground border-t">
        No events found — run event segmentation first
      </div>
    );
  }

  const aboveLabels = labels
    .filter(
      (l) =>
        (l.aboveThreshold && l.correction !== "remove") ||
        l.correction === "add",
    )
    .sort((a, b) => b.score - a.score);
  const belowLabels = labels
    .filter(
      (l) =>
        (!l.aboveThreshold || l.correction === "remove") &&
        l.correction !== "add",
    )
    .sort((a, b) => b.score - a.score);

  const addableTypes = vocabulary.filter(
    (t) => !aboveLabels.some((l) => l.typeName === t),
  );

  return (
    <div className="px-4 py-3 border-t space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          Event: {selectedEvent.start_sec.toFixed(1)}s –{" "}
          {selectedEvent.end_sec.toFixed(1)}s
          {region && (
            <span className="ml-2">
              Region: {region.region_id.slice(0, 8)}
            </span>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {aboveLabels.map((l) => (
          <Badge
            key={l.typeName}
            className="cursor-pointer text-xs text-white select-none"
            style={{
              backgroundColor: typeColor(l.typeName),
              outline: l.isPending
                ? "2px solid rgba(250, 204, 21, 0.8)"
                : undefined,
              outlineOffset: l.isPending ? "1px" : undefined,
            }}
            onClick={() => onBadgeClick(l.typeName)}
          >
            {l.typeName} {(l.score * 100).toFixed(0)}%
            {l.correction === "add" && (
              <Plus className="h-2.5 w-2.5 ml-0.5 inline" />
            )}
          </Badge>
        ))}

        {belowLabels.map((l) => (
          <Badge
            key={l.typeName}
            variant="outline"
            className="cursor-pointer text-xs select-none opacity-50 hover:opacity-75"
            style={{
              borderColor: typeColor(l.typeName),
              color: typeColor(l.typeName),
              outline: l.isPending
                ? "2px solid rgba(250, 204, 21, 0.8)"
                : undefined,
              outlineOffset: l.isPending ? "1px" : undefined,
            }}
            onClick={() => onBadgeClick(l.typeName)}
          >
            {l.typeName} {(l.score * 100).toFixed(0)}%
            {l.correction === "remove" && (
              <X className="h-2.5 w-2.5 ml-0.5 inline" />
            )}
          </Badge>
        ))}

        <div className="relative">
          <button
            className="px-1.5 py-0.5 rounded text-xs border border-dashed border-muted-foreground/40 text-muted-foreground hover:border-muted-foreground hover:text-foreground"
            onClick={() => setShowAddPopover((v) => !v)}
          >
            <Plus className="h-3 w-3" />
          </button>
          {showAddPopover && addableTypes.length > 0 && (
            <div className="absolute z-50 bottom-full mb-1 left-0 bg-popover border rounded-md shadow-md py-1 min-w-[120px]">
              {addableTypes.map((t) => (
                <button
                  key={t}
                  className="w-full px-3 py-1 text-xs text-left hover:bg-accent flex items-center gap-1.5"
                  onClick={() => {
                    onAddType(t);
                    setShowAddPopover(false);
                  }}
                >
                  <span
                    className="w-2 h-2 rounded-full inline-block"
                    style={{ backgroundColor: typeColor(t) }}
                  />
                  {t}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {labels.length > 0 && (
        <div className="text-xs">
          <span className="text-muted-foreground font-medium">
            Max scores (overlapping windows):
          </span>
          <div className="mt-1 grid grid-cols-3 gap-x-4 gap-y-0.5">
            {labels
              .slice()
              .sort((a, b) => b.score - a.score)
              .map((l) => (
                <div key={l.typeName} className="flex items-center gap-1">
                  <span
                    className="w-2 h-2 rounded-full inline-block"
                    style={{ backgroundColor: typeColor(l.typeName) }}
                  />
                  <span
                    className={l.aboveThreshold ? "font-medium" : ""}
                  >
                    {l.typeName}
                  </span>
                  <span className="text-muted-foreground ml-auto">
                    {l.score.toFixed(3)}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
