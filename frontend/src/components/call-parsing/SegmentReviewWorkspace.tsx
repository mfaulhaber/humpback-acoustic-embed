import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useSegmentationJobs,
  useSegmentationJobEvents,
  useRegionDetectionJobs,
  useRegionJobRegions,
  useEventBoundaryCorrections,
  useUpsertEventBoundaryCorrections,
  useCreateSegmentationJob,
  useSegmentationModels,
  useQuickRetrain,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { regionTileUrl } from "@/api/client";
import type {
  EventSegmentationJob,
  EventBoundaryCorrectionItem,
  EventBoundaryCorrectionResponse,
  Region,
} from "@/api/types";
import { toast } from "@/components/ui/use-toast";
import { useTimelineContext } from "@/components/timeline/provider/useTimelineContext";
import { REVIEW_ZOOM } from "@/components/timeline/provider/types";
import type { TimelinePlaybackHandle } from "@/components/timeline/provider/types";
import { Spectrogram } from "@/components/timeline/spectrogram/Spectrogram";
import { RegionBoundaryMarkers } from "@/components/timeline/overlays/RegionBoundaryMarkers";
import { RegionBandOverlay } from "@/components/timeline/overlays/RegionBandOverlay";
import { EventBarOverlay, type EffectiveEvent } from "@/components/timeline/overlays/EventBarOverlay";
import { ZoomSelector } from "@/components/timeline/controls/ZoomSelector";
import { EventDetailPanel } from "./EventDetailPanel";
import { RegionAudioTimeline } from "./RegionAudioTimeline";
import { RegionTable } from "./RegionTable";
import { ReviewToolbar, type RetrainStatus } from "./ReviewToolbar";
import { useRegionEpoch, type RegionEpoch } from "./useRegionEpoch";

export function SegmentReviewWorkspace({
  initialJobId,
}: {
  initialJobId?: string;
}) {
  const { data: segJobs = [] } = useSegmentationJobs();
  const { data: regionJobs = [] } = useRegionDetectionJobs();
  const { data: hydrophones = [] } = useHydrophones();
  const { data: segModels = [], refetch: refetchModels } =
    useSegmentationModels();

  const completeJobs = useMemo(
    () => segJobs.filter((j) => j.status === "complete"),
    [segJobs],
  );

  const [selectedJobId, setSelectedJobId] = useState<string | null>(
    initialJobId ?? null,
  );

  // Auto-select initialJobId when jobs load
  useEffect(() => {
    if (initialJobId && completeJobs.some((j) => j.id === initialJobId)) {
      setSelectedJobId(initialJobId);
    }
  }, [initialJobId, completeJobs]);

  const selectedJob = completeJobs.find((j) => j.id === selectedJobId) ?? null;
  const regionDetectionJobId = selectedJob?.region_detection_job_id ?? null;
  const regionEpoch = useRegionEpoch(regionDetectionJobId);

  const { data: regions = [] } = useRegionJobRegions(regionDetectionJobId);
  const { data: events = [] } = useSegmentationJobEvents(selectedJobId);
  const { data: savedCorrections = [] } = useEventBoundaryCorrections(
    regionDetectionJobId,
    selectedJobId,
  );

  const timelineExtent = useMemo(() => {
    if (regions.length === 0) return undefined;
    const start = Math.min(...regions.map((r) => r.padded_start_sec));
    const end = Math.max(...regions.map((r) => r.padded_end_sec));
    return { start, end };
  }, [regions]);

  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [addMode, setAddMode] = useState(false);
  const [viewStart, setViewStart] = useState<number | undefined>(undefined);
  const [viewSpan, setViewSpan] = useState(60);
  const scrollSeqRef = useRef(0);
  const [scrollToCenter, setScrollToCenter] = useState<
    { target: number; seq: number } | undefined
  >(undefined);

  // Playback via TimelineProvider ref handle
  const playbackRef = useRef<TimelinePlaybackHandle>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [userZoom, setUserZoom] = useState("1m");

  // Pending corrections: Map<eventId, EventBoundaryCorrectionItem>
  const [pendingCorrections, setPendingCorrections] = useState<
    Map<string, EventBoundaryCorrectionItem>
  >(new Map());

  const isDirty = pendingCorrections.size > 0;
  const pendingChangeCount = pendingCorrections.size;

  // Reset state when job changes
  useEffect(() => {
    setPendingCorrections(new Map());
    setSelectedEventId(null);
    setCurrentEventIndex(0);
    setAddMode(false);
    setActiveTrainingJobId(null);
    setRetrainError(null);
  }, [selectedJobId]);

  // Auto-select first region when regions load or job changes
  useEffect(() => {
    if (regions.length > 0) {
      setSelectedRegionId(regions[0].region_id);
    } else {
      setSelectedRegionId(null);
    }
  }, [regions]);

  const selectedRegion =
    regions.find((r) => r.region_id === selectedRegionId) ?? null;

  // Build effective events: merge original events + saved corrections + pending corrections
  const effectiveEvents: EffectiveEvent[] = useMemo(() => {
    // Index saved corrections by (region_id, original_start, original_end) for adjust/delete
    const savedByKey = new Map<string, EventBoundaryCorrectionResponse>();
    const savedByEventId = new Map<string, EventBoundaryCorrectionResponse>();
    for (const c of savedCorrections) {
      if (c.correction_type !== "add") {
        const key = `${c.region_id}:${c.original_start_sec}:${c.original_end_sec}`;
        savedByKey.set(key, c);
        if (c.source_event_id) savedByEventId.set(c.source_event_id, c);
      }
    }

    const noType = { effectiveType: null, typeSource: null } as const;

    const result: EffectiveEvent[] = events.map((ev) => {
      const pending = pendingCorrections.get(ev.event_id);
      const savedKey = `${ev.region_id}:${ev.start_sec}:${ev.end_sec}`;
      const saved = savedByEventId.get(ev.event_id) ?? savedByKey.get(savedKey);

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
          ...noType,
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
          ...noType,
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
        ...noType,
      };
    });

    // Add saved "add" corrections (they have no corresponding original event)
    for (const corr of savedCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.corrected_start_sec != null &&
        corr.corrected_end_sec != null
      ) {
        const addId = `saved-add-${corr.id}`;
        if (!pendingCorrections.has(addId)) {
          result.push({
            eventId: addId,
            regionId: corr.region_id,
            startSec: corr.corrected_start_sec,
            endSec: corr.corrected_end_sec,
            originalStartSec: corr.corrected_start_sec,
            originalEndSec: corr.corrected_end_sec,
            confidence: 0,
            correctionType: "add",
            ...noType,
          });
        }
      }
    }

    // Add pending "add" corrections (they have no corresponding event)
    for (const [key, corr] of pendingCorrections) {
      if (corr.correction_type === "add" && corr.corrected_start_sec != null && corr.corrected_end_sec != null) {
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
  }, [events, savedCorrections, pendingCorrections]);

  // Filter to current region
  const regionEvents = useMemo(
    () => effectiveEvents.filter((e) => e.regionId === selectedRegionId),
    [effectiveEvents, selectedRegionId],
  );

  // Flat navigable events across all regions (excludes deleted), sorted by startSec
  const navigableEvents = useMemo(
    () =>
      effectiveEvents
        .filter((e) => e.correctionType !== "delete")
        .sort((a, b) => a.startSec - b.startSec),
    [effectiveEvents],
  );

  const [currentEventIndex, setCurrentEventIndex] = useState(0);

  // Clamp index when navigable list shrinks
  useEffect(() => {
    if (navigableEvents.length > 0 && currentEventIndex >= navigableEvents.length) {
      setCurrentEventIndex(navigableEvents.length - 1);
    }
  }, [navigableEvents.length, currentEventIndex]);

  // Derive selected event from index
  const currentNavEvent = navigableEvents[currentEventIndex] ?? null;

  // Sync selectedEventId with currentNavEvent (one-way: index drives selection)
  useEffect(() => {
    if (currentNavEvent) {
      setSelectedEventId(currentNavEvent.eventId);
    }
  }, [currentNavEvent]);

  // Track whether the region change was caused by event navigation (skip auto-select)
  const regionChangeFromNavRef = useRef(false);

  // Auto-select first event when region changes via region buttons (not event nav)
  const prevRegionIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (!selectedRegionId || selectedRegionId === prevRegionIdRef.current) return;
    prevRegionIdRef.current = selectedRegionId;
    if (regionChangeFromNavRef.current) {
      regionChangeFromNavRef.current = false;
      return;
    }
    // Find the first navigable event in this region and set the index
    const idx = navigableEvents.findIndex((e) => e.regionId === selectedRegionId);
    if (idx >= 0) {
      setCurrentEventIndex(idx);
    } else {
      setSelectedEventId(null);
    }
  }, [selectedRegionId, navigableEvents]);

  // Navigation callbacks — cross-region navigation switches the active region
  const navDirectionRef = useRef<"forward" | "backward">("forward");

  const goPrevEvent = useCallback(() => {
    navDirectionRef.current = "backward";
    setCurrentEventIndex((prev) => {
      const next = Math.max(0, prev - 1);
      const event = navigableEvents[next];
      if (event && event.regionId !== selectedRegionId) {
        regionChangeFromNavRef.current = true;
        setSelectedRegionId(event.regionId);
      }
      return next;
    });
  }, [navigableEvents, selectedRegionId]);
  const goNextEvent = useCallback(() => {
    navDirectionRef.current = "forward";
    setCurrentEventIndex((prev) => {
      const next = Math.min(navigableEvents.length - 1, prev + 1);
      const event = navigableEvents[next];
      if (event && event.regionId !== selectedRegionId) {
        regionChangeFromNavRef.current = true;
        setSelectedRegionId(event.regionId);
      }
      return next;
    });
  }, [navigableEvents, selectedRegionId]);

  // Auto-scroll spectrogram when current event is not fully visible
  useEffect(() => {
    if (!currentNavEvent || viewStart === undefined || !regionEpoch) return;
    const viewEnd = viewStart + viewSpan;
    const pad = viewSpan * 0.15;
    const eventStartEpoch = regionEpoch.toEpoch(currentNavEvent.startSec);
    const eventEndEpoch = regionEpoch.toEpoch(currentNavEvent.endSec);
    const fullyVisible =
      eventStartEpoch >= viewStart + pad && eventEndEpoch <= viewEnd - pad;
    if (!fullyVisible) {
      let target: number;
      if (navDirectionRef.current === "forward") {
        target = eventEndEpoch + pad - viewSpan / 2;
      } else {
        target = eventStartEpoch - pad + viewSpan / 2;
      }
      scrollSeqRef.current += 1;
      setScrollToCenter({ target, seq: scrollSeqRef.current });
    }
  }, [currentNavEvent, viewStart, viewSpan, regionEpoch]);

  // Callbacks for overlay
  const handleAdjust = useCallback(
    (eventId: string, startSec: number, endSec: number) => {
      setPendingCorrections((prev) => {
        const next = new Map(prev);
        const existing = prev.get(eventId);
        const ev = events.find((e) => e.event_id === eventId);
        const effectiveEv = effectiveEvents.find((e) => e.eventId === eventId);
        const isAdd = existing?.correction_type === "add" || effectiveEv?.correctionType === "add";
        const regionId = ev?.region_id ?? existing?.region_id ?? selectedRegionId ?? "";
        const savedAddId = eventId.startsWith("saved-add-")
          ? eventId.slice("saved-add-".length)
          : null;
        if (isAdd) {
          next.set(eventId, {
            id: existing?.id ?? savedAddId,
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
            source_event_id: ev?.event_id ?? eventId,
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
    [events, effectiveEvents, selectedRegionId],
  );

  const handleAdd = useCallback(
    (regionId: string, startSec: number, endSec: number) => {
      const tempId = `add-${crypto.randomUUID()}`;
      setPendingCorrections((prev) => {
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

  const handleDelete = useCallback(
    (eventId: string) => {
      setPendingCorrections((prev) => {
        const next = new Map(prev);
        const existing = next.get(eventId);
        if (existing?.correction_type === "delete") {
          next.delete(eventId);
        } else if (existing?.correction_type === "add") {
          next.delete(eventId);
        } else if (eventId.startsWith("saved-add-")) {
          next.set(eventId, {
            id: eventId.slice("saved-add-".length),
            region_id: selectedRegionId ?? "",
            correction_type: "delete",
            original_start_sec: null,
            original_end_sec: null,
            corrected_start_sec: null,
            corrected_end_sec: null,
          });
        } else {
          const ev = events.find((e) => e.event_id === eventId);
          next.set(eventId, {
            region_id: ev?.region_id ?? selectedRegionId ?? "",
            source_event_id: ev?.event_id ?? eventId,
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
    [events, selectedRegionId],
  );

  const saveMutation = useUpsertEventBoundaryCorrections();

  const handleToggleAddMode = useCallback(() => {
    setAddMode((prev) => !prev);
  }, []);

  const handleSave = useCallback(() => {
    if (!regionDetectionJobId || pendingCorrections.size === 0) return;
    const activeEvents = effectiveEvents
      .filter((event) => event.correctionType !== "delete")
      .sort((a, b) =>
        a.regionId === b.regionId
          ? a.startSec - b.startSec || a.endSec - b.endSec
          : a.regionId.localeCompare(b.regionId),
      );
    for (let i = 1; i < activeEvents.length; i += 1) {
      const prev = activeEvents[i - 1];
      const current = activeEvents[i];
      if (prev.regionId === current.regionId && prev.endSec > current.startSec) {
        toast({
          title: "Overlapping events",
          description: "Adjust the event boundaries before saving.",
          variant: "destructive",
        });
        return;
      }
    }
    const corrections = Array.from(pendingCorrections.values());
    if (!selectedJobId) return;
    saveMutation.mutate(
      { regionDetectionJobId, eventSegmentationJobId: selectedJobId, corrections },
      {
        onSuccess: () => {
          setPendingCorrections(new Map());
          setSelectedEventId(null);
        },
      },
    );
  }, [
    effectiveEvents,
    regionDetectionJobId,
    selectedJobId,
    pendingCorrections,
    saveMutation,
  ]);

  const handleCancel = useCallback(() => {
    setPendingCorrections(new Map());
    setSelectedEventId(null);
    setAddMode(false);
  }, []);

  // Switch region — edits accumulate across regions within the job
  const handleSelectRegion = useCallback((regionId: string) => {
    setAddMode(false);
    setSelectedRegionId(regionId);
  }, []);

  // Find the selected effective event for the detail panel
  const selectedEvent =
    effectiveEvents.find((e) => e.eventId === selectedEventId) ?? null;

  // Keyboard shortcuts
  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      playbackRef.current?.pause();
    } else if (selectedEvent && regionEpoch) {
      const duration = selectedEvent.endSec - selectedEvent.startSec;
      playbackRef.current?.play(
        regionEpoch.toEpoch(selectedEvent.startSec),
        duration,
      );
    } else if (selectedRegion && regionEpoch) {
      // viewStart is already in epoch space when the timeline is mounted;
      // padded_start_sec is region-relative and needs translation.
      const playStart =
        viewStart ?? regionEpoch.toEpoch(selectedRegion.padded_start_sec);
      const regionEndEpoch = regionEpoch.toEpoch(
        selectedRegion.padded_end_sec,
      );
      const duration = Math.min(regionEndEpoch - playStart, 30);
      playbackRef.current?.play(playStart, duration);
    }
  }, [isPlaying, selectedEvent, selectedRegion, viewStart, regionEpoch]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      const tag = el.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      switch (e.code) {
        case "Space":
          e.preventDefault();
          togglePlayback();
          break;
        case "KeyA":
          e.preventDefault();
          goPrevEvent();
          break;
        case "KeyD":
          e.preventDefault();
          goNextEvent();
          break;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [togglePlayback, goPrevEvent, goNextEvent]);

  // Resolve source label for the selected job
  const sourceLabel = useMemo(() => {
    if (!selectedJob) return "";
    const rj = regionJobs.find(
      (r) => r.id === selectedJob.region_detection_job_id,
    );
    if (!rj) return selectedJob.region_detection_job_id.slice(0, 8);
    if (rj.hydrophone_id) {
      const h = hydrophones.find((hp) => hp.id === rj.hydrophone_id);
      return h?.name ?? rj.hydrophone_id;
    }
    return rj.audio_file_id?.slice(0, 8) ?? "unknown";
  }, [selectedJob, regionJobs, hydrophones]);

  // ---- Retrain / Re-segment ----

  const hasCorrections = savedCorrections.length > 0;

  const [activeTrainingJobId, setActiveTrainingJobId] = useState<string | null>(
    null,
  );
  const [retrainError, setRetrainError] = useState<string | null>(null);

  const isPolling = activeTrainingJobId !== null && retrainError === null;

  const retrainStatus: RetrainStatus | null = useMemo(() => {
    if (retrainError) {
      return { status: "failed", error: retrainError };
    }
    if (!activeTrainingJobId) return null;
    const model = segModels.find(
      (m) => m.training_job_id === activeTrainingJobId,
    );
    if (model) {
      return { status: "complete", modelId: model.id, modelName: model.name };
    }
    return { status: "running" };
  }, [activeTrainingJobId, retrainError, segModels]);

  // Poll models at 3s while training is in progress
  useEffect(() => {
    if (!isPolling) return;
    const interval = setInterval(() => {
      void refetchModels();
    }, 3000);
    return () => clearInterval(interval);
  }, [isPolling, refetchModels]);

  const quickRetrain = useQuickRetrain();
  const createSegJob = useCreateSegmentationJob();

  const handleRetrain = useCallback(() => {
    if (!selectedJobId) return;
    const ok = window.confirm(
      "Train a new segmentation model from corrections on this job?",
    );
    if (!ok) return;
    setRetrainError(null);
    quickRetrain.mutate(
      { segmentation_job_id: selectedJobId },
      {
        onSuccess: (data) => {
          setActiveTrainingJobId(data.training_job_id);
          toast({
            title: "Training job started",
            description: `${data.sample_count} samples. The model will train in the background.`,
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
  }, [selectedJobId, quickRetrain]);

  const handleResegment = useCallback(() => {
    if (!retrainStatus?.modelId || !regionDetectionJobId) return;
    const modelName = retrainStatus.modelName ?? retrainStatus.modelId.slice(0, 8);
    const ok = window.confirm(
      `Create a new segmentation job using model "${modelName}" on the same regions?`,
    );
    if (!ok) return;
    createSegJob.mutate(
      {
        region_detection_job_id: regionDetectionJobId,
        segmentation_model_id: retrainStatus.modelId,
      },
      {
        onSuccess: () => {
          toast({
            title: "Segmentation job created",
            description:
              "It will appear in the job selector when complete.",
          });
        },
        onError: (err) => {
          toast({
            title: "Failed to create segmentation job",
            description: (err as Error).message,
            variant: "destructive",
          });
        },
      },
    );
  }, [retrainStatus, regionDetectionJobId, createSegJob]);

  const tileUrlBuilder = useCallback(
    (_jobId: string, zoomLevel: string, tileIndex: number) =>
      regionTileUrl(regionDetectionJobId ?? "", zoomLevel, tileIndex),
    [regionDetectionJobId],
  );

  return (
    <div className="space-y-4">
      {/* Job selector */}
      <div className="flex items-center gap-4">
        <label htmlFor="review-job-select" className="text-sm font-medium">
          Job
        </label>
        <select
          id="review-job-select"
          className="rounded-md border bg-background px-3 py-2 text-sm"
          value={selectedJobId ?? ""}
          onChange={(e) => setSelectedJobId(e.target.value || null)}
        >
          <option value="">Select a completed segmentation job</option>
          {completeJobs.map((j) => (
            <option key={j.id} value={j.id}>
              {jobLabel(j, regionJobs, hydrophones)} — {j.event_count ?? 0}{" "}
              events
            </option>
          ))}
        </select>
        {selectedJob && (
          <span className="text-xs text-muted-foreground">
            Source: {sourceLabel}
          </span>
        )}
        {isDirty && (
          <span className="text-xs text-yellow-500">
            {pendingChangeCount} unsaved change
            {pendingChangeCount !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Workspace body */}
      {selectedJob && selectedRegion && timelineExtent && regionEpoch ? (
        <div className="rounded-md border">
          <ReviewToolbar
            region={selectedRegion}
            eventCount={regionEvents.length}
            pendingChangeCount={pendingChangeCount}
            isDirty={isDirty}
            addMode={addMode}
            onToggleAddMode={handleToggleAddMode}
            onSave={handleSave}
            onCancel={handleCancel}
            isPlaying={isPlaying}
            onPlay={() => {
              if (isPlaying) {
                playbackRef.current?.pause();
                return;
              }
              if (!selectedRegion) return;
              const playStart =
                viewStart ?? regionEpoch.toEpoch(selectedRegion.padded_start_sec);
              const regionEndEpoch = regionEpoch.toEpoch(
                selectedRegion.padded_end_sec,
              );
              const duration = Math.min(regionEndEpoch - playStart, 30);
              playbackRef.current?.play(playStart, duration);
            }}
            hasCorrections={hasCorrections}
            onRetrain={handleRetrain}
            retrainStatus={retrainStatus}
            onResegment={handleResegment}
            regions={regions}
            selectedRegionId={selectedRegionId}
            onPrevRegion={() => {
              const idx = regions.findIndex((r) => r.region_id === selectedRegionId);
              if (idx > 0) handleSelectRegion(regions[idx - 1].region_id);
            }}
            onNextRegion={() => {
              const idx = regions.findIndex((r) => r.region_id === selectedRegionId);
              if (idx >= 0 && idx < regions.length - 1) handleSelectRegion(regions[idx + 1].region_id);
            }}
            onPrevEvent={goPrevEvent}
            onNextEvent={goNextEvent}
            currentEventIndex={currentEventIndex}
            totalEventCount={navigableEvents.length}
            jobStartEpoch={regionEpoch.regionStartTimestamp}
          />
          <RegionAudioTimeline
            ref={playbackRef}
            regionDetectionJobId={regionDetectionJobId!}
            regionEpoch={regionEpoch}
            resetKey={`${selectedJobId}-${selectedRegionId}`}
            zoomLevels={REVIEW_ZOOM}
            defaultZoom={userZoom}
            disableKeyboardShortcuts
            scrollOnPlayback={false}
            onZoomChange={setUserZoom}
            onPlayStateChange={setIsPlaying}
          >
            <SegmentViewerBody
              regionDetectionJobId={regionDetectionJobId!}
              region={selectedRegion}
              regions={regions}
              selectedRegionId={selectedRegionId!}
              onSelectRegion={handleSelectRegion}
              regionEvents={regionEvents}
              selectedEventId={selectedEventId}
              onSelectEvent={(eventId) => {
                setSelectedEventId(eventId);
                const idx = navigableEvents.findIndex((e) => e.eventId === eventId);
                if (idx >= 0) setCurrentEventIndex(idx);
              }}
              onAdjust={handleAdjust}
              onAdd={handleAdd}
              addMode={addMode}
              scrollToCenter={scrollToCenter}
              onViewStartChange={setViewStart}
              onViewSpanChange={setViewSpan}
              tileUrlBuilder={tileUrlBuilder}
              regionEpoch={regionEpoch}
            />
          </RegionAudioTimeline>
          <EventDetailPanel
            event={selectedEvent}
            onDelete={handleDelete}
            isPlaying={isPlaying}
            onPlaySlice={() => {
              if (isPlaying) {
                playbackRef.current?.pause();
                return;
              }
              if (!selectedEvent) return;
              const duration = selectedEvent.endSec - selectedEvent.startSec;
              playbackRef.current?.play(
                regionEpoch.toEpoch(selectedEvent.startSec),
                duration,
              );
            }}
            jobStartEpoch={regionEpoch.regionStartTimestamp}
          />
          <RegionTable
            regions={regions}
            events={events}
            corrections={savedCorrections}
            selectedRegionId={selectedRegionId}
            onSelectRegion={handleSelectRegion}
            jobStartEpoch={regionEpoch.regionStartTimestamp}
          />
        </div>
      ) : selectedJob ? (
        <div className="rounded-md border">
          <div className="p-4 text-center text-sm text-muted-foreground">
            {regionEpoch
              ? "Select a region to view its spectrogram"
              : "Loading region…"}
          </div>
          <RegionTable
            regions={regions}
            events={events}
            corrections={savedCorrections}
            selectedRegionId={selectedRegionId}
            onSelectRegion={handleSelectRegion}
            jobStartEpoch={regionEpoch?.regionStartTimestamp ?? 0}
          />
        </div>
      ) : (
        <div className="py-8 text-center text-muted-foreground">
          Select a completed segmentation job to begin reviewing event
          boundaries.
        </div>
      )}
    </div>
  );
}

interface SegmentViewerBodyProps {
  regionDetectionJobId: string;
  region: Region;
  regions: Region[];
  selectedRegionId: string;
  onSelectRegion: (regionId: string) => void;
  regionEvents: EffectiveEvent[];
  selectedEventId: string | null;
  onSelectEvent: (eventId: string | null) => void;
  onAdjust: (eventId: string, startSec: number, endSec: number) => void;
  onAdd: (regionId: string, startSec: number, endSec: number) => void;
  addMode: boolean;
  scrollToCenter: { target: number; seq: number } | undefined;
  onViewStartChange: (viewStart: number) => void;
  onViewSpanChange: (viewSpan: number) => void;
  tileUrlBuilder: (jobId: string, zoomLevel: string, tileIndex: number, freqMin: number, freqMax: number) => string;
  regionEpoch: RegionEpoch;
}

function SegmentViewerBody({
  regionDetectionJobId,
  region,
  regions,
  selectedRegionId,
  onSelectRegion,
  regionEvents,
  selectedEventId,
  onSelectEvent,
  onAdjust,
  onAdd,
  addMode,
  scrollToCenter,
  onViewStartChange,
  onViewSpanChange,
  tileUrlBuilder,
  regionEpoch,
}: SegmentViewerBodyProps) {
  const ctx = useTimelineContext();
  const regionStart = regionEpoch.regionStartTimestamp;

  // Re-center when region changes
  const prevRegionRef = useRef<string>(region.region_id);
  useEffect(() => {
    if (region.region_id !== prevRegionRef.current) {
      prevRegionRef.current = region.region_id;
      const dur = region.padded_end_sec - region.padded_start_sec;
      const span = ctx.activePreset.span;
      ctx.seekTo(
        regionStart + region.padded_start_sec + Math.min(dur, span) / 2,
      );
    }
  }, [
    region.region_id,
    region.padded_start_sec,
    region.padded_end_sec,
    ctx,
    regionStart,
  ]);

  // External scroll-to-center (for event navigation)
  const scrollSeqRef = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!scrollToCenter || scrollToCenter.seq === scrollSeqRef.current) return;
    scrollSeqRef.current = scrollToCenter.seq;
    ctx.seekTo(scrollToCenter.target);
  }, [scrollToCenter, ctx]);

  // Report viewStart/viewSpan to parent for event visibility checks
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

  // Show region bands at wide zoom (1m or 5m)
  const showBands = ctx.activePreset.key === "5m" || ctx.activePreset.key === "1m";

  // Translate region/event coordinates from job-relative to absolute epoch
  // before handing them to overlays, which operate in the timeline provider's
  // (now epoch) coordinate space.
  const regionsEpoch = useMemo<Region[]>(
    () =>
      regions.map((r) => ({
        ...r,
        start_sec: regionStart + r.start_sec,
        end_sec: regionStart + r.end_sec,
        padded_start_sec: regionStart + r.padded_start_sec,
        padded_end_sec: regionStart + r.padded_end_sec,
      })),
    [regions, regionStart],
  );
  const regionEventsEpoch = useMemo<EffectiveEvent[]>(
    () =>
      regionEvents.map((e) => ({
        ...e,
        startSec: regionStart + e.startSec,
        endSec: regionStart + e.endSec,
        originalStartSec: regionStart + e.originalStartSec,
        originalEndSec: regionStart + e.originalEndSec,
      })),
    [regionEvents, regionStart],
  );
  const handleAdjustEpoch = useCallback(
    (eventId: string, startEpoch: number, endEpoch: number) => {
      onAdjust(eventId, startEpoch - regionStart, endEpoch - regionStart);
    },
    [onAdjust, regionStart],
  );
  const handleAddEpoch = useCallback(
    (regionId: string, startEpoch: number, endEpoch: number) => {
      onAdd(regionId, startEpoch - regionStart, endEpoch - regionStart);
    },
    [onAdd, regionStart],
  );

  return (
    <div className="w-full select-none">
      <div className="flex flex-col" style={{ height: 240 }}>
        <Spectrogram
          jobId={regionDetectionJobId}
          tileUrlBuilder={tileUrlBuilder}
          freqRange={[0, 3000]}
        >
          <RegionBoundaryMarkers
            startEpoch={regionStart + region.start_sec}
            endEpoch={regionStart + region.end_sec}
          />
          {showBands && (
            <RegionBandOverlay
              regions={regionsEpoch}
              activeRegionId={selectedRegionId}
              onSelectRegion={onSelectRegion}
            />
          )}
          <EventBarOverlay
            events={regionEventsEpoch}
            selectedEventId={selectedEventId}
            onSelectEvent={onSelectEvent}
            onAdjust={handleAdjustEpoch}
            onAdd={handleAddEpoch}
            addMode={addMode}
            activeRegionId={selectedRegionId}
          />
        </Spectrogram>
      </div>
      <div className="border-t border-border px-2 py-1">
        <ZoomSelector />
      </div>
    </div>
  );
}

function jobLabel(
  job: EventSegmentationJob,
  regionJobs: {
    id: string;
    hydrophone_id: string | null;
    audio_file_id: string | null;
  }[],
  hydrophones: { id: string; name: string }[],
): string {
  const shortId = job.id.slice(0, 8);
  const rj = regionJobs.find((r) => r.id === job.region_detection_job_id);
  if (rj?.hydrophone_id) {
    const h = hydrophones.find((hp) => hp.id === rj.hydrophone_id);
    const name = h?.name ?? rj.hydrophone_id;
    return `${name} - ${shortId}`;
  }
  const source = rj?.audio_file_id?.slice(0, 8) ?? shortId;
  if (source === shortId) return shortId;
  return `${source} - ${shortId}`;
}
