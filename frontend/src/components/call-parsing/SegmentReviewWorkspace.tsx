import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useSegmentationJobs,
  useSegmentationJobEvents,
  useRegionDetectionJobs,
  useRegionJobRegions,
  useBoundaryCorrections,
  useSaveBoundaryCorrections,
  useSegmentationFeedbackTrainingJobs,
  useCreateSegmentationFeedbackTrainingJob,
  useCreateSegmentationJob,
  useSegmentationModels,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { regionAudioSliceUrl } from "@/api/client";
import type {
  EventSegmentationJob,
  BoundaryCorrection,
} from "@/api/types";
import { toast } from "@/components/ui/use-toast";
import { EventDetailPanel } from "./EventDetailPanel";
import { RegionTable } from "./RegionTable";
import { RegionSpectrogramViewer } from "./RegionSpectrogramViewer";
import { ReviewToolbar, type RetrainStatus } from "./ReviewToolbar";
import { EventBarOverlay, type EffectiveEvent } from "./EventBarOverlay";

export function SegmentReviewWorkspace({
  initialJobId,
}: {
  initialJobId?: string;
}) {
  const { data: segJobs = [] } = useSegmentationJobs();
  const { data: regionJobs = [] } = useRegionDetectionJobs();
  const { data: hydrophones = [] } = useHydrophones();
  const { data: segModels = [] } = useSegmentationModels();

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

  const { data: regions = [] } = useRegionJobRegions(regionDetectionJobId);
  const { data: events = [] } = useSegmentationJobEvents(selectedJobId);
  const { data: savedCorrections = [] } = useBoundaryCorrections(selectedJobId);

  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [addMode, setAddMode] = useState(false);
  const [viewStart, setViewStart] = useState<number | undefined>(undefined);

  // Shared audio playback state
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackOriginSec, setPlaybackOriginSec] = useState(0);

  const startPlayback = useCallback(
    (startSec: number, duration: number) => {
      const audio = audioRef.current;
      if (!audio || !regionDetectionJobId) return;
      // Stop any current playback first
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

  // Pending corrections: Map<eventId, BoundaryCorrection>
  const [pendingCorrections, setPendingCorrections] = useState<
    Map<string, BoundaryCorrection>
  >(new Map());

  const isDirty = pendingCorrections.size > 0;
  const pendingChangeCount = pendingCorrections.size;

  // Reset state when job changes
  useEffect(() => {
    setPendingCorrections(new Map());
    setSelectedEventId(null);
    setAddMode(false);
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
    const savedMap = new Map(
      savedCorrections.map((c) => [c.event_id, c]),
    );

    const result: EffectiveEvent[] = events.map((ev) => {
      const pending = pendingCorrections.get(ev.event_id);
      const saved = savedMap.get(ev.event_id);

      if (pending) {
        return {
          eventId: ev.event_id,
          regionId: ev.region_id,
          startSec: pending.start_sec ?? ev.start_sec,
          endSec: pending.end_sec ?? ev.end_sec,
          originalStartSec: ev.start_sec,
          originalEndSec: ev.end_sec,
          confidence: ev.segmentation_confidence,
          correctionType: pending.correction_type,
        };
      }

      if (saved) {
        return {
          eventId: ev.event_id,
          regionId: ev.region_id,
          startSec: saved.start_sec ?? ev.start_sec,
          endSec: saved.end_sec ?? ev.end_sec,
          originalStartSec: ev.start_sec,
          originalEndSec: ev.end_sec,
          confidence: ev.segmentation_confidence,
          correctionType: saved.correction_type as EffectiveEvent["correctionType"],
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
      };
    });

    // Add saved "add" corrections (they have no corresponding original event)
    const originalEventIds = new Set(events.map((e) => e.event_id));
    for (const corr of savedCorrections) {
      if (
        corr.correction_type === "add" &&
        corr.start_sec != null &&
        corr.end_sec != null &&
        !originalEventIds.has(corr.event_id) &&
        !pendingCorrections.has(corr.event_id)
      ) {
        result.push({
          eventId: corr.event_id,
          regionId: corr.region_id,
          startSec: corr.start_sec,
          endSec: corr.end_sec,
          originalStartSec: corr.start_sec,
          originalEndSec: corr.end_sec,
          confidence: 0,
          correctionType: "add",
        });
      }
    }

    // Add pending "add" corrections (they have no corresponding event)
    for (const [key, corr] of pendingCorrections) {
      if (corr.correction_type === "add" && corr.start_sec != null && corr.end_sec != null) {
        result.push({
          eventId: key,
          regionId: corr.region_id,
          startSec: corr.start_sec,
          endSec: corr.end_sec,
          originalStartSec: corr.start_sec,
          originalEndSec: corr.end_sec,
          confidence: 0,
          correctionType: "add",
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

  // Auto-select first event when region changes (not on every regionEvents update)
  const prevRegionIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (!selectedRegionId || selectedRegionId === prevRegionIdRef.current) return;
    prevRegionIdRef.current = selectedRegionId;
    const sorted = regionEvents
      .filter((e) => e.correctionType !== "delete")
      .sort((a, b) => a.startSec - b.startSec);
    setSelectedEventId(sorted.length > 0 ? sorted[0].eventId : null);
  }, [selectedRegionId, regionEvents]);

  // Callbacks for overlay
  const handleAdjust = useCallback(
    (eventId: string, startSec: number, endSec: number) => {
      setPendingCorrections((prev) => {
        const next = new Map(prev);
        const existing = prev.get(eventId);
        const saved = savedCorrections.find((c) => c.event_id === eventId);
        // Preserve "add" type when adjusting an added event (pending or saved)
        const isAdd =
          existing?.correction_type === "add" ||
          saved?.correction_type === "add";
        const correctionType = isAdd ? "add" : "adjust";
        const ev = events.find((e) => e.event_id === eventId);
        next.set(eventId, {
          event_id: eventId,
          region_id: ev?.region_id ?? existing?.region_id ?? saved?.region_id ?? selectedRegionId ?? "",
          correction_type: correctionType,
          start_sec: startSec,
          end_sec: endSec,
        });
        return next;
      });
    },
    [events, savedCorrections, selectedRegionId],
  );

  const handleAdd = useCallback(
    (regionId: string, startSec: number, endSec: number) => {
      const tempId = `add-${crypto.randomUUID()}`;
      setPendingCorrections((prev) => {
        const next = new Map(prev);
        next.set(tempId, {
          event_id: tempId,
          region_id: regionId,
          correction_type: "add",
          start_sec: startSec,
          end_sec: endSec,
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
          // Undo delete
          next.delete(eventId);
        } else if (existing?.correction_type === "add") {
          // Remove the add entirely
          next.delete(eventId);
        } else {
          const ev = events.find((e) => e.event_id === eventId);
          next.set(eventId, {
            event_id: eventId,
            region_id: ev?.region_id ?? selectedRegionId ?? "",
            correction_type: "delete",
            start_sec: null,
            end_sec: null,
          });
        }
        return next;
      });
    },
    [events, selectedRegionId],
  );

  const saveMutation = useSaveBoundaryCorrections();

  const handleToggleAddMode = useCallback(() => {
    setAddMode((prev) => !prev);
  }, []);

  const handleSave = useCallback(() => {
    if (!selectedJobId || pendingCorrections.size === 0) return;
    const corrections = Array.from(pendingCorrections.values());
    saveMutation.mutate(
      { jobId: selectedJobId, body: { corrections } },
      {
        onSuccess: () => {
          setPendingCorrections(new Map());
          setSelectedEventId(null);
        },
      },
    );
  }, [selectedJobId, pendingCorrections, saveMutation]);

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

  // Spacebar toggles playback on selected event
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      const el = e.target as HTMLElement;
      const tag = el.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      // Avoid double-action: space on a focused button fires both click and this handler
      if (tag === "BUTTON" || el.closest("button")) return;
      e.preventDefault();
      if (isPlaying) {
        stopPlayback();
      } else if (selectedEvent) {
        const duration = selectedEvent.endSec - selectedEvent.startSec;
        startPlayback(selectedEvent.startSec, duration);
      } else if (selectedRegion) {
        const playStart = viewStart ?? selectedRegion.padded_start_sec;
        const duration = Math.min(selectedRegion.padded_end_sec - playStart, 30);
        startPlayback(playStart, duration);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isPlaying, selectedEvent, selectedRegion, viewStart, startPlayback, stopPlayback]);

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

  // Poll feedback training jobs — only poll at 3s when a job is in progress
  const { data: feedbackJobs = [] } = useSegmentationFeedbackTrainingJobs();

  // Find the most recent feedback training job for this segmentation job
  const latestFeedbackJob = useMemo(() => {
    if (!selectedJobId) return null;
    const matching = feedbackJobs.filter((j) => {
      try {
        const ids = JSON.parse(j.source_job_ids) as string[];
        return ids.includes(selectedJobId);
      } catch {
        return false;
      }
    });
    if (matching.length === 0) return null;
    // Most recent by created_at
    return matching.sort(
      (a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
    )[0];
  }, [selectedJobId, feedbackJobs]);

  const feedbackJobActive =
    latestFeedbackJob?.status === "queued" ||
    latestFeedbackJob?.status === "running";

  // Re-fetch feedback jobs at 3s interval when a job is active
  useSegmentationFeedbackTrainingJobs(feedbackJobActive ? 3000 : undefined);

  const retrainStatus: RetrainStatus | null = useMemo(() => {
    if (!latestFeedbackJob) return null;
    const modelId = latestFeedbackJob.segmentation_model_id ?? undefined;
    const model = modelId
      ? segModels.find((m) => m.id === modelId)
      : undefined;
    return {
      status: latestFeedbackJob.status,
      modelId,
      modelName: model?.name,
      error: latestFeedbackJob.error_message ?? undefined,
    };
  }, [latestFeedbackJob, segModels]);

  const createFeedbackJob = useCreateSegmentationFeedbackTrainingJob();
  const createSegJob = useCreateSegmentationJob();

  const handleRetrain = useCallback(() => {
    if (!selectedJobId) return;
    const ok = window.confirm(
      "Train a new segmentation model from corrections on this job?",
    );
    if (!ok) return;
    createFeedbackJob.mutate(
      { source_job_ids: [selectedJobId] },
      {
        onSuccess: () => {
          toast({
            title: "Training job started",
            description:
              "The model will train in the background. Status will update here.",
          });
        },
        onError: (err) => {
          toast({
            title: "Failed to start training",
            description: (err as Error).message,
            variant: "destructive",
          });
        },
      },
    );
  }, [selectedJobId, createFeedbackJob]);

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
      {selectedJob ? (
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
                stopPlayback();
                return;
              }
              if (!selectedRegion) return;
              const playStart = viewStart ?? selectedRegion.padded_start_sec;
              const duration = Math.min(selectedRegion.padded_end_sec - playStart, 30);
              startPlayback(playStart, duration);
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
          />
          {selectedRegion ? (
            <>
              <RegionSpectrogramViewer
                regionJobId={regionDetectionJobId!}
                region={selectedRegion}
                onViewStartChange={setViewStart}
                audioRef={audioRef}
                isPlaying={isPlaying}
                playbackOriginSec={playbackOriginSec}
              >
                <EventBarOverlay
                  events={regionEvents}
                  selectedEventId={selectedEventId}
                  onSelectEvent={setSelectedEventId}
                  onAdjust={handleAdjust}
                  onAdd={handleAdd}
                  addMode={addMode}
                  activeRegionId={selectedRegionId!}
                />
              </RegionSpectrogramViewer>
              <EventDetailPanel
                event={selectedEvent}
                onDelete={handleDelete}
                isPlaying={isPlaying}
                onPlaySlice={() => {
                  if (isPlaying) {
                    stopPlayback();
                    return;
                  }
                  if (!selectedEvent) return;
                  const duration = selectedEvent.endSec - selectedEvent.startSec;
                  startPlayback(selectedEvent.startSec, duration);
                }}
              />
            </>
          ) : (
            <div className="p-4 text-center text-sm text-muted-foreground">
              Select a region to view its spectrogram
            </div>
          )}
          <RegionTable
            regions={regions}
            events={events}
            corrections={savedCorrections}
            selectedRegionId={selectedRegionId}
            onSelectRegion={handleSelectRegion}
          />
        </div>
      ) : (
        <div className="py-8 text-center text-muted-foreground">
          Select a completed segmentation job to begin reviewing event
          boundaries.
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
