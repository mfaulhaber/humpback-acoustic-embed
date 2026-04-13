import { useCallback, useEffect, useMemo, useState } from "react";
import {
  useSegmentationJobs,
  useSegmentationJobEvents,
  useRegionDetectionJobs,
  useRegionJobRegions,
  useBoundaryCorrections,
  useSaveBoundaryCorrections,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import type {
  EventSegmentationJob,
  BoundaryCorrection,
} from "@/api/types";
import { EventDetailPanel } from "./EventDetailPanel";
import { RegionSidebar } from "./RegionSidebar";
import { RegionSpectrogramViewer } from "./RegionSpectrogramViewer";
import { ReviewToolbar } from "./ReviewToolbar";
import { EventBarOverlay, type EffectiveEvent } from "./EventBarOverlay";

export function SegmentReviewWorkspace({
  initialJobId,
}: {
  initialJobId?: string;
}) {
  const { data: segJobs = [] } = useSegmentationJobs();
  const { data: regionJobs = [] } = useRegionDetectionJobs();
  const { data: hydrophones = [] } = useHydrophones();

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

  // Callbacks for overlay
  const handleAdjust = useCallback(
    (eventId: string, startSec: number, endSec: number) => {
      setPendingCorrections((prev) => {
        const next = new Map(prev);
        const ev = events.find((e) => e.event_id === eventId);
        next.set(eventId, {
          event_id: eventId,
          region_id: ev?.region_id ?? selectedRegionId ?? "",
          correction_type: "adjust",
          start_sec: startSec,
          end_sec: endSec,
        });
        return next;
      });
    },
    [events, selectedRegionId],
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
    setSelectedEventId(null);
    setAddMode(false);
    setSelectedRegionId(regionId);
  }, []);

  // Find the selected effective event for the detail panel
  const selectedEvent =
    effectiveEvents.find((e) => e.eventId === selectedEventId) ?? null;

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
        <div className="flex gap-4">
          <RegionSidebar
            regions={regions}
            events={events}
            corrections={savedCorrections}
            selectedRegionId={selectedRegionId}
            onSelectRegion={handleSelectRegion}
          />
          <div className="flex-1 rounded-md border">
            <ReviewToolbar
              region={selectedRegion}
              regionJobId={regionDetectionJobId}
              eventCount={regionEvents.length}
              pendingChangeCount={pendingChangeCount}
              isDirty={isDirty}
              addMode={addMode}
              onToggleAddMode={handleToggleAddMode}
              onSave={handleSave}
              onCancel={handleCancel}
              viewStart={viewStart}
            />
            {selectedRegion ? (
              <>
                <RegionSpectrogramViewer
                  regionJobId={regionDetectionJobId!}
                  region={selectedRegion}
                  onViewStartChange={setViewStart}
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
                  regionJobId={regionDetectionJobId}
                  onDelete={handleDelete}
                />
              </>
            ) : (
              <div className="p-4 text-center text-sm text-muted-foreground">
                Select a region to view its spectrogram
              </div>
            )}
          </div>
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

function jobLabel(
  job: EventSegmentationJob,
  regionJobs: {
    id: string;
    hydrophone_id: string | null;
    audio_file_id: string | null;
  }[],
  hydrophones: { id: string; name: string }[],
): string {
  const rj = regionJobs.find((r) => r.id === job.region_detection_job_id);
  if (rj?.hydrophone_id) {
    const h = hydrophones.find((hp) => hp.id === rj.hydrophone_id);
    return h?.name ?? rj.hydrophone_id;
  }
  return rj?.audio_file_id?.slice(0, 8) ?? job.id.slice(0, 8);
}
