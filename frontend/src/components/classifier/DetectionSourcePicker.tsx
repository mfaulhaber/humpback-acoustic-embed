import { useState, useMemo, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fetchDetectionJobs, fetchHydrophoneDetectionJobs } from "@/api/client";
import { useModels } from "@/hooks/queries/useAdmin";
import {
  useReembeddingStatus,
  useEnqueueReembedding,
} from "@/api/detectionEmbeddingJobs";
import { ReembeddingStatusTable } from "./ReembeddingStatusTable";
import type { DetectionJob } from "@/api/types";

function fmtDetectionJobLabel(j: DetectionJob): string {
  const name = j.hydrophone_name ?? j.audio_folder ?? j.id.slice(0, 8);
  if (j.start_timestamp != null && j.end_timestamp != null) {
    const fmt = (ts: number) => {
      const d = new Date(ts * 1000);
      return (
        d.getUTCFullYear() +
        "-" +
        String(d.getUTCMonth() + 1).padStart(2, "0") +
        "-" +
        String(d.getUTCDate()).padStart(2, "0") +
        " " +
        String(d.getUTCHours()).padStart(2, "0") +
        ":" +
        String(d.getUTCMinutes()).padStart(2, "0") +
        " UTC"
      );
    };
    return `${name}\u2003${fmt(j.start_timestamp)} — ${fmt(j.end_timestamp)}`;
  }
  return name;
}

export interface DetectionSourcePickerValue {
  selectedDetectionJobIds: string[];
  embeddingModelVersion: string;
  isReady: boolean;
}

interface DetectionSourcePickerProps {
  value: DetectionSourcePickerValue;
  onChange: (v: DetectionSourcePickerValue) => void;
}

export function DetectionSourcePicker({
  value,
  onChange,
}: DetectionSourcePickerProps) {
  const { data: allDetectionJobs = [] } = useQuery({
    queryKey: ["allDetectionJobsForPicker"],
    queryFn: async () => {
      const [local, hydro] = await Promise.all([
        fetchDetectionJobs(),
        fetchHydrophoneDetectionJobs(),
      ]);
      return [...local, ...hydro];
    },
  });
  const { data: modelConfigs = [] } = useModels();

  const labeledJobs = useMemo(
    () =>
      allDetectionJobs
        .filter(
          (j) => j.status === "complete" && j.has_positive_labels === true,
        )
        .sort(
          (a, b) =>
            new Date(b.created_at).getTime() -
            new Date(a.created_at).getTime(),
        ),
    [allDetectionJobs],
  );

  const selectedIds = value.selectedDetectionJobIds;
  const modelVersion = value.embeddingModelVersion;

  // Determine whether any selected pair needs re-embedding.
  const hasSelection = selectedIds.length > 0 && !!modelVersion;
  const [shouldPoll, setShouldPoll] = useState(false);

  const { data: statuses = [] } = useReembeddingStatus(
    selectedIds,
    modelVersion,
    hasSelection,
    shouldPoll ? 2000 : false,
  );

  // Update polling state based on status results.
  useEffect(() => {
    const hasActive = statuses.some(
      (s) => s.status === "queued" || s.status === "running",
    );
    setShouldPoll(hasActive);
  }, [statuses]);

  const allComplete = useMemo(
    () =>
      hasSelection &&
      statuses.length === selectedIds.length &&
      statuses.every((s) => s.status === "complete"),
    [hasSelection, statuses, selectedIds.length],
  );

  // Push isReady upward whenever it changes.
  useEffect(() => {
    if (value.isReady !== allComplete) {
      onChange({ ...value, isReady: allComplete });
    }
  }, [allComplete]); // eslint-disable-line react-hooks/exhaustive-deps

  const enqueueMutation = useEnqueueReembedding();

  const toggleJob = useCallback(
    (jobId: string) => {
      const next = selectedIds.includes(jobId)
        ? selectedIds.filter((id) => id !== jobId)
        : [...selectedIds, jobId];
      onChange({ ...value, selectedDetectionJobIds: next, isReady: false });
    },
    [selectedIds, value, onChange],
  );

  const setModelVersion = useCallback(
    (mv: string) => {
      onChange({ ...value, embeddingModelVersion: mv, isReady: false });
    },
    [value, onChange],
  );

  // Show status table only when there are pairs that aren't all complete.
  const showStatusTable =
    hasSelection && statuses.length > 0 && !allComplete;

  return (
    <div className="space-y-3">
      {/* Detection Job Multi-Select */}
      <div>
        <label className="text-xs font-medium text-muted-foreground mb-1 block">
          Detection Jobs (labeled)
        </label>
        <div className="max-h-40 overflow-y-auto border rounded-md p-2 space-y-1">
          {labeledJobs.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No labeled detection jobs available.
            </p>
          ) : (
            labeledJobs.map((j) => (
              <label
                key={j.id}
                className="flex items-center gap-2 text-xs cursor-pointer hover:bg-muted/30 rounded px-1 py-0.5"
              >
                <Checkbox
                  checked={selectedIds.includes(j.id)}
                  onCheckedChange={() => toggleJob(j.id)}
                />
                <span className="truncate">{fmtDetectionJobLabel(j)}</span>
              </label>
            ))
          )}
        </div>
      </div>

      {/* Embedding Model Selector */}
      <div>
        <label className="text-xs font-medium text-muted-foreground mb-1 block">
          Embedding Model
        </label>
        <Select value={modelVersion} onValueChange={setModelVersion}>
          <SelectTrigger className="h-8 text-xs">
            <SelectValue placeholder="Select embedding model" />
          </SelectTrigger>
          <SelectContent>
            {modelConfigs.map((mc) => (
              <SelectItem key={mc.id} value={mc.name} className="text-xs">
                {mc.display_name} ({mc.vector_dim}d)
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Re-embedding Status Table */}
      {showStatusTable && (
        <ReembeddingStatusTable
          rows={statuses}
          detectionJobs={allDetectionJobs}
          onReembed={(detJobId) =>
            enqueueMutation.mutate({
              detectionJobId: detJobId,
              modelVersion: modelVersion,
            })
          }
          isReembedding={enqueueMutation.isPending}
        />
      )}
    </div>
  );
}
