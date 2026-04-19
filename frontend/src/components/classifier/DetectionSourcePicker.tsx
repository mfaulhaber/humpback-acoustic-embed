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
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Loader2, AlertCircle } from "lucide-react";
import { fetchDetectionJobs, fetchHydrophoneDetectionJobs } from "@/api/client";
import { useModels } from "@/hooks/queries/useAdmin";
import {
  useReembeddingStatus,
  useEnqueueReembedding,
} from "@/api/detectionEmbeddingJobs";
import { useDetectionJobLabelCounts } from "@/hooks/queries/useClassifier";
import type { DetectionJob, DetectionEmbeddingJobStatus } from "@/api/types";

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

function EmbeddingCell({
  status,
  onEmbed,
  isEmbedding,
}: {
  status: DetectionEmbeddingJobStatus | undefined;
  onEmbed: () => void;
  isEmbedding: boolean;
}) {
  if (!status) return <span className="text-muted-foreground">—</span>;

  switch (status.status) {
    case "complete":
      return null;
    case "not_started":
      return (
        <Badge variant="outline" className="text-[10px] text-amber-500 border-amber-500/40">
          Missing
        </Badge>
      );
    case "queued":
      return (
        <Badge variant="outline" className="text-[10px]">
          Queued
        </Badge>
      );
    case "running": {
      const pct =
        status.rows_total != null && status.rows_total > 0
          ? Math.round((status.rows_processed / status.rows_total) * 100)
          : null;
      return (
        <div className="flex items-center gap-1">
          <Badge variant="secondary" className="text-[10px]">
            Running{pct != null ? ` ${pct}%` : ""}
          </Badge>
          <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
        </div>
      );
    }
    case "failed":
      return (
        <div className="flex items-center gap-1">
          <Badge variant="destructive" className="text-[10px]">
            Failed
          </Badge>
          {status.error_message && (
            <Popover>
              <PopoverTrigger asChild>
                <button className="text-destructive">
                  <AlertCircle className="h-3 w-3" />
                </button>
              </PopoverTrigger>
              <PopoverContent className="text-xs max-w-sm">
                {status.error_message}
              </PopoverContent>
            </Popover>
          )}
        </div>
      );
    default:
      return null;
  }
}

function ActionCell({
  status,
  onEmbed,
  isEmbedding,
}: {
  status: DetectionEmbeddingJobStatus | undefined;
  onEmbed: () => void;
  isEmbedding: boolean;
}) {
  if (!status) return null;

  if (status.status === "not_started") {
    return (
      <Button
        size="sm"
        variant="ghost"
        className="h-6 px-2 text-xs"
        disabled={isEmbedding}
        onClick={onEmbed}
      >
        Embed
      </Button>
    );
  }
  if (status.status === "failed") {
    return (
      <Button
        size="sm"
        variant="ghost"
        className="h-6 px-2 text-xs"
        disabled={isEmbedding}
        onClick={onEmbed}
      >
        Retry
      </Button>
    );
  }
  return null;
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

  const labeledJobIds = useMemo(
    () => labeledJobs.map((j) => j.id),
    [labeledJobs],
  );

  const { data: labelCounts = [] } = useDetectionJobLabelCounts(labeledJobIds);

  const labelCountMap = useMemo(() => {
    const m = new Map<string, { positive: number; negative: number }>();
    for (const lc of labelCounts) {
      m.set(lc.detection_job_id, { positive: lc.positive, negative: lc.negative });
    }
    return m;
  }, [labelCounts]);

  const selectedIds = value.selectedDetectionJobIds;
  const modelVersion = value.embeddingModelVersion;

  const hasModel = !!modelVersion;
  const hasSelection = selectedIds.length > 0 && hasModel;
  const [shouldPoll, setShouldPoll] = useState(false);

  const { data: statuses = [] } = useReembeddingStatus(
    labeledJobIds,
    modelVersion,
    hasModel && labeledJobIds.length > 0,
    shouldPoll ? 2000 : false,
  );

  useEffect(() => {
    const hasActive = statuses.some(
      (s) => s.status === "queued" || s.status === "running",
    );
    setShouldPoll(hasActive);
  }, [statuses]);

  const statusMap = useMemo(() => {
    const m = new Map<string, DetectionEmbeddingJobStatus>();
    for (const s of statuses) m.set(s.detection_job_id, s);
    return m;
  }, [statuses]);

  const allComplete = useMemo(() => {
    if (!hasSelection) return false;
    return selectedIds.every((id) => {
      const s = statusMap.get(id);
      return s != null && s.status === "complete";
    });
  }, [hasSelection, selectedIds, statusMap]);

  useEffect(() => {
    console.log("[DetectionSourcePicker] allComplete=%s, value.isReady=%s, selectedIds=%d, statusMap.size=%d",
      allComplete, value.isReady, selectedIds.length, statusMap.size);
    if (value.isReady !== allComplete) {
      console.log("[DetectionSourcePicker] pushing isReady=%s", allComplete);
      onChange({ ...value, isReady: allComplete });
    }
  }, [allComplete, value.isReady]); // eslint-disable-line react-hooks/exhaustive-deps

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

  const showEmbeddingColumns = !!modelVersion;

  return (
    <div className="space-y-3">
      {/* Detection Job Table */}
      <div>
        <label className="text-xs font-medium text-muted-foreground mb-1 block">
          Detection Jobs (labeled)
        </label>
        <div className="max-h-48 overflow-y-auto border rounded-md">
          {labeledJobs.length === 0 ? (
            <p className="text-xs text-muted-foreground p-2">
              No labeled detection jobs available.
            </p>
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b bg-muted/40 text-left text-muted-foreground sticky top-0">
                  <th className="py-1.5 px-2 w-8" />
                  <th className="py-1.5 px-2 font-medium">Detection Job</th>
                  <th className="py-1.5 px-2 font-medium text-right w-12">Pos</th>
                  <th className="py-1.5 px-2 font-medium text-right w-12">Neg</th>
                  <th className="py-1.5 px-2 font-medium w-24">Embedding</th>
                  <th className="py-1.5 px-2 w-16" />
                </tr>
              </thead>
              <tbody>
                {labeledJobs.map((j) => {
                  const counts = labelCountMap.get(j.id);
                  const embStatus = showEmbeddingColumns
                    ? statusMap.get(j.id)
                    : undefined;

                  return (
                    <tr
                      key={j.id}
                      className="border-b last:border-0 hover:bg-muted/30 cursor-pointer"
                      onClick={() => toggleJob(j.id)}
                    >
                      <td className="py-1.5 px-2" onClick={(e) => e.stopPropagation()}>
                        <Checkbox
                          checked={selectedIds.includes(j.id)}
                          onCheckedChange={() => toggleJob(j.id)}
                        />
                      </td>
                      <td className="py-1.5 px-2 truncate max-w-[300px]">
                        {fmtDetectionJobLabel(j)}
                      </td>
                      <td className="py-1.5 px-2 text-right font-semibold text-green-500">
                        {counts ? counts.positive.toLocaleString() : "—"}
                      </td>
                      <td className="py-1.5 px-2 text-right font-semibold text-slate-400">
                        {counts ? counts.negative.toLocaleString() : "—"}
                      </td>
                      <td className="py-1.5 px-2" onClick={(e) => e.stopPropagation()}>
                        {showEmbeddingColumns ? (
                          <EmbeddingCell
                            status={embStatus}
                            onEmbed={() =>
                              enqueueMutation.mutate({
                                detectionJobId: j.id,
                                modelVersion,
                              })
                            }
                            isEmbedding={enqueueMutation.isPending}
                          />
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </td>
                      <td className="py-1.5 px-2" onClick={(e) => e.stopPropagation()}>
                        {showEmbeddingColumns && (
                          <ActionCell
                            status={embStatus}
                            onEmbed={() =>
                              enqueueMutation.mutate({
                                detectionJobId: j.id,
                                modelVersion,
                              })
                            }
                            isEmbedding={enqueueMutation.isPending}
                          />
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
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
    </div>
  );
}
