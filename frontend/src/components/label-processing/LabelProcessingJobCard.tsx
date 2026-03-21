import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { ChevronRight, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { LabelProcessingJob } from "@/api/types";

interface Props {
  job: LabelProcessingJob;
  onDelete: (id: string) => void;
}

export function LabelProcessingJobCard({ job, onDelete }: Props) {
  const [expanded, setExpanded] = useState(false);

  const progress =
    job.files_total && job.files_total > 0
      ? Math.round(((job.files_processed ?? 0) / job.files_total) * 100)
      : 0;

  const isActive = job.status === "queued" || job.status === "running";

  const summary = job.result_summary as Record<string, unknown> | null;
  const isSampleBuilder = job.workflow === "sample_builder";
  const treatmentCounts = summary?.treatment_counts as Record<string, number> | undefined;
  const callTypeCounts = summary?.call_type_counts as Record<string, number> | undefined;
  const scoreStatsByLabel = summary?.score_stats_by_label as
    | Record<string, { count: number; mean: number; median: number; std: number; min: number; max: number }>
    | undefined;
  const rejectionCounts = summary?.rejection_counts as Record<string, number> | undefined;
  const totalAccepted = summary?.total_accepted as number | undefined;
  const totalRejected = summary?.total_rejected as number | undefined;
  const acceptanceRate = summary?.acceptance_rate as number | undefined;

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-0">
        {/* Header row */}
        <div
          className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-slate-50"
          onClick={() => setExpanded(!expanded)}
        >
          <ChevronRight
            className={cn(
              "h-4 w-4 shrink-0 transition-transform",
              expanded && "rotate-90",
            )}
          />
          <StatusBadge status={job.status} />
          {isSampleBuilder && (
            <span className="px-1.5 py-0.5 text-[10px] font-medium bg-indigo-100 text-indigo-700 rounded">
              Sample Builder
            </span>
          )}
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium truncate">
              {job.annotation_folder.split("/").pop()} / {job.audio_folder.split("/").pop()}
            </div>
            <div className="text-xs text-muted-foreground">
              {new Date(job.created_at).toLocaleString()}
              {job.annotations_total != null && (
                <> &middot; {job.annotations_total} annotations</>
              )}
            </div>
          </div>

          {/* Progress or counts */}
          {isActive && job.files_total != null && (
            <div className="w-32 flex items-center gap-2">
              <div className="flex-1 h-2 bg-slate-100 rounded overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <span className="text-xs text-muted-foreground whitespace-nowrap">
                {job.files_processed ?? 0}/{job.files_total}
              </span>
            </div>
          )}

          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(job.id);
            }}
          >
            <Trash2 className="h-3.5 w-3.5 text-muted-foreground" />
          </Button>
        </div>

        {/* Expanded detail */}
        {expanded && (
          <div className="border-t px-4 py-3 space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-x-8 gap-y-1">
              <div>
                <span className="text-muted-foreground">Annotation folder:</span>{" "}
                <span className="font-mono text-xs">{job.annotation_folder}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Audio folder:</span>{" "}
                <span className="font-mono text-xs">{job.audio_folder}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Output root:</span>{" "}
                <span className="font-mono text-xs">{job.output_root}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Classifier model:</span>{" "}
                <span className="font-mono text-xs">{job.classifier_model_id ? job.classifier_model_id.slice(0, 8) : "none"}</span>
              </div>
            </div>

            {job.error_message && (
              <div className="text-red-600 text-xs bg-red-50 rounded p-2">
                {job.error_message}
              </div>
            )}

            {/* Sample builder acceptance stats */}
            {isSampleBuilder && totalAccepted != null && (
              <div>
                <div className="text-muted-foreground mb-1 text-xs font-medium">
                  Acceptance
                </div>
                <div className="flex gap-4 text-xs">
                  <div>
                    <span className="text-green-600 font-medium">{totalAccepted}</span> accepted
                  </div>
                  <div>
                    <span className="text-red-600 font-medium">{totalRejected ?? 0}</span> rejected
                  </div>
                  {acceptanceRate != null && (
                    <div className="text-muted-foreground">
                      ({(acceptanceRate * 100).toFixed(0)}% acceptance)
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Rejection breakdown (sample_builder) */}
            {isSampleBuilder && rejectionCounts && Object.keys(rejectionCounts).length > 0 && (
              <div>
                <div className="text-muted-foreground mb-1 text-xs font-medium">
                  Rejection Breakdown
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  {Object.entries(rejectionCounts)
                    .sort(([, a], [, b]) => b - a)
                    .map(([reason, count]) => (
                      <div key={reason} className="text-xs">
                        <span className="text-muted-foreground">{reason.replace(/_/g, " ")}:</span>{" "}
                        <span className="font-mono">{count}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Treatment distribution */}
            {treatmentCounts && Object.keys(treatmentCounts).length > 0 && (
              <div>
                <div className="text-muted-foreground mb-1 text-xs font-medium">
                  Treatment Distribution
                </div>
                <div className="flex gap-4">
                  {Object.entries(treatmentCounts).map(([treatment, count]) => (
                    <div
                      key={treatment}
                      className="flex items-center gap-1.5 text-xs"
                    >
                      <div
                        className={cn(
                          "h-2.5 w-2.5 rounded-full",
                          treatment === "clean" && "bg-green-400",
                          treatment === "recentered" && "bg-blue-400",
                          treatment === "synthesized" && "bg-purple-400",
                          treatment === "fallback" && "bg-amber-400",
                          treatment === "skipped" && "bg-gray-400",
                        )}
                      />
                      <span className="capitalize">{treatment}</span>
                      <span className="font-mono">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Per-call-type counts */}
            {callTypeCounts && Object.keys(callTypeCounts).length > 0 && (
              <div>
                <div className="text-muted-foreground mb-1 text-xs font-medium">
                  Per Call Type
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  {Object.entries(callTypeCounts)
                    .sort(([, a], [, b]) => b - a)
                    .map(([callType, count]) => (
                      <div key={callType} className="text-xs">
                        <span className="text-muted-foreground">{callType}:</span>{" "}
                        <span className="font-mono">{count}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Score statistics by label */}
            {scoreStatsByLabel && Object.keys(scoreStatsByLabel).length > 0 && (
              <div>
                <div className="text-muted-foreground mb-1 text-xs font-medium">
                  Score Statistics by Label
                </div>
                <div className="overflow-x-auto">
                  <table className="text-xs w-full">
                    <thead>
                      <tr className="text-muted-foreground border-b">
                        <th className="text-left py-1 pr-3 font-medium">Call Type</th>
                        <th className="text-right py-1 px-2 font-medium">Count</th>
                        <th className="text-right py-1 px-2 font-medium">Mean</th>
                        <th className="text-right py-1 px-2 font-medium">Median</th>
                        <th className="text-right py-1 px-2 font-medium">Std</th>
                        <th className="text-right py-1 px-2 font-medium">Min</th>
                        <th className="text-right py-1 px-2 font-medium">Max</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(scoreStatsByLabel)
                        .sort(([, a], [, b]) => b.count - a.count)
                        .map(([callType, stats]) => (
                          <tr key={callType} className="border-b border-dashed last:border-0">
                            <td className="py-1 pr-3">{callType}</td>
                            <td className="text-right py-1 px-2 font-mono">{stats.count}</td>
                            <td className="text-right py-1 px-2 font-mono">{stats.mean.toFixed(3)}</td>
                            <td className="text-right py-1 px-2 font-mono">{stats.median.toFixed(3)}</td>
                            <td className="text-right py-1 px-2 font-mono">{stats.std.toFixed(3)}</td>
                            <td className="text-right py-1 px-2 font-mono">{stats.min.toFixed(3)}</td>
                            <td className="text-right py-1 px-2 font-mono">{stats.max.toFixed(3)}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Parameters if present */}
            {job.parameters && Object.keys(job.parameters).length > 0 && (
              <div>
                <div className="text-muted-foreground mb-1 text-xs font-medium">
                  Parameters
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  {Object.entries(job.parameters).map(([key, val]) => (
                    <div key={key} className="text-xs">
                      <span className="text-muted-foreground">{key}:</span>{" "}
                      <span className="font-mono">{String(val)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
