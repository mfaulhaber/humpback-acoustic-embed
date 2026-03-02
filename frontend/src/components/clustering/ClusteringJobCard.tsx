import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { ClusterTable } from "./ClusterTable";
import { UmapPlot } from "./UmapPlot";
import { EvaluationPanel } from "./EvaluationPanel";
import { LabelDotPlot } from "./LabelDotPlot";
import { ExportReport } from "./ExportReport";
import { useClusters, useMetrics } from "@/hooks/queries/useClustering";
import { useCollapseState } from "@/hooks/useCollapseState";
import { shortId, fmtDate, jsonPretty } from "@/utils/format";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ClusteringJob } from "@/api/types";

interface ClusteringJobCardProps {
  job: ClusteringJob;
}

export function ClusteringJobCard({ job }: ClusteringJobCardProps) {
  const [showUmap, setShowUmap] = useState(false);
  const [showEval, setShowEval] = useState(false);
  const [showLabelPlot, setShowLabelPlot] = useState(false);
  const isComplete = job.status === "complete";

  const { isExpanded, toggle } = useCollapseState("cjob", "cj");
  const expanded = isExpanded(job.id);

  const { data: clusters = [] } = useClusters(isComplete ? job.id : null);
  const { data: metrics } = useMetrics(isComplete ? job.id : null);

  const hasConfusionMatrix =
    !!metrics?.confusion_matrix &&
    typeof metrics.confusion_matrix === "object" &&
    Object.keys(metrics.confusion_matrix as object).length > 0;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center gap-3 flex-wrap">
          {isComplete && (
            <button onClick={() => toggle(job.id)} className="p-0.5 -ml-1 hover:bg-accent rounded">
              <ChevronRight
                className={cn("h-4 w-4 shrink-0 transition-transform", expanded && "rotate-90")}
              />
            </button>
          )}
          <span className="font-mono text-sm font-medium">{shortId(job.id)}</span>
          <StatusBadge status={job.status} />
          {job.parameters && (
            <span className="text-xs text-muted-foreground" title={jsonPretty(job.parameters)}>
              {Object.entries(job.parameters)
                .filter(([, v]) => v != null)
                .map(([k, v]) => `${k}=${v}`)
                .join(", ")}
            </span>
          )}
          <span className="text-xs text-muted-foreground ml-auto">{fmtDate(job.created_at)}</span>
        </div>
        {job.error_message && (
          <p className="text-sm text-red-600 mt-1">{job.error_message}</p>
        )}
      </CardHeader>

      {isComplete && expanded && (
        <CardContent className="space-y-4">
          <ClusterTable clusters={clusters} />

          <div className="flex gap-2 flex-wrap">
            <Button variant="outline" size="sm" onClick={() => setShowUmap((v) => !v)}>
              {showUmap ? "Hide UMAP" : "Show UMAP Plot"}
            </Button>
            <Button variant="outline" size="sm" onClick={() => setShowEval((v) => !v)}>
              {showEval ? "Hide Evaluation" : "Show Evaluation"}
            </Button>
            {hasConfusionMatrix && (
              <Button variant="outline" size="sm" onClick={() => setShowLabelPlot((v) => !v)}>
                {showLabelPlot ? "Hide Label Plot" : "Show Label Plot"}
              </Button>
            )}
            <ExportReport jobId={job.id} />
          </div>

          {showUmap && <UmapPlot jobId={job.id} />}
          {showEval && <EvaluationPanel jobId={job.id} />}
          {showLabelPlot && <LabelDotPlot jobId={job.id} />}
        </CardContent>
      )}
    </Card>
  );
}
