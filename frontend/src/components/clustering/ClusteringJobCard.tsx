import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { ClusterTable } from "./ClusterTable";
import { UmapPlot } from "./UmapPlot";
import { EvaluationPanel } from "./EvaluationPanel";
import { ExportReport } from "./ExportReport";
import { useClusters } from "@/hooks/queries/useClustering";
import { shortId, fmtDate, jsonPretty } from "@/utils/format";
import type { ClusteringJob } from "@/api/types";

interface ClusteringJobCardProps {
  job: ClusteringJob;
}

export function ClusteringJobCard({ job }: ClusteringJobCardProps) {
  const [showUmap, setShowUmap] = useState(false);
  const [showEval, setShowEval] = useState(false);
  const isComplete = job.status === "complete";

  const { data: clusters = [] } = useClusters(isComplete ? job.id : null);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center gap-3 flex-wrap">
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

      {isComplete && (
        <CardContent className="space-y-4">
          <ClusterTable clusters={clusters} />

          <div className="flex gap-2 flex-wrap">
            <Button variant="outline" size="sm" onClick={() => setShowUmap((v) => !v)}>
              {showUmap ? "Hide UMAP" : "Show UMAP Plot"}
            </Button>
            <Button variant="outline" size="sm" onClick={() => setShowEval((v) => !v)}>
              {showEval ? "Hide Evaluation" : "Show Evaluation"}
            </Button>
            <ExportReport jobId={job.id} />
          </div>

          {showUmap && <UmapPlot jobId={job.id} />}
          {showEval && <EvaluationPanel jobId={job.id} />}
        </CardContent>
      )}
    </Card>
  );
}
