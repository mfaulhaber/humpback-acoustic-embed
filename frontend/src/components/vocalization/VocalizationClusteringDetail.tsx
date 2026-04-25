import { useParams, Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { ClusterTable } from "@/components/clustering/ClusterTable";
import { EvaluationPanel } from "@/components/clustering/EvaluationPanel";
import { VocalizationUmapPlot } from "./VocalizationUmapPlot";
import {
  useVocalizationClusteringJob,
  useVocClusteringClusters,
} from "@/hooks/queries/useVocalization";

export function VocalizationClusteringDetail() {
  const { jobId } = useParams<{ jobId: string }>();
  const { data: job } = useVocalizationClusteringJob(jobId ?? null);
  const isComplete = job?.status === "complete";
  const { data: clusters = [] } = useVocClusteringClusters(
    isComplete ? jobId! : null,
  );

  if (!job) {
    return (
      <div className="text-center py-12 text-muted-foreground">Loading…</div>
    );
  }

  const detectionJobIds = job.detection_job_ids;

  return (
    <div className="space-y-6">
      <Link
        to="/app/vocalization/clustering"
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Back to Vocalization Clustering
      </Link>

      <div className="border rounded-md p-4 space-y-2">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold">Clustering Job</h2>
          <StatusBadge status={job.status} />
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>
            Created: {new Date(job.created_at).toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC")}
          </div>
          {detectionJobIds && (
            <div>
              Source: {detectionJobIds.length} detection job
              {detectionJobIds.length !== 1 ? "s" : ""}
            </div>
          )}
          {job.parameters && (
            <div>
              Parameters:{" "}
              {Object.entries(job.parameters)
                .filter(([, v]) => v != null)
                .map(([k, v]) => `${k}=${v}`)
                .join(", ")}
            </div>
          )}
        </div>
      </div>

      {job.status === "failed" && job.error_message && (
        <div className="border border-red-200 bg-red-50 rounded-md p-4">
          <p className="text-sm text-red-700">{job.error_message}</p>
        </div>
      )}

      {isComplete && (
        <>
          <ClusterTable clusters={clusters} />
          <VocalizationUmapPlot jobId={jobId!} />
          <EvaluationPanel jobId={jobId!} />
        </>
      )}
    </div>
  );
}
