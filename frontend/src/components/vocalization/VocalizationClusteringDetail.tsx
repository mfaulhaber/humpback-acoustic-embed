import { useParams, Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { useQueries } from "@tanstack/react-query";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { VocalizationUmapPlot } from "./VocalizationUmapPlot";
import {
  useVocalizationClusteringJob,
  useVocClusteringMetrics,
} from "@/hooks/queries/useVocalization";
import { fetchDetectionJob } from "@/api/client";
import type { DetectionJob } from "@/api/types";

function fmt(v: number | undefined | null, decimals = 4): string {
  if (v == null) return "—";
  return v.toFixed(decimals);
}

function formatUtcRange(start: number | null, end: number | null): string {
  if (start == null) return "Unknown dates";
  const s = new Date(start * 1000).toISOString().slice(0, 10);
  if (end == null) return s;
  const e = new Date(end * 1000).toISOString().slice(0, 10);
  return s === e ? s : `${s} – ${e}`;
}

function DetectionJobList({ jobIds }: { jobIds: string[] }) {
  const results = useQueries({
    queries: jobIds.map((id) => ({
      queryKey: ["detectionJob", id],
      queryFn: () => fetchDetectionJob(id),
      staleTime: Infinity,
    })),
  });

  return (
    <div className="text-xs text-muted-foreground space-y-0.5">
      <div className="font-medium text-foreground">Detection Jobs:</div>
      {results.map((r, i) => {
        if (r.isLoading) return <div key={jobIds[i]}>Loading...</div>;
        const dj = r.data as DetectionJob | undefined;
        if (!dj) return <div key={jobIds[i]}>{jobIds[i].slice(0, 8)}...</div>;
        const name = dj.hydrophone_name || dj.audio_folder || jobIds[i].slice(0, 8);
        return (
          <div key={jobIds[i]}>
            {name} — {formatUtcRange(dj.start_timestamp, dj.end_timestamp)}
          </div>
        );
      })}
    </div>
  );
}

export function VocalizationClusteringDetail() {
  const { jobId } = useParams<{ jobId: string }>();
  const { data: job } = useVocalizationClusteringJob(jobId ?? null);
  const isComplete = job?.status === "complete";
  const { data: metrics } = useVocClusteringMetrics(isComplete ? jobId! : null);

  if (!job) {
    return (
      <div className="text-center py-12 text-muted-foreground">Loading…</div>
    );
  }

  const detectionJobIds = job.detection_job_ids;

  const mainMetrics = [
    { label: "Silhouette Score", value: metrics?.silhouette_score },
    { label: "Davies-Bouldin Index", value: metrics?.davies_bouldin_index },
    { label: "Calinski-Harabasz Score", value: metrics?.calinski_harabasz_score },
    { label: "N Clusters", value: metrics?.n_clusters },
    { label: "Noise Points", value: metrics?.noise_count },
  ];

  const hasMetrics = mainMetrics.some((m) => m.value != null);

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
        {detectionJobIds && detectionJobIds.length > 0 && (
          <DetectionJobList jobIds={detectionJobIds} />
        )}
      </div>

      {job.status === "failed" && job.error_message && (
        <div className="border border-red-200 bg-red-50 rounded-md p-4">
          <p className="text-sm text-red-700">{job.error_message}</p>
        </div>
      )}

      {isComplete && (
        <>
          <VocalizationUmapPlot jobId={jobId!} />

          {hasMetrics && (
            <div className="border rounded-md">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left py-2 px-3 font-medium">Metric</th>
                    <th className="text-left py-2 px-3 font-medium">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {mainMetrics.map(
                    (m) =>
                      m.value != null && (
                        <tr key={m.label} className="border-b last:border-0">
                          <td className="py-1.5 px-3">{m.label}</td>
                          <td className="py-1.5 px-3 font-mono text-xs">
                            {typeof m.value === "number"
                              ? Number.isInteger(m.value) ? String(m.value) : fmt(m.value)
                              : m.value}
                          </td>
                        </tr>
                      ),
                  )}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
