import { useMemo } from "react";
import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/shared/StatusBadge";
import {
  isMaskedTransformerJobActive,
  useCancelMaskedTransformerJob,
  useDeleteMaskedTransformerJob,
  useMaskedTransformerJobs,
  type MaskedTransformerJob,
} from "@/api/sequenceModels";
import { MTTrainingCreateForm } from "./MTTrainingCreateForm";

export function MTTrainingJobsPage() {
  const { data: jobs = [], isLoading } = useMaskedTransformerJobs();
  const active = useMemo(() => jobs.filter(isMaskedTransformerJobActive), [jobs]);
  const previous = useMemo(
    () => jobs.filter((job) => !isMaskedTransformerJobActive(job)),
    [jobs],
  );

  return (
    <div className="space-y-6 p-2" data-testid="mt-training-jobs-page">
      <MTTrainingCreateForm />
      {isLoading ? (
        <div className="text-sm text-muted-foreground">Loading...</div>
      ) : (
        <>
          <TrainingTable title="Active Jobs" jobs={active} mode="active" />
          <TrainingTable title="Previous Jobs" jobs={previous} mode="previous" />
        </>
      )}
    </div>
  );
}

function TrainingTable({
  title,
  jobs,
  mode,
}: {
  title: string;
  jobs: MaskedTransformerJob[];
  mode: "active" | "previous";
}) {
  const cancelMutation = useCancelMaskedTransformerJob();
  const deleteMutation = useDeleteMaskedTransformerJob();
  if (jobs.length === 0 && mode === "active") return null;
  return (
    <div className="rounded-md border">
      <div className="flex items-center gap-2 border-b px-4 py-3">
        <h2 className="text-sm font-semibold">{title}</h2>
        <Badge variant="secondary">{jobs.length}</Badge>
      </div>
      <table className="w-full text-sm" data-testid="mt-training-job-table">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="px-3 py-2 text-left font-medium">Status</th>
            <th className="px-3 py-2 text-left font-medium">Created</th>
            <th className="px-3 py-2 text-left font-medium">Sources</th>
            <th className="px-3 py-2 text-left font-medium">Preset</th>
            <th className="px-3 py-2 text-left font-medium">k</th>
            <th className="px-3 py-2 text-left font-medium">Chunks</th>
            <th className="px-3 py-2 text-left font-medium">Retrieval</th>
            <th className="px-3 py-2 text-left font-medium">Device</th>
            <th className="px-3 py-2 text-left font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {jobs.length === 0 ? (
            <tr>
              <td colSpan={9} className="px-3 py-4 text-center text-xs text-muted-foreground">
                No jobs found.
              </td>
            </tr>
          ) : null}
          {jobs.map((job) => (
            <tr key={job.id} className="border-b hover:bg-muted/30">
              <td className="px-3 py-2">
                <StatusBadge status={job.status} />
              </td>
              <td className="px-3 py-2 text-xs">
                {new Date(job.created_at).toLocaleString()}
              </td>
              <td className="px-3 py-2 text-xs">
                <Badge variant="outline" data-testid={`mt-training-source-count-${job.id}`}>
                  {job.source_count ?? 1}
                </Badge>
              </td>
              <td className="px-3 py-2 text-xs">{job.preset}</td>
              <td className="px-3 py-2 text-xs">{job.k_values.join(", ")}</td>
              <td className="px-3 py-2 text-xs">{job.total_chunks ?? "-"}</td>
              <td className="px-3 py-2 text-xs">
                {job.retrieval_head_enabled ? job.retrieval_head_arch : "off"}
              </td>
              <td className="px-3 py-2 text-xs">
                {job.chosen_device ? (
                  <Badge variant={job.fallback_reason ? "destructive" : "secondary"}>
                    {job.chosen_device}
                  </Badge>
                ) : (
                  "-"
                )}
              </td>
              <td className="px-3 py-2 text-xs">
                <div className="flex items-center gap-2">
                  <Link
                    to={`/app/sequence-models/mt-training/${job.id}`}
                    className="rounded-md border px-2 py-1 text-xs hover:bg-accent"
                    data-testid={`mt-training-open-${job.id}`}
                  >
                    Open
                  </Link>
                  {mode === "active" ? (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => cancelMutation.mutate(job.id)}
                      disabled={cancelMutation.isPending}
                    >
                      Cancel
                    </Button>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-red-600 hover:text-red-700"
                      onClick={() => deleteMutation.mutate(job.id)}
                      disabled={deleteMutation.isPending}
                    >
                      Delete
                    </Button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
