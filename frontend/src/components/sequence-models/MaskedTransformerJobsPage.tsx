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
import { MaskedTransformerCreateForm } from "./MaskedTransformerCreateForm";

export function MaskedTransformerJobsPage() {
  const { data: jobs = [], isLoading } = useMaskedTransformerJobs();

  const { active, previous } = useMemo(() => {
    const a = jobs.filter(isMaskedTransformerJobActive);
    const p = jobs.filter((j) => !isMaskedTransformerJobActive(j));
    return { active: a, previous: p };
  }, [jobs]);

  return (
    <div className="space-y-6" data-testid="masked-transformer-jobs-page">
      <MaskedTransformerCreateForm />
      {isLoading ? (
        <div className="text-sm text-slate-500">Loading…</div>
      ) : (
        <>
          <JobTablePanel title="Active Jobs" jobs={active} mode="active" />
          <JobTablePanel title="Previous Jobs" jobs={previous} mode="previous" />
        </>
      )}
    </div>
  );
}

function JobTablePanel({
  title,
  jobs,
  mode,
}: {
  title: string;
  jobs: MaskedTransformerJob[];
  mode: "active" | "previous";
}) {
  if (jobs.length === 0 && mode === "active") return null;
  return (
    <div className="border rounded-md">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">{title}</h3>
          <Badge variant="secondary">{jobs.length}</Badge>
        </div>
      </div>
      <JobTable jobs={jobs} mode={mode} />
    </div>
  );
}

function JobTable({
  jobs,
  mode,
}: {
  jobs: MaskedTransformerJob[];
  mode: "active" | "previous";
}) {
  const cancelMutation = useCancelMaskedTransformerJob();
  const deleteMutation = useDeleteMaskedTransformerJob();
  return (
    <table className="w-full text-sm" data-testid="mt-job-table">
      <thead>
        <tr className="border-b bg-muted/50">
          <th className="px-3 py-2 text-left font-medium">Status</th>
          <th className="px-3 py-2 text-left font-medium">Created</th>
          <th className="px-3 py-2 text-left font-medium">Source CE Job</th>
          <th className="px-3 py-2 text-left font-medium">Preset</th>
          <th className="px-3 py-2 text-left font-medium">k_values</th>
          <th className="px-3 py-2 text-left font-medium">Device</th>
          <th className="px-3 py-2 text-left font-medium">Actions</th>
        </tr>
      </thead>
      <tbody>
        {jobs.length === 0 && (
          <tr>
            <td colSpan={7} className="px-3 py-4 text-center text-muted-foreground text-xs">
              No jobs found.
            </td>
          </tr>
        )}
        {jobs.map((job) => (
          <tr key={job.id} className="border-b hover:bg-muted/30" data-testid={`mt-job-row-${job.id}`}>
            <td className="px-3 py-2">
              <StatusBadge status={job.status} />
            </td>
            <td className="px-3 py-2 text-xs whitespace-nowrap">
              {new Date(job.created_at).toLocaleString()}
            </td>
            <td className="px-3 py-2 text-xs">
              {job.continuous_embedding_job_id.slice(0, 8)}
            </td>
            <td className="px-3 py-2 text-xs">{job.preset}</td>
            <td className="px-3 py-2 text-xs">
              <span data-testid={`mt-job-k-values-${job.id}`}>
                {(job.k_values ?? []).join(", ") || "—"}
              </span>
            </td>
            <td className="px-3 py-2 text-xs">
              {job.chosen_device ? (
                <Badge variant={job.fallback_reason ? "destructive" : "secondary"}>
                  {job.chosen_device}
                  {job.fallback_reason ? ` (fallback)` : ""}
                </Badge>
              ) : (
                "—"
              )}
            </td>
            <td className="px-3 py-2 text-xs">
              <div className="flex items-center gap-2">
                {job.status === "complete" && (
                  <Link
                    to={`/app/sequence-models/masked-transformer/${job.id}`}
                    className="rounded-md border px-2 py-1 text-xs hover:bg-accent"
                    data-testid={`mt-job-open-${job.id}`}
                  >
                    Open
                  </Link>
                )}
                {mode === "active" ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => cancelMutation.mutate(job.id)}
                    disabled={cancelMutation.isPending}
                    data-testid={`mt-job-cancel-${job.id}`}
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
                    data-testid={`mt-job-delete-${job.id}`}
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
  );
}
