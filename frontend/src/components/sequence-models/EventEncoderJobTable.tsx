import { Link } from "react-router-dom";
import { X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { DeleteConfirmButton } from "@/components/shared/DeleteConfirmationDialog";
import {
  type EventEncoderJob,
  useCancelEventEncoderJob,
  useDeleteEventEncoderJob,
} from "@/api/sequenceModels";

interface TableProps {
  jobs: EventEncoderJob[];
  mode: "active" | "previous";
}

export function EventEncoderJobTable({ jobs, mode }: TableProps) {
  const cancelMutation = useCancelEventEncoderJob();
  const deleteMutation = useDeleteEventEncoderJob();

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b bg-muted/50">
          <th className="px-3 py-2 text-left font-medium">Status</th>
          <th className="px-3 py-2 text-left font-medium">Created</th>
          <th className="px-3 py-2 text-left font-medium">Segmentation</th>
          <th className="px-3 py-2 text-left font-medium">Source</th>
          <th className="px-3 py-2 text-left font-medium">Events</th>
          <th className="px-3 py-2 text-left font-medium">Vector Dim</th>
          <th className="px-3 py-2 text-left font-medium">k</th>
          <th className="px-3 py-2 text-left font-medium">
            {mode === "active" ? "Actions" : ""}
          </th>
        </tr>
      </thead>
      <tbody>
        {jobs.map((job) => (
          <tr key={job.id} className="border-b hover:bg-muted/30">
            <td className="px-3 py-2">
              <StatusBadge status={job.status} />
            </td>
            <td className="px-3 py-2 text-xs whitespace-nowrap">
              {new Date(job.created_at).toLocaleString()}
            </td>
            <td className="px-3 py-2 text-xs">
              {job.event_segmentation_job_id.slice(0, 8)}
            </td>
            <td className="px-3 py-2 text-xs">
              <Badge variant="outline">{job.event_source_mode}</Badge>
            </td>
            <td className="px-3 py-2 text-xs">
              {job.encoded_events ?? "-"} / {job.total_events ?? "-"}
            </td>
            <td className="px-3 py-2 text-xs">
              {job.event_vector_dim ?? "-"}
            </td>
            <td className="px-3 py-2 text-xs">{formatKValues(job)}</td>
            <td className="px-3 py-2">
              {mode === "active" ? (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => cancelMutation.mutate(job.id)}
                  disabled={cancelMutation.isPending}
                >
                  <X className="h-3.5 w-3.5 mr-1" />
                  Cancel
                </Button>
              ) : (
                <div className="flex items-center gap-2">
                  {job.status === "complete" ? (
                    <Link
                      to={`/app/sequence-models/event-encoder/${job.id}`}
                      className="rounded-md border px-2 py-1 text-xs hover:bg-accent"
                    >
                      Report
                    </Link>
                  ) : null}
                  <DeleteConfirmButton
                    size="sm"
                    resourceType="event encoder job"
                    resourceName={job.id.slice(0, 8)}
                    consequence="This Event Encoder job and its token/vector artifacts will be removed."
                    onConfirm={() => deleteMutation.mutateAsync(job.id)}
                    isPending={deleteMutation.isPending}
                  >
                    Delete
                  </DeleteConfirmButton>
                </div>
              )}
            </td>
          </tr>
        ))}
        {jobs.length === 0 ? (
          <tr>
            <td
              colSpan={8}
              className="px-3 py-4 text-center text-muted-foreground text-xs"
            >
              No jobs found.
            </td>
          </tr>
        ) : null}
      </tbody>
    </table>
  );
}

export function EventEncoderJobTablePanel({
  title,
  jobs,
  mode,
}: {
  title: string;
  jobs: EventEncoderJob[];
  mode: "active" | "previous";
}) {
  if (jobs.length === 0 && mode === "active") return null;
  return (
    <div className="border rounded-md">
      <div className="flex items-center gap-2 px-4 py-3 border-b">
        <h3 className="text-sm font-semibold">{title}</h3>
        <Badge variant="secondary">{jobs.length}</Badge>
      </div>
      <EventEncoderJobTable jobs={jobs} mode={mode} />
    </div>
  );
}

function formatKValues(job: EventEncoderJob): string {
  try {
    const values = JSON.parse(job.k_values_json);
    return Array.isArray(values) ? values.join(", ") : job.k_values_json;
  } catch {
    return job.k_values_json;
  }
}
