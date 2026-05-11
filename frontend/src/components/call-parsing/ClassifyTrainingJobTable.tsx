import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { DeleteConfirmButton } from "@/components/shared/DeleteConfirmationDialog";
import {
  useClassifierTrainingJobs,
  useDeleteClassifierTrainingJob,
} from "@/hooks/queries/useCallParsing";
import { toast } from "@/components/ui/use-toast";

export function ClassifyTrainingJobTable() {
  const { data: jobs = [] } = useClassifierTrainingJobs(3000);
  const deleteMutation = useDeleteClassifierTrainingJob();

  const handleDelete = async (jobId: string) => {
    try {
      await deleteMutation.mutateAsync(jobId);
    } catch (err) {
      toast({
        title: "Cannot delete training job",
        description: (err as Error).message,
        variant: "destructive",
      });
      throw err;
    }
  };

  function sourceCount(sourceJobIds: string): number {
    try {
      const ids = JSON.parse(sourceJobIds) as string[];
      return ids.length;
    } catch {
      return 0;
    }
  }

  return (
    <div className="border rounded-md">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">Training Jobs</h3>
          <Badge variant="secondary">{jobs.length}</Badge>
        </div>
      </div>

      {jobs.length === 0 ? (
        <div className="px-4 py-6 text-center text-sm text-muted-foreground">
          No training jobs yet.
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-3 py-2 text-left font-medium">ID</th>
              <th className="px-3 py-2 text-left font-medium">Status</th>
              <th className="px-3 py-2 text-right font-medium">
                Source Jobs
              </th>
              <th className="px-3 py-2 text-left font-medium">Model</th>
              <th className="px-3 py-2 text-left font-medium">Created</th>
              <th className="px-3 py-2 text-left font-medium" />
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id} className="border-b hover:bg-muted/30">
                <td className="px-3 py-2 font-mono text-xs">
                  {job.id.slice(0, 8)}
                </td>
                <td className="px-3 py-2">
                  <StatusBadge status={job.status} />
                </td>
                <td className="px-3 py-2 text-right">
                  {sourceCount(job.source_job_ids)}
                </td>
                <td className="px-3 py-2 text-xs">
                  {job.vocalization_model_id
                    ? job.vocalization_model_id.slice(0, 8)
                    : job.status === "failed"
                      ? (
                          <span className="text-red-600" title={job.error_message ?? ""}>
                            failed
                          </span>
                        )
                      : "—"}
                </td>
                <td className="px-3 py-2 text-xs text-muted-foreground whitespace-nowrap">
                  {new Date(job.created_at).toLocaleDateString()}
                </td>
                <td className="px-3 py-2 text-right">
                  <DeleteConfirmButton
                    size="sm"
                    resourceType="training job"
                    resourceName={job.id.slice(0, 8)}
                    consequence="This event classifier training job will be removed. Models already produced by completed jobs will remain."
                    onConfirm={() => handleDelete(job.id)}
                    isPending={deleteMutation.isPending}
                  >
                    Delete
                  </DeleteConfirmButton>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
