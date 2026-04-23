import { Button } from "@/components/ui/button";
import { useDeleteWindowClassificationJob } from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import type {
  WindowClassificationJob,
  RegionDetectionJob,
} from "@/api/types";

function sourceLabel(
  job: WindowClassificationJob,
  regionJobs: RegionDetectionJob[],
  hydrophones: Array<{ id: string; name: string }>,
): string {
  const rj = regionJobs.find((r) => r.id === job.region_detection_job_id);
  if (!rj) return "—";
  if (rj.hydrophone_id) {
    return (
      hydrophones.find((hp) => hp.id === rj.hydrophone_id)?.name ??
      rj.hydrophone_id
    );
  }
  return "file";
}

interface WindowClassifyJobTableProps {
  jobs: WindowClassificationJob[];
  regionJobs: RegionDetectionJob[];
  title: string;
  showReview?: boolean;
  onReview?: (jobId: string) => void;
}

export function WindowClassifyJobTable({
  jobs,
  regionJobs,
  title,
  showReview,
  onReview,
}: WindowClassifyJobTableProps) {
  const { data: hydrophones = [] } = useHydrophones();
  const deleteMutation = useDeleteWindowClassificationJob();

  if (jobs.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold">{title}</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-slate-500 text-xs">
            <th className="text-left py-2 px-2">Source</th>
            <th className="text-left py-2 px-2">Region Job</th>
            <th className="text-left py-2 px-2">Model</th>
            <th className="text-left py-2 px-2">Windows</th>
            <th className="text-left py-2 px-2">Status</th>
            <th className="py-2 px-2" />
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id} className="border-b hover:bg-slate-50">
              <td className="py-2 px-2">
                {sourceLabel(job, regionJobs, hydrophones)}
              </td>
              <td className="py-2 px-2 font-mono text-xs">
                {job.region_detection_job_id.slice(0, 8)}
              </td>
              <td className="py-2 px-2 font-mono text-xs">
                {job.vocalization_model_id.slice(0, 8)}
              </td>
              <td className="py-2 px-2">{job.window_count ?? "—"}</td>
              <td className="py-2 px-2">
                <span
                  className={
                    job.status === "complete"
                      ? "text-green-600"
                      : job.status === "failed"
                        ? "text-red-600"
                        : job.status === "running"
                          ? "text-blue-600"
                          : ""
                  }
                >
                  {job.status}
                </span>
              </td>
              <td className="py-2 px-2 text-right space-x-2">
                {showReview && job.status === "complete" && onReview && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => onReview(job.id)}
                  >
                    Review
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => deleteMutation.mutate(job.id)}
                  disabled={deleteMutation.isPending}
                >
                  Delete
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
