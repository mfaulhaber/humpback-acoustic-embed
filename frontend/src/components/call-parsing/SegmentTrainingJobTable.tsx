import { Badge } from "@/components/ui/badge";
import { useSegmentationTrainingJobs } from "@/hooks/queries/useCallParsing";
import type { SegmentationTrainingJob } from "@/api/types";

const statusColor: Record<string, string> = {
  queued: "bg-yellow-100 text-yellow-800",
  running: "bg-blue-100 text-blue-800",
  complete: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
};

function shortId(id: string | null): string {
  return id ? id.slice(0, 8) : "—";
}

function formatDate(value: string | null): string {
  if (!value) return "—";
  return new Date(value).toLocaleDateString();
}

function configSummary(job: SegmentationTrainingJob): string {
  try {
    const config = JSON.parse(job.config_json) as Record<string, unknown>;
    const epochs = config.epochs;
    const batchSize = config.batch_size;
    const nMels = config.n_mels;
    if (
      typeof epochs === "number" &&
      typeof batchSize === "number" &&
      typeof nMels === "number"
    ) {
      return `${epochs} ep / batch ${batchSize} / ${nMels} mels`;
    }
  } catch {
    return "—";
  }
  return "—";
}

function modelLabel(job: SegmentationTrainingJob): string {
  if (job.segmentation_model_id) return shortId(job.segmentation_model_id);
  if (job.status === "queued") return `queued-${shortId(job.id)}`;
  if (job.status === "running") return `training-${shortId(job.id)}`;
  return "—";
}

function SegmentTrainingJobRow({ job }: { job: SegmentationTrainingJob }) {
  return (
    <tr className="border-b last:border-0 hover:bg-muted/30">
      <td className="px-3 py-2">
        <Badge className={statusColor[job.status] ?? ""}>{job.status}</Badge>
      </td>
      <td className="px-3 py-2 font-medium">{modelLabel(job)}</td>
      <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
        {shortId(job.training_dataset_id)}
      </td>
      <td className="px-3 py-2 text-xs text-muted-foreground">
        {configSummary(job)}
      </td>
      <td className="px-3 py-2 text-xs text-muted-foreground whitespace-nowrap">
        {formatDate(job.created_at)}
      </td>
      <td className="px-3 py-2 text-xs text-muted-foreground whitespace-nowrap">
        {formatDate(job.completed_at)}
      </td>
      <td className="px-3 py-2">
        {job.error_message && (
          <span className="block max-w-64 truncate text-xs text-red-600">
            {job.error_message}
          </span>
        )}
      </td>
    </tr>
  );
}

export function SegmentTrainingJobTable() {
  const { data: jobs = [] } = useSegmentationTrainingJobs(3000);
  const hasActiveJobs = jobs.some(
    (job) => job.status === "queued" || job.status === "running",
  );

  return (
    <div className="border rounded-md">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">Previous Jobs</h3>
          <Badge variant="secondary">{jobs.length}</Badge>
          {hasActiveJobs && (
            <span className="text-xs text-muted-foreground">(polling...)</span>
          )}
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
              <th className="px-3 py-2 text-left font-medium">Status</th>
              <th className="px-3 py-2 text-left font-medium">Model</th>
              <th className="px-3 py-2 text-left font-medium">Dataset</th>
              <th className="px-3 py-2 text-left font-medium">Config</th>
              <th className="px-3 py-2 text-left font-medium">Created</th>
              <th className="px-3 py-2 text-left font-medium">Completed</th>
              <th className="px-3 py-2 text-left font-medium">Error</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <SegmentTrainingJobRow key={job.id} job={job} />
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
