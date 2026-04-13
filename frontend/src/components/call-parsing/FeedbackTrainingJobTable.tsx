import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/shared/StatusBadge";
import {
  useSegmentationFeedbackTrainingJobs,
  useDeleteSegmentationFeedbackTrainingJob,
  useSegmentationModels,
} from "@/hooks/queries/useCallParsing";
import { toast } from "@/components/ui/use-toast";
import type {
  SegmentationFeedbackTrainingJob,
  SegmentationModel,
} from "@/api/types";

function configSummary(job: SegmentationFeedbackTrainingJob): string {
  if (!job.config_json) return "defaults";
  try {
    const cfg = JSON.parse(job.config_json) as Record<string, unknown>;
    const ep = typeof cfg.epochs === "number" ? cfg.epochs : "?";
    const lr = typeof cfg.learning_rate === "number" ? cfg.learning_rate : "?";
    return `${ep} ep · lr=${lr}`;
  } catch {
    return "—";
  }
}

function sourceJobsSummary(job: SegmentationFeedbackTrainingJob): string {
  try {
    const ids = JSON.parse(job.source_job_ids) as string[];
    return ids.map((id) => id.slice(0, 8)).join(", ");
  } catch {
    return "—";
  }
}

interface Metrics {
  framewise_f1: number | null;
  event_f1: number | null;
}

function parseMetrics(job: SegmentationFeedbackTrainingJob): Metrics {
  if (!job.result_summary) return { framewise_f1: null, event_f1: null };
  try {
    const rs = JSON.parse(job.result_summary) as Record<string, unknown>;
    const fw = rs.framewise as Record<string, unknown> | undefined;
    const ev = rs.event as Record<string, unknown> | undefined;
    return {
      framewise_f1: typeof fw?.f1 === "number" ? fw.f1 : null,
      event_f1: typeof ev?.f1 === "number" ? ev.f1 : null,
    };
  } catch {
    return { framewise_f1: null, event_f1: null };
  }
}

function modelNameById(
  modelId: string | null,
  models: SegmentationModel[],
): string | null {
  if (!modelId) return null;
  const m = models.find((mo) => mo.id === modelId);
  return m ? m.name : modelId.slice(0, 8);
}

export function FeedbackTrainingJobTable() {
  const { data: jobs = [] } = useSegmentationFeedbackTrainingJobs(3000);
  const { data: models = [] } = useSegmentationModels();
  const deleteMutation = useDeleteSegmentationFeedbackTrainingJob();

  const handleDelete = (jobId: string) => {
    if (!confirm("Delete this training job?")) return;
    deleteMutation.mutate(jobId, {
      onError: (err) => {
        toast({
          title: "Cannot delete training job",
          description: (err as Error).message,
          variant: "destructive",
        });
      },
    });
  };

  return (
    <div className="border rounded-md">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">Feedback Training Jobs</h3>
          <Badge variant="secondary">{jobs.length}</Badge>
        </div>
      </div>

      {jobs.length === 0 ? (
        <div className="px-4 py-6 text-center text-sm text-muted-foreground">
          No training jobs yet. Use the Retrain button in Segment Review to
          start a feedback training job.
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-3 py-2 text-left font-medium">Status</th>
              <th className="px-3 py-2 text-left font-medium">Created</th>
              <th className="px-3 py-2 text-left font-medium">Source Jobs</th>
              <th className="px-3 py-2 text-left font-medium">Config</th>
              <th className="px-3 py-2 text-left font-medium">Model</th>
              <th className="px-3 py-2 text-left font-medium">Metrics</th>
              <th className="px-3 py-2 text-left font-medium" />
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => {
              const metrics = parseMetrics(job);
              const mName = modelNameById(
                job.segmentation_model_id,
                models,
              );
              return (
                <tr key={job.id} className="border-b hover:bg-muted/30">
                  <td className="px-3 py-2">
                    <StatusBadge status={job.status} />
                  </td>
                  <td className="px-3 py-2 text-xs whitespace-nowrap">
                    {new Date(job.created_at).toLocaleString()}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {sourceJobsSummary(job)}
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">
                    {configSummary(job)}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {mName ? (
                      <span className="text-blue-600">{mName}</span>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {metrics.framewise_f1 != null || metrics.event_f1 != null
                      ? [
                          metrics.framewise_f1 != null
                            ? `F1\u1DA0=${metrics.framewise_f1.toFixed(2)}`
                            : null,
                          metrics.event_f1 != null
                            ? `F1\u1D49=${metrics.event_f1.toFixed(2)}`
                            : null,
                        ]
                          .filter(Boolean)
                          .join(" · ")
                      : "—"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-red-600 hover:text-red-700"
                      onClick={() => handleDelete(job.id)}
                      disabled={deleteMutation.isPending}
                    >
                      Delete
                    </Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
