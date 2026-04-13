import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/shared/StatusBadge";
import {
  useSegmentationTrainingJobs,
  useDeleteSegmentationTrainingJob,
  useSegmentationModels,
  useSegmentationTrainingDatasets,
} from "@/hooks/queries/useCallParsing";
import { toast } from "@/components/ui/use-toast";
import type { SegmentationTrainingJob, SegmentationModel } from "@/api/types";

function configSummary(job: SegmentationTrainingJob): string {
  try {
    const cfg = JSON.parse(job.config_json) as Record<string, unknown>;
    const ep = typeof cfg.epochs === "number" ? cfg.epochs : "?";
    const lr = typeof cfg.learning_rate === "number" ? cfg.learning_rate : "?";
    return `${ep} ep · lr=${lr}`;
  } catch {
    return "—";
  }
}

interface Metrics {
  framewise_f1: number | null;
  event_f1: number | null;
}

function parseMetrics(job: SegmentationTrainingJob): Metrics {
  if (!job.result_summary) return { framewise_f1: null, event_f1: null };
  try {
    const rs = JSON.parse(job.result_summary) as Record<string, unknown>;
    return {
      framewise_f1:
        typeof rs.framewise_f1 === "number" ? rs.framewise_f1 : null,
      event_f1:
        typeof rs.event_f1_iou_0_3 === "number" ? rs.event_f1_iou_0_3 : null,
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

function datasetNameById(
  datasetId: string,
  datasets: { id: string; name: string }[],
): string {
  const d = datasets.find((ds) => ds.id === datasetId);
  return d ? d.name : datasetId.slice(0, 8);
}

export function SegmentTrainingJobTable() {
  const { data: jobs = [] } = useSegmentationTrainingJobs(3000);
  const { data: models = [] } = useSegmentationModels();
  const { data: datasets = [] } = useSegmentationTrainingDatasets();
  const deleteMutation = useDeleteSegmentationTrainingJob();

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
              <th className="px-3 py-2 text-left font-medium">Status</th>
              <th className="px-3 py-2 text-left font-medium">Created</th>
              <th className="px-3 py-2 text-left font-medium">Dataset</th>
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
                  <td className="px-3 py-2 text-xs">
                    {datasetNameById(job.training_dataset_id, datasets)}
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">
                    {configSummary(job)}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {mName ? (
                      <Link
                        to="/app/call-parsing/segment-training"
                        className="text-blue-600 underline"
                      >
                        {mName}
                      </Link>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {metrics.framewise_f1 != null || metrics.event_f1 != null
                      ? [
                          metrics.framewise_f1 != null
                            ? `F1ᶠ=${metrics.framewise_f1.toFixed(2)}`
                            : null,
                          metrics.event_f1 != null
                            ? `F1ᵉ=${metrics.event_f1.toFixed(2)}`
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
