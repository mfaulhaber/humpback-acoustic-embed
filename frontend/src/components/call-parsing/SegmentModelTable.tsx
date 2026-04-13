import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  useSegmentationModels,
  useDeleteSegmentationModel,
} from "@/hooks/queries/useCallParsing";
import { toast } from "@/components/ui/use-toast";
import type { SegmentationModel } from "@/api/types";

interface Metrics {
  framewise_f1: number | null;
  event_f1: number | null;
}

function parseMetrics(model: SegmentationModel): Metrics {
  if (!model.config_json) return { framewise_f1: null, event_f1: null };
  try {
    const cfg = JSON.parse(model.config_json) as Record<string, unknown>;
    return {
      framewise_f1:
        typeof cfg.framewise_f1 === "number" ? cfg.framewise_f1 : null,
      event_f1:
        typeof cfg.event_f1_iou_0_3 === "number"
          ? cfg.event_f1_iou_0_3
          : null,
    };
  } catch {
    return { framewise_f1: null, event_f1: null };
  }
}

function f1Color(val: number | null): string {
  if (val == null) return "";
  if (val >= 0.7) return "text-green-600";
  if (val >= 0.5) return "text-amber-600";
  return "";
}

export function SegmentModelTable() {
  const { data: models = [] } = useSegmentationModels();
  const deleteMutation = useDeleteSegmentationModel();

  const handleDelete = (modelId: string) => {
    if (!confirm("Delete this segmentation model?")) return;
    deleteMutation.mutate(modelId, {
      onError: (err) => {
        toast({
          title: "Cannot delete model",
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
          <h3 className="text-sm font-semibold">Segmentation Models</h3>
          <Badge variant="secondary">{models.length}</Badge>
        </div>
      </div>

      {models.length === 0 ? (
        <div className="px-4 py-6 text-center text-sm text-muted-foreground">
          No models yet.
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-3 py-2 text-left font-medium">Name</th>
              <th className="px-3 py-2 text-left font-medium">Family</th>
              <th className="px-3 py-2 text-left font-medium">
                Framewise F1
              </th>
              <th className="px-3 py-2 text-left font-medium">
                Event F1 (IoU≥0.3)
              </th>
              <th className="px-3 py-2 text-left font-medium">Created</th>
              <th className="px-3 py-2 text-left font-medium" />
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const metrics = parseMetrics(m);
              return (
                <tr key={m.id} className="border-b hover:bg-muted/30">
                  <td className="px-3 py-2 font-medium">{m.name}</td>
                  <td className="px-3 py-2 text-muted-foreground text-xs">
                    {m.model_family}
                  </td>
                  <td className={`px-3 py-2 ${f1Color(metrics.framewise_f1)}`}>
                    {metrics.framewise_f1 != null
                      ? metrics.framewise_f1.toFixed(2)
                      : "—"}
                  </td>
                  <td className={`px-3 py-2 ${f1Color(metrics.event_f1)}`}>
                    {metrics.event_f1 != null
                      ? metrics.event_f1.toFixed(2)
                      : "—"}
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground whitespace-nowrap">
                    {new Date(m.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-red-600 hover:text-red-700"
                      onClick={() => handleDelete(m.id)}
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
