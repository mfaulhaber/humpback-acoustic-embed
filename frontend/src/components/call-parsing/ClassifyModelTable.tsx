import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  useEventClassifierModels,
  useDeleteEventClassifierModel,
} from "@/hooks/queries/useCallParsing";
import { toast } from "@/components/ui/use-toast";
import type { EventClassifierModel } from "@/api/types";
import { typeColor } from "./TypePalette";

interface PerTypeMetric {
  type_name: string;
  precision: number;
  recall: number;
  f1: number;
  threshold: number;
  support: number;
}

function parsePerClassMetrics(model: EventClassifierModel): PerTypeMetric[] {
  if (!model.per_class_metrics) return [];
  try {
    const metrics = JSON.parse(model.per_class_metrics) as Record<
      string,
      Record<string, number>
    >;
    return Object.entries(metrics)
      .map(([type_name, m]) => ({
        type_name,
        precision: m.precision ?? 0,
        recall: m.recall ?? 0,
        f1: m.f1 ?? 0,
        threshold: m.threshold ?? 0.5,
        support: m.support ?? 0,
      }))
      .sort((a, b) => b.f1 - a.f1);
  } catch {
    return [];
  }
}

function avgF1(model: EventClassifierModel): number | null {
  const metrics = parsePerClassMetrics(model);
  if (metrics.length === 0) return null;
  return metrics.reduce((sum, m) => sum + m.f1, 0) / metrics.length;
}

function f1Color(val: number | null): string {
  if (val == null) return "";
  if (val >= 0.8) return "text-green-600";
  if (val >= 0.6) return "text-amber-600";
  return "text-red-600";
}

export function ClassifyModelTable() {
  const { data: models = [] } = useEventClassifierModels();
  const deleteMutation = useDeleteEventClassifierModel();
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggleExpand = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const handleDelete = (modelId: string) => {
    if (!confirm("Delete this event classifier model?")) return;
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
          <h3 className="text-sm font-semibold">Event Classifier Models</h3>
          <Badge variant="secondary">{models.length}</Badge>
        </div>
      </div>

      {models.length === 0 ? (
        <div className="px-4 py-6 text-center text-sm text-muted-foreground">
          No event classifier models yet.
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="w-8" />
              <th className="px-3 py-2 text-left font-medium">Name</th>
              <th className="px-3 py-2 text-left font-medium">Types</th>
              <th className="px-3 py-2 text-left font-medium">Avg F1</th>
              <th className="px-3 py-2 text-left font-medium">Created</th>
              <th className="px-3 py-2 text-left font-medium" />
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const metrics = parsePerClassMetrics(m);
              const avg = avgF1(m);
              return (
                <ModelRow
                  key={m.id}
                  model={m}
                  metrics={metrics}
                  avgF1={avg}
                  isExpanded={expanded.has(m.id)}
                  onToggleExpand={() => toggleExpand(m.id)}
                  onDelete={() => handleDelete(m.id)}
                  isDeleting={deleteMutation.isPending}
                />
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

function ModelRow({
  model,
  metrics,
  avgF1: avg,
  isExpanded,
  onToggleExpand,
  onDelete,
  isDeleting,
}: {
  model: EventClassifierModel;
  metrics: PerTypeMetric[];
  avgF1: number | null;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onDelete: () => void;
  isDeleting: boolean;
}) {
  return (
    <>
      <tr className="border-b hover:bg-muted/30">
        <td className="px-2 py-2 text-center">
          {metrics.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0"
              onClick={onToggleExpand}
            >
              <ChevronDown
                className={`h-3 w-3 transition-transform ${isExpanded ? "rotate-180" : ""}`}
              />
            </Button>
          )}
        </td>
        <td className="px-3 py-2 font-medium">{model.name}</td>
        <td className="px-3 py-2">{metrics.length || "—"}</td>
        <td className={`px-3 py-2 ${f1Color(avg)}`}>
          {avg != null ? avg.toFixed(2) : "—"}
        </td>
        <td className="px-3 py-2 text-xs text-muted-foreground whitespace-nowrap">
          {new Date(model.created_at).toLocaleDateString()}
        </td>
        <td className="px-3 py-2 text-right">
          <Button
            variant="ghost"
            size="sm"
            className="text-red-600 hover:text-red-700"
            onClick={onDelete}
            disabled={isDeleting}
          >
            Delete
          </Button>
        </td>
      </tr>
      {isExpanded && metrics.length > 0 && (
        <tr>
          <td colSpan={6} className="bg-muted/20">
            <div className="px-8 py-2">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-muted-foreground">
                    <th className="text-left py-1">Type</th>
                    <th className="text-right py-1">Precision</th>
                    <th className="text-right py-1">Recall</th>
                    <th className="text-right py-1">F1</th>
                    <th className="text-right py-1">Threshold</th>
                    <th className="text-right py-1">Samples</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m) => (
                    <tr key={m.type_name} className="border-t border-muted">
                      <td className="py-1">
                        <Badge
                          variant="outline"
                          className="text-xs"
                          style={{
                            borderColor: typeColor(m.type_name),
                            color: typeColor(m.type_name),
                          }}
                        >
                          {m.type_name}
                        </Badge>
                      </td>
                      <td className="text-right py-1">
                        {m.precision.toFixed(3)}
                      </td>
                      <td className="text-right py-1">
                        {m.recall.toFixed(3)}
                      </td>
                      <td className={`text-right py-1 ${f1Color(m.f1)}`}>
                        {m.f1.toFixed(3)}
                      </td>
                      <td className="text-right py-1">
                        {m.threshold.toFixed(3)}
                      </td>
                      <td className="text-right py-1">{m.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
