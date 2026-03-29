import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronRight, Star } from "lucide-react";
import {
  useVocClassifierModels,
  useActivateVocClassifierModel,
} from "@/hooks/queries/useVocalization";
import type { VocClassifierModel } from "@/api/types";
import { fmtDate, shortId } from "@/utils/format";

export function VocalizationModelList() {
  const { data: models = [], isLoading } = useVocClassifierModels();
  const activateMut = useActivateVocClassifierModel();

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Models</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Models</CardTitle>
      </CardHeader>
      <CardContent>
        {models.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No vocalization models trained yet. Train one using the form above.
          </p>
        ) : (
          <div className="border rounded-md divide-y">
            {models.map((m) => (
              <ModelRow
                key={m.id}
                model={m}
                onActivate={() => activateMut.mutate(m.id)}
                isActivating={activateMut.isPending}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ModelRow({
  model,
  onActivate,
  isActivating,
}: {
  model: VocClassifierModel;
  onActivate: () => void;
  isActivating: boolean;
}) {
  const [open, setOpen] = useState(false);
  const metrics = model.per_class_metrics as Record<
    string,
    { ap?: number; f1?: number; precision?: number; recall?: number; n_samples?: number }
  > | null;

  const meanF1 = computeMeanF1(metrics);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className="px-3 py-2">
        <CollapsibleTrigger className="flex items-center justify-between w-full text-sm">
          <div className="flex items-center gap-2 min-w-0">
            <ChevronRight
              className={`h-3.5 w-3.5 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
            />
            <span className="font-medium truncate">{model.name || shortId(model.id)}</span>
            <Badge variant="outline" className="shrink-0">
              {model.vocabulary_snapshot.length} types
            </Badge>
            {meanF1 !== null && (
              <span className="text-xs text-muted-foreground shrink-0">
                F1 {(meanF1 * 100).toFixed(1)}%
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {model.is_active ? (
              <Badge className="bg-green-100 text-green-800 border-green-200">
                <Star className="h-3 w-3 mr-0.5 fill-current" />
                Active
              </Badge>
            ) : (
              <Button
                size="sm"
                variant="outline"
                className="h-7"
                onClick={(e) => {
                  e.stopPropagation();
                  onActivate();
                }}
                disabled={isActivating}
              >
                Activate
              </Button>
            )}
            <span className="text-xs text-muted-foreground">{fmtDate(model.created_at)}</span>
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent className="mt-2">
          <div className="space-y-2 pl-5">
            {/* Vocabulary */}
            <div>
              <span className="text-xs font-medium text-muted-foreground">Vocabulary:</span>
              <div className="flex flex-wrap gap-1 mt-0.5">
                {model.vocabulary_snapshot.map((t) => (
                  <Badge key={t} variant="secondary" className="text-xs">
                    {t}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Per-class thresholds */}
            <div>
              <span className="text-xs font-medium text-muted-foreground">Thresholds:</span>
              <div className="flex flex-wrap gap-2 mt-0.5 text-xs">
                {Object.entries(model.per_class_thresholds).map(([type, thresh]) => (
                  <span key={type}>
                    {type}: {thresh.toFixed(3)}
                  </span>
                ))}
              </div>
            </div>

            {/* Per-class metrics table */}
            {metrics && Object.keys(metrics).length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr className="border-b text-muted-foreground">
                      <th className="text-left py-1 pr-3">Type</th>
                      <th className="text-right py-1 px-2">AP</th>
                      <th className="text-right py-1 px-2">F1</th>
                      <th className="text-right py-1 px-2">Precision</th>
                      <th className="text-right py-1 px-2">Recall</th>
                      <th className="text-right py-1 px-2">Samples</th>
                      <th className="text-right py-1 pl-2">Threshold</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics).map(([type, m]) => (
                      <tr key={type} className="border-b last:border-0">
                        <td className="py-1 pr-3 font-medium">{type}</td>
                        <td className="text-right py-1 px-2">{pct(m.ap)}</td>
                        <td className="text-right py-1 px-2">{pct(m.f1)}</td>
                        <td className="text-right py-1 px-2">{pct(m.precision)}</td>
                        <td className="text-right py-1 px-2">{pct(m.recall)}</td>
                        <td className="text-right py-1 px-2">{m.n_samples ?? "—"}</td>
                        <td className="text-right py-1 pl-2">
                          {model.per_class_thresholds[type]?.toFixed(3) ?? "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Training summary */}
            {model.training_summary && (
              <div className="text-xs text-muted-foreground">
                <span className="font-medium">Training:</span>{" "}
                {typeof model.training_summary.n_types === "number" &&
                  `${model.training_summary.n_types} types, `}
                {typeof model.training_summary.total_samples === "number" &&
                  `${model.training_summary.total_samples} samples`}
              </div>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

function pct(v: number | undefined): string {
  if (v === undefined || v === null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function computeMeanF1(
  metrics: Record<string, { f1?: number }> | null,
): number | null {
  if (!metrics) return null;
  const vals = Object.values(metrics)
    .map((m) => m.f1)
    .filter((v): v is number => v !== undefined && v !== null);
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}
