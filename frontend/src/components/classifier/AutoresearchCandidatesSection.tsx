import { useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Loader2,
} from "lucide-react";

import type {
  AutoresearchCandidateDetail,
  AutoresearchCandidateSummary,
} from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import {
  useAutoresearchCandidate,
  useAutoresearchCandidates,
  useCreateAutoresearchCandidateTrainingJob,
  useImportAutoresearchCandidate,
} from "@/hooks/queries/useClassifier";
import { cn } from "@/lib/utils";

const candidateStatusColor: Record<string, string> = {
  imported: "bg-slate-100 text-slate-800",
  promotable: "bg-green-100 text-green-800",
  blocked: "bg-red-100 text-red-800",
  training: "bg-blue-100 text-blue-800",
  complete: "bg-emerald-100 text-emerald-800",
  failed: "bg-red-100 text-red-800",
};

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
}

function asList(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function emptyToUndefined(value: string): string | undefined {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function formatConfigValue(value: unknown): string {
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => formatConfigValue(item)).join(", ");
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return "—";
}

function pickPreferredSplit(payload: Record<string, unknown> | null | undefined) {
  const record = asRecord(payload);
  if (!record) {
    return null;
  }
  for (const key of ["test", "val", "validation", "train"]) {
    if (asRecord(record[key])) {
      return key;
    }
  }
  return Object.keys(record)[0] ?? null;
}

function getSplitRecord(
  payload: Record<string, unknown> | null | undefined,
  splitName: string | null,
) {
  if (!splitName) {
    return null;
  }
  return asRecord(asRecord(payload)?.[splitName]);
}

function formatMetricValue(metric: string, value: unknown): string {
  const numeric = asNumber(value);
  if (numeric == null) {
    return "—";
  }
  if (
    metric === "precision" ||
    metric === "recall" ||
    metric.endsWith("_rate") ||
    metric === "threshold"
  ) {
    return `${(numeric * 100).toFixed(1)}%`;
  }
  return Number.isInteger(numeric) ? numeric.toString() : numeric.toFixed(3);
}

function formatDeltaValue(metric: string, value: unknown): string {
  const numeric = asNumber(value);
  if (numeric == null) {
    return "—";
  }
  const prefix = numeric > 0 ? "+" : "";
  if (metric === "precision" || metric === "recall" || metric.endsWith("_rate")) {
    return `${prefix}${(numeric * 100).toFixed(1)} pp`;
  }
  return `${prefix}${Number.isInteger(numeric) ? numeric : numeric.toFixed(3)}`;
}

function metricLabel(metric: string): string {
  const labels: Record<string, string> = {
    precision: "Precision",
    recall: "Recall",
    fp_rate: "FP Rate",
    high_conf_fp_rate: "High-Conf FP",
    tp: "TP",
    fp: "FP",
    fn: "FN",
    tn: "TN",
  };
  return labels[metric] ?? metric;
}

function buildPrimaryDeltas(candidate: AutoresearchCandidateSummary) {
  const splitName = pickPreferredSplit(candidate.metric_deltas);
  const deltas = getSplitRecord(candidate.metric_deltas, splitName);
  if (!deltas) {
    return [];
  }
  const priorities = [
    "precision",
    "recall",
    "fp_rate",
    "high_conf_fp_rate",
    "fp",
  ];
  return priorities
    .filter((metric) => deltas[metric] != null)
    .slice(0, 3)
    .map((metric) => ({
      metric,
      splitName,
      value: deltas[metric],
    }));
}

function formatReplaySummary(summary: Record<string, unknown> | null | undefined) {
  const replay = asRecord(summary);
  if (!replay) {
    return "Replay summary unavailable";
  }
  const replayed = asNumber(replay.replayed_hard_negatives);
  const available = asNumber(replay.available_hard_negatives);
  const usedReplay = replay.used_replay_manifest === true;
  if (replayed != null || available != null) {
    return `Replay ${replayed ?? 0}/${available ?? 0} hard negatives`;
  }
  return usedReplay ? "Replay manifest used" : "No replayed hard negatives";
}

function formatSplitCounts(sourceCounts: Record<string, unknown> | null | undefined) {
  const splitCounts = asRecord(asRecord(sourceCounts)?.split_counts);
  if (!splitCounts) {
    return null;
  }
  const entries = Object.entries(splitCounts)
    .map(([splitName, count]) => {
      const numeric = asNumber(count);
      return numeric != null ? `${splitName}: ${numeric}` : null;
    })
    .filter((value): value is string => value !== null);
  return entries.length > 0 ? entries.join(" • ") : null;
}

function defaultPromotionName(candidate: AutoresearchCandidateSummary) {
  const base = candidate.name.trim().replace(/\s+/g, "-").toLowerCase();
  return base.includes("promoted") ? base : `${base}-promoted`;
}

function summarizePreviewItem(item: unknown) {
  const record = asRecord(item);
  if (!record) {
    return formatConfigValue(item);
  }

  const preferredKeys = [
    "row_id",
    "audio_file_id",
    "start_utc",
    "label",
    "source_type",
    "negative_group",
    "autoresearch_score",
    "production_score",
    "confidence",
  ];
  const entries = preferredKeys
    .filter((key) => record[key] != null)
    .slice(0, 4)
    .map((key) => `${key}=${formatConfigValue(record[key])}`);
  if (entries.length > 0) {
    return entries.join(" • ");
  }
  const fallback = Object.entries(record)
    .slice(0, 4)
    .map(([key, value]) => `${key}=${formatConfigValue(value)}`);
  return fallback.join(" • ");
}

function DetailField({
  label,
  value,
}: {
  label: string;
  value: string | null | undefined;
}) {
  return (
    <div className="space-y-0.5">
      <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className="text-sm">{value && value.length > 0 ? value : "—"}</div>
    </div>
  );
}

function PreviewPanel({
  title,
  items,
  emptyLabel,
}: {
  title: string;
  items: unknown[];
  emptyLabel: string;
}) {
  return (
    <div className="rounded-md border p-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </div>
      <div className="mt-2 space-y-2">
        {items.length === 0 ? (
          <div className="text-xs text-muted-foreground">{emptyLabel}</div>
        ) : (
          items.slice(0, 5).map((item, index) => (
            <div
              key={`${title}-${index}`}
              className="rounded-md bg-muted/40 px-2.5 py-2 text-xs text-muted-foreground"
            >
              {summarizePreviewItem(item)}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function ComparisonTable({
  detail,
  splitName,
}: {
  detail: AutoresearchCandidateDetail | null;
  splitName: string | null;
}) {
  const splitMetrics = getSplitRecord(detail?.split_metrics, splitName);
  const deltas = getSplitRecord(detail?.metric_deltas, splitName);
  const autoresearch = asRecord(splitMetrics?.autoresearch);
  const production = asRecord(splitMetrics?.production);

  const metrics = [
    "precision",
    "recall",
    "fp_rate",
    "high_conf_fp_rate",
    "fp",
    "fn",
  ].filter(
    (metric) =>
      autoresearch?.[metric] != null ||
      production?.[metric] != null ||
      deltas?.[metric] != null,
  );

  return (
    <div className="rounded-md border p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          Comparison Metrics
        </div>
        {splitName && (
          <Badge variant="outline" className="text-[10px]">
            {splitName} split
          </Badge>
        )}
      </div>
      {metrics.length === 0 ? (
        <div className="mt-2 text-xs text-muted-foreground">
          No production comparison metrics were imported for this candidate.
        </div>
      ) : (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-1.5 font-medium">Metric</th>
                <th className="py-1.5 font-medium">Autoresearch</th>
                <th className="py-1.5 font-medium">Production</th>
                <th className="py-1.5 font-medium">Delta</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((metric) => (
                <tr key={metric} className="border-b last:border-0">
                  <td className="py-1.5">{metricLabel(metric)}</td>
                  <td className="py-1.5 text-muted-foreground">
                    {formatMetricValue(metric, autoresearch?.[metric])}
                  </td>
                  <td className="py-1.5 text-muted-foreground">
                    {formatMetricValue(metric, production?.[metric])}
                  </td>
                  <td className="py-1.5">
                    <span
                      className={cn(
                        "font-medium",
                        asNumber(deltas?.[metric]) != null &&
                          asNumber(deltas?.[metric])! < 0 &&
                          (metric === "precision" ||
                            metric === "recall" ||
                            metric === "tp")
                          ? "text-red-700"
                          : "text-green-700",
                      )}
                    >
                      {formatDeltaValue(metric, deltas?.[metric])}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function CandidateCard({
  candidate,
  open,
  onOpenChange,
}: {
  candidate: AutoresearchCandidateSummary;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [newModelName, setNewModelName] = useState(defaultPromotionName(candidate));
  const [notes, setNotes] = useState("");

  const { data: detail, isLoading, isError } = useAutoresearchCandidate(
    open ? candidate.id : null,
  );
  const promoteMutation = useCreateAutoresearchCandidateTrainingJob();

  const deltaEntries = useMemo(() => buildPrimaryDeltas(candidate), [candidate]);
  const splitName = useMemo(
    () => pickPreferredSplit((detail ?? candidate).split_metrics),
    [candidate, detail],
  );
  const disagreementPreview = useMemo(() => {
    const preview = asRecord(detail?.prediction_disagreements_preview);
    return asList(splitName ? preview?.[splitName] : null);
  }, [detail, splitName]);
  const falsePositivePreview = useMemo(() => {
    const preview = asRecord(detail?.top_false_positives_preview);
    const splitPreview = splitName ? asRecord(preview?.[splitName]) : null;
    const splitItems = asList(splitPreview?.autoresearch);
    if (splitItems.length > 0) {
      return splitItems;
    }
    return asList(preview?.imported);
  }, [detail, splitName]);

  const sourceCounts = asRecord((detail ?? candidate).source_counts);
  const splitCountsLabel = formatSplitCounts(sourceCounts);
  const exampleCount = asNumber(sourceCounts?.example_count);
  const warnings = (detail ?? candidate).warnings;
  const sourceModelLabel =
    candidate.source_model_name ?? candidate.source_model_id ?? "Unknown model";

  const handlePromote = () => {
    if (!newModelName.trim()) {
      return;
    }
    promoteMutation.mutate({
      candidateId: candidate.id,
      body: {
        new_model_name: newModelName.trim(),
        notes: emptyToUndefined(notes),
      },
    });
  };

  return (
    <Collapsible open={open} onOpenChange={onOpenChange}>
      <div className="rounded-lg border">
        <CollapsibleTrigger asChild>
          <button
            type="button"
            className="w-full px-4 py-3 text-left transition-colors hover:bg-muted/30"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0 flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  {open ? (
                    <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
                  )}
                  <span className="truncate font-medium">{candidate.name}</span>
                  <Badge className={candidateStatusColor[candidate.status] ?? ""}>
                    {candidate.status}
                  </Badge>
                  {candidate.phase && (
                    <Badge variant="outline">{candidate.phase}</Badge>
                  )}
                  {candidate.is_reproducible_exact && (
                    <Badge className="bg-emerald-100 text-emerald-800">
                      exact replay
                    </Badge>
                  )}
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                  <span>Source model: {sourceModelLabel}</span>
                  <span>Objective: {candidate.objective_name ?? "default"}</span>
                  <span>{formatReplaySummary(candidate.replay_summary)}</span>
                  {exampleCount != null && <span>{exampleCount} manifest rows</span>}
                  {splitCountsLabel && <span>{splitCountsLabel}</span>}
                </div>
              </div>
              <div className="flex flex-wrap justify-end gap-2">
                {deltaEntries.length > 0 ? (
                  deltaEntries.map(({ metric, splitName: deltaSplit, value }) => (
                    <div
                      key={`${candidate.id}-${metric}`}
                      className="rounded-md border bg-background px-2.5 py-1 text-right text-xs"
                    >
                      <div className="text-muted-foreground">
                        {metricLabel(metric)}
                        {deltaSplit ? ` (${deltaSplit})` : ""}
                      </div>
                      <div className="font-medium">{formatDeltaValue(metric, value)}</div>
                    </div>
                  ))
                ) : (
                  <div className="rounded-md border bg-background px-2.5 py-1 text-xs text-muted-foreground">
                    No comparison deltas
                  </div>
                )}
              </div>
            </div>
          </button>
        </CollapsibleTrigger>

        <CollapsibleContent className="border-t px-4 py-4">
          {warnings.length > 0 && (
            <div className="mb-4 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
              <div className="flex items-center gap-2 font-medium">
                <AlertTriangle className="h-3.5 w-3.5" />
                Promotion warnings
              </div>
              <div className="mt-1 space-y-1">
                {warnings.map((warning) => (
                  <div key={warning}>{warning}</div>
                ))}
              </div>
            </div>
          )}

          {isLoading ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading candidate details...
            </div>
          ) : isError || !detail ? (
            <div className="text-sm text-muted-foreground">
              Candidate detail is unavailable right now.
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid gap-4 lg:grid-cols-3">
                <div className="rounded-md border p-3">
                  <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                    Promoted Config
                  </div>
                  <div className="mt-2 space-y-2">
                    {Object.entries(detail.promoted_config).map(([key, value]) => (
                      <DetailField
                        key={key}
                        label={key}
                        value={formatConfigValue(value)}
                      />
                    ))}
                  </div>
                </div>

                <ComparisonTable detail={detail} splitName={splitName} />

                <div className="rounded-md border p-3">
                  <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                    Manifest Summary
                  </div>
                  <div className="mt-2 space-y-2">
                    <DetailField label="Source model" value={sourceModelLabel} />
                    <DetailField
                      label="Compared against"
                      value={candidate.comparison_target}
                    />
                    <DetailField
                      label="Threshold"
                      value={
                        candidate.threshold != null
                          ? formatMetricValue("threshold", candidate.threshold)
                          : null
                      }
                    />
                    <DetailField
                      label="Replay"
                      value={formatReplaySummary(candidate.replay_summary)}
                    />
                    <DetailField label="Splits" value={splitCountsLabel} />
                  </div>
                </div>
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <PreviewPanel
                  title={`Prediction Disagreements${splitName ? ` (${splitName})` : ""}`}
                  items={disagreementPreview}
                  emptyLabel="No disagreement preview was imported."
                />
                <PreviewPanel
                  title="Top False Positives"
                  items={falsePositivePreview}
                  emptyLabel="No false-positive preview was imported."
                />
              </div>

              <div className="rounded-md border p-3">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Stored Artifacts
                </div>
                <div className="mt-2 grid gap-2 text-xs lg:grid-cols-2">
                  <div className="rounded bg-muted/40 px-2.5 py-2">
                    <div className="font-medium text-foreground">Manifest</div>
                    <div className="mt-1 break-all font-mono text-[11px] text-muted-foreground">
                      {detail.artifact_paths.manifest_path}
                    </div>
                  </div>
                  <div className="rounded bg-muted/40 px-2.5 py-2">
                    <div className="font-medium text-foreground">Best Run</div>
                    <div className="mt-1 break-all font-mono text-[11px] text-muted-foreground">
                      {detail.artifact_paths.best_run_path}
                    </div>
                  </div>
                  {detail.artifact_paths.comparison_path && (
                    <div className="rounded bg-muted/40 px-2.5 py-2">
                      <div className="font-medium text-foreground">Comparison</div>
                      <div className="mt-1 break-all font-mono text-[11px] text-muted-foreground">
                        {detail.artifact_paths.comparison_path}
                      </div>
                    </div>
                  )}
                  {detail.artifact_paths.top_false_positives_path && (
                    <div className="rounded bg-muted/40 px-2.5 py-2">
                      <div className="font-medium text-foreground">
                        Top False Positives
                      </div>
                      <div className="mt-1 break-all font-mono text-[11px] text-muted-foreground">
                        {detail.artifact_paths.top_false_positives_path}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <Separator />

              <div className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-medium">Promotion</div>
                    <div className="text-xs text-muted-foreground">
                      Create a candidate-backed training job using the manifest
                      train split.
                    </div>
                  </div>
                  {candidate.status === "complete" && (
                    <Badge className="bg-emerald-100 text-emerald-800">
                      <CheckCircle2 className="mr-1 h-3 w-3" />
                      promoted
                    </Badge>
                  )}
                </div>

                {candidate.status === "promotable" ? (
                  <div className="grid gap-3 lg:grid-cols-[1fr_1fr_auto]">
                    <div>
                      <label
                        htmlFor={`promotion-name-${candidate.id}`}
                        className="text-sm font-medium"
                      >
                        New Model Name
                      </label>
                      <Input
                        id={`promotion-name-${candidate.id}`}
                        value={newModelName}
                        onChange={(event) => setNewModelName(event.target.value)}
                        placeholder="e.g. lr-v13"
                      />
                    </div>
                    <div>
                      <label
                        htmlFor={`promotion-notes-${candidate.id}`}
                        className="text-sm font-medium"
                      >
                        Promotion Notes
                      </label>
                      <Input
                        id={`promotion-notes-${candidate.id}`}
                        value={notes}
                        onChange={(event) => setNotes(event.target.value)}
                        placeholder="optional notes"
                      />
                    </div>
                    <div className="flex items-end">
                      <Button
                        onClick={handlePromote}
                        disabled={
                          !newModelName.trim() || promoteMutation.isPending
                        }
                      >
                        {promoteMutation.isPending ? "Starting…" : "Start Candidate Training"}
                      </Button>
                    </div>
                  </div>
                ) : candidate.status === "training" ? (
                  <div className="rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-900">
                    Candidate training is in progress.
                    {candidate.training_job_id && (
                      <span className="ml-1 font-mono text-xs">
                        job={candidate.training_job_id.slice(0, 8)}
                      </span>
                    )}
                  </div>
                ) : candidate.status === "complete" ? (
                  <div className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
                    Promotion finished successfully.
                    {candidate.new_model_id && (
                      <span className="ml-1 font-mono text-xs">
                        model={candidate.new_model_id.slice(0, 8)}
                      </span>
                    )}
                  </div>
                ) : candidate.status === "failed" ? (
                  <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-900">
                    Training failed for this candidate.
                    {candidate.error_message && (
                      <span className="ml-1">{candidate.error_message}</span>
                    )}
                  </div>
                ) : (
                  <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                    Promotion is currently blocked until the candidate can be
                    reproduced exactly by the production trainer.
                  </div>
                )}
              </div>
            </div>
          )}
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

export function AutoresearchCandidatesSection() {
  const { data: candidates = [], isLoading } = useAutoresearchCandidates(3000);
  const importMutation = useImportAutoresearchCandidate();

  const [name, setName] = useState("");
  const [manifestPath, setManifestPath] = useState("");
  const [bestRunPath, setBestRunPath] = useState("");
  const [comparisonPath, setComparisonPath] = useState("");
  const [topFalsePositivesPath, setTopFalsePositivesPath] = useState("");
  const [openCandidateId, setOpenCandidateId] = useState<string | null>(null);

  const promotableCount = useMemo(
    () => candidates.filter((candidate) => candidate.status === "promotable").length,
    [candidates],
  );

  const handleImport = () => {
    if (!manifestPath.trim() || !bestRunPath.trim()) {
      return;
    }
    importMutation.mutate(
      {
        name: emptyToUndefined(name) ?? null,
        manifest_path: manifestPath.trim(),
        best_run_path: bestRunPath.trim(),
        comparison_path: emptyToUndefined(comparisonPath) ?? null,
        top_false_positives_path: emptyToUndefined(topFalsePositivesPath) ?? null,
      },
      {
        onSuccess: (candidate) => {
          setName("");
          setManifestPath("");
          setBestRunPath("");
          setComparisonPath("");
          setTopFalsePositivesPath("");
          setOpenCandidateId(candidate.id);
        },
      },
    );
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
          <div>
            <CardTitle className="text-base">Autoresearch Candidates</CardTitle>
            <div className="mt-1 text-sm text-muted-foreground">
              Import server-side autoresearch artifacts, review comparison
              evidence against production models, and launch candidate-backed
              training jobs.
            </div>
          </div>
          <div className="flex gap-2">
            <Badge variant="secondary">{candidates.length} imported</Badge>
            <Badge className="bg-emerald-100 text-emerald-800">
              {promotableCount} promotable
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-lg border bg-muted/20 p-4">
          <div className="flex items-start gap-2 text-sm text-muted-foreground">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              Paths are read on the backend, so the files must already exist on
              this machine and be accessible from the API server process.
            </div>
          </div>

          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            <div>
              <label htmlFor="candidate-import-name" className="text-sm font-medium">
                Candidate Name
              </label>
              <Input
                id="candidate-import-name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="optional display name"
              />
            </div>
            <div>
              <label htmlFor="candidate-import-manifest" className="text-sm font-medium">
                Manifest Path
              </label>
              <Input
                id="candidate-import-manifest"
                value={manifestPath}
                onChange={(event) => setManifestPath(event.target.value)}
                placeholder="/abs/path/manifest.json"
              />
            </div>
            <div>
              <label htmlFor="candidate-import-best-run" className="text-sm font-medium">
                Best Run Path
              </label>
              <Input
                id="candidate-import-best-run"
                value={bestRunPath}
                onChange={(event) => setBestRunPath(event.target.value)}
                placeholder="/abs/path/phase1/best_run.json"
              />
            </div>
            <div>
              <label htmlFor="candidate-import-comparison" className="text-sm font-medium">
                Comparison Path
              </label>
              <Input
                id="candidate-import-comparison"
                value={comparisonPath}
                onChange={(event) => setComparisonPath(event.target.value)}
                placeholder="/abs/path/comparison.json"
              />
              <div className="mt-1 text-[11px] text-muted-foreground">
                Prefer `phase*/lr-v12-comparison.json` for full deltas. Summary files
                such as `comparison_summary.json` import with limited detail.
              </div>
            </div>
            <div>
              <label
                htmlFor="candidate-import-false-positives"
                className="text-sm font-medium"
              >
                Top False Positives Path
              </label>
              <Input
                id="candidate-import-false-positives"
                value={topFalsePositivesPath}
                onChange={(event) => setTopFalsePositivesPath(event.target.value)}
                placeholder="/abs/path/top_false_positives.json"
              />
            </div>
          </div>

          <div className="mt-4">
            <Button
              onClick={handleImport}
              disabled={
                !manifestPath.trim() ||
                !bestRunPath.trim() ||
                importMutation.isPending
              }
            >
              {importMutation.isPending ? "Importing…" : "Import Candidate"}
            </Button>
          </div>
        </div>

        {isLoading && candidates.length === 0 ? (
          <div className="flex items-center gap-2 rounded-lg border border-dashed px-4 py-6 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading imported candidates...
          </div>
        ) : candidates.length === 0 ? (
          <div className="rounded-lg border border-dashed px-4 py-6 text-sm text-muted-foreground">
            No autoresearch candidates have been imported yet.
          </div>
        ) : (
          <div className="space-y-3">
            {candidates.map((candidate) => (
              <CandidateCard
                key={candidate.id}
                candidate={candidate}
                open={openCandidateId === candidate.id}
                onOpenChange={(nextOpen) =>
                  setOpenCandidateId(nextOpen ? candidate.id : null)
                }
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
