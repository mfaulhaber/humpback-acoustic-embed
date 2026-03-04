import { useState, useMemo } from "react";
import { useMetrics, useFragmentation, useStability, useClassifier, useLabelQueue, useRefinement } from "@/hooks/queries/useClustering";
import type { CategoryFragmentation, ClassifierReport, ClusterFragmentation, LabelQueueEntry, RefinementReport, StabilitySummary } from "@/api/types";

interface EvaluationPanelProps {
  jobId: string;
}

type SortDir = "asc" | "desc";

function useSortable<T>(data: [string, T][], defaultKey: string) {
  const [sortKey, setSortKey] = useState(defaultKey);
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sorted = useMemo(() => {
    const copy = [...data];
    copy.sort((a, b) => {
      const av = (a[1] as Record<string, unknown>)[sortKey];
      const bv = (b[1] as Record<string, unknown>)[sortKey];
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av;
      }
      const as = String(av ?? "");
      const bs = String(bv ?? "");
      return sortDir === "asc" ? as.localeCompare(bs) : bs.localeCompare(as);
    });
    return copy;
  }, [data, sortKey, sortDir]);

  const toggle = (key: string) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  return { sorted, sortKey, sortDir, toggle };
}

function SortHeader({
  label,
  field,
  sortKey,
  sortDir,
  onToggle,
}: {
  label: string;
  field: string;
  sortKey: string;
  sortDir: SortDir;
  onToggle: (f: string) => void;
}) {
  const arrow = sortKey === field ? (sortDir === "asc" ? " \u25B2" : " \u25BC") : "";
  return (
    <th
      className="text-left py-2 px-3 font-medium cursor-pointer select-none hover:bg-muted/80"
      onClick={() => onToggle(field)}
    >
      {label}{arrow}
    </th>
  );
}

function fmt(v: number | undefined | null, decimals = 4): string {
  if (v == null) return "\u2014";
  return v.toFixed(decimals);
}

export function EvaluationPanel({ jobId }: EvaluationPanelProps) {
  const { data: metrics, isLoading: metricsLoading } = useMetrics(jobId);
  const { data: fragReport } = useFragmentation(jobId);
  const { data: stability } = useStability(jobId);
  const { data: classifierReport } = useClassifier(jobId);
  const { data: labelQueue } = useLabelQueue(jobId);
  const { data: refinement } = useRefinement(jobId);

  if (metricsLoading) return <p className="text-sm text-muted-foreground">Loading metrics...</p>;

  const mainMetrics = [
    { label: "Silhouette Score", value: metrics?.silhouette_score },
    { label: "Davies-Bouldin Index", value: metrics?.davies_bouldin_index },
    { label: "Calinski-Harabasz Score", value: metrics?.calinski_harabasz_score },
    { label: "N Clusters", value: metrics?.n_clusters },
    { label: "Noise Points", value: metrics?.noise_count },
  ];

  const supervisedMetrics = [
    { label: "ARI (Adjusted Rand Index)", value: metrics?.adjusted_rand_index },
    { label: "NMI (Normalized Mutual Info)", value: metrics?.normalized_mutual_info },
    { label: "N Categories", value: metrics?.n_categories },
  ];

  const categoryMetrics = metrics?.category_metrics as
    | Record<string, { purity?: number; count?: number; homogeneity?: number; completeness?: number; v_measure?: number }>
    | undefined;

  return (
    <div className="space-y-4">
      <div className="border rounded-md">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="text-left py-2 px-3 font-medium">Metric</th>
              <th className="text-left py-2 px-3 font-medium">Value</th>
            </tr>
          </thead>
          <tbody>
            {mainMetrics.map(
              (m) =>
                m.value != null && (
                  <tr key={m.label} className="border-b last:border-0">
                    <td className="py-1.5 px-3">{m.label}</td>
                    <td className="py-1.5 px-3 font-mono text-xs">
                      {typeof m.value === "number" ? m.value.toFixed(4) : m.value}
                    </td>
                  </tr>
                ),
            )}
            {supervisedMetrics.map(
              (m) =>
                m.value != null && (
                  <tr key={m.label} className="border-b last:border-0">
                    <td className="py-1.5 px-3">{m.label}</td>
                    <td className="py-1.5 px-3 font-mono text-xs">
                      {typeof m.value === "number" ? m.value.toFixed(4) : m.value}
                    </td>
                  </tr>
                ),
            )}
          </tbody>
        </table>
      </div>

      {categoryMetrics && Object.keys(categoryMetrics).length > 0 && (
        <div className="border rounded-md">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="text-left py-2 px-3 font-medium">Category</th>
                <th className="text-left py-2 px-3 font-medium">Purity</th>
                <th className="text-left py-2 px-3 font-medium">Count</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(categoryMetrics).map(([cat, vals]) => (
                <tr key={cat} className="border-b last:border-0">
                  <td className="py-1.5 px-3">{cat}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">
                    {vals.purity != null ? vals.purity.toFixed(4) : "\u2014"}
                  </td>
                  <td className="py-1.5 px-3 font-mono text-xs">{vals.count ?? "\u2014"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {fragReport && (
        <>
          {/* Global Fragmentation Summary */}
          <div>
            <h4 className="text-sm font-medium mb-2">Global Fragmentation Summary</h4>
            <div className="border rounded-md">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left py-2 px-3 font-medium">Metric</th>
                    <th className="text-left py-2 px-3 font-medium">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { label: "Categories", value: fragReport.summary.n_categories },
                    { label: "Clusters", value: fragReport.summary.n_clusters },
                    { label: "Total Points", value: fragReport.summary.n_total },
                    { label: "Noise Points", value: fragReport.summary.n_noise_total },
                    { label: "Overall Noise Rate", value: fragReport.summary.overall_noise_rate },
                    { label: "Mean Entropy (norm)", value: fragReport.global_fragmentation.mean_entropy_norm },
                    { label: "Mean N_eff", value: fragReport.global_fragmentation.mean_neff },
                    { label: "Mean Noise Rate", value: fragReport.global_fragmentation.mean_noise_rate },
                    { label: "Mean Cluster Entropy (norm)", value: fragReport.global_fragmentation.mean_cluster_entropy_norm },
                  ].map((m) => (
                    <tr key={m.label} className="border-b last:border-0">
                      <td className="py-1.5 px-3">{m.label}</td>
                      <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Per-Category Fragmentation */}
          <CategoryFragmentationTable data={fragReport.category_fragmentation} />

          {/* Per-Cluster Composition */}
          <ClusterFragmentationTable data={fragReport.cluster_fragmentation} />
        </>
      )}

      {stability && <StabilitySection data={stability} />}

      {classifierReport && <ClassifierSection data={classifierReport} />}

      {labelQueue && labelQueue.length > 0 && <LabelQueueSection data={labelQueue} />}

      {refinement && <RefinementSection data={refinement} />}
    </div>
  );
}

function CategoryFragmentationTable({ data }: { data: Record<string, CategoryFragmentation> }) {
  const entries = useMemo(() => Object.entries(data), [data]);
  const { sorted, sortKey, sortDir, toggle } = useSortable(entries, "n_total");

  if (entries.length === 0) return null;

  return (
    <div>
      <h4 className="text-sm font-medium mb-2">Per-Category Fragmentation</h4>
      <div className="border rounded-md overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="text-left py-2 px-3 font-medium">Category</th>
              <SortHeader label="N Total" field="n_total" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Non-Noise" field="n_non_noise" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Noise Rate" field="noise_rate" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Top-1" field="top1_mass" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Top-2" field="top2_mass" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Top-3" field="top3_mass" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Entropy (norm)" field="normalized_entropy" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="N_eff" field="neff" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Gini" field="gini" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
            </tr>
          </thead>
          <tbody>
            {sorted.map(([cat, m]) => (
              <tr key={cat} className="border-b last:border-0">
                <td className="py-1.5 px-3">{cat}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{m.n_total}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{m.n_non_noise}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.noise_rate)}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.top1_mass)}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.top2_mass)}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.top3_mass)}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.normalized_entropy)}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.neff, 2)}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.gini)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ClusterFragmentationTable({ data }: { data: Record<string, ClusterFragmentation> }) {
  const entries = useMemo(() => Object.entries(data), [data]);
  const { sorted, sortKey, sortDir, toggle } = useSortable(entries, "size");

  if (entries.length === 0) return null;

  return (
    <div>
      <h4 className="text-sm font-medium mb-2">Per-Cluster Composition</h4>
      <div className="border rounded-md overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="text-left py-2 px-3 font-medium">Cluster</th>
              <SortHeader label="Size" field="size" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Dominant Category" field="dominant_category" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Dominant Mass" field="dominant_mass" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Entropy (norm)" field="cluster_entropy_norm" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
            </tr>
          </thead>
          <tbody>
            {sorted.map(([cl, m]) => (
              <tr key={cl} className="border-b last:border-0">
                <td className="py-1.5 px-3">{cl}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{m.size}</td>
                <td className="py-1.5 px-3">{m.dominant_category}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.dominant_mass)}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.cluster_entropy_norm)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const STABILITY_METRICS = [
  { key: "n_clusters", label: "N Clusters" },
  { key: "noise_fraction", label: "Noise Fraction" },
  { key: "silhouette_score", label: "Silhouette Score" },
  { key: "adjusted_rand_index", label: "ARI" },
  { key: "normalized_mutual_info", label: "NMI" },
  { key: "fragmentation_index", label: "Fragmentation Index" },
] as const;

function StabilitySection({ data }: { data: StabilitySummary }) {
  const perRunEntries = useMemo(
    () => data.per_run.map((r) => [String(r.run_index), r] as [string, typeof r]),
    [data.per_run],
  );
  const { sorted, sortKey, sortDir, toggle } = useSortable(perRunEntries, "run_index");

  const pla = data.pairwise_label_agreement;

  return (
    <>
      {/* Stability Overview */}
      <div>
        <h4 className="text-sm font-medium mb-2">Stability Overview</h4>
        <div className="border rounded-md">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="text-left py-2 px-3 font-medium">Metric</th>
                <th className="text-left py-2 px-3 font-medium">Value</th>
              </tr>
            </thead>
            <tbody>
              {[
                { label: "N Runs", value: data.n_runs },
                { label: "Mean Pairwise ARI", value: pla.mean_pairwise_ari },
                { label: "Std Pairwise ARI", value: pla.std_pairwise_ari },
                { label: "Min Pairwise ARI", value: pla.min_pairwise_ari },
                { label: "Max Pairwise ARI", value: pla.max_pairwise_ari },
              ].map((m) => (
                <tr key={m.label} className="border-b last:border-0">
                  <td className="py-1.5 px-3">{m.label}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Metric Stability */}
      <div>
        <h4 className="text-sm font-medium mb-2">Metric Stability</h4>
        <div className="border rounded-md overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="text-left py-2 px-3 font-medium">Metric</th>
                <th className="text-left py-2 px-3 font-medium">Mean</th>
                <th className="text-left py-2 px-3 font-medium">Std</th>
                <th className="text-left py-2 px-3 font-medium">Min</th>
                <th className="text-left py-2 px-3 font-medium">Max</th>
              </tr>
            </thead>
            <tbody>
              {STABILITY_METRICS.map(({ key, label }) => (
                <tr key={key} className="border-b last:border-0">
                  <td className="py-1.5 px-3">{label}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{fmt(data.aggregate_metrics[`${key}_mean`])}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{fmt(data.aggregate_metrics[`${key}_std`])}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{fmt(data.aggregate_metrics[`${key}_min`])}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{fmt(data.aggregate_metrics[`${key}_max`])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-Run Details */}
      <div>
        <h4 className="text-sm font-medium mb-2">Per-Run Details</h4>
        <div className="border rounded-md overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <SortHeader label="Run" field="run_index" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
                <th className="text-left py-2 px-3 font-medium">Seed</th>
                <SortHeader label="N Clusters" field="n_clusters" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
                <SortHeader label="Noise%" field="noise_fraction" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
                <SortHeader label="Silhouette" field="silhouette_score" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
                <SortHeader label="ARI" field="adjusted_rand_index" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
                <SortHeader label="NMI" field="normalized_mutual_info" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
                <SortHeader label="Frag Index" field="fragmentation_index" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              </tr>
            </thead>
            <tbody>
              {sorted.map(([, r]) => (
                <tr key={r.run_index} className="border-b last:border-0">
                  <td className="py-1.5 px-3 font-mono text-xs">{r.run_index}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{r.seed}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{r.n_clusters}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{fmt(r.noise_fraction)}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{fmt(r.silhouette_score)}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{fmt(r.adjusted_rand_index)}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{fmt(r.normalized_mutual_info)}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{fmt(r.fragmentation_index)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

function ClassifierSection({ data }: { data: ClassifierReport }) {
  const perClassEntries = useMemo(
    () => Object.entries(data.per_class).map(([cat, m]) => [cat, m] as [string, typeof m]),
    [data.per_class],
  );
  const { sorted, sortKey, sortDir, toggle } = useSortable(perClassEntries, "f1_score");

  return (
    <>
      {/* Classifier Summary */}
      <div>
        <h4 className="text-sm font-medium mb-2">Classifier Baseline</h4>
        <div className="border rounded-md">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="text-left py-2 px-3 font-medium">Metric</th>
                <th className="text-left py-2 px-3 font-medium">Value</th>
              </tr>
            </thead>
            <tbody>
              {[
                { label: "Overall Accuracy", value: data.overall_accuracy },
                { label: "N Samples", value: data.n_samples },
                { label: "N Categories", value: data.n_categories },
                { label: "N Folds", value: data.n_folds },
                { label: "Macro Precision", value: data.macro_avg.precision },
                { label: "Macro Recall", value: data.macro_avg.recall },
                { label: "Macro F1", value: data.macro_avg.f1_score },
                { label: "Weighted F1", value: data.weighted_avg.f1_score },
              ].map((m) => (
                <tr key={m.label} className="border-b last:border-0">
                  <td className="py-1.5 px-3">{m.label}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {data.categories_excluded.length > 0 && (
          <p className="text-xs text-muted-foreground mt-1">
            Excluded categories (too few samples): {data.categories_excluded.join(", ")}
          </p>
        )}
      </div>

      {/* Per-Class Metrics */}
      {perClassEntries.length > 0 && (
        <div>
          <h4 className="text-sm font-medium mb-2">Per-Class Metrics</h4>
          <div className="border rounded-md overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="text-left py-2 px-3 font-medium">Category</th>
                  <SortHeader label="Precision" field="precision" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
                  <SortHeader label="Recall" field="recall" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
                  <SortHeader label="F1" field="f1_score" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
                  <SortHeader label="Support" field="support" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
                </tr>
              </thead>
              <tbody>
                {sorted.map(([cat, m]) => (
                  <tr key={cat} className="border-b last:border-0">
                    <td className="py-1.5 px-3">{cat}</td>
                    <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.precision)}</td>
                    <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.recall)}</td>
                    <td className="py-1.5 px-3 font-mono text-xs">{fmt(m.f1_score)}</td>
                    <td className="py-1.5 px-3 font-mono text-xs">{m.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}

function RefinementSection({ data }: { data: RefinementReport }) {
  const tp = data.training_params;

  return (
    <>
      {/* Training Summary */}
      <div>
        <h4 className="text-sm font-medium mb-2">Metric Learning Refinement</h4>
        <div className="border rounded-md">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="text-left py-2 px-3 font-medium">Parameter</th>
                <th className="text-left py-2 px-3 font-medium">Value</th>
              </tr>
            </thead>
            <tbody>
              {[
                { label: "N Labeled", value: data.n_labeled_samples },
                { label: "N Categories", value: data.n_categories },
                { label: "N Total", value: data.n_total_samples },
                { label: "Output Dim", value: tp.output_dim },
                { label: "Hidden Dim", value: tp.hidden_dim },
                { label: "Epochs", value: tp.n_epochs },
                { label: "Learning Rate", value: tp.lr },
                { label: "Margin", value: tp.margin },
                { label: "Mining Strategy", value: tp.mining_strategy },
                { label: "Final Loss", value: data.final_loss },
              ].map((m) => (
                <tr key={m.label} className="border-b last:border-0">
                  <td className="py-1.5 px-3">{m.label}</td>
                  <td className="py-1.5 px-3 font-mono text-xs">
                    {typeof m.value === "number" ? fmt(m.value) : m.value}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Categories used: {data.categories_used.join(", ")}
        </p>
      </div>

      {/* Base vs Refined Comparison */}
      <div>
        <h4 className="text-sm font-medium mb-2">Base vs Refined Comparison</h4>
        <div className="border rounded-md overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="text-left py-2 px-3 font-medium">Metric</th>
                <th className="text-left py-2 px-3 font-medium">Base</th>
                <th className="text-left py-2 px-3 font-medium">Refined</th>
                <th className="text-left py-2 px-3 font-medium">Delta</th>
              </tr>
            </thead>
            <tbody>
              {data.comparison.map((c) => {
                let deltaClass = "";
                if (c.improved === true) deltaClass = "text-green-600";
                else if (c.improved === false) deltaClass = "text-red-600";

                return (
                  <tr key={c.key} className="border-b last:border-0">
                    <td className="py-1.5 px-3">{c.metric}</td>
                    <td className="py-1.5 px-3 font-mono text-xs">{fmt(c.base)}</td>
                    <td className="py-1.5 px-3 font-mono text-xs">{fmt(c.refined)}</td>
                    <td className={`py-1.5 px-3 font-mono text-xs ${deltaClass}`}>
                      {c.delta != null ? (c.delta > 0 ? "+" : "") + fmt(c.delta) : "\u2014"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

const LABEL_QUEUE_LIMIT = 50;

function LabelQueueSection({ data }: { data: LabelQueueEntry[] }) {
  const displayData = useMemo(
    () => data.slice(0, LABEL_QUEUE_LIMIT).map((e) => [String(e.rank), e] as [string, LabelQueueEntry]),
    [data],
  );
  const { sorted, sortKey, sortDir, toggle } = useSortable(displayData, "priority");

  return (
    <div>
      <h4 className="text-sm font-medium mb-2">Active Learning Queue</h4>
      {data.length > LABEL_QUEUE_LIMIT && (
        <p className="text-xs text-muted-foreground mb-1">
          Showing top {LABEL_QUEUE_LIMIT} of {data.length}
        </p>
      )}
      <div className="border rounded-md overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <SortHeader label="Rank" field="rank" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <th className="text-left py-2 px-3 font-medium">ES ID</th>
              <th className="text-left py-2 px-3 font-medium">Row</th>
              <th className="text-left py-2 px-3 font-medium">Current</th>
              <th className="text-left py-2 px-3 font-medium">Predicted</th>
              <SortHeader label="Entropy" field="entropy" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Margin" field="margin" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Frag Boost" field="fragmentation_boost" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
              <SortHeader label="Priority" field="priority" sortKey={sortKey} sortDir={sortDir} onToggle={toggle} />
            </tr>
          </thead>
          <tbody>
            {sorted.map(([, e]) => (
              <tr key={e.global_index} className="border-b last:border-0">
                <td className="py-1.5 px-3 font-mono text-xs">{e.rank}</td>
                <td className="py-1.5 px-3 font-mono text-xs truncate max-w-[100px]" title={e.embedding_set_id}>
                  {e.embedding_set_id.slice(0, 8)}
                </td>
                <td className="py-1.5 px-3 font-mono text-xs">{e.embedding_row_index}</td>
                <td className="py-1.5 px-3 text-xs">{e.current_category ?? "\u2014"}</td>
                <td className="py-1.5 px-3 text-xs">{e.predicted_category ?? "\u2014"}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(e.entropy)}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(e.margin)}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(e.fragmentation_boost)}</td>
                <td className="py-1.5 px-3 font-mono text-xs">{fmt(e.priority)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
