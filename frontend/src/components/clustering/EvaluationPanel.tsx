import { useMetrics } from "@/hooks/queries/useClustering";

interface EvaluationPanelProps {
  jobId: string;
}

export function EvaluationPanel({ jobId }: EvaluationPanelProps) {
  const { data: metrics, isLoading: metricsLoading } = useMetrics(jobId);

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
                    {vals.purity != null ? vals.purity.toFixed(4) : "—"}
                  </td>
                  <td className="py-1.5 px-3 font-mono text-xs">{vals.count ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

    </div>
  );
}
