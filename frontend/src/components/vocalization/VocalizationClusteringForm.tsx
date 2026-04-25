import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  useClusteringEligibleJobs,
  useCreateVocalizationClusteringJob,
} from "@/hooks/queries/useVocalization";
import type { ClusteringEligibleDetectionJob } from "@/api/types";

function formatUtcRange(start: number | null, end: number | null): string {
  if (start == null || end == null) return "—";
  const fmt = (ts: number) => {
    const d = new Date(ts * 1000);
    const p = (n: number) => String(n).padStart(2, "0");
    return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())} UTC`;
  };
  return `${fmt(start)} — ${fmt(end)}`;
}

export function VocalizationClusteringForm() {
  const { data: eligibleJobs = [] } = useClusteringEligibleJobs();
  const createMutation = useCreateVocalizationClusteringJob();

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [minClusterSize, setMinClusterSize] = useState(5);
  const [minSamples, setMinSamples] = useState<string>("");
  const [selectionMethod, setSelectionMethod] = useState("leaf");

  const [umapEnabled, setUmapEnabled] = useState(true);
  const [umapComponents, setUmapComponents] = useState(5);
  const [umapNeighbors, setUmapNeighbors] = useState(15);
  const [umapMinDist, setUmapMinDist] = useState(0.1);
  const [stabilityRuns, setStabilityRuns] = useState(0);

  const allSelected =
    eligibleJobs.length > 0 &&
    eligibleJobs.every((j) => selectedIds.has(j.id));

  const toggleAll = () => {
    if (allSelected) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(eligibleJobs.map((j) => j.id)));
    }
  };

  const toggleOne = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const canSubmit = selectedIds.size > 0 && !createMutation.isPending;

  const handleSubmit = () => {
    if (!canSubmit) return;

    const params: Record<string, unknown> = {
      min_cluster_size: minClusterSize,
      cluster_selection_method: selectionMethod,
    };
    if (minSamples) params.min_samples = parseInt(minSamples);
    if (umapEnabled) {
      params.umap_cluster_n_components = umapComponents;
      params.umap_n_neighbors = umapNeighbors;
      params.umap_min_dist = umapMinDist;
    }
    if (stabilityRuns > 0) params.stability_runs = stabilityRuns;

    createMutation.mutate({
      detection_job_ids: [...selectedIds],
      parameters: params,
    });
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Vocalization Clustering</CardTitle>
        {eligibleJobs.length > 0 && (
          <Button variant="ghost" size="sm" onClick={toggleAll}>
            {allSelected ? "Deselect all" : "Select all"}
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {eligibleJobs.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No eligible detection jobs found. Run vocalization inference with the
            active model first.
          </p>
        ) : (
          <div className="border rounded-md max-h-64 overflow-y-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50 sticky top-0">
                  <th className="w-10 px-3 py-2" />
                  <th className="px-3 py-2 text-left font-medium">Hydrophone</th>
                  <th className="px-3 py-2 text-left font-medium">Date Range (UTC)</th>
                  <th className="px-3 py-2 text-left font-medium">Detections</th>
                </tr>
              </thead>
              <tbody>
                {eligibleJobs.map((job: ClusteringEligibleDetectionJob) => (
                  <tr key={job.id} className="border-b hover:bg-muted/30">
                    <td className="px-3 py-2">
                      <Checkbox
                        checked={selectedIds.has(job.id)}
                        onCheckedChange={() => toggleOne(job.id)}
                      />
                    </td>
                    <td className="px-3 py-2 text-xs">
                      {job.hydrophone_name ?? "—"}
                    </td>
                    <td className="px-3 py-2 text-xs whitespace-nowrap">
                      {formatUtcRange(job.start_timestamp, job.end_timestamp)}
                    </td>
                    <td className="px-3 py-2 text-xs">
                      {job.detection_count ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="text-sm font-medium">Min Cluster Size</label>
            <Input
              type="number"
              min={2}
              value={minClusterSize}
              onChange={(e) => setMinClusterSize(parseInt(e.target.value) || 5)}
              className="mt-1"
            />
          </div>
          <div>
            <label className="text-sm font-medium">Min Samples</label>
            <Input
              type="number"
              min={1}
              placeholder="auto"
              value={minSamples}
              onChange={(e) => setMinSamples(e.target.value)}
              className="mt-1"
            />
          </div>
          <div>
            <label className="text-sm font-medium">Selection Method</label>
            <select
              className="w-full border rounded px-3 py-2 text-sm mt-1"
              value={selectionMethod}
              onChange={(e) => setSelectionMethod(e.target.value)}
            >
              <option value="leaf">leaf</option>
              <option value="eom">eom</option>
            </select>
          </div>
        </div>

        <Collapsible defaultOpen={false}>
          <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium group">
            <ChevronRight className="h-3 w-3 transition-transform group-data-[state=open]:rotate-90" />
            Advanced Settings
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-2 space-y-3">
              <div className="flex items-center gap-2">
                <Checkbox
                  checked={umapEnabled}
                  onCheckedChange={(v) => setUmapEnabled(!!v)}
                />
                <label className="text-xs font-medium">Enable UMAP Pre-Reduction</label>
              </div>
              {umapEnabled && (
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="text-xs font-medium">Components</label>
                    <Input
                      type="number"
                      min={2}
                      max={50}
                      value={umapComponents}
                      onChange={(e) => setUmapComponents(parseInt(e.target.value) || 5)}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium">Neighbors</label>
                    <Input
                      type="number"
                      min={2}
                      max={200}
                      value={umapNeighbors}
                      onChange={(e) => setUmapNeighbors(parseInt(e.target.value) || 15)}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium">Min Dist</label>
                    <Input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={umapMinDist}
                      onChange={(e) => setUmapMinDist(parseFloat(e.target.value) || 0.1)}
                      className="mt-1"
                    />
                  </div>
                </div>
              )}
              <div className="w-48">
                <label className="text-xs font-medium">Stability Runs (0 = off)</label>
                <Input
                  type="number"
                  min={0}
                  max={50}
                  value={stabilityRuns}
                  onChange={(e) => setStabilityRuns(parseInt(e.target.value) || 0)}
                  className="mt-1"
                />
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>

        <Button onClick={handleSubmit} disabled={!canSubmit}>
          {createMutation.isPending ? "Creating…" : "Start Clustering"}
        </Button>
        {createMutation.isError && (
          <p className="text-sm text-red-600">
            {(createMutation.error as Error).message}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
