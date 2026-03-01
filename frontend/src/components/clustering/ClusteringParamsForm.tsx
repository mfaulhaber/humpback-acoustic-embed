import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export interface ClusteringParams {
  minClusterSize: number;
  minSamples: number | null;
  selectionMethod: string;
  enableUmap: boolean;
  umapComponents: number;
  umapNeighbors: number;
  umapMinDist: number;
}

interface ClusteringParamsFormProps {
  onSubmit: (params: ClusteringParams) => void;
  disabled: boolean;
}

export function ClusteringParamsForm({ onSubmit, disabled }: ClusteringParamsFormProps) {
  const [minClusterSize, setMinClusterSize] = useState("5");
  const [minSamples, setMinSamples] = useState("");
  const [selectionMethod, setSelectionMethod] = useState("leaf");
  const [enableUmap, setEnableUmap] = useState(true);
  const [umapComponents, setUmapComponents] = useState("5");
  const [umapNeighbors, setUmapNeighbors] = useState("15");
  const [umapMinDist, setUmapMinDist] = useState("0.1");

  const handleSubmit = () => {
    onSubmit({
      minClusterSize: parseInt(minClusterSize) || 5,
      minSamples: minSamples ? parseInt(minSamples) : null,
      selectionMethod,
      enableUmap,
      umapComponents: parseInt(umapComponents) || 5,
      umapNeighbors: parseInt(umapNeighbors) || 15,
      umapMinDist: parseFloat(umapMinDist) || 0.1,
    });
  };

  return (
    <div className="space-y-3 border-t pt-3">
      <h4 className="text-sm font-medium">HDBSCAN Parameters</h4>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div>
          <label className="text-xs text-muted-foreground">Min Cluster Size</label>
          <Input
            type="number"
            value={minClusterSize}
            onChange={(e) => setMinClusterSize(e.target.value)}
            className="h-8"
            min={2}
          />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">Min Samples (optional)</label>
          <Input
            type="number"
            value={minSamples}
            onChange={(e) => setMinSamples(e.target.value)}
            className="h-8"
            placeholder="auto"
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">Selection Method</label>
          <Select value={selectionMethod} onValueChange={setSelectionMethod}>
            <SelectTrigger className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="leaf">leaf</SelectItem>
              <SelectItem value="eom">eom</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Checkbox
          checked={enableUmap}
          onCheckedChange={(v) => setEnableUmap(!!v)}
          id="enable-umap"
        />
        <label htmlFor="enable-umap" className="text-sm font-medium cursor-pointer">
          Enable UMAP Dimensionality Reduction
        </label>
      </div>

      {enableUmap && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div>
            <label className="text-xs text-muted-foreground">Cluster Dimensions</label>
            <Input
              type="number"
              value={umapComponents}
              onChange={(e) => setUmapComponents(e.target.value)}
              className="h-8"
              min={2}
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Neighbors</label>
            <Input
              type="number"
              value={umapNeighbors}
              onChange={(e) => setUmapNeighbors(e.target.value)}
              className="h-8"
              min={2}
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Min Distance</label>
            <Input
              type="number"
              value={umapMinDist}
              onChange={(e) => setUmapMinDist(e.target.value)}
              className="h-8"
              step={0.01}
              min={0}
            />
          </div>
        </div>
      )}

      <Button size="sm" onClick={handleSubmit} disabled={disabled}>
        Queue Clustering Job
      </Button>
    </div>
  );
}
