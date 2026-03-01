import { useState, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { useCreateClusteringJob } from "@/hooks/queries/useClustering";
import { ClusteringParamsForm, type ClusteringParams } from "./ClusteringParamsForm";
import { showMsg } from "@/components/shared/MessageToast";
import { shortId, audioDisplayName } from "@/utils/format";
import type { EmbeddingSet } from "@/api/types";

interface EmbeddingSetSelectorProps {
  embeddingSets: EmbeddingSet[];
}

export function EmbeddingSetSelector({ embeddingSets }: EmbeddingSetSelectorProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const { data: audioFiles = [] } = useAudioFiles();
  const createJob = useCreateClusteringJob();

  const audioMap = new Map(audioFiles.map((af) => [af.id, af]));

  // Group by model version
  const grouped = useMemo(() => {
    const map = new Map<string, EmbeddingSet[]>();
    for (const es of embeddingSets) {
      const key = es.model_version;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(es);
    }
    return map;
  }, [embeddingSets]);

  const toggleOne = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleGroup = useCallback(
    (modelVersion: string) => {
      const group = grouped.get(modelVersion) ?? [];
      const allSelected = group.every((es) => selected.has(es.id));
      setSelected((prev) => {
        const next = new Set(prev);
        for (const es of group) {
          if (allSelected) next.delete(es.id);
          else next.add(es.id);
        }
        return next;
      });
    },
    [grouped, selected],
  );

  const toggleAll = useCallback(() => {
    const allSelected = embeddingSets.every((es) => selected.has(es.id));
    if (allSelected) {
      setSelected(new Set());
    } else {
      setSelected(new Set(embeddingSets.map((es) => es.id)));
    }
  }, [embeddingSets, selected]);

  const handleSubmit = useCallback(
    (params: ClusteringParams) => {
      if (selected.size === 0) {
        showMsg("error", "Select at least one embedding set");
        return;
      }

      // Dimension validation
      const dims = new Set(
        embeddingSets.filter((es) => selected.has(es.id)).map((es) => es.vector_dim),
      );
      if (dims.size > 1) {
        showMsg("error", `Cannot cluster sets with mixed dimensions: ${[...dims].join(", ")}`);
        return;
      }

      const parameters: Record<string, unknown> = {};
      if (params.minClusterSize) parameters.min_cluster_size = params.minClusterSize;
      if (params.minSamples) parameters.min_samples = params.minSamples;
      if (params.selectionMethod) parameters.cluster_selection_method = params.selectionMethod;
      if (params.enableUmap) {
        parameters.umap_cluster_n_components = params.umapComponents;
        parameters.umap_n_neighbors = params.umapNeighbors;
        parameters.umap_min_dist = params.umapMinDist;
      } else {
        parameters.reduction_method = "none";
      }

      createJob.mutate(
        { embedding_set_ids: [...selected], parameters },
        {
          onSuccess: () => showMsg("success", "Clustering job queued"),
          onError: (e) => showMsg("error", `Failed: ${e.message}`),
        },
      );
    },
    [selected, embeddingSets, createJob],
  );

  const allSelected = embeddingSets.length > 0 && embeddingSets.every((es) => selected.has(es.id));

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Queue Clustering Job</CardTitle>
          <Button variant="outline" size="sm" onClick={toggleAll}>
            {allSelected ? "Deselect All" : "Select All"}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Embedding set checkboxes grouped by model */}
        <div className="space-y-3 max-h-60 overflow-y-auto">
          {Array.from(grouped.entries()).map(([modelVersion, sets]) => {
            const groupAllSelected = sets.every((es) => selected.has(es.id));
            return (
              <div key={modelVersion}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-medium">{modelVersion}</span>
                  <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={() => toggleGroup(modelVersion)}>
                    {groupAllSelected ? "Deselect" : "Select All"}
                  </Button>
                </div>
                <div className="space-y-1 ml-2">
                  {sets.map((es) => {
                    const af = audioMap.get(es.audio_file_id);
                    const name = af ? audioDisplayName(af.filename, af.folder_path) : es.audio_file_id;
                    return (
                      <label key={es.id} className="flex items-center gap-2 text-sm cursor-pointer">
                        <Checkbox
                          checked={selected.has(es.id)}
                          onCheckedChange={() => toggleOne(es.id)}
                        />
                        <span className="font-mono text-xs text-muted-foreground">{shortId(es.id)}</span>
                        <span className="truncate">{name}</span>
                        <span className="text-xs text-muted-foreground ml-auto">{es.vector_dim}d</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            );
          })}
          {embeddingSets.length === 0 && (
            <p className="text-sm text-muted-foreground">No embedding sets available. Process some audio first.</p>
          )}
        </div>

        <ClusteringParamsForm
          onSubmit={handleSubmit}
          disabled={createJob.isPending || selected.size === 0}
        />
      </CardContent>
    </Card>
  );
}
