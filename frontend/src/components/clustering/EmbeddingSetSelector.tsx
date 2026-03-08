import { useState, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { ChevronRight, ChevronDown } from "lucide-react";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { useCreateClusteringJob } from "@/hooks/queries/useClustering";
import { ClusteringParamsForm, type ClusteringParams } from "./ClusteringParamsForm";
import { showMsg } from "@/components/shared/MessageToast";
import { ModelFilter } from "@/components/shared/ModelFilter";
import type { EmbeddingSet } from "@/api/types";

const ROOT_SENTINEL = "__root__";

interface FolderNode {
  /** child folder name (e.g. "Ascending_moan") or ROOT_SENTINEL */
  child: string;
  /** embedding sets for audio files in this subfolder */
  sets: EmbeddingSet[];
}

interface ParentNode {
  parent: string;
  children: FolderNode[];
  totalSets: number;
}

interface EmbeddingSetSelectorProps {
  embeddingSets: EmbeddingSet[];
}

export function EmbeddingSetSelector({ embeddingSets }: EmbeddingSetSelectorProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [collapsed, setCollapsed] = useState<Set<string> | null>(null);
  const [modelFilter, setModelFilter] = useState("__all__");
  const { data: audioFiles = [] } = useAudioFiles();
  const createJob = useCreateClusteringJob();

  const audioMap = useMemo(() => new Map(audioFiles.map((af) => [af.id, af])), [audioFiles]);

  const filteredSets = useMemo(
    () => modelFilter === "__all__" ? embeddingSets : embeddingSets.filter((es) => es.model_version === modelFilter),
    [embeddingSets, modelFilter],
  );

  // Build two-level folder tree: parent → child → embedding sets
  const folderTree = useMemo(() => {
    // Map each embedding set to its parent/child folder via audioMap
    const tree = new Map<string, Map<string, EmbeddingSet[]>>();

    for (const es of filteredSets) {
      const af = audioMap.get(es.audio_file_id);
      const folderPath = af?.folder_path || "";
      const slashIdx = folderPath.indexOf("/");
      const parent = folderPath ? (slashIdx >= 0 ? folderPath.slice(0, slashIdx) : folderPath) : ROOT_SENTINEL;
      const child = folderPath && slashIdx >= 0 ? folderPath.slice(slashIdx + 1) : ROOT_SENTINEL;

      if (!tree.has(parent)) tree.set(parent, new Map());
      const childMap = tree.get(parent)!;
      if (!childMap.has(child)) childMap.set(child, []);
      childMap.get(child)!.push(es);
    }

    // Convert to sorted array structure
    const result: ParentNode[] = [];
    const sortedParents = [...tree.keys()].sort((a, b) => {
      if (a === ROOT_SENTINEL) return -1;
      if (b === ROOT_SENTINEL) return 1;
      return a.localeCompare(b);
    });

    for (const parent of sortedParents) {
      const childMap = tree.get(parent)!;
      const sortedChildren = [...childMap.keys()].sort((a, b) => {
        if (a === ROOT_SENTINEL) return -1;
        if (b === ROOT_SENTINEL) return 1;
        return a.localeCompare(b);
      });
      const children: FolderNode[] = sortedChildren.map((child) => ({
        child,
        sets: childMap.get(child)!,
      }));
      const totalSets = children.reduce((sum, c) => sum + c.sets.length, 0);
      result.push({ parent, children, totalSets });
    }

    return result;
  }, [filteredSets, audioMap]);

  const toggleChild = useCallback((sets: EmbeddingSet[]) => {
    setSelected((prev) => {
      const next = new Set(prev);
      const allIn = sets.every((es) => next.has(es.id));
      for (const es of sets) {
        if (allIn) next.delete(es.id);
        else next.add(es.id);
      }
      return next;
    });
  }, []);

  const toggleParent = useCallback((node: ParentNode) => {
    setSelected((prev) => {
      const next = new Set(prev);
      const allSets = node.children.flatMap((c) => c.sets);
      const allIn = allSets.every((es) => next.has(es.id));
      for (const es of allSets) {
        if (allIn) next.delete(es.id);
        else next.add(es.id);
      }
      return next;
    });
  }, []);

  const toggleAll = useCallback(() => {
    const allSelected = filteredSets.length > 0 && filteredSets.every((es) => selected.has(es.id));
    if (allSelected) {
      setSelected(new Set());
    } else {
      setSelected(new Set(filteredSets.map((es) => es.id)));
    }
  }, [filteredSets, selected]);

  // null = initial state (all collapsed); once user interacts, explicit Set
  const allParentKeys = useMemo(() => new Set(folderTree.map((n) => n.parent)), [folderTree]);
  const effectiveCollapsed = collapsed ?? allParentKeys;

  const toggleCollapse = useCallback((parent: string) => {
    setCollapsed((prev) => {
      const base = prev ?? allParentKeys;
      const next = new Set(base);
      if (next.has(parent)) next.delete(parent);
      else next.add(parent);
      return next;
    });
  }, [allParentKeys]);

  const handleSubmit = useCallback(
    (params: ClusteringParams) => {
      if (selected.size === 0) {
        showMsg("error", "Select at least one embedding set");
        return;
      }

      const selectedSets = embeddingSets.filter((es) => selected.has(es.id));

      // Dimension validation
      const dims = new Set(selectedSets.map((es) => es.vector_dim));
      if (dims.size > 1) {
        showMsg("error", `Cannot cluster sets with mixed dimensions: ${[...dims].join(", ")}`);
        return;
      }

      // Model version validation
      const models = new Set(selectedSets.map((es) => es.model_version));
      if (models.size > 1) {
        showMsg("error", `Cannot cluster sets from different models: ${[...models].join(", ")}`);
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
      if (params.stabilityRuns >= 2) {
        parameters.stability_runs = params.stabilityRuns;
      }
      if (params.runClassifier) {
        parameters.run_classifier = true;
      }
      if (params.enableMetricLearning) {
        parameters.enable_metric_learning = true;
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

  const allSelected = filteredSets.length > 0 && filteredSets.every((es) => selected.has(es.id));
  const displayName = (key: string) => (key === ROOT_SENTINEL ? "(root)" : key);

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
        <ModelFilter items={embeddingSets} value={modelFilter} onChange={setModelFilter} />
        {/* Two-level folder tree */}
        <div className="space-y-1 max-h-72 overflow-y-auto">
          {folderTree.map((node) => {
            const allParentSets = node.children.flatMap((c) => c.sets);
            const parentAllSelected = allParentSets.every((es) => selected.has(es.id));
            const parentSomeSelected = allParentSets.some((es) => selected.has(es.id));
            const isCollapsed = effectiveCollapsed.has(node.parent);

            return (
              <div key={node.parent}>
                {/* Parent folder row */}
                <div className="flex items-center gap-1.5 py-1">
                  <button
                    className="p-0.5 hover:bg-muted rounded"
                    onClick={() => toggleCollapse(node.parent)}
                  >
                    {isCollapsed ? (
                      <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                    )}
                  </button>
                  <Checkbox
                    checked={parentAllSelected ? true : parentSomeSelected ? "indeterminate" : false}
                    onCheckedChange={() => toggleParent(node)}
                  />
                  <span
                    className="text-sm font-medium cursor-pointer select-none"
                    onClick={() => toggleCollapse(node.parent)}
                  >
                    {displayName(node.parent)}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    ({node.totalSets} sets)
                  </span>
                </div>

                {/* Child folder rows */}
                {!isCollapsed && (
                  <div className="ml-6 space-y-0.5">
                    {node.children.map((child) => {
                      const childAllSelected = child.sets.every((es) => selected.has(es.id));
                      const childSomeSelected = child.sets.some((es) => selected.has(es.id));
                      return (
                        <label
                          key={child.child}
                          className="flex items-center gap-2 py-0.5 text-sm cursor-pointer"
                        >
                          <Checkbox
                            checked={childAllSelected ? true : childSomeSelected ? "indeterminate" : false}
                            onCheckedChange={() => toggleChild(child.sets)}
                          />
                          <span className="truncate">{displayName(child.child)}</span>
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0 font-normal shrink-0">
                            {child.sets[0]?.model_version}
                          </Badge>
                          <span className="text-xs text-muted-foreground ml-auto shrink-0">
                            {child.sets.length} {child.sets.length === 1 ? "set" : "sets"}
                          </span>
                        </label>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
          {filteredSets.length === 0 && (
            <p className="text-sm text-muted-foreground">No embedding sets available. Process some audio first.</p>
          )}
        </div>

        {selected.size > 0 && (
          <p className="text-xs text-muted-foreground">
            {selected.size} of {filteredSets.length} embedding sets selected
          </p>
        )}

        <ClusteringParamsForm
          onSubmit={handleSubmit}
          disabled={createJob.isPending || selected.size === 0}
        />
      </CardContent>
    </Card>
  );
}
