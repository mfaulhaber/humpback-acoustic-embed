import { useState, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { ChevronRight, ChevronDown } from "lucide-react";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { useEmbeddingSets } from "@/hooks/queries/useProcessing";
import {
  useTrainingJobs,
  useClassifierModels,
  useCreateTrainingJob,
  useDeleteClassifierModel,
} from "@/hooks/queries/useClassifier";
import type {
  ClassifierTrainingJob,
  ClassifierModelInfo,
  EmbeddingSet,
} from "@/api/types";

const ROOT_SENTINEL = "__root__";

interface FolderNode {
  child: string;
  sets: EmbeddingSet[];
}

interface ParentNode {
  parent: string;
  children: FolderNode[];
  totalSets: number;
}

export function TrainingTab() {
  const { data: embeddingSets = [] } = useEmbeddingSets();
  const { data: audioFiles = [] } = useAudioFiles();
  const { data: trainingJobs = [] } = useTrainingJobs(3000);
  const { data: models = [] } = useClassifierModels();
  const createMutation = useCreateTrainingJob();
  const deleteMutation = useDeleteClassifierModel();

  const [name, setName] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [collapsed, setCollapsed] = useState<Set<string> | null>(null);
  const [negativeFolder, setNegativeFolder] = useState("");

  const hasActiveJobs = trainingJobs.some(
    (j) => j.status === "queued" || j.status === "running"
  );

  const audioMap = useMemo(
    () => new Map(audioFiles.map((af) => [af.id, af])),
    [audioFiles]
  );

  // Build two-level folder tree: parent → child → embedding sets
  const folderTree = useMemo(() => {
    const tree = new Map<string, Map<string, EmbeddingSet[]>>();

    for (const es of embeddingSets) {
      const af = audioMap.get(es.audio_file_id);
      const folderPath = af?.folder_path || "";
      const slashIdx = folderPath.indexOf("/");
      const parent = folderPath
        ? slashIdx >= 0
          ? folderPath.slice(0, slashIdx)
          : folderPath
        : ROOT_SENTINEL;
      const child =
        folderPath && slashIdx >= 0
          ? folderPath.slice(slashIdx + 1)
          : ROOT_SENTINEL;

      if (!tree.has(parent)) tree.set(parent, new Map());
      const childMap = tree.get(parent)!;
      if (!childMap.has(child)) childMap.set(child, []);
      childMap.get(child)!.push(es);
    }

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
  }, [embeddingSets, audioMap]);

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
    const allSelected =
      embeddingSets.length > 0 &&
      embeddingSets.every((es) => selected.has(es.id));
    if (allSelected) {
      setSelected(new Set());
    } else {
      setSelected(new Set(embeddingSets.map((es) => es.id)));
    }
  }, [embeddingSets, selected]);

  const allParentKeys = useMemo(
    () => new Set(folderTree.map((n) => n.parent)),
    [folderTree]
  );
  const effectiveCollapsed = collapsed ?? allParentKeys;

  const toggleCollapse = useCallback(
    (parent: string) => {
      setCollapsed((prev) => {
        const base = prev ?? allParentKeys;
        const next = new Set(base);
        if (next.has(parent)) next.delete(parent);
        else next.add(parent);
        return next;
      });
    },
    [allParentKeys]
  );

  const handleSubmit = () => {
    if (!name || selected.size === 0 || !negativeFolder) return;
    createMutation.mutate(
      {
        name,
        positive_embedding_set_ids: [...selected],
        negative_audio_folder: negativeFolder,
      },
      {
        onSuccess: () => {
          setName("");
          setSelected(new Set());
          setNegativeFolder("");
        },
      }
    );
  };

  const allSelected =
    embeddingSets.length > 0 &&
    embeddingSets.every((es) => selected.has(es.id));
  const displayName = (key: string) =>
    key === ROOT_SENTINEL ? "(root)" : key;

  return (
    <div className="space-y-4">
      {/* Training Form */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Train Binary Classifier</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <label className="text-sm font-medium">Model Name</label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. humpback-detector-v1"
            />
          </div>

          {/* Folder-based embedding set selector */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-sm font-medium">
                Positive Embedding Sets
              </label>
              <Button variant="outline" size="sm" onClick={toggleAll}>
                {allSelected ? "Deselect All" : "Select All"}
              </Button>
            </div>
            <div className="space-y-1 max-h-72 overflow-y-auto border rounded p-2">
              {folderTree.map((node) => {
                const allParentSets = node.children.flatMap((c) => c.sets);
                const parentAllSelected = allParentSets.every((es) =>
                  selected.has(es.id)
                );
                const parentSomeSelected = allParentSets.some((es) =>
                  selected.has(es.id)
                );
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
                        checked={
                          parentAllSelected
                            ? true
                            : parentSomeSelected
                              ? "indeterminate"
                              : false
                        }
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
                          const childAllSelected = child.sets.every((es) =>
                            selected.has(es.id)
                          );
                          const childSomeSelected = child.sets.some((es) =>
                            selected.has(es.id)
                          );
                          return (
                            <label
                              key={child.child}
                              className="flex items-center gap-2 py-0.5 text-sm cursor-pointer"
                            >
                              <Checkbox
                                checked={
                                  childAllSelected
                                    ? true
                                    : childSomeSelected
                                      ? "indeterminate"
                                      : false
                                }
                                onCheckedChange={() => toggleChild(child.sets)}
                              />
                              <span className="truncate">
                                {displayName(child.child)}
                              </span>
                              <span className="text-xs text-muted-foreground ml-auto shrink-0">
                                {child.sets.length}{" "}
                                {child.sets.length === 1 ? "set" : "sets"}
                              </span>
                            </label>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
              {embeddingSets.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  No embedding sets available. Process audio first.
                </p>
              )}
            </div>
            {selected.size > 0 && (
              <p className="text-xs text-muted-foreground mt-1">
                {selected.size} of {embeddingSets.length} embedding sets
                selected
              </p>
            )}
          </div>

          <div>
            <label className="text-sm font-medium">
              Negative Audio Folder Path
            </label>
            <Input
              value={negativeFolder}
              onChange={(e) => setNegativeFolder(e.target.value)}
              placeholder="/path/to/negative/audio"
            />
          </div>
          <Button
            onClick={handleSubmit}
            disabled={
              !name ||
              selected.size === 0 ||
              !negativeFolder ||
              createMutation.isPending
            }
          >
            {createMutation.isPending ? "Creating…" : "Train Classifier"}
          </Button>
          {createMutation.isError && (
            <p className="text-sm text-red-600">
              {(createMutation.error as Error).message}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Training Jobs */}
      {trainingJobs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              Training Jobs{" "}
              {hasActiveJobs && (
                <span className="text-xs text-muted-foreground">
                  (polling…)
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {trainingJobs.map((job) => (
                <TrainingJobRow key={job.id} job={job} />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Trained Models */}
      {models.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Trained Models</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {models.map((m) => (
                <ModelRow
                  key={m.id}
                  model={m}
                  onDelete={() => deleteMutation.mutate(m.id)}
                />
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function TrainingJobRow({ job }: { job: ClassifierTrainingJob }) {
  const statusColor: Record<string, string> = {
    queued: "bg-yellow-100 text-yellow-800",
    running: "bg-blue-100 text-blue-800",
    complete: "bg-green-100 text-green-800",
    failed: "bg-red-100 text-red-800",
    canceled: "bg-gray-100 text-gray-800",
  };
  return (
    <div className="flex items-center justify-between p-2 border rounded text-sm">
      <div className="flex items-center gap-2">
        <Badge className={statusColor[job.status] ?? ""}>
          {job.status}
        </Badge>
        <span className="font-medium">{job.name}</span>
        <span className="text-muted-foreground">
          {job.positive_embedding_set_ids.length} set(s) — {job.model_version}
        </span>
      </div>
      {job.error_message && (
        <span className="text-red-600 text-xs truncate max-w-64">
          {job.error_message}
        </span>
      )}
    </div>
  );
}

function ModelRow({
  model,
  onDelete,
}: {
  model: ClassifierModelInfo;
  onDelete: () => void;
}) {
  const summary = model.training_summary as Record<string, number> | null;
  return (
    <div className="flex items-center justify-between p-2 border rounded text-sm">
      <div className="flex items-center gap-3">
        <span className="font-medium">{model.name}</span>
        <span className="text-muted-foreground">{model.model_version}</span>
        {summary && (
          <span className="text-muted-foreground">
            acc={((summary.cv_accuracy ?? 0) * 100).toFixed(1)}% AUC=
            {((summary.cv_roc_auc ?? 0) * 100).toFixed(1)}%
          </span>
        )}
      </div>
      <Button variant="outline" size="sm" onClick={onDelete}>
        Delete
      </Button>
    </div>
  );
}
