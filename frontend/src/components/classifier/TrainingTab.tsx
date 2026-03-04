import { useState, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { ChevronRight, ChevronDown, FolderOpen } from "lucide-react";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { useEmbeddingSets } from "@/hooks/queries/useProcessing";
import {
  useTrainingJobs,
  useClassifierModels,
  useCreateTrainingJob,
  useDeleteClassifierModel,
  useBulkDeleteTrainingJobs,
  useBulkDeleteClassifierModels,
} from "@/hooks/queries/useClassifier";
import { FolderBrowser } from "@/components/shared/FolderBrowser";
import { BulkDeleteDialog } from "./BulkDeleteDialog";
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
  const bulkDeleteJobsMutation = useBulkDeleteTrainingJobs();
  const bulkDeleteModelsMutation = useBulkDeleteClassifierModels();

  const [name, setName] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [collapsed, setCollapsed] = useState<Set<string> | null>(null);
  const [negativeFolder, setNegativeFolder] = useState("");
  const [folderBrowserOpen, setFolderBrowserOpen] = useState(false);

  // Job table selection
  const [selectedJobIds, setSelectedJobIds] = useState<Set<string>>(new Set());
  const [showJobDeleteDialog, setShowJobDeleteDialog] = useState(false);

  // Model table selection
  const [selectedModelIds, setSelectedModelIds] = useState<Set<string>>(
    new Set(),
  );
  const [showModelDeleteDialog, setShowModelDeleteDialog] = useState(false);

  const hasActiveJobs = trainingJobs.some(
    (j) => j.status === "queued" || j.status === "running",
  );

  const audioMap = useMemo(
    () => new Map(audioFiles.map((af) => [af.id, af])),
    [audioFiles],
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
    [folderTree],
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
    [allParentKeys],
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
      },
    );
  };

  // Job table helpers
  const toggleJobId = (id: string) => {
    setSelectedJobIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const allJobsSelected =
    trainingJobs.length > 0 &&
    trainingJobs.every((j) => selectedJobIds.has(j.id));
  const someJobsSelected = trainingJobs.some((j) => selectedJobIds.has(j.id));

  const toggleAllJobs = () => {
    if (allJobsSelected) setSelectedJobIds(new Set());
    else setSelectedJobIds(new Set(trainingJobs.map((j) => j.id)));
  };

  // Model table helpers
  const toggleModelId = (id: string) => {
    setSelectedModelIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const allModelsSelected =
    models.length > 0 && models.every((m) => selectedModelIds.has(m.id));
  const someModelsSelected = models.some((m) => selectedModelIds.has(m.id));

  const toggleAllModels = () => {
    if (allModelsSelected) setSelectedModelIds(new Set());
    else setSelectedModelIds(new Set(models.map((m) => m.id)));
  };

  const allSelected =
    embeddingSets.length > 0 &&
    embeddingSets.every((es) => selected.has(es.id));
  const displayName = (key: string) =>
    key === ROOT_SENTINEL ? "(root)" : key;

  const jobsWithModels = useMemo(() => {
    const count = [...selectedJobIds].filter((id) => {
      const job = trainingJobs.find((j) => j.id === id);
      return job?.classifier_model_id;
    }).length;
    return count;
  }, [selectedJobIds, trainingJobs]);

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
                  selected.has(es.id),
                );
                const parentSomeSelected = allParentSets.some((es) =>
                  selected.has(es.id),
                );
                const isCollapsed = effectiveCollapsed.has(node.parent);

                return (
                  <div key={node.parent}>
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

                    {!isCollapsed && (
                      <div className="ml-6 space-y-0.5">
                        {node.children.map((child) => {
                          const childAllSelected = child.sets.every((es) =>
                            selected.has(es.id),
                          );
                          const childSomeSelected = child.sets.some((es) =>
                            selected.has(es.id),
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
            <div className="flex gap-2">
              <Input
                value={negativeFolder}
                onChange={(e) => setNegativeFolder(e.target.value)}
                placeholder="/path/to/negative/audio"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={() => setFolderBrowserOpen(true)}
                title="Browse folders"
              >
                <FolderOpen className="h-4 w-4" />
              </Button>
            </div>
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

      {/* Training Jobs Table */}
      {trainingJobs.length > 0 && (
        <div className="border rounded-md">
          <div className="flex items-center justify-between px-4 py-3 border-b">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold">Training Jobs</h3>
              <Badge variant="secondary">{trainingJobs.length}</Badge>
              {hasActiveJobs && (
                <span className="text-xs text-muted-foreground">
                  (polling…)
                </span>
              )}
            </div>
            {selectedJobIds.size > 0 && (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => setShowJobDeleteDialog(true)}
              >
                Delete ({selectedJobIds.size})
              </Button>
            )}
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="w-10 px-3 py-2">
                  <Checkbox
                    checked={
                      allJobsSelected
                        ? true
                        : someJobsSelected
                          ? "indeterminate"
                          : false
                    }
                    onCheckedChange={toggleAllJobs}
                  />
                </th>
                <th className="px-3 py-2 text-left font-medium">Status</th>
                <th className="px-3 py-2 text-left font-medium">Name</th>
                <th className="px-3 py-2 text-left font-medium">
                  Positive Sets
                </th>
                <th className="px-3 py-2 text-left font-medium">Model</th>
                <th className="px-3 py-2 text-left font-medium">Created</th>
                <th className="px-3 py-2 text-left font-medium">Error</th>
              </tr>
            </thead>
            <tbody>
              {trainingJobs.map((job) => (
                <TrainingJobTableRow
                  key={job.id}
                  job={job}
                  checked={selectedJobIds.has(job.id)}
                  onToggle={() => toggleJobId(job.id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Trained Models Table */}
      {models.length > 0 && (
        <div className="border rounded-md">
          <div className="flex items-center justify-between px-4 py-3 border-b">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold">Trained Models</h3>
              <Badge variant="secondary">{models.length}</Badge>
            </div>
            {selectedModelIds.size > 0 && (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => setShowModelDeleteDialog(true)}
              >
                Delete ({selectedModelIds.size})
              </Button>
            )}
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="w-10 px-3 py-2">
                  <Checkbox
                    checked={
                      allModelsSelected
                        ? true
                        : someModelsSelected
                          ? "indeterminate"
                          : false
                    }
                    onCheckedChange={toggleAllModels}
                  />
                </th>
                <th className="px-3 py-2 text-left font-medium">Name</th>
                <th className="px-3 py-2 text-left font-medium">Model</th>
                <th className="px-3 py-2 text-left font-medium">Accuracy</th>
                <th className="px-3 py-2 text-left font-medium">AUC</th>
                <th className="px-3 py-2 text-left font-medium">Created</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <ModelTableRow
                  key={m.id}
                  model={m}
                  checked={selectedModelIds.has(m.id)}
                  onToggle={() => toggleModelId(m.id)}
                  onDelete={() => deleteMutation.mutate(m.id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Dialogs */}
      <FolderBrowser
        open={folderBrowserOpen}
        onOpenChange={setFolderBrowserOpen}
        onSelect={setNegativeFolder}
        initialPath={negativeFolder || "/"}
      />

      <BulkDeleteDialog
        open={showJobDeleteDialog}
        onOpenChange={setShowJobDeleteDialog}
        count={selectedJobIds.size}
        entityName="training job"
        warningText={
          jobsWithModels > 0
            ? `${jobsWithModels} job(s) have trained models that will also be deleted.`
            : undefined
        }
        onConfirm={() => {
          bulkDeleteJobsMutation.mutate([...selectedJobIds], {
            onSuccess: () => {
              setSelectedJobIds(new Set());
              setShowJobDeleteDialog(false);
            },
          });
        }}
        isPending={bulkDeleteJobsMutation.isPending}
      />

      <BulkDeleteDialog
        open={showModelDeleteDialog}
        onOpenChange={setShowModelDeleteDialog}
        count={selectedModelIds.size}
        entityName="classifier model"
        onConfirm={() => {
          bulkDeleteModelsMutation.mutate([...selectedModelIds], {
            onSuccess: () => {
              setSelectedModelIds(new Set());
              setShowModelDeleteDialog(false);
            },
          });
        }}
        isPending={bulkDeleteModelsMutation.isPending}
      />
    </div>
  );
}

const statusColor: Record<string, string> = {
  queued: "bg-yellow-100 text-yellow-800",
  running: "bg-blue-100 text-blue-800",
  complete: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
  canceled: "bg-gray-100 text-gray-800",
};

function TrainingJobTableRow({
  job,
  checked,
  onToggle,
}: {
  job: ClassifierTrainingJob;
  checked: boolean;
  onToggle: () => void;
}) {
  return (
    <tr className="border-b last:border-0 hover:bg-muted/30">
      <td className="px-3 py-2">
        <Checkbox checked={checked} onCheckedChange={onToggle} />
      </td>
      <td className="px-3 py-2">
        <Badge className={statusColor[job.status] ?? ""}>{job.status}</Badge>
      </td>
      <td className="px-3 py-2 font-medium">{job.name}</td>
      <td className="px-3 py-2 text-muted-foreground">
        {job.positive_embedding_set_ids.length} set(s)
      </td>
      <td className="px-3 py-2 text-muted-foreground">{job.model_version}</td>
      <td className="px-3 py-2 text-muted-foreground">
        {new Date(job.created_at).toLocaleDateString()}
      </td>
      <td className="px-3 py-2">
        {job.error_message && (
          <span className="text-red-600 text-xs truncate block max-w-48">
            {job.error_message}
          </span>
        )}
      </td>
    </tr>
  );
}

function ModelTableRow({
  model,
  checked,
  onToggle,
  onDelete,
}: {
  model: ClassifierModelInfo;
  checked: boolean;
  onToggle: () => void;
  onDelete: () => void;
}) {
  const summary = model.training_summary as Record<string, number> | null;
  return (
    <tr className="border-b last:border-0 hover:bg-muted/30">
      <td className="px-3 py-2">
        <Checkbox checked={checked} onCheckedChange={onToggle} />
      </td>
      <td className="px-3 py-2 font-medium">{model.name}</td>
      <td className="px-3 py-2 text-muted-foreground">
        {model.model_version}
      </td>
      <td className="px-3 py-2 text-muted-foreground">
        {summary ? `${((summary.cv_accuracy ?? 0) * 100).toFixed(1)}%` : "—"}
      </td>
      <td className="px-3 py-2 text-muted-foreground">
        {summary ? `${((summary.cv_roc_auc ?? 0) * 100).toFixed(1)}%` : "—"}
      </td>
      <td className="px-3 py-2 text-muted-foreground">
        {new Date(model.created_at).toLocaleDateString()}
      </td>
    </tr>
  );
}
