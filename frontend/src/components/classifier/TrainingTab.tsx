import { useState, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronRight, ChevronDown, Settings2, AlertTriangle, RotateCcw, Loader2, CheckCircle2, XCircle } from "lucide-react";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { useEmbeddingSets } from "@/hooks/queries/useProcessing";
import { ModelFilter } from "@/components/shared/ModelFilter";
import {
  useTrainingJobs,
  useClassifierModels,
  useCreateTrainingJob,
  useDeleteClassifierModel,
  useBulkDeleteTrainingJobs,
  useBulkDeleteClassifierModels,
  useRetrainInfo,
  useRetrainWorkflows,
  useCreateRetrainWorkflow,
} from "@/hooks/queries/useClassifier";
import { BulkDeleteDialog } from "./BulkDeleteDialog";
import type {
  ClassifierTrainingJob,
  ClassifierModelInfo,
  EmbeddingSet,
  RetrainWorkflow as RetrainWorkflowType,
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

  // Retrain workflows — poll at 3s to track active workflows
  const { data: retrainWorkflows = [] } = useRetrainWorkflows(3000);

  const [name, setName] = useState("");
  const [modelFilter, setModelFilter] = useState("__all__");
  const [posSelected, setPosSelected] = useState<Set<string>>(new Set());
  const [posCollapsed, setPosCollapsed] = useState<Set<string> | null>(null);
  const [negSelected, setNegSelected] = useState<Set<string>>(new Set());
  const [negCollapsed, setNegCollapsed] = useState<Set<string> | null>(null);

  // Advanced options
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [classifierType, setClassifierType] = useState("logistic_regression");
  const [l2Normalize, setL2Normalize] = useState(false);
  const [regularizationC, setRegularizationC] = useState(1.0);
  const [classWeight, setClassWeight] = useState("balanced");

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

  const filteredSets = useMemo(
    () => modelFilter === "__all__" ? embeddingSets : embeddingSets.filter((es) => es.model_version === modelFilter),
    [embeddingSets, modelFilter],
  );

  // Build two-level folder tree: parent → child → embedding sets
  const folderTree = useMemo(() => {
    const tree = new Map<string, Map<string, EmbeddingSet[]>>();

    for (const es of filteredSets) {
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
  }, [filteredSets, audioMap]);

  const allParentKeys = useMemo(
    () => new Set(folderTree.map((n) => n.parent)),
    [folderTree],
  );

  // Generic toggle helpers that work for both pos and neg panels
  const makeToggleChild = (
    setSel: React.Dispatch<React.SetStateAction<Set<string>>>,
  ) =>
    (sets: EmbeddingSet[]) => {
      setSel((prev) => {
        const next = new Set(prev);
        const allIn = sets.every((es) => next.has(es.id));
        for (const es of sets) {
          if (allIn) next.delete(es.id);
          else next.add(es.id);
        }
        return next;
      });
    };

  const makeToggleParent = (
    setSel: React.Dispatch<React.SetStateAction<Set<string>>>,
  ) =>
    (node: ParentNode) => {
      setSel((prev) => {
        const next = new Set(prev);
        const allSets = node.children.flatMap((c) => c.sets);
        const allIn = allSets.every((es) => next.has(es.id));
        for (const es of allSets) {
          if (allIn) next.delete(es.id);
          else next.add(es.id);
        }
        return next;
      });
    };

  const makeToggleAll = (
    sel: Set<string>,
    setSel: React.Dispatch<React.SetStateAction<Set<string>>>,
  ) =>
    () => {
      const allSel =
        filteredSets.length > 0 &&
        filteredSets.every((es) => sel.has(es.id));
      if (allSel) setSel(new Set());
      else setSel(new Set(filteredSets.map((es) => es.id)));
    };

  const makeToggleCollapse = (
    collapsed: Set<string> | null,
    setCollapsed: React.Dispatch<React.SetStateAction<Set<string> | null>>,
  ) =>
    (parent: string) => {
      setCollapsed((prev) => {
        const base = prev ?? allParentKeys;
        const next = new Set(base);
        if (next.has(parent)) next.delete(parent);
        else next.add(parent);
        return next;
      });
    };

  const togglePosChild = useCallback(makeToggleChild(setPosSelected), []);
  const togglePosParent = useCallback(makeToggleParent(setPosSelected), []);
  const togglePosAll = useCallback(
    makeToggleAll(posSelected, setPosSelected),
    [filteredSets, posSelected],
  );
  const togglePosCollapse = useCallback(
    makeToggleCollapse(posCollapsed, setPosCollapsed),
    [allParentKeys],
  );

  const toggleNegChild = useCallback(makeToggleChild(setNegSelected), []);
  const toggleNegParent = useCallback(makeToggleParent(setNegSelected), []);
  const toggleNegAll = useCallback(
    makeToggleAll(negSelected, setNegSelected),
    [filteredSets, negSelected],
  );
  const toggleNegCollapse = useCallback(
    makeToggleCollapse(negCollapsed, setNegCollapsed),
    [allParentKeys],
  );

  const selectedModels = useMemo(() => {
    const ids = new Set([...posSelected, ...negSelected]);
    return new Set(embeddingSets.filter((es) => ids.has(es.id)).map((es) => es.model_version));
  }, [posSelected, negSelected, embeddingSets]);

  const modelMismatch = selectedModels.size > 1;

  const handleSubmit = () => {
    if (!name || posSelected.size === 0 || negSelected.size === 0) return;
    const parameters: Record<string, unknown> = {
      classifier_type: classifierType,
      l2_normalize: l2Normalize,
      class_weight: classWeight === "balanced" ? "balanced" : null,
    };
    if (classifierType === "logistic_regression") {
      parameters.C = regularizationC;
    }
    createMutation.mutate(
      {
        name,
        positive_embedding_set_ids: [...posSelected],
        negative_embedding_set_ids: [...negSelected],
        parameters,
      },
      {
        onSuccess: () => {
          setName("");
          setPosSelected(new Set());
          setNegSelected(new Set());
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

          <ModelFilter items={embeddingSets} value={modelFilter} onChange={setModelFilter} />

          {/* Positive Embedding Sets */}
          <EmbeddingSetPanel
            label="Positive Embedding Sets"
            selected={posSelected}
            collapsed={posCollapsed ?? allParentKeys}
            folderTree={folderTree}
            embeddingSets={filteredSets}
            onToggleChild={togglePosChild}
            onToggleParent={togglePosParent}
            onToggleAll={togglePosAll}
            onToggleCollapse={togglePosCollapse}
            displayName={displayName}
          />

          {/* Negative Embedding Sets */}
          <EmbeddingSetPanel
            label="Negative Embedding Sets"
            selected={negSelected}
            collapsed={negCollapsed ?? allParentKeys}
            folderTree={folderTree}
            embeddingSets={filteredSets}
            onToggleChild={toggleNegChild}
            onToggleParent={toggleNegParent}
            onToggleAll={toggleNegAll}
            onToggleCollapse={toggleNegCollapse}
            displayName={displayName}
          />

          {/* Advanced Options */}
          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground">
                <Settings2 className="h-3.5 w-3.5" />
                Advanced Options
                {advancedOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-3 pt-2 border rounded p-3 mt-1">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-sm font-medium">Classifier Type</label>
                  <Select value={classifierType} onValueChange={setClassifierType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="logistic_regression">Logistic Regression</SelectItem>
                      <SelectItem value="mlp">Neural Network (MLP)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="text-sm font-medium">Class Weight</label>
                  <Select value={classWeight} onValueChange={setClassWeight}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="balanced">Balanced</SelectItem>
                      <SelectItem value="none">None (uniform)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Checkbox
                    checked={l2Normalize}
                    onCheckedChange={(v) => setL2Normalize(v === true)}
                  />
                  L2 Normalize Embeddings
                </label>
                {classifierType === "logistic_regression" && (
                  <div className="flex items-center gap-2">
                    <label className="text-sm font-medium whitespace-nowrap">Regularization (C)</label>
                    <Input
                      type="number"
                      min={0.001}
                      step={0.1}
                      value={regularizationC}
                      onChange={(e) => setRegularizationC(parseFloat(e.target.value) || 1.0)}
                      className="w-24"
                    />
                  </div>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>

          {modelMismatch && (
            <div className="flex items-center gap-2 rounded-md border border-yellow-300 bg-yellow-50 px-3 py-2 text-sm text-yellow-800">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              Cannot train with embedding sets from different models: {[...selectedModels].join(", ")}
            </div>
          )}

          <Button
            onClick={handleSubmit}
            disabled={
              !name ||
              posSelected.size === 0 ||
              negSelected.size === 0 ||
              modelMismatch ||
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
            <Button
              variant="destructive"
              size="sm"
              disabled={selectedJobIds.size === 0}
              onClick={() => setShowJobDeleteDialog(true)}
            >
              Delete ({selectedJobIds.size})
            </Button>
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
            <Button
              variant="destructive"
              size="sm"
              disabled={selectedModelIds.size === 0}
              onClick={() => setShowModelDeleteDialog(true)}
            >
              Delete ({selectedModelIds.size})
            </Button>
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
                <th className="px-3 py-2 text-left font-medium">Samples</th>
                <th className="px-3 py-2 text-left font-medium">Accuracy</th>
                <th className="px-3 py-2 text-left font-medium">AUC</th>
                <th className="px-3 py-2 text-left font-medium">Precision</th>
                <th className="px-3 py-2 text-left font-medium">F1</th>
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
                  retrainWorkflow={retrainWorkflows.find(
                    (w) => w.source_model_id === m.id
                  )}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Dialogs */}
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

// ---- Embedding Set Panel (shared by positive and negative) ----

function EmbeddingSetPanel({
  label,
  selected,
  collapsed,
  folderTree,
  embeddingSets,
  onToggleChild,
  onToggleParent,
  onToggleAll,
  onToggleCollapse,
  displayName,
}: {
  label: string;
  selected: Set<string>;
  collapsed: Set<string>;
  folderTree: ParentNode[];
  embeddingSets: EmbeddingSet[];
  onToggleChild: (sets: EmbeddingSet[]) => void;
  onToggleParent: (node: ParentNode) => void;
  onToggleAll: () => void;
  onToggleCollapse: (parent: string) => void;
  displayName: (key: string) => string;
}) {
  const allSelected =
    embeddingSets.length > 0 &&
    embeddingSets.every((es) => selected.has(es.id));
  const someSelected = embeddingSets.some((es) => selected.has(es.id));

  return (
    <div>
      <div className="space-y-1 max-h-72 overflow-y-auto border rounded p-2">
        <div className="flex items-center gap-2 pb-1 border-b mb-1">
          <Checkbox
            checked={
              allSelected ? true : someSelected ? "indeterminate" : false
            }
            onCheckedChange={onToggleAll}
          />
          <span className="text-sm font-medium">{label}</span>
        </div>
        {folderTree.map((node) => {
          const allParentSets = node.children.flatMap((c) => c.sets);
          const parentAllSelected = allParentSets.every((es) =>
            selected.has(es.id),
          );
          const parentSomeSelected = allParentSets.some((es) =>
            selected.has(es.id),
          );
          const isCollapsed = collapsed.has(node.parent);

          return (
            <div key={node.parent}>
              <div className="flex items-center gap-1.5 py-1">
                <button
                  className="p-0.5 hover:bg-muted rounded"
                  onClick={() => onToggleCollapse(node.parent)}
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
                  onCheckedChange={() => onToggleParent(node)}
                />
                <span
                  className="text-sm font-medium cursor-pointer select-none"
                  onClick={() => onToggleCollapse(node.parent)}
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
                          onCheckedChange={() => onToggleChild(child.sets)}
                        />
                        <span className="truncate">
                          {displayName(child.child)}
                        </span>
                        <Badge variant="outline" className="text-[10px] px-1.5 py-0 font-normal shrink-0">
                          {child.sets[0]?.model_version}
                        </Badge>
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
          {selected.size} of {embeddingSets.length} embedding sets selected
        </p>
      )}
    </div>
  );
}

// ---- Table Row Components ----

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

function RetrainPanel({ model, workflow }: { model: ClassifierModelInfo; workflow?: RetrainWorkflowType }) {
  const [showForm, setShowForm] = useState(false);
  const [newName, setNewName] = useState(`${model.name}-retrained`);
  const retrainInfo = useRetrainInfo(showForm ? model.id : null);
  const createRetrain = useCreateRetrainWorkflow();

  const handleSubmit = () => {
    createRetrain.mutate(
      {
        source_model_id: model.id,
        new_model_name: newName,
        parameters: retrainInfo.data?.parameters ?? undefined,
      },
      { onSuccess: () => setShowForm(false) }
    );
  };

  const stepLabels = ["Importing", "Processing", "Training", "Complete"];
  const stepIndex = workflow
    ? { importing: 0, processing: 1, training: 2, complete: 3, failed: -1 }[workflow.status] ?? -1
    : -1;

  if (workflow && !showForm) {
    return (
      <div className="mt-3 border-t pt-3">
        <div className="flex items-center gap-2 text-xs font-medium mb-2">
          <RotateCcw className="h-3.5 w-3.5" />
          Retrain Workflow
        </div>
        {workflow.status === "failed" ? (
          <div className="flex items-center gap-2 text-xs text-destructive">
            <XCircle className="h-3.5 w-3.5" />
            Failed: {workflow.error_message}
          </div>
        ) : workflow.status === "complete" ? (
          <div className="flex items-center gap-2 text-xs text-green-700">
            <CheckCircle2 className="h-3.5 w-3.5" />
            Complete — new model: {workflow.new_model_name}
          </div>
        ) : (
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              {stepLabels.map((label, i) => (
                <div key={label} className="flex items-center gap-1">
                  <div
                    className={`h-2 w-2 rounded-full ${
                      i < stepIndex
                        ? "bg-green-500"
                        : i === stepIndex
                        ? "bg-blue-500 animate-pulse"
                        : "bg-muted"
                    }`}
                  />
                  <span className={`text-[10px] ${i === stepIndex ? "font-medium" : "text-muted-foreground"}`}>
                    {label}
                  </span>
                </div>
              ))}
            </div>
            {workflow.status === "processing" && workflow.processing_total != null && (
              <div className="text-[10px] text-muted-foreground">
                Processing: {workflow.processing_complete ?? 0}/{workflow.processing_total} jobs
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="mt-3 border-t pt-3">
      {!showForm ? (
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-xs gap-1.5"
          onClick={() => setShowForm(true)}
        >
          <RotateCcw className="h-3 w-3" />
          Retrain
        </Button>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-xs font-medium">
            <RotateCcw className="h-3.5 w-3.5" />
            Retrain from {model.name}
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] font-medium text-muted-foreground">
                New Model Name
              </label>
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="h-7 text-xs mt-0.5"
              />
            </div>
          </div>
          {retrainInfo.isLoading ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading folder info...
            </div>
          ) : retrainInfo.data ? (
            <div className="text-[10px] space-y-1">
              <div>
                <span className="font-medium">Positive folders:</span>{" "}
                <span className="text-muted-foreground">
                  {retrainInfo.data.positive_folder_roots.map((r) => r.split("/").pop()).join(", ")}
                </span>
              </div>
              <div>
                <span className="font-medium">Negative folders:</span>{" "}
                <span className="text-muted-foreground">
                  {retrainInfo.data.negative_folder_roots.map((r) => r.split("/").pop()).join(", ")}
                </span>
              </div>
              <div>
                <span className="font-medium">Parameters:</span>{" "}
                <span className="text-muted-foreground">
                  {Object.entries(retrainInfo.data.parameters)
                    .map(([k, v]) => `${k}=${v}`)
                    .join(", ") || "defaults"}
                </span>
              </div>
            </div>
          ) : retrainInfo.isError ? (
            <div className="text-xs text-destructive">
              Unable to load retrain info. Model may lack training provenance.
            </div>
          ) : null}
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              className="h-7 text-xs"
              onClick={handleSubmit}
              disabled={
                !newName.trim() ||
                createRetrain.isPending ||
                !retrainInfo.data
              }
            >
              {createRetrain.isPending && (
                <Loader2 className="h-3 w-3 animate-spin mr-1" />
              )}
              Start Retrain
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={() => setShowForm(false)}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function ModelTableRow({
  model,
  checked,
  onToggle,
  onDelete,
  retrainWorkflow,
}: {
  model: ClassifierModelInfo;
  checked: boolean;
  onToggle: () => void;
  onDelete: () => void;
  retrainWorkflow?: RetrainWorkflowType;
}) {
  const [expanded, setExpanded] = useState(false);
  const summary = model.training_summary as Record<string, unknown> | null;
  const nPos = summary?.n_positive as number | undefined;
  const nNeg = summary?.n_negative as number | undefined;
  const balanceRatio = summary?.balance_ratio as number | undefined;
  const imbalanceWarning = summary?.imbalance_warning as string | undefined;
  const cvAccuracy = summary?.cv_accuracy as number | undefined;
  const cvRocAuc = summary?.cv_roc_auc as number | undefined;
  const cvPrecision = summary?.cv_precision as number | undefined;
  const cvF1 = summary?.cv_f1 as number | undefined;
  const scoreSeparation = summary?.score_separation as number | undefined;
  const classifierType = summary?.classifier_type as string | undefined;
  const trainConfusion = summary?.train_confusion as Record<string, number> | undefined;
  const effectiveWeights = summary?.effective_class_weights as Record<string, string> | undefined;
  const configWarning = summary?._config_mismatch_warning as string | undefined;

  const classifierTag = classifierType === "mlp" ? "MLP" : classifierType === "logistic_regression" ? "LR" : null;

  return (
    <>
      <tr
        className="border-b last:border-0 hover:bg-muted/30 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
          <Checkbox checked={checked} onCheckedChange={onToggle} />
        </td>
        <td className="px-3 py-2 font-medium">
          <span className="flex items-center gap-1.5">
            {expanded ? <ChevronDown className="h-3.5 w-3.5 text-muted-foreground shrink-0" /> : <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />}
            {model.name}
            {classifierTag && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 font-normal">
                {classifierTag}
              </Badge>
            )}
          </span>
        </td>
        <td className="px-3 py-2 text-muted-foreground">
          {model.model_version}
        </td>
        <td className="px-3 py-2">
          {nPos != null && nNeg != null ? (
            <span className="flex items-center gap-1.5">
              <span className="text-muted-foreground">
                {nPos}+ / {nNeg}&minus;
              </span>
              {imbalanceWarning && (
                <Badge
                  className="bg-amber-100 text-amber-800 text-[10px] px-1.5 py-0"
                  title={imbalanceWarning}
                >
                  {balanceRatio}:1
                </Badge>
              )}
            </span>
          ) : (
            "—"
          )}
        </td>
        <td className="px-3 py-2 text-muted-foreground">
          {cvAccuracy != null ? `${(cvAccuracy * 100).toFixed(1)}%` : "—"}
        </td>
        <td className="px-3 py-2 text-muted-foreground">
          {cvRocAuc != null ? `${(cvRocAuc * 100).toFixed(1)}%` : "—"}
        </td>
        <td className="px-3 py-2">
          <span className="flex items-center gap-1">
            <span className="text-muted-foreground">
              {cvPrecision != null ? `${(cvPrecision * 100).toFixed(1)}%` : "—"}
            </span>
            {cvPrecision != null && cvPrecision < 0.7 && (
              <Badge className="bg-amber-100 text-amber-800 text-[10px] px-1.5 py-0">Low</Badge>
            )}
          </span>
        </td>
        <td className="px-3 py-2 text-muted-foreground">
          {cvF1 != null ? `${(cvF1 * 100).toFixed(1)}%` : "—"}
        </td>
        <td className="px-3 py-2">
          <span className="flex items-center gap-1">
            <span className="text-muted-foreground">
              {new Date(model.created_at).toLocaleDateString()}
            </span>
            {scoreSeparation != null && scoreSeparation < 1.0 && (
              <Badge className="bg-amber-100 text-amber-800 text-[10px] px-1.5 py-0" title={`Score separation: ${scoreSeparation.toFixed(2)}`}>
                Poor sep.
              </Badge>
            )}
          </span>
        </td>
      </tr>
      {expanded && (
        <tr className="border-b last:border-0 bg-muted/20">
          <td colSpan={9} className="px-6 py-3">
            <div className="grid grid-cols-3 gap-4 text-xs">
              {scoreSeparation != null && (
                <div>
                  <span className="font-medium">Score Separation:</span>{" "}
                  <span className="text-muted-foreground">{scoreSeparation.toFixed(3)}</span>
                </div>
              )}
              {effectiveWeights && (
                <div>
                  <span className="font-medium">Class Weights:</span>{" "}
                  <span className="text-muted-foreground">neg={effectiveWeights["0"]}, pos={effectiveWeights["1"]}</span>
                </div>
              )}
              {summary?.l2_normalize != null && (
                <div>
                  <span className="font-medium">L2 Normalize:</span>{" "}
                  <span className="text-muted-foreground">{summary.l2_normalize ? "Yes" : "No"}</span>
                </div>
              )}
              {trainConfusion && (
                <div className="col-span-3">
                  <span className="font-medium">Train Confusion:</span>{" "}
                  <span className="text-muted-foreground">
                    TP={trainConfusion.tp} FP={trainConfusion.fp} TN={trainConfusion.tn} FN={trainConfusion.fn}
                  </span>
                </div>
              )}
              {configWarning && (
                <div className="col-span-3">
                  <Badge className="bg-amber-100 text-amber-800 text-[10px]">{configWarning}</Badge>
                </div>
              )}
            </div>
            <RetrainPanel model={model} workflow={retrainWorkflow} />
          </td>
        </tr>
      )}
    </>
  );
}
