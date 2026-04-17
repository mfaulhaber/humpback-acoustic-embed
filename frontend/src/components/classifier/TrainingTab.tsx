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
  ROOT_SENTINEL,
  buildFolderTree,
  makeToggleChild,
  makeToggleParent,
  makeToggleAll,
  makeToggleCollapse,
  EmbeddingSetPanel,
} from "@/components/shared/EmbeddingSetPanel";
import type { ParentNode } from "@/components/shared/EmbeddingSetPanel";
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
  useTrainingDataSummary,
} from "@/hooks/queries/useClassifier";
import { Separator } from "@/components/ui/separator";
import { BulkDeleteDialog } from "./BulkDeleteDialog";
import {
  DetectionSourcePicker,
  type DetectionSourcePickerValue,
} from "./DetectionSourcePicker";
import { ActiveEmbeddingBanner } from "./ActiveEmbeddingBanner";
import type {
  ClassifierTrainingJob,
  ClassifierModelInfo,
  EmbeddingSet,
  TrainingSourceInfo,
  RetrainWorkflow as RetrainWorkflowType,
} from "@/api/types";

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
  const [sourceMode, setSourceMode] = useState<"embedding_sets" | "detection_jobs">("embedding_sets");
  const [detectionSource, setDetectionSource] =
    useState<DetectionSourcePickerValue>({
      selectedDetectionJobIds: [],
      embeddingModelVersion: "",
      isReady: false,
    });
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
  const folderTree = useMemo(
    () => buildFolderTree(filteredSets, audioMap),
    [filteredSets, audioMap],
  );

  const allParentKeys = useMemo(
    () => new Set(folderTree.map((n) => n.parent)),
    [folderTree],
  );

  // Toggle helpers using shared factories
  const togglePosChild = useCallback(makeToggleChild(setPosSelected), []);
  const togglePosParent = useCallback(makeToggleParent(setPosSelected), []);
  const togglePosAll = useCallback(
    makeToggleAll(filteredSets, posSelected, setPosSelected),
    [filteredSets, posSelected],
  );
  const togglePosCollapse = useCallback(
    makeToggleCollapse(allParentKeys, posCollapsed, setPosCollapsed),
    [allParentKeys],
  );

  const toggleNegChild = useCallback(makeToggleChild(setNegSelected), []);
  const toggleNegParent = useCallback(makeToggleParent(setNegSelected), []);
  const toggleNegAll = useCallback(
    makeToggleAll(filteredSets, negSelected, setNegSelected),
    [filteredSets, negSelected],
  );
  const toggleNegCollapse = useCallback(
    makeToggleCollapse(allParentKeys, negCollapsed, setNegCollapsed),
    [allParentKeys],
  );

  const selectedModels = useMemo(() => {
    const ids = new Set([...posSelected, ...negSelected]);
    return new Set(embeddingSets.filter((es) => ids.has(es.id)).map((es) => es.model_version));
  }, [posSelected, negSelected, embeddingSets]);

  const modelMismatch = selectedModels.size > 1;

  const handleSubmit = () => {
    if (!name) return;
    const parameters: Record<string, unknown> = {
      classifier_type: classifierType,
      l2_normalize: l2Normalize,
      class_weight: classWeight === "balanced" ? "balanced" : null,
    };
    if (classifierType === "logistic_regression") {
      parameters.C = regularizationC;
    }

    if (sourceMode === "detection_jobs") {
      if (
        detectionSource.selectedDetectionJobIds.length === 0 ||
        !detectionSource.isReady
      )
        return;
      createMutation.mutate(
        {
          name,
          detection_job_ids: detectionSource.selectedDetectionJobIds,
          embedding_model_version: detectionSource.embeddingModelVersion,
          parameters,
        },
        {
          onSuccess: () => {
            setName("");
            setDetectionSource({
              selectedDetectionJobIds: [],
              embeddingModelVersion: "",
              isReady: false,
            });
          },
        },
      );
    } else {
      if (posSelected.size === 0 || negSelected.size === 0) return;
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
    }
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
      <ActiveEmbeddingBanner />

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

          {/* Source Mode Radio */}
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium">Source</label>
            <label className="flex items-center gap-1.5 text-sm cursor-pointer">
              <input
                type="radio"
                name="sourceMode"
                checked={sourceMode === "embedding_sets"}
                onChange={() => setSourceMode("embedding_sets")}
                className="accent-primary"
              />
              Embedding sets
            </label>
            <label className="flex items-center gap-1.5 text-sm cursor-pointer">
              <input
                type="radio"
                name="sourceMode"
                checked={sourceMode === "detection_jobs"}
                onChange={() => setSourceMode("detection_jobs")}
                className="accent-primary"
              />
              Detection jobs
            </label>
          </div>

          {sourceMode === "embedding_sets" ? (
            <>
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
            </>
          ) : (
            <DetectionSourcePicker
              value={detectionSource}
              onChange={setDetectionSource}
            />
          )}

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

          {sourceMode === "embedding_sets" && modelMismatch && (
            <div className="flex items-center gap-2 rounded-md border border-yellow-300 bg-yellow-50 px-3 py-2 text-sm text-yellow-800">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              Cannot train with embedding sets from different models: {[...selectedModels].join(", ")}
            </div>
          )}

          <Button
            onClick={handleSubmit}
            disabled={
              !name ||
              createMutation.isPending ||
              (sourceMode === "embedding_sets"
                ? posSelected.size === 0 ||
                  negSelected.size === 0 ||
                  modelMismatch
                : !detectionSource.isReady)
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
                  trainingJobs={trainingJobs}
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
  const sourceContext = job.source_comparison_context as Record<string, unknown> | null;
  const candidateName = sourceContext?.candidate_name as string | undefined;
  return (
    <tr className="border-b last:border-0 hover:bg-muted/30">
      <td className="px-3 py-2">
        <Checkbox checked={checked} onCheckedChange={onToggle} />
      </td>
      <td className="px-3 py-2">
        <Badge className={statusColor[job.status] ?? ""}>{job.status}</Badge>
      </td>
      <td className="px-3 py-2 font-medium">
        <div className="flex items-center gap-2">
          <span>{job.name}</span>
          {job.source_mode === "autoresearch_candidate" && (
            <Badge variant="outline" className="text-[10px]">
              candidate
            </Badge>
          )}
          {job.source_mode === "detection_manifest" && (
            <Badge variant="outline" className="text-[10px]">
              detection
            </Badge>
          )}
        </div>
      </td>
      <td className="px-3 py-2 text-muted-foreground">
        {job.source_mode === "autoresearch_candidate" ? (
          <div>
            <div>Manifest {job.training_split_name ?? "train"}</div>
            {candidateName && (
              <div className="text-[10px] text-muted-foreground">{candidateName}</div>
            )}
          </div>
        ) : job.source_mode === "detection_manifest" ? (
          `${(job.source_detection_job_ids ?? []).length} detection job(s)`
        ) : (
          `${job.positive_embedding_set_ids.length} set(s)`
        )}
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

  if (model.training_source_mode === "autoresearch_candidate" && !workflow) {
    return (
      <div className="mt-3 border-t pt-3 text-xs text-muted-foreground">
        Candidate-backed models do not support folder-root retrain yet.
      </div>
    );
  }

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
  trainingJobs,
}: {
  model: ClassifierModelInfo;
  checked: boolean;
  onToggle: () => void;
  onDelete: () => void;
  retrainWorkflow?: RetrainWorkflowType;
  trainingJobs: ClassifierTrainingJob[];
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
  const cvRecall = summary?.cv_recall as number | undefined;
  const cvF1 = summary?.cv_f1 as number | undefined;
  const cvAccuracyStd = summary?.cv_accuracy_std as number | undefined;
  const cvRocAucStd = summary?.cv_roc_auc_std as number | undefined;
  const cvPrecisionStd = summary?.cv_precision_std as number | undefined;
  const cvRecallStd = summary?.cv_recall_std as number | undefined;
  const cvF1Std = summary?.cv_f1_std as number | undefined;
  const nCvFolds = summary?.n_cv_folds as number | undefined;
  const posMeanScore = summary?.pos_mean_score as number | undefined;
  const negMeanScore = summary?.neg_mean_score as number | undefined;
  const scoreSeparation = summary?.score_separation as number | undefined;
  const classifierType = summary?.classifier_type as string | undefined;
  const classWeightStrategy = summary?.class_weight_strategy as string | undefined;
  const trainConfusion = summary?.train_confusion as Record<string, number> | undefined;
  const effectiveWeights = summary?.effective_class_weights as Record<string, string> | undefined;
  const configWarning = summary?._config_mismatch_warning as string | undefined;

  const classifierTag =
    classifierType === "mlp"
      ? "MLP"
      : classifierType === "logistic_regression"
        ? "LR"
        : classifierType === "linear_svm"
          ? "SVM"
          : null;
  const classifierLabel =
    classifierType === "mlp"
      ? "Neural Network (MLP)"
      : classifierType === "logistic_regression"
        ? "Logistic Regression"
        : classifierType === "linear_svm"
          ? "Linear SVM"
          : classifierType ?? "—";
  const promotionProvenance = model.promotion_provenance as Record<string, unknown> | null;
  const promotedCandidateName = promotionProvenance?.candidate_name as string | undefined;
  const promotedSourceModelName = promotionProvenance?.source_model_name as string | undefined;

  // Look up training job for regularization C
  const trainingJob = model.training_job_id
    ? trainingJobs.find(j => j.id === model.training_job_id)
    : undefined;
  const regularizationC = (trainingJob?.parameters as Record<string, unknown> | null)?.C as number | undefined;

  // Lazy-load training data summary when expanded
  const { data: dataSummary, isLoading: dataSummaryLoading, isError: dataSummaryError } =
    useTrainingDataSummary(expanded ? model.id : null);

  const fmtPct = (v: number | undefined) => v != null ? `${(v * 100).toFixed(1)}%` : "—";
  const fmtStd = (v: number | undefined) => v != null ? `\u00B1${(v * 100).toFixed(1)}%` : "";

  const sepColor = scoreSeparation != null
    ? scoreSeparation >= 2 ? "text-green-700" : scoreSeparation >= 1 ? "text-amber-700" : "text-red-700"
    : "";

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
            {model.training_source_mode === "autoresearch_candidate" && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 font-normal">
                Candidate
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
            {/* Two-column layout: Training Parameters | Performance */}
            <div className="grid grid-cols-2 gap-6 text-xs">
              {/* Left: Training Parameters */}
              <div className="max-w-xs">
                <h4 className="font-semibold text-[11px] uppercase tracking-wide text-muted-foreground mb-2">Training Parameters</h4>
                <div className="space-y-1.5">
                  <div className="flex gap-2">
                    <span className="text-muted-foreground whitespace-nowrap">Classifier Type</span>
                    <span className="font-medium">{classifierLabel}</span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-muted-foreground whitespace-nowrap">Training Source</span>
                    <span className="font-medium">
                      {model.training_source_mode === "autoresearch_candidate"
                        ? `Candidate${promotedCandidateName ? `: ${promotedCandidateName}` : ""}`
                        : "Embedding Sets"}
                    </span>
                  </div>
                  {model.training_source_mode === "autoresearch_candidate" && (
                    <div className="flex gap-2">
                      <span className="text-muted-foreground whitespace-nowrap">Compared Against</span>
                      <span className="font-medium">{promotedSourceModelName ?? model.source_model_id ?? "—"}</span>
                    </div>
                  )}
                  <div className="flex gap-2">
                    <span className="text-muted-foreground whitespace-nowrap">Class Weight</span>
                    <span className="font-medium">{classWeightStrategy ?? (effectiveWeights ? "balanced" : "—")}</span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-muted-foreground whitespace-nowrap">L2 Normalize</span>
                    <span className="font-medium">{summary?.l2_normalize != null ? (summary.l2_normalize ? "Yes" : "No") : "—"}</span>
                  </div>
                  {classifierType !== "mlp" && (
                    <div className="flex gap-2">
                      <span className="text-muted-foreground whitespace-nowrap">Regularization C</span>
                      <span className="font-medium">{regularizationC != null ? regularizationC : "1.0 (default)"}</span>
                    </div>
                  )}
                  {effectiveWeights && (
                    <div className="flex gap-2">
                      <span className="text-muted-foreground whitespace-nowrap">Effective Weights</span>
                      <span className="font-medium">neg={effectiveWeights["0"]}, pos={effectiveWeights["1"]}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Right: Performance */}
              <div className="max-w-xs">
                <h4 className="font-semibold text-[11px] uppercase tracking-wide text-muted-foreground mb-2">Performance</h4>
                <div className="space-y-1.5">
                  <div className="flex gap-2">
                    <span className="text-muted-foreground whitespace-nowrap">Accuracy</span>
                    <span><span className="font-medium">{fmtPct(cvAccuracy)}</span> <span className="text-muted-foreground text-[10px]">{fmtStd(cvAccuracyStd)}</span></span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-muted-foreground whitespace-nowrap">ROC AUC</span>
                    <span><span className="font-medium">{fmtPct(cvRocAuc)}</span> <span className="text-muted-foreground text-[10px]">{fmtStd(cvRocAucStd)}</span></span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-muted-foreground whitespace-nowrap">Precision</span>
                    <span><span className="font-medium">{fmtPct(cvPrecision)}</span> <span className="text-muted-foreground text-[10px]">{fmtStd(cvPrecisionStd)}</span></span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-muted-foreground whitespace-nowrap">Recall</span>
                    <span><span className="font-medium">{fmtPct(cvRecall)}</span> <span className="text-muted-foreground text-[10px]">{fmtStd(cvRecallStd)}</span></span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-muted-foreground whitespace-nowrap">F1</span>
                    <span><span className="font-medium">{fmtPct(cvF1)}</span> <span className="text-muted-foreground text-[10px]">{fmtStd(cvF1Std)}</span></span>
                  </div>
                  {nCvFolds != null && (
                    <div className="text-[10px] text-muted-foreground">{nCvFolds}-fold cross-validation</div>
                  )}
                  {scoreSeparation != null && (
                    <div className="flex gap-2 pt-1">
                      <span className="text-muted-foreground whitespace-nowrap">Score Separation</span>
                      <span className={`font-medium ${sepColor}`}>{scoreSeparation.toFixed(3)}</span>
                    </div>
                  )}
                  {posMeanScore != null && negMeanScore != null && (
                    <div className="text-[10px] text-muted-foreground">
                      Mean scores: pos={posMeanScore.toFixed(3)}, neg={negMeanScore.toFixed(3)}
                    </div>
                  )}
                  {trainConfusion && (
                    <div className="pt-1">
                      <span className="text-muted-foreground block mb-1">Confusion Matrix</span>
                      <div className="grid grid-cols-2 gap-px w-fit text-[10px]">
                        <div className="bg-green-100 text-green-800 px-2 py-0.5 text-center rounded-tl" title="True Positive">
                          TP {trainConfusion.tp}
                        </div>
                        <div className="bg-red-50 text-red-700 px-2 py-0.5 text-center rounded-tr" title="False Positive">
                          FP {trainConfusion.fp}
                        </div>
                        <div className="bg-red-50 text-red-700 px-2 py-0.5 text-center rounded-bl" title="False Negative">
                          FN {trainConfusion.fn}
                        </div>
                        <div className="bg-green-100 text-green-800 px-2 py-0.5 text-center rounded-br" title="True Negative">
                          TN {trainConfusion.tn}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Training Data Section */}
            <Separator className="my-3" />
            <div>
              <h4 className="font-semibold text-[11px] uppercase tracking-wide text-muted-foreground mb-2">Training Data</h4>
              {dataSummaryLoading ? (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Loading training data...
                </div>
              ) : dataSummaryError ? (
                <div className="text-xs text-muted-foreground">Training data provenance unavailable</div>
              ) : dataSummary ? (
                model.training_source_mode === "autoresearch_candidate" ? (
                  <div className="space-y-2 text-xs">
                    <div className="rounded-md border bg-muted/30 px-3 py-2 text-muted-foreground">
                      This model was trained from an imported candidate manifest rather than
                      legacy embedding-set selections, so folder-root provenance is not available
                      in the retrain summary view.
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="font-medium">Positive</div>
                        <div className="text-muted-foreground">
                          {dataSummary.total_positive} manifest vectors
                        </div>
                      </div>
                      <div>
                        <div className="font-medium">Negative</div>
                        <div className="text-muted-foreground">
                          {dataSummary.total_negative} manifest vectors
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <SourceList
                      label="Positive"
                      sources={dataSummary.positive_sources}
                      total={dataSummary.total_positive}
                    />
                    <SourceList
                      label="Negative"
                      sources={dataSummary.negative_sources}
                      total={dataSummary.total_negative}
                    />
                  </div>
                )
              ) : null}
            </div>

            {/* Config mismatch warning */}
            {configWarning && (
              <>
                <Separator className="my-3" />
                <Badge className="bg-amber-100 text-amber-800 text-[10px]">{configWarning}</Badge>
              </>
            )}

            <RetrainPanel model={model} workflow={retrainWorkflow} />
          </td>
        </tr>
      )}
    </>
  );
}

function SourceList({ label, sources, total }: {
  label: string;
  sources: TrainingSourceInfo[];
  total: number;
}) {
  // Unique top-level parent folder names
  const folders = useMemo(() => {
    const seen = new Set<string>();
    for (const s of sources) {
      const fp = s.folder_path ?? "";
      const slashIdx = fp.indexOf("/");
      const parent = fp ? (slashIdx >= 0 ? fp.slice(0, slashIdx) : fp) : "(root)";
      seen.add(parent);
    }
    return [...seen].sort();
  }, [sources]);

  return (
    <div>
      <div className="font-medium">
        {label}{" "}
        <span className="text-muted-foreground font-normal">
          ({sources.length} {sources.length === 1 ? "set" : "sets"}, {total} vectors)
        </span>
      </div>
      <div className="text-muted-foreground mt-0.5">
        <span className="font-medium text-foreground">Sets:</span>
        {folders.map((folder) => (
          <div key={folder} className="truncate">{folder}</div>
        ))}
      </div>
    </div>
  );
}
