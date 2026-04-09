import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ChevronRight,
  ChevronDown,
  Loader2,
  Plus,
  Trash2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "@/components/ui/use-toast";
import {
  useManifests,
  useCreateManifest,
  useDeleteManifest,
  useSearches,
  useCreateSearch,
  useDeleteSearch,
  useSearch,
  useSearchSpaceDefaults,
  useImportCandidateFromSearch,
  useTrainingJobs,
  useClassifierModels,
} from "@/hooks/queries/useClassifier";
import { fetchDetectionJobs } from "@/api/client";
import type {
  DetectionJob,
  HyperparameterManifestSummary,
} from "@/api/types";
import { AutoresearchCandidatesSection } from "./AutoresearchCandidatesSection";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function statusVariant(
  status: string,
): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "complete":
      return "default";
    case "running":
      return "secondary";
    case "queued":
      return "outline";
    case "failed":
      return "destructive";
    default:
      return "outline";
  }
}

function fmtDate(iso: string) {
  return new Date(iso).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function sourceSummary(m: HyperparameterManifestSummary) {
  const parts: string[] = [];
  if (m.training_job_ids.length > 0)
    parts.push(`${m.training_job_ids.length} training`);
  if (m.detection_job_ids.length > 0)
    parts.push(`${m.detection_job_ids.length} detection`);
  return parts.join(", ") || "none";
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "none";
  return String(v);
}

// ---------------------------------------------------------------------------
// Manifest Section
// ---------------------------------------------------------------------------

function ManifestDetail({
  manifestId,
}: {
  manifestId: string;
}) {
  // We use the summary list data; for split_summary we fetch detail on expand
  const { data: manifests = [] } = useManifests();
  const manifest = manifests.find((m) => m.id === manifestId);
  if (!manifest) return null;

  // Only show extra detail for complete manifests
  if (manifest.status !== "complete") {
    if (manifest.error_message) {
      return (
        <p className="text-xs text-destructive mt-1">{manifest.error_message}</p>
      );
    }
    return null;
  }

  return (
    <div className="text-xs text-muted-foreground mt-2 space-y-1">
      <p>Examples: {manifest.example_count ?? "—"}</p>
      <p>Sources: {sourceSummary(manifest)}</p>
    </div>
  );
}

function ManifestsSection() {
  const { data: manifests = [], isLoading } = useManifests(3000);
  const createMutation = useCreateManifest();
  const deleteMutation = useDeleteManifest();
  const { data: trainingJobs = [] } = useTrainingJobs();
  const { data: detectionJobs = [] } = useQuery({
    queryKey: ["detectionJobs"],
    queryFn: fetchDetectionJobs,
  });

  const [showDialog, setShowDialog] = useState(false);
  const [name, setName] = useState("");
  const [selectedTrainingJobIds, setSelectedTrainingJobIds] = useState<
    Set<string>
  >(new Set());
  const [selectedDetectionJobIds, setSelectedDetectionJobIds] = useState<
    Set<string>
  >(new Set());
  const [splitRatio, setSplitRatio] = useState("70,15,15");
  const [seed, setSeed] = useState("42");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const completedTrainingJobs = useMemo(
    () => trainingJobs.filter((j) => j.status === "complete"),
    [trainingJobs],
  );
  const labeledDetectionJobs = useMemo(
    () =>
      detectionJobs.filter(
        (j) => j.status === "complete" && j.has_positive_labels,
      ),
    [detectionJobs],
  );

  const handleCreate = () => {
    const ratio = splitRatio
      .split(",")
      .map((s) => parseInt(s.trim(), 10));
    if (ratio.length !== 3 || ratio.some(isNaN)) {
      toast({
        title: "Invalid split ratio",
        description: "Enter 3 comma-separated integers",
        variant: "destructive",
      });
      return;
    }
    createMutation.mutate(
      {
        name: name.trim() || "Untitled manifest",
        training_job_ids: Array.from(selectedTrainingJobIds),
        detection_job_ids: Array.from(selectedDetectionJobIds),
        split_ratio: ratio,
        seed: parseInt(seed, 10) || 42,
      },
      {
        onSuccess: () => {
          setShowDialog(false);
          setName("");
          setSelectedTrainingJobIds(new Set());
          setSelectedDetectionJobIds(new Set());
          setSplitRatio("70,15,15");
          setSeed("42");
        },
      },
    );
  };

  return (
    <Card>
      <CardHeader className="py-3 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Manifests</CardTitle>
          <Button size="sm" variant="outline" onClick={() => setShowDialog(true)}>
            <Plus className="h-3.5 w-3.5 mr-1" /> New Manifest
          </Button>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin mx-auto" />
        ) : manifests.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No manifests yet. Create one to define the training data split.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-left text-muted-foreground">
                  <th className="py-2 pr-3 font-medium">Name</th>
                  <th className="py-2 pr-3 font-medium">Status</th>
                  <th className="py-2 pr-3 font-medium">Sources</th>
                  <th className="py-2 pr-3 font-medium">Examples</th>
                  <th className="py-2 pr-3 font-medium">Split</th>
                  <th className="py-2 pr-3 font-medium">Created</th>
                  <th className="py-2 font-medium" />
                </tr>
              </thead>
              <tbody>
                {manifests.map((m) => (
                  <tr
                    key={m.id}
                    className="border-b last:border-0 cursor-pointer hover:bg-muted/30"
                    onClick={() =>
                      setExpandedId(expandedId === m.id ? null : m.id)
                    }
                  >
                    <td className="py-2 pr-3">
                      <span className="flex items-center gap-1">
                        {expandedId === m.id ? (
                          <ChevronDown className="h-3 w-3" />
                        ) : (
                          <ChevronRight className="h-3 w-3" />
                        )}
                        {m.name}
                      </span>
                      {expandedId === m.id && (
                        <ManifestDetail manifestId={m.id} />
                      )}
                    </td>
                    <td className="py-2 pr-3">
                      <Badge
                        variant={statusVariant(m.status)}
                        className="text-[10px]"
                      >
                        {m.status}
                      </Badge>
                    </td>
                    <td className="py-2 pr-3">{sourceSummary(m)}</td>
                    <td className="py-2 pr-3">{m.example_count ?? "—"}</td>
                    <td className="py-2 pr-3">{m.split_ratio.join("/")}</td>
                    <td className="py-2 pr-3">{fmtDate(m.created_at)}</td>
                    <td className="py-2">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteMutation.mutate(m.id);
                        }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* New Manifest Dialog */}
        <Dialog open={showDialog} onOpenChange={setShowDialog}>
          <DialogContent className="sm:max-w-lg">
            <DialogHeader>
              <DialogTitle>New Manifest</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-2">
              <div className="space-y-1">
                <label className="text-xs font-medium">Name</label>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. v12-full-data"
                  className="text-sm"
                />
              </div>

              {/* Training jobs multi-select */}
              <div className="space-y-1">
                <label className="text-xs font-medium">
                  Training Jobs ({selectedTrainingJobIds.size} selected)
                </label>
                <div className="max-h-32 overflow-y-auto border rounded p-2 space-y-1">
                  {completedTrainingJobs.length === 0 ? (
                    <p className="text-xs text-muted-foreground">
                      No completed training jobs
                    </p>
                  ) : (
                    completedTrainingJobs.map((j) => (
                      <label
                        key={j.id}
                        className="flex items-center gap-2 text-xs"
                      >
                        <Checkbox
                          checked={selectedTrainingJobIds.has(j.id)}
                          onCheckedChange={(checked) => {
                            const next = new Set(selectedTrainingJobIds);
                            if (checked) next.add(j.id);
                            else next.delete(j.id);
                            setSelectedTrainingJobIds(next);
                          }}
                        />
                        {j.name}
                      </label>
                    ))
                  )}
                </div>
              </div>

              {/* Detection jobs multi-select */}
              <div className="space-y-1">
                <label className="text-xs font-medium">
                  Detection Jobs ({selectedDetectionJobIds.size} selected)
                </label>
                <div className="max-h-32 overflow-y-auto border rounded p-2 space-y-1">
                  {labeledDetectionJobs.length === 0 ? (
                    <p className="text-xs text-muted-foreground">
                      No labeled detection jobs
                    </p>
                  ) : (
                    labeledDetectionJobs.map((j) => (
                      <label
                        key={j.id}
                        className="flex items-center gap-2 text-xs"
                      >
                        <Checkbox
                          checked={selectedDetectionJobIds.has(j.id)}
                          onCheckedChange={(checked) => {
                            const next = new Set(selectedDetectionJobIds);
                            if (checked) next.add(j.id);
                            else next.delete(j.id);
                            setSelectedDetectionJobIds(next);
                          }}
                        />
                        {j.hydrophone_name ?? j.audio_folder ?? j.id}
                      </label>
                    ))
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs font-medium">
                    Split Ratio (train,val,test)
                  </label>
                  <Input
                    value={splitRatio}
                    onChange={(e) => setSplitRatio(e.target.value)}
                    placeholder="70,15,15"
                    className="text-sm"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium">Seed</label>
                  <Input
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                    placeholder="42"
                    className="text-sm"
                  />
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setShowDialog(false)}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleCreate}
                disabled={
                  createMutation.isPending ||
                  (selectedTrainingJobIds.size === 0 &&
                    selectedDetectionJobIds.size === 0)
                }
              >
                {createMutation.isPending && (
                  <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
                )}
                Create
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Search Section
// ---------------------------------------------------------------------------

function SearchExpandedDetail({ searchId }: { searchId: string }) {
  const { data: detail } = useSearch(searchId);
  const importMutation = useImportCandidateFromSearch();

  if (!detail) return <Loader2 className="h-3 w-3 animate-spin" />;

  return (
    <div className="text-xs text-muted-foreground mt-2 space-y-2">
      {detail.best_config && (
        <div>
          <p className="font-medium text-foreground mb-1">Best Config</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
            {Object.entries(detail.best_config).map(([k, v]) => (
              <div key={k}>
                <span className="text-muted-foreground">{k}:</span>{" "}
                <span className="text-foreground">{formatValue(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {detail.best_metrics && (
        <div>
          <p className="font-medium text-foreground mb-1">Best Metrics</p>
          <div className="grid grid-cols-3 gap-x-4 gap-y-0.5">
            {Object.entries(detail.best_metrics).map(([k, v]) => (
              <div key={k}>
                <span className="text-muted-foreground">{k}:</span>{" "}
                <span className="text-foreground">
                  {typeof v === "number" ? v.toFixed(4) : formatValue(v)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      {detail.comparison_result && (
        <div>
          <p className="font-medium text-foreground mb-1">
            Comparison vs Production
          </p>
          <pre className="text-[10px] bg-muted p-2 rounded overflow-x-auto">
            {JSON.stringify(detail.comparison_result, null, 2)}
          </pre>
        </div>
      )}
      {detail.status === "complete" && (
        <Button
          size="sm"
          variant="outline"
          className="mt-1"
          disabled={importMutation.isPending}
          onClick={(e) => {
            e.stopPropagation();
            importMutation.mutate(searchId);
          }}
        >
          {importMutation.isPending && (
            <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
          )}
          Import as Candidate
        </Button>
      )}
      {detail.error_message && (
        <p className="text-destructive">{detail.error_message}</p>
      )}
    </div>
  );
}

function SearchSpaceConfigurator({
  value,
  onChange,
}: {
  value: Record<string, unknown[]>;
  onChange: (v: Record<string, unknown[]>) => void;
}) {
  return (
    <div className="space-y-2">
      {Object.entries(value).map(([dim, values]) => (
        <div key={dim} className="space-y-0.5">
          <label className="text-xs font-medium">{dim}</label>
          <div className="flex flex-wrap gap-2">
            {(values as unknown[]).map((v, i) => {
              const label = formatValue(v);
              return (
                <label key={i} className="flex items-center gap-1 text-xs">
                  <Checkbox
                    checked={true}
                    onCheckedChange={(checked) => {
                      if (!checked) {
                        // Remove this value — but keep at least one
                        const filtered = values.filter((_, j) => j !== i);
                        if (filtered.length === 0) return;
                        onChange({ ...value, [dim]: filtered });
                      }
                    }}
                  />
                  {label}
                </label>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function SearchesSection() {
  const { data: searches = [], isLoading } = useSearches(3000);
  const { data: manifests = [] } = useManifests();
  const { data: models = [] } = useClassifierModels();
  const { data: defaults } = useSearchSpaceDefaults();
  const createMutation = useCreateSearch();
  const deleteMutation = useDeleteSearch();

  const [showDialog, setShowDialog] = useState(false);
  const [name, setName] = useState("");
  const [manifestId, setManifestId] = useState("");
  const [nTrials, setNTrials] = useState("100");
  const [seed, setSeed] = useState("42");
  const [comparisonModelId, setComparisonModelId] = useState<string>("");
  const [comparisonThreshold, setComparisonThreshold] = useState("0.85");
  const [searchSpace, setSearchSpace] = useState<Record<
    string,
    unknown[]
  > | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const completedManifests = useMemo(
    () => manifests.filter((m) => m.status === "complete"),
    [manifests],
  );

  const openDialog = () => {
    if (defaults) {
      setSearchSpace({ ...defaults.search_space });
    }
    setShowDialog(true);
  };

  const handleCreate = () => {
    createMutation.mutate(
      {
        name: name.trim() || "Untitled search",
        manifest_id: manifestId,
        search_space: searchSpace ?? undefined,
        n_trials: parseInt(nTrials, 10) || 100,
        seed: parseInt(seed, 10) || 42,
        comparison_model_id: comparisonModelId || null,
        comparison_threshold: comparisonModelId
          ? parseFloat(comparisonThreshold) || 0.85
          : null,
      },
      {
        onSuccess: () => {
          setShowDialog(false);
          setName("");
          setManifestId("");
          setNTrials("100");
          setSeed("42");
          setComparisonModelId("");
          setComparisonThreshold("0.85");
          setSearchSpace(null);
        },
      },
    );
  };

  const hasRunning = searches.some(
    (s) => s.status === "queued" || s.status === "running",
  );

  return (
    <Card>
      <CardHeader className="py-3 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Searches</CardTitle>
          <Button size="sm" variant="outline" onClick={openDialog}>
            <Plus className="h-3.5 w-3.5 mr-1" /> New Search
          </Button>
        </div>
        <p className="text-[10px] text-muted-foreground mt-1">
          Objective: recall - 15 &times; high_conf_fp_rate - 3 &times; fp_rate
        </p>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin mx-auto" />
        ) : searches.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No searches yet. Create one to run a hyperparameter search.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-left text-muted-foreground">
                  <th className="py-2 pr-3 font-medium">Name</th>
                  <th className="py-2 pr-3 font-medium">Status</th>
                  <th className="py-2 pr-3 font-medium">Manifest</th>
                  <th className="py-2 pr-3 font-medium">Best Obj.</th>
                  <th className="py-2 pr-3 font-medium">Comparison</th>
                  <th className="py-2 pr-3 font-medium">Created</th>
                  <th className="py-2 font-medium" />
                </tr>
              </thead>
              <tbody>
                {searches.map((s) => (
                  <tr
                    key={s.id}
                    className="border-b last:border-0 cursor-pointer hover:bg-muted/30"
                    onClick={() =>
                      setExpandedId(expandedId === s.id ? null : s.id)
                    }
                  >
                    <td className="py-2 pr-3">
                      <span className="flex items-center gap-1">
                        {expandedId === s.id ? (
                          <ChevronDown className="h-3 w-3" />
                        ) : (
                          <ChevronRight className="h-3 w-3" />
                        )}
                        {s.name}
                      </span>
                      {expandedId === s.id && (
                        <SearchExpandedDetail searchId={s.id} />
                      )}
                    </td>
                    <td className="py-2 pr-3">
                      <Badge
                        variant={statusVariant(s.status)}
                        className="text-[10px]"
                      >
                        {s.status === "running"
                          ? `running ${s.trials_completed}/${s.n_trials}`
                          : s.status}
                      </Badge>
                    </td>
                    <td className="py-2 pr-3">
                      {s.manifest_name ?? s.manifest_id.slice(0, 8)}
                    </td>
                    <td className="py-2 pr-3">
                      {s.best_objective != null
                        ? s.best_objective.toFixed(4)
                        : "—"}
                    </td>
                    <td className="py-2 pr-3">
                      {s.comparison_model_id
                        ? models.find((m) => m.id === s.comparison_model_id)
                            ?.name ?? s.comparison_model_id.slice(0, 8)
                        : "—"}
                    </td>
                    <td className="py-2 pr-3">{fmtDate(s.created_at)}</td>
                    <td className="py-2">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteMutation.mutate(s.id);
                        }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* New Search Dialog */}
        <Dialog open={showDialog} onOpenChange={setShowDialog}>
          <DialogContent className="sm:max-w-xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>New Hyperparameter Search</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-2">
              <div className="space-y-1">
                <label className="text-xs font-medium">Name</label>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. search-v12-broad"
                  className="text-sm"
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs font-medium">Manifest</label>
                <Select value={manifestId} onValueChange={setManifestId}>
                  <SelectTrigger className="text-sm">
                    <SelectValue placeholder="Select a completed manifest" />
                  </SelectTrigger>
                  <SelectContent>
                    {completedManifests.map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        {m.name} ({m.example_count ?? "?"} examples)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs font-medium">Trials</label>
                  <Input
                    value={nTrials}
                    onChange={(e) => setNTrials(e.target.value)}
                    placeholder="100"
                    className="text-sm"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium">Seed</label>
                  <Input
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                    placeholder="42"
                    className="text-sm"
                  />
                </div>
              </div>

              {/* Search space configurator */}
              {searchSpace && (
                <Collapsible defaultOpen={false}>
                  <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium group">
                    <ChevronRight className="h-3 w-3 transition-transform group-data-[state=open]:rotate-90" />
                    Search Space
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="mt-2 border rounded p-3">
                      <SearchSpaceConfigurator
                        value={searchSpace}
                        onChange={setSearchSpace}
                      />
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              )}

              {/* Comparison model */}
              <Collapsible defaultOpen={false}>
                <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium group">
                  <ChevronRight className="h-3 w-3 transition-transform group-data-[state=open]:rotate-90" />
                  Compare vs Production Model
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="mt-2 space-y-2">
                    <Select
                      value={comparisonModelId}
                      onValueChange={setComparisonModelId}
                    >
                      <SelectTrigger className="text-sm">
                        <SelectValue placeholder="None (skip comparison)" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="">None</SelectItem>
                        {models.map((m) => (
                          <SelectItem key={m.id} value={m.id}>
                            {m.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {comparisonModelId && (
                      <div className="space-y-1">
                        <label className="text-xs font-medium">
                          Decision Threshold
                        </label>
                        <Input
                          value={comparisonThreshold}
                          onChange={(e) =>
                            setComparisonThreshold(e.target.value)
                          }
                          placeholder="0.85"
                          className="text-sm"
                        />
                      </div>
                    )}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </div>
            <DialogFooter>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setShowDialog(false)}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleCreate}
                disabled={createMutation.isPending || !manifestId}
              >
                {createMutation.isPending && (
                  <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
                )}
                Create
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// TuningTab (main export)
// ---------------------------------------------------------------------------

export function TuningTab() {
  return (
    <div className="space-y-4 p-4">
      <h2 className="text-lg font-semibold">Hyperparameter Tuning</h2>

      <Collapsible defaultOpen={true}>
        <CollapsibleTrigger className="flex items-center gap-1 text-sm font-medium group w-full">
          <ChevronRight className="h-4 w-4 transition-transform group-data-[state=open]:rotate-90" />
          Manifests
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-2">
            <ManifestsSection />
          </div>
        </CollapsibleContent>
      </Collapsible>

      <Collapsible defaultOpen={true}>
        <CollapsibleTrigger className="flex items-center gap-1 text-sm font-medium group w-full">
          <ChevronRight className="h-4 w-4 transition-transform group-data-[state=open]:rotate-90" />
          Searches
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-2">
            <SearchesSection />
          </div>
        </CollapsibleContent>
      </Collapsible>

      <Collapsible defaultOpen={true}>
        <CollapsibleTrigger className="flex items-center gap-1 text-sm font-medium group w-full">
          <ChevronRight className="h-4 w-4 transition-transform group-data-[state=open]:rotate-90" />
          Candidates
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-2">
            <AutoresearchCandidatesSection />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
