import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ChevronRight, Loader2, Settings2 } from "lucide-react";
import { useEmbeddingSets } from "@/hooks/queries/useProcessing";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import {
  useVocClassifierTrainingJobs,
  useCreateVocClassifierTrainingJob,
} from "@/hooks/queries/useVocalization";
import { fetchDetectionJobs, fetchHydrophoneDetectionJobs } from "@/api/client";
import { useQuery } from "@tanstack/react-query";
import type { DetectionJob, VocClassifierTrainingJob } from "@/api/types";
import { fmtDate, shortId } from "@/utils/format";

function formatUtcDateTime(epochSec: number): string {
  const d = new Date(epochSec * 1000);
  return d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

function detectionJobLabel(dj: DetectionJob): string {
  if (dj.hydrophone_name && dj.start_timestamp != null && dj.end_timestamp != null) {
    return `${dj.hydrophone_name}    ${formatUtcDateTime(dj.start_timestamp)} — ${formatUtcDateTime(dj.end_timestamp)}`;
  }
  if (dj.audio_folder) return dj.audio_folder;
  return shortId(dj.id);
}

export function VocalizationTrainForm() {
  const { data: embeddingSets = [] } = useEmbeddingSets();
  const { data: audioFiles = [] } = useAudioFiles();
  const { data: localDetJobs = [] } = useQuery({
    queryKey: ["detectionJobs"],
    queryFn: fetchDetectionJobs,
  });
  const { data: hydroDetJobs = [] } = useQuery({
    queryKey: ["hydrophoneDetectionJobs"],
    queryFn: fetchHydrophoneDetectionJobs,
  });
  const detectionJobs = useMemo(
    () => [...localDetJobs, ...hydroDetJobs],
    [localDetJobs, hydroDetJobs],
  );

  // Group embedding sets by dataset (parent folder), with call types as children
  const datasetGroups = useMemo(() => {
    const audioMap = new Map(audioFiles.map((af) => [af.id, af.folder_path]));
    // dataset → { type → embedding_set_ids[] }
    const datasets = new Map<string, Map<string, string[]>>();
    for (const es of embeddingSets) {
      const folderPath = audioMap.get(es.audio_file_id) ?? "";
      if (!folderPath) continue;
      const slashIdx = folderPath.indexOf("/");
      const dataset = slashIdx >= 0 ? folderPath.slice(0, slashIdx) : folderPath;
      const typeName = slashIdx >= 0 ? folderPath.slice(slashIdx + 1) : "";
      if (!dataset || !typeName) continue;
      let types = datasets.get(dataset);
      if (!types) {
        types = new Map();
        datasets.set(dataset, types);
      }
      const ids = types.get(typeName);
      if (ids) ids.push(es.id);
      else types.set(typeName, [es.id]);
    }
    return Array.from(datasets.entries())
      .map(([dataset, types]) => ({
        dataset,
        types: Array.from(types.entries())
          .map(([name, ids]) => ({ name, count: ids.length }))
          .sort((a, b) => a.name.localeCompare(b.name)),
        allIds: Array.from(types.values()).flat(),
      }))
      .sort((a, b) => a.dataset.localeCompare(b.dataset));
  }, [embeddingSets, audioFiles]);

  const { data: trainingJobs = [] } = useVocClassifierTrainingJobs();
  const createMut = useCreateVocClassifierTrainingJob();

  const [selDatasets, setSelDatasets] = useState<Set<string>>(new Set());
  const [selDetJobs, setSelDetJobs] = useState<Set<string>>(new Set());

  // Advanced params
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [classifierType, setClassifierType] = useState("logistic_regression");
  const [l2Normalize, setL2Normalize] = useState(false);
  const [classWeight, setClassWeight] = useState("balanced");
  const [minExamples, setMinExamples] = useState(4);

  const completedDetJobs = detectionJobs.filter((j) => j.status === "complete");

  function toggleDataset(dataset: string) {
    setSelDatasets((prev) => {
      const next = new Set(prev);
      if (next.has(dataset)) next.delete(dataset);
      else next.add(dataset);
      return next;
    });
  }

  function toggleDetJob(id: string) {
    setSelDetJobs((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  // Expand selected datasets to their embedding set IDs
  const selectedEmbSetIds = useMemo(() => {
    const ids: string[] = [];
    for (const g of datasetGroups) {
      if (selDatasets.has(g.dataset)) ids.push(...g.allIds);
    }
    return ids;
  }, [selDatasets, datasetGroups]);

  // Preview: which call types will be trained from selected datasets
  const selectedTypes = useMemo(() => {
    const types = new Map<string, number>();
    for (const g of datasetGroups) {
      if (!selDatasets.has(g.dataset)) continue;
      for (const t of g.types) {
        types.set(t.name, (types.get(t.name) ?? 0) + t.count);
      }
    }
    return Array.from(types.entries())
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [selDatasets, datasetGroups]);

  function handleTrain() {
    createMut.mutate({
      source_config: {
        embedding_set_ids: selectedEmbSetIds,
        detection_job_ids: Array.from(selDetJobs),
      },
      parameters: {
        classifier_type: classifierType,
        l2_normalize: l2Normalize,
        class_weight: classWeight === "none" ? null : classWeight,
        min_examples_per_type: minExamples,
      },
    });
  }

  const hasSource = selDatasets.size > 0 || selDetJobs.size > 0;
  const activeJob = trainingJobs.find(
    (j) => j.status === "queued" || j.status === "running",
  );

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Train Model</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Embedding set sources — grouped by dataset */}
        <div>
          <h4 className="text-sm font-medium mb-1">Curated Datasets</h4>
          <p className="text-xs text-muted-foreground mb-2">
            Each dataset contains call type subfolders. One classifier is trained per type;
            other types provide negative examples.
          </p>
          {datasetGroups.length === 0 ? (
            <p className="text-sm text-muted-foreground">No embedding set datasets available.</p>
          ) : (
            <div className="border rounded-md max-h-48 overflow-y-auto divide-y">
              {datasetGroups.map((g) => (
                <label
                  key={g.dataset}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm cursor-pointer hover:bg-muted/50"
                >
                  <Checkbox
                    checked={selDatasets.has(g.dataset)}
                    onCheckedChange={() => toggleDataset(g.dataset)}
                  />
                  <span className="truncate font-medium">{g.dataset}</span>
                  <span className="text-xs text-muted-foreground shrink-0">
                    {g.types.length} {g.types.length === 1 ? "type" : "types"}, {g.allIds.length} files
                  </span>
                </label>
              ))}
            </div>
          )}

          {/* Preview of call types that will be trained */}
          {selectedTypes.length > 0 && (
            <div className="mt-2 text-xs">
              <span className="text-muted-foreground">
                Will train {selectedTypes.length} {selectedTypes.length === 1 ? "classifier" : "classifiers"}:
              </span>
              <div className="flex flex-wrap gap-1 mt-1">
                {selectedTypes.map((t) => (
                  <Badge key={t.name} variant="secondary" className="text-xs">
                    {t.name} ({t.count})
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Detection job sources */}
        <div>
          <h4 className="text-sm font-medium mb-1">Detection Jobs (labeled)</h4>
          {completedDetJobs.length === 0 ? (
            <p className="text-sm text-muted-foreground">No completed detection jobs.</p>
          ) : (
            <div className="border rounded-md max-h-40 overflow-y-auto divide-y">
              {completedDetJobs.map((dj) => (
                <label
                  key={dj.id}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm cursor-pointer hover:bg-muted/50"
                >
                  <Checkbox
                    checked={selDetJobs.has(dj.id)}
                    onCheckedChange={() => toggleDetJob(dj.id)}
                  />
                  <span className="truncate">
                    {detectionJobLabel(dj)}
                  </span>
                </label>
              ))}
            </div>
          )}
        </div>

        {/* Advanced parameters */}
        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
            <ChevronRight
              className={`h-3.5 w-3.5 transition-transform ${advancedOpen ? "rotate-90" : ""}`}
            />
            <Settings2 className="h-3.5 w-3.5" />
            Advanced Parameters
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-2 space-y-3 pl-5">
            <div className="flex items-center gap-3">
              <label className="text-sm w-36">Classifier Type</label>
              <Select value={classifierType} onValueChange={setClassifierType}>
                <SelectTrigger className="h-8 w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="logistic_regression">Logistic Regression</SelectItem>
                  <SelectItem value="mlp">MLP</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-3">
              <label className="text-sm w-36">Class Weight</label>
              <Select value={classWeight} onValueChange={setClassWeight}>
                <SelectTrigger className="h-8 w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="balanced">Balanced</SelectItem>
                  <SelectItem value="none">None</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={l2Normalize}
                onCheckedChange={(v) => setL2Normalize(v === true)}
              />
              L2 Normalize embeddings
            </label>
            <div className="flex items-center gap-3">
              <label className="text-sm w-36">Min examples/type</label>
              <Input
                type="number"
                min={1}
                value={minExamples}
                onChange={(e) => setMinExamples(Number(e.target.value))}
                className="h-8 w-24"
              />
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* Train button */}
        <Button onClick={handleTrain} disabled={!hasSource || createMut.isPending}>
          {createMut.isPending && <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />}
          Train Multi-Label Classifier
        </Button>

        {/* Active job status */}
        {activeJob && <TrainingJobStatus job={activeJob} />}

        {/* Recent jobs */}
        {trainingJobs.length > 0 && (
          <div className="space-y-1">
            <h4 className="text-sm font-medium">Recent Training Jobs</h4>
            <div className="border rounded-md divide-y text-sm">
              {trainingJobs.slice(0, 5).map((j) => (
                <div key={j.id} className="flex items-center justify-between px-3 py-1.5">
                  <span className="text-muted-foreground">{shortId(j.id)}</span>
                  <StatusBadge status={j.status} />
                  <span className="text-xs text-muted-foreground">{fmtDate(j.created_at)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StatusBadge({ status }: { status: string }) {
  const variant =
    status === "complete"
      ? "default"
      : status === "failed"
        ? "destructive"
        : "secondary";
  return <Badge variant={variant}>{status}</Badge>;
}

function TrainingJobStatus({ job }: { job: VocClassifierTrainingJob }) {
  return (
    <div className="rounded-md border bg-muted/50 p-3 text-sm space-y-1">
      <div className="flex items-center gap-2">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        <span className="font-medium">Training in progress</span>
        <StatusBadge status={job.status} />
      </div>
      <p className="text-muted-foreground">Job {shortId(job.id)}</p>
      {job.error_message && (
        <p className="text-destructive">{job.error_message}</p>
      )}
    </div>
  );
}
