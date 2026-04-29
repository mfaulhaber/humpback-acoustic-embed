import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
import { Badge } from "@/components/ui/badge";
import { ChevronRight, Loader2, Settings2 } from "lucide-react";
import {
  useVocClassifierTrainingJobs,
  useCreateVocClassifierTrainingJob,
} from "@/hooks/queries/useVocalization";
import { useTrainingSummary } from "@/hooks/queries/useLabeling";
import { fetchDetectionJobs, fetchHydrophoneDetectionJobs } from "@/api/client";
import { useQuery } from "@tanstack/react-query";
import type { DetectionJob, VocClassifierTrainingJob } from "@/api/types";
import { shortId } from "@/utils/format";

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
  const { data: localDetJobs = [] } = useQuery({
    queryKey: ["detectionJobs"],
    queryFn: fetchDetectionJobs,
  });
  const { data: hydroDetJobs = [] } = useQuery({
    queryKey: ["hydrophoneDetectionJobs"],
    queryFn: fetchHydrophoneDetectionJobs,
  });
  const { data: trainingSummary } = useTrainingSummary();
  const detectionJobs = useMemo(
    () => [...localDetJobs, ...hydroDetJobs],
    [localDetJobs, hydroDetJobs],
  );
  const labeledJobIds = new Set(trainingSummary?.labeled_job_ids ?? []);
  const labeledCompletedJobs = detectionJobs.filter(
    (job) => job.status === "complete" && labeledJobIds.has(job.id),
  );

  const { data: trainingJobs = [] } = useVocClassifierTrainingJobs();
  const createMut = useCreateVocClassifierTrainingJob();

  const [selDetJobs, setSelDetJobs] = useState<Set<string>>(new Set());
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [classifierType, setClassifierType] = useState("logistic_regression");
  const [l2Normalize, setL2Normalize] = useState(false);
  const [classWeight, setClassWeight] = useState("balanced");
  const [minExamples, setMinExamples] = useState(4);

  function toggleDetJob(id: string) {
    setSelDetJobs((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function handleTrain() {
    createMut.mutate({
      source_config: {
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

  const activeJob = trainingJobs.find(
    (j) => j.status === "queued" || j.status === "running",
  );

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Train Model</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h4 className="text-sm font-medium mb-1">Labeled Detection Jobs</h4>
          <p className="text-xs text-muted-foreground mb-2">
            Training snapshots now come only from detection jobs with saved vocalization labels.
          </p>
          {labeledCompletedJobs.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No completed detection jobs with vocalization labels yet.
            </p>
          ) : (
            <div className="border rounded-md max-h-48 overflow-y-auto divide-y">
              {labeledCompletedJobs.map((dj) => (
                <label
                  key={dj.id}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm cursor-pointer hover:bg-muted/50"
                >
                  <Checkbox
                    checked={selDetJobs.has(dj.id)}
                    onCheckedChange={() => toggleDetJob(dj.id)}
                  />
                  <span className="truncate">{detectionJobLabel(dj)}</span>
                </label>
              ))}
            </div>
          )}

          {selDetJobs.size > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {Array.from(selDetJobs).map((jobId) => (
                <Badge key={jobId} variant="secondary" className="text-xs">
                  {shortId(jobId)}
                </Badge>
              ))}
            </div>
          )}
        </div>

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
                  <SelectItem value="random_forest">Random Forest</SelectItem>
                  <SelectItem value="linear_svm">Linear SVM</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={l2Normalize}
                onCheckedChange={(value) => setL2Normalize(Boolean(value))}
              />
              L2 normalize embeddings
            </label>

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

            <div className="flex items-center gap-3">
              <label className="text-sm w-36">Min examples / type</label>
              <input
                type="number"
                min={1}
                value={minExamples}
                onChange={(e) => setMinExamples(Number(e.target.value) || 1)}
                className="h-8 w-24 rounded-md border px-2 text-sm"
              />
            </div>
          </CollapsibleContent>
        </Collapsible>

        <div className="flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            {selDetJobs.size} detection job{selDetJobs.size !== 1 ? "s" : ""} selected
          </div>
          <Button
            size="sm"
            onClick={handleTrain}
            disabled={selDetJobs.size === 0 || createMut.isPending || Boolean(activeJob)}
          >
            {createMut.isPending ? (
              <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
            ) : null}
            {activeJob
              ? `Training in progress (${shortId((activeJob as VocClassifierTrainingJob).id)})`
              : "Start Training"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
