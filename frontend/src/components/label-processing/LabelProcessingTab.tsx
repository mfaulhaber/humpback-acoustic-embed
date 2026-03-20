import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
import { ChevronRight, Loader2, Eye, Play } from "lucide-react";
import { cn } from "@/lib/utils";
import { useClassifierModels } from "@/hooks/queries/useClassifier";
import {
  useLabelProcessingJobs,
  useCreateLabelProcessingJob,
  useDeleteLabelProcessingJob,
  useLabelProcessingPreview,
} from "@/hooks/queries/useLabelProcessing";
import { LabelProcessingJobCard } from "./LabelProcessingJobCard";
import { LabelProcessingPreview } from "./LabelProcessingPreview";
import type { ClassifierModelInfo } from "@/api/types";

export function LabelProcessingTab() {
  const { data: models = [] } = useClassifierModels();
  const { data: jobs = [] } = useLabelProcessingJobs(3000);
  const createMutation = useCreateLabelProcessingJob();
  const deleteMutation = useDeleteLabelProcessingJob();

  // Form state
  const [classifierModelId, setClassifierModelId] = useState("");
  const [annotationFolder, setAnnotationFolder] = useState("");
  const [audioFolder, setAudioFolder] = useState("");
  const [outputRoot, setOutputRoot] = useState("");

  // Advanced parameters
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [thresholdHigh, setThresholdHigh] = useState("0.7");
  const [smoothingWindow, setSmoothingWindow] = useState("3");
  const [enableRecentered, setEnableRecentered] = useState(true);
  const [enableSynthesized, setEnableSynthesized] = useState(true);
  const [backgroundThreshold, setBackgroundThreshold] = useState("0.1");
  const [synthesisVariants, setSynthesisVariants] = useState("3");

  // Preview state
  const [showPreview, setShowPreview] = useState(false);
  const {
    data: previewData,
    isLoading: previewLoading,
    error: previewError,
  } = useLabelProcessingPreview(
    showPreview ? annotationFolder : "",
    showPreview ? audioFolder : "",
  );

  const canSubmit =
    classifierModelId.length > 0 &&
    annotationFolder.length > 0 &&
    audioFolder.length > 0 &&
    outputRoot.length > 0 &&
    !createMutation.isPending;

  const handleSubmit = () => {
    const parameters: Record<string, unknown> = {};
    if (thresholdHigh !== "0.7") parameters.threshold_high = parseFloat(thresholdHigh);
    if (smoothingWindow !== "3") parameters.smoothing_window = parseInt(smoothingWindow);
    if (!enableRecentered) parameters.enable_recentered = false;
    if (!enableSynthesized) parameters.enable_synthesized = false;
    if (backgroundThreshold !== "0.1")
      parameters.background_threshold = parseFloat(backgroundThreshold);
    if (synthesisVariants !== "3")
      parameters.synthesis_variants = parseInt(synthesisVariants);

    createMutation.mutate({
      classifier_model_id: classifierModelId,
      annotation_folder: annotationFolder,
      audio_folder: audioFolder,
      output_root: outputRoot,
      parameters: Object.keys(parameters).length > 0 ? parameters : null,
    });
  };

  const activeJobs = jobs.filter(
    (j) => j.status === "queued" || j.status === "running",
  );
  const completedJobs = jobs.filter(
    (j) => j.status !== "queued" && j.status !== "running",
  );

  return (
    <div className="space-y-6">
      {/* Job creation form */}
      <Card>
        <CardHeader>
          <CardTitle>Create Label Processing Job</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Classifier model selector */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium">Classifier Model</label>
            <Select value={classifierModelId} onValueChange={setClassifierModelId}>
              <SelectTrigger>
                <SelectValue placeholder="Select a classifier model..." />
              </SelectTrigger>
              <SelectContent>
                {models.map((m: ClassifierModelInfo) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.name} ({m.model_version})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Folder paths */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Annotation Folder</label>
              <Input
                placeholder="/path/to/raven/annotations"
                value={annotationFolder}
                onChange={(e) => {
                  setAnnotationFolder(e.target.value);
                  setShowPreview(false);
                }}
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Audio Folder</label>
              <Input
                placeholder="/path/to/audio/recordings"
                value={audioFolder}
                onChange={(e) => {
                  setAudioFolder(e.target.value);
                  setShowPreview(false);
                }}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium">Output Root</label>
            <Input
              placeholder="/path/to/output"
              value={outputRoot}
              onChange={(e) => setOutputRoot(e.target.value)}
            />
          </div>

          {/* Advanced parameters */}
          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
            <CollapsibleTrigger className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
              <ChevronRight
                className={cn(
                  "h-3.5 w-3.5 transition-transform",
                  advancedOpen && "rotate-90",
                )}
              />
              Advanced Parameters
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="mt-3 space-y-3 pl-5">
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-xs text-muted-foreground">
                      Peak Threshold
                    </label>
                    <Input
                      type="number"
                      step="0.05"
                      min="0"
                      max="1"
                      value={thresholdHigh}
                      onChange={(e) => setThresholdHigh(e.target.value)}
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs text-muted-foreground">
                      Smoothing Window
                    </label>
                    <Input
                      type="number"
                      step="1"
                      min="1"
                      max="11"
                      value={smoothingWindow}
                      onChange={(e) => setSmoothingWindow(e.target.value)}
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs text-muted-foreground">
                      Background Threshold
                    </label>
                    <Input
                      type="number"
                      step="0.05"
                      min="0"
                      max="1"
                      value={backgroundThreshold}
                      onChange={(e) => setBackgroundThreshold(e.target.value)}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-xs text-muted-foreground">
                      Synthesis Variants
                    </label>
                    <Input
                      type="number"
                      step="1"
                      min="1"
                      max="5"
                      value={synthesisVariants}
                      onChange={(e) => setSynthesisVariants(e.target.value)}
                    />
                  </div>
                  <label className="flex items-center gap-2 text-sm pt-5">
                    <input
                      type="checkbox"
                      checked={enableRecentered}
                      onChange={(e) => setEnableRecentered(e.target.checked)}
                      className="rounded border-gray-300"
                    />
                    Enable re-centering
                  </label>
                  <label className="flex items-center gap-2 text-sm pt-5">
                    <input
                      type="checkbox"
                      checked={enableSynthesized}
                      onChange={(e) => setEnableSynthesized(e.target.checked)}
                      className="rounded border-gray-300"
                    />
                    Enable synthesis
                  </label>
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>

          {/* Actions */}
          <div className="flex items-center gap-3 pt-2">
            <Button
              variant="outline"
              size="sm"
              disabled={
                annotationFolder.length === 0 ||
                audioFolder.length === 0 ||
                previewLoading
              }
              onClick={() => setShowPreview(true)}
            >
              {previewLoading ? (
                <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
              ) : (
                <Eye className="h-4 w-4 mr-1.5" />
              )}
              Preview
            </Button>
            <Button
              size="sm"
              disabled={!canSubmit}
              onClick={handleSubmit}
            >
              {createMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
              ) : (
                <Play className="h-4 w-4 mr-1.5" />
              )}
              Start Processing
            </Button>
            {createMutation.isError && (
              <span className="text-sm text-red-600">
                {(createMutation.error as Error).message}
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Preview panel */}
      {showPreview && previewData && (
        <LabelProcessingPreview data={previewData} />
      )}
      {showPreview && previewError && (
        <Card>
          <CardContent className="py-4 text-sm text-red-600">
            Preview failed: {(previewError as Error).message}
          </CardContent>
        </Card>
      )}

      {/* Active jobs */}
      {activeJobs.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-muted-foreground">
            Active Jobs ({activeJobs.length})
          </h3>
          {activeJobs.map((job) => (
            <LabelProcessingJobCard
              key={job.id}
              job={job}
              onDelete={(id) => deleteMutation.mutate(id)}
            />
          ))}
        </div>
      )}

      {/* Completed / failed jobs */}
      {completedJobs.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-muted-foreground">
            Previous Jobs ({completedJobs.length})
          </h3>
          {completedJobs.map((job) => (
            <LabelProcessingJobCard
              key={job.id}
              job={job}
              onDelete={(id) => deleteMutation.mutate(id)}
            />
          ))}
        </div>
      )}

      {/* Empty state */}
      {jobs.length === 0 && (
        <div className="text-center text-sm text-muted-foreground py-8">
          No label processing jobs yet. Create one above to get started.
        </div>
      )}
    </div>
  );
}
