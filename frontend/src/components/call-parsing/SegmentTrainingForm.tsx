import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "@/components/ui/use-toast";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import {
  useCreateSegmentationTrainingJob,
  useSegmentationJobsWithCorrectionCounts,
} from "@/hooks/queries/useCallParsing";
import type {
  SegmentationJobWithCorrectionCount,
  SegmentationTrainingConfig,
} from "@/api/types";
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Settings2,
} from "lucide-react";

const PAGE_SIZE = 10;

const DEFAULT_CONFIG: SegmentationTrainingConfig = {
  epochs: 30,
  batch_size: 16,
  learning_rate: 0.001,
  weight_decay: 0.0001,
  early_stopping_patience: 5,
  grad_clip: 1.0,
  seed: 42,
  val_fraction: 0.2,
  n_mels: 64,
  conv_channels: [32, 64, 96, 128],
  gru_hidden: 64,
  gru_layers: 2,
  feature_config: {
    sample_rate: 16000,
    n_fft: 2048,
    hop_length: 512,
    n_mels: 64,
    fmin: 20.0,
    fmax: 4000.0,
    normalize: "per_region_zscore",
  },
};

function hydrophoneLabel(
  job: SegmentationJobWithCorrectionCount,
  hydrophones: { id: string; name: string }[],
): string {
  if (!job.hydrophone_id) return "—";
  const match = hydrophones.find((hp) => hp.id === job.hydrophone_id);
  return match?.name ?? job.hydrophone_id;
}

function dateRange(job: SegmentationJobWithCorrectionCount): string {
  if (job.start_timestamp == null || job.end_timestamp == null) return "—";
  const fmt = (ts: number) =>
    new Date(ts * 1000).toISOString().slice(0, 16).replace("T", " ") + "Z";
  return `${fmt(job.start_timestamp)} - ${fmt(job.end_timestamp)}`;
}

function parseNumber(value: string, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseChannels(value: string): number[] {
  const channels = value
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((part) => Number.isInteger(part) && part > 0);
  return channels.length > 0 ? channels : DEFAULT_CONFIG.conv_channels;
}

export function SegmentTrainingForm() {
  const { data: jobs = [] } = useSegmentationJobsWithCorrectionCounts(3000);
  const { data: hydrophones = [] } = useHydrophones();
  const createMutation = useCreateSegmentationTrainingJob();

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [page, setPage] = useState(0);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [config, setConfig] = useState<SegmentationTrainingConfig>(DEFAULT_CONFIG);
  const [convChannelsText, setConvChannelsText] = useState(
    DEFAULT_CONFIG.conv_channels.join(", "),
  );

  const completedJobs = useMemo(
    () => jobs.filter((job) => job.status === "complete"),
    [jobs],
  );
  const selectableJobs = useMemo(
    () => completedJobs.filter((job) => job.correction_count > 0),
    [completedJobs],
  );
  const selectableIds = useMemo(
    () => new Set(selectableJobs.map((job) => job.id)),
    [selectableJobs],
  );
  const totalPages = Math.max(1, Math.ceil(completedJobs.length / PAGE_SIZE));
  const effectivePage = Math.min(page, totalPages - 1);
  const pageJobs = completedJobs.slice(
    effectivePage * PAGE_SIZE,
    (effectivePage + 1) * PAGE_SIZE,
  );

  const updateConfig = <K extends keyof SegmentationTrainingConfig>(
    key: K,
    value: SegmentationTrainingConfig[K],
  ) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const updateFeatureConfig = <
    K extends keyof SegmentationTrainingConfig["feature_config"],
  >(
    key: K,
    value: SegmentationTrainingConfig["feature_config"][K],
  ) => {
    setConfig((prev) => ({
      ...prev,
      feature_config: {
        ...prev.feature_config,
        [key]: value,
      },
    }));
  };

  const updateMelBins = (next: number) => {
    setConfig((prev) => ({
      ...prev,
      n_mels: next,
      feature_config: {
        ...prev.feature_config,
        n_mels: next,
      },
    }));
  };

  const toggleJob = (jobId: string) => {
    if (!selectableIds.has(jobId)) return;
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(jobId)) next.delete(jobId);
      else next.add(jobId);
      return next;
    });
  };

  const pageSelectableIds = pageJobs
    .filter((job) => selectableIds.has(job.id))
    .map((job) => job.id);
  const allPageSelected =
    pageSelectableIds.length > 0 &&
    pageSelectableIds.every((jobId) => selected.has(jobId));

  const toggleAllPage = () => {
    setSelected((prev) => {
      const next = new Set(prev);
      for (const jobId of pageSelectableIds) {
        if (allPageSelected) next.delete(jobId);
        else next.add(jobId);
      }
      return next;
    });
  };

  const handleTrain = () => {
    const segmentationJobIds = Array.from(selected).filter((id) =>
      selectableIds.has(id),
    );
    if (segmentationJobIds.length === 0) return;
    createMutation.mutate(
      {
        segmentation_job_ids: segmentationJobIds,
        config: {
          ...config,
          conv_channels: parseChannels(convChannelsText),
        },
      },
      {
        onSuccess: () => {
          toast({
            title: "Training job queued",
            description: `${segmentationJobIds.length} segmentation job${
              segmentationJobIds.length !== 1 ? "s" : ""
            } selected for training.`,
          });
          setSelected(new Set());
        },
        onError: (err) => {
          toast({
            title: "Failed to start training",
            description: (err as Error).message,
            variant: "destructive",
          });
        },
      },
    );
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-3">
          <CardTitle className="text-base">Train Segmentation Model</CardTitle>
          <Badge variant="secondary">
            {selectableJobs.length} job{selectableJobs.length !== 1 ? "s" : ""}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {completedJobs.length === 0 ? (
          <div className="py-6 text-center text-sm text-muted-foreground">
            No completed segmentation jobs with corrections available.
          </div>
        ) : (
          <div className="overflow-hidden rounded-md border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="w-10 px-3 py-2">
                    <Checkbox
                      checked={allPageSelected}
                      onCheckedChange={toggleAllPage}
                      aria-label="Select all training jobs on this page"
                    />
                  </th>
                  <th className="px-3 py-2 text-left font-medium">
                    Hydrophone
                  </th>
                  <th className="px-3 py-2 text-left font-medium">
                    Date Range
                  </th>
                  <th className="px-3 py-2 text-right font-medium">
                    Corrections
                  </th>
                </tr>
              </thead>
              <tbody>
                {pageJobs.map((job) => {
                  const disabled = !selectableIds.has(job.id);
                  return (
                    <tr
                      key={job.id}
                      className={`border-b last:border-0 hover:bg-muted/30 ${
                        disabled ? "text-muted-foreground" : "cursor-pointer"
                      }`}
                      onClick={() => toggleJob(job.id)}
                    >
                      <td className="px-3 py-2">
                        <Checkbox
                          checked={selected.has(job.id)}
                          disabled={disabled}
                          onCheckedChange={() => toggleJob(job.id)}
                          onClick={(e) => e.stopPropagation()}
                          aria-label={`Select segmentation job ${job.id.slice(0, 8)}`}
                        />
                      </td>
                      <td className="px-3 py-2 text-xs">
                        {hydrophoneLabel(job, hydrophones)}
                      </td>
                      <td className="px-3 py-2 text-xs text-muted-foreground whitespace-nowrap">
                        {dateRange(job)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-xs">
                        {job.correction_count}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {totalPages > 1 && (
              <div className="flex items-center justify-between border-t px-3 py-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={effectivePage === 0}
                  onClick={() => setPage((prev) => Math.max(0, prev - 1))}
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                </Button>
                <span className="text-xs text-muted-foreground">
                  Page {effectivePage + 1} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={effectivePage >= totalPages - 1}
                  onClick={() =>
                    setPage((prev) => Math.min(totalPages - 1, prev + 1))
                  }
                >
                  <ChevronRight className="h-3.5 w-3.5" />
                </Button>
              </div>
            )}
          </div>
        )}

        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="gap-1.5 text-muted-foreground"
            >
              <Settings2 className="h-3.5 w-3.5" />
              Advanced Options
              {advancedOpen ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" />
              )}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-1 space-y-3 rounded-md border p-3">
            <div className="grid gap-3 md:grid-cols-4">
              <label className="space-y-1 text-sm font-medium">
                Epochs
                <Input
                  type="number"
                  min={1}
                  value={config.epochs}
                  onChange={(event) =>
                    updateConfig(
                      "epochs",
                      parseNumber(event.target.value, DEFAULT_CONFIG.epochs),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                Batch Size
                <Input
                  type="number"
                  min={1}
                  value={config.batch_size}
                  onChange={(event) =>
                    updateConfig(
                      "batch_size",
                      parseNumber(event.target.value, DEFAULT_CONFIG.batch_size),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                Learning Rate
                <Input
                  type="number"
                  min={0}
                  step={0.0001}
                  value={config.learning_rate}
                  onChange={(event) =>
                    updateConfig(
                      "learning_rate",
                      parseNumber(
                        event.target.value,
                        DEFAULT_CONFIG.learning_rate,
                      ),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                Weight Decay
                <Input
                  type="number"
                  min={0}
                  step={0.0001}
                  value={config.weight_decay}
                  onChange={(event) =>
                    updateConfig(
                      "weight_decay",
                      parseNumber(
                        event.target.value,
                        DEFAULT_CONFIG.weight_decay,
                      ),
                    )
                  }
                />
              </label>
            </div>
            <div className="grid gap-3 md:grid-cols-4">
              <label className="space-y-1 text-sm font-medium">
                Early Stop Patience
                <Input
                  type="number"
                  min={0}
                  value={config.early_stopping_patience}
                  onChange={(event) =>
                    updateConfig(
                      "early_stopping_patience",
                      parseNumber(
                        event.target.value,
                        DEFAULT_CONFIG.early_stopping_patience,
                      ),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                Grad Clip
                <Input
                  type="number"
                  min={0}
                  step={0.1}
                  value={config.grad_clip}
                  onChange={(event) =>
                    updateConfig(
                      "grad_clip",
                      parseNumber(event.target.value, DEFAULT_CONFIG.grad_clip),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                Validation Fraction
                <Input
                  type="number"
                  min={0}
                  max={0.95}
                  step={0.05}
                  value={config.val_fraction}
                  onChange={(event) =>
                    updateConfig(
                      "val_fraction",
                      parseNumber(
                        event.target.value,
                        DEFAULT_CONFIG.val_fraction,
                      ),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                Seed
                <Input
                  type="number"
                  value={config.seed}
                  onChange={(event) =>
                    updateConfig(
                      "seed",
                      parseNumber(event.target.value, DEFAULT_CONFIG.seed),
                    )
                  }
                />
              </label>
            </div>
            <div className="grid gap-3 md:grid-cols-4">
              <label className="space-y-1 text-sm font-medium">
                Mel Bins
                <Input
                  type="number"
                  min={1}
                  value={config.n_mels}
                  onChange={(event) =>
                    updateMelBins(
                      parseNumber(event.target.value, DEFAULT_CONFIG.n_mels),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                Conv Channels
                <Input
                  value={convChannelsText}
                  onChange={(event) => setConvChannelsText(event.target.value)}
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                GRU Hidden
                <Input
                  type="number"
                  min={1}
                  value={config.gru_hidden}
                  onChange={(event) =>
                    updateConfig(
                      "gru_hidden",
                      parseNumber(
                        event.target.value,
                        DEFAULT_CONFIG.gru_hidden,
                      ),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                GRU Layers
                <Input
                  type="number"
                  min={1}
                  value={config.gru_layers}
                  onChange={(event) =>
                    updateConfig(
                      "gru_layers",
                      parseNumber(
                        event.target.value,
                        DEFAULT_CONFIG.gru_layers,
                      ),
                    )
                  }
                />
              </label>
            </div>
            <div className="grid gap-3 md:grid-cols-4">
              <label className="space-y-1 text-sm font-medium">
                Sample Rate
                <Input
                  type="number"
                  min={1}
                  value={config.feature_config.sample_rate}
                  onChange={(event) =>
                    updateFeatureConfig(
                      "sample_rate",
                      parseNumber(
                        event.target.value,
                        DEFAULT_CONFIG.feature_config.sample_rate,
                      ),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                FFT Size
                <Input
                  type="number"
                  min={1}
                  value={config.feature_config.n_fft}
                  onChange={(event) =>
                    updateFeatureConfig(
                      "n_fft",
                      parseNumber(
                        event.target.value,
                        DEFAULT_CONFIG.feature_config.n_fft,
                      ),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                Hop Length
                <Input
                  type="number"
                  min={1}
                  value={config.feature_config.hop_length}
                  onChange={(event) =>
                    updateFeatureConfig(
                      "hop_length",
                      parseNumber(
                        event.target.value,
                        DEFAULT_CONFIG.feature_config.hop_length,
                      ),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                Normalization
                <Select
                  value={config.feature_config.normalize}
                  onValueChange={(value) =>
                    updateFeatureConfig("normalize", value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="per_region_zscore">
                      Per-region z-score
                    </SelectItem>
                  </SelectContent>
                </Select>
              </label>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <label className="space-y-1 text-sm font-medium">
                Min Frequency
                <Input
                  type="number"
                  min={0}
                  step={1}
                  value={config.feature_config.fmin}
                  onChange={(event) =>
                    updateFeatureConfig(
                      "fmin",
                      parseNumber(
                        event.target.value,
                        DEFAULT_CONFIG.feature_config.fmin,
                      ),
                    )
                  }
                />
              </label>
              <label className="space-y-1 text-sm font-medium">
                Max Frequency
                <Input
                  type="number"
                  min={1}
                  step={1}
                  value={config.feature_config.fmax}
                  onChange={(event) =>
                    updateFeatureConfig(
                      "fmax",
                      parseNumber(
                        event.target.value,
                        DEFAULT_CONFIG.feature_config.fmax,
                      ),
                    )
                  }
                />
              </label>
            </div>
          </CollapsibleContent>
        </Collapsible>

        <Button
          onClick={handleTrain}
          disabled={selected.size === 0 || createMutation.isPending}
        >
          {createMutation.isPending ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Queuing...
            </>
          ) : (
            `Train Model (${selected.size} job${selected.size !== 1 ? "s" : ""})`
          )}
        </Button>
      </CardContent>
    </Card>
  );
}
