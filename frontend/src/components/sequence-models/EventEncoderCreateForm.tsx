import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { useSegmentationJobs } from "@/hooks/queries/useCallParsing";
import {
  continuousEmbeddingSourceKind,
  useContinuousEmbeddingJobs,
  useCreateEventEncoderJob,
} from "@/api/sequenceModels";
import type {
  CreateEventEncoderJobRequest,
  EventEncoderPoolingConfig,
} from "@/api/sequenceModels";

const ALL_POOLS = [
  "mean_pool",
  "top_k_pool",
  "start_pool",
  "middle_pool",
  "end_pool",
] as const;

const DEFAULT_DESCRIPTOR_WEIGHT = 0.364;

export function EventEncoderCreateForm() {
  const { data: segJobs = [] } = useSegmentationJobs(0);
  const { data: continuousJobs = [] } = useContinuousEmbeddingJobs(0);
  const createMutation = useCreateEventEncoderJob();

  const [segJobId, setSegJobId] = useState("");
  const [continuousJobId, setContinuousJobId] = useState("");
  const [eventSourceMode, setEventSourceMode] =
    useState<"raw" | "effective">("raw");
  const [pcaDim, setPcaDim] = useState<64 | 128>(128);
  const [kValues, setKValues] = useState<Set<number>>(
    () => new Set([50, 100, 200]),
  );
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [enabledPools, setEnabledPools] = useState<Set<string>>(
    () => new Set(ALL_POOLS),
  );
  const [topKFraction, setTopKFraction] = useState(0.25);
  const [minOverlap, setMinOverlap] = useState(0.25);
  const [l2NormalizePools, setL2NormalizePools] = useState(true);
  const [embeddingWeight, setEmbeddingWeight] = useState(1.0);
  const [descriptorWeight, setDescriptorWeight] = useState(
    DEFAULT_DESCRIPTOR_WEIGHT,
  );
  const [randomSeed, setRandomSeed] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const completedSegJobs = useMemo(
    () => segJobs.filter((job) => job.status === "complete"),
    [segJobs],
  );

  const crnnEmbeddingJobs = useMemo(
    () =>
      continuousJobs.filter(
        (job) =>
          job.status === "complete" &&
          continuousEmbeddingSourceKind(job) === "region_crnn",
      ),
    [continuousJobs],
  );

  const selectedContinuousJob = useMemo(
    () => crnnEmbeddingJobs.find((job) => job.id === continuousJobId) ?? null,
    [crnnEmbeddingJobs, continuousJobId],
  );

  const compatibleSegJobs = useMemo(
    () =>
      selectedContinuousJob
        ? completedSegJobs.filter(
            (job) => job.id === selectedContinuousJob.event_segmentation_job_id,
          )
        : [],
    [completedSegJobs, selectedContinuousJob],
  );

  const canSubmit =
    segJobId !== "" &&
    continuousJobId !== "" &&
    kValues.size > 0 &&
    enabledPools.size > 0 &&
    !createMutation.isPending;

  const selectSegmentationJob = (nextSegJobId: string) => {
    if (
      selectedContinuousJob &&
      nextSegJobId &&
      selectedContinuousJob.event_segmentation_job_id !== nextSegJobId
    ) {
      return;
    }
    setSegJobId(nextSegJobId);
  };

  const selectContinuousEmbeddingJob = (nextContinuousJobId: string) => {
    setContinuousJobId(nextContinuousJobId);
    const selectedJob = crnnEmbeddingJobs.find(
      (job) => job.id === nextContinuousJobId,
    );
    const selectedSegJobId = selectedJob?.event_segmentation_job_id ?? "";
    const matchingSegJob = completedSegJobs.find(
      (job) => job.id === selectedSegJobId,
    );
    if (matchingSegJob) {
      setSegJobId(selectedSegJobId);
    } else {
      setSegJobId("");
    }
  };

  const submit = () => {
    if (!canSubmit) return;
    setError(null);
    const body: CreateEventEncoderJobRequest = {
      event_segmentation_job_id: segJobId,
      event_source_mode: eventSourceMode,
      continuous_embedding_job_id: continuousJobId,
      preprocessing: {
        l2_normalize_pools: l2NormalizePools,
        pca_dim: pcaDim,
        embedding_weight: embeddingWeight,
        descriptor_weight: descriptorWeight,
      },
      pooling: {
        enabled_pools: [...enabledPools] as EventEncoderPoolingConfig["enabled_pools"],
        top_k_fraction: topKFraction,
        min_overlap_fraction: minOverlap,
      },
      k_values: [...kValues].sort((a, b) => a - b),
      random_seed: randomSeed,
    };
    createMutation.mutate(body, {
      onError: (err: Error) => setError(err.message),
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>New Event Encoder Job</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-sm font-medium block mb-1">
              CRNN Continuous Embedding
            </label>
            <select
              data-testid="eej-continuous-job-select"
              className="w-full border rounded-md px-2 py-1 text-sm"
              value={continuousJobId}
              onChange={(e) => selectContinuousEmbeddingJob(e.target.value)}
            >
              <option value="">- select a CRNN embedding -</option>
              {crnnEmbeddingJobs.map((job) => (
                <option key={job.id} value={job.id}>
                  {job.id.slice(0, 8)} - {job.total_chunks ?? 0} chunks
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">
              Segmentation Job
            </label>
            <select
              data-testid="eej-seg-job-select"
              className="w-full border rounded-md px-2 py-1 text-sm"
              value={segJobId}
              onChange={(e) => selectSegmentationJob(e.target.value)}
              disabled={!continuousJobId}
            >
              <option value="">
                {continuousJobId
                  ? "- select a compatible Pass-2 job -"
                  : "- pick a CRNN embedding first -"}
              </option>
              {compatibleSegJobs.map((job) => (
                <option key={job.id} value={job.id}>
                  {job.id.slice(0, 8)} - {job.event_count ?? 0} events
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">Event Source</label>
            <div className="flex gap-2" data-testid="eej-event-source-mode">
              {(["raw", "effective"] as const).map((mode) => (
                <Button
                  key={mode}
                  type="button"
                  variant={eventSourceMode === mode ? "default" : "outline"}
                  size="sm"
                  onClick={() => setEventSourceMode(mode)}
                >
                  {mode}
                </Button>
              ))}
            </div>
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">PCA Dim</label>
            <div className="flex gap-2" data-testid="eej-pca-dim">
              {[64, 128].map((dim) => (
                <Button
                  key={dim}
                  type="button"
                  variant={pcaDim === dim ? "default" : "outline"}
                  size="sm"
                  onClick={() => setPcaDim(dim as 64 | 128)}
                >
                  {dim}
                </Button>
              ))}
            </div>
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">k Values</label>
            <div className="flex gap-3" data-testid="eej-k-values">
              {[50, 100, 200].map((k) => (
                <label key={k} className="flex items-center gap-2 text-sm">
                  <Checkbox
                    checked={kValues.has(k)}
                    onCheckedChange={(checked) =>
                      setKValues((prev) => {
                        const next = new Set(prev);
                        if (checked) next.add(k);
                        else next.delete(k);
                        return next;
                      })
                    }
                  />
                  {k}
                </label>
              ))}
            </div>
          </div>
        </div>

        {continuousJobId && compatibleSegJobs.length === 0 ? (
          <div className="rounded-md border px-3 py-2 text-sm text-slate-600">
            No completed Segmentation Job matches this CRNN Continuous
            Embedding job.
          </div>
        ) : null}

        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => setShowAdvanced((v) => !v)}
          data-testid="eej-advanced-toggle"
        >
          Advanced
        </Button>

        {showAdvanced ? (
          <div className="grid grid-cols-2 gap-3 border-t pt-3">
            <div>
              <label className="text-sm font-medium block mb-1">Pools</label>
              <div className="space-y-1" data-testid="eej-pools">
                {ALL_POOLS.map((pool) => (
                  <label key={pool} className="flex items-center gap-2 text-sm">
                    <Checkbox
                      checked={enabledPools.has(pool)}
                      onCheckedChange={(checked) =>
                        setEnabledPools((prev) => {
                          const next = new Set(prev);
                          if (checked) next.add(pool);
                          else next.delete(pool);
                          return next;
                        })
                      }
                    />
                    {pool}
                  </label>
                ))}
              </div>
            </div>
            <div>
              <label className="text-sm font-medium block mb-1">
                Preprocessing
              </label>
              <label className="flex items-center gap-2 text-sm">
                <Checkbox
                  data-testid="eej-l2-normalize-pools"
                  checked={l2NormalizePools}
                  onCheckedChange={(checked) =>
                    setL2NormalizePools(checked === true)
                  }
                />
                L2-normalize embedding pools
              </label>
            </div>
            <NumberField
              label="Top-k Fraction"
              testId="eej-top-k"
              value={topKFraction}
              min={0.01}
              max={1}
              step={0.01}
              onChange={setTopKFraction}
            />
            <NumberField
              label="Min Overlap"
              testId="eej-min-overlap"
              value={minOverlap}
              min={0}
              max={1}
              step={0.01}
              onChange={setMinOverlap}
            />
            <NumberField
              label="Embedding Weight"
              testId="eej-embedding-weight"
              value={embeddingWeight}
              min={0}
              step={0.1}
              onChange={setEmbeddingWeight}
            />
            <NumberField
              label="Descriptor Weight"
              testId="eej-descriptor-weight"
              value={descriptorWeight}
              min={0}
              step={0.1}
              onChange={setDescriptorWeight}
            />
            <NumberField
              label="Random Seed"
              testId="eej-random-seed"
              value={randomSeed}
              step={1}
              onChange={setRandomSeed}
            />
          </div>
        ) : null}

        {error ? (
          <div className="text-sm text-red-700" data-testid="eej-create-error">
            {error}
          </div>
        ) : null}

        <Button
          disabled={!canSubmit}
          onClick={submit}
          data-testid="eej-create-submit"
        >
          {createMutation.isPending ? "Creating..." : "Create job"}
        </Button>
      </CardContent>
    </Card>
  );
}

function NumberField({
  label,
  testId,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  testId: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div>
      <label className="text-sm font-medium block mb-1">{label}</label>
      <Input
        type="number"
        data-testid={testId}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
