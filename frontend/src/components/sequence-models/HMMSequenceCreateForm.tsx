import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  type CreateHMMSequenceJobRequest,
  continuousEmbeddingSourceKind,
  useContinuousEmbeddingJobs,
  useCreateHMMSequenceJob,
} from "@/api/sequenceModels";

type TrainingMode = "full_region" | "event_balanced" | "event_only";

const DEFAULT_PROPORTIONS = {
  event_core: 0.4,
  near_event: 0.35,
  background: 0.25,
};

export function HMMSequenceCreateForm() {
  const { data: cejJobs = [] } = useContinuousEmbeddingJobs();
  const createMutation = useCreateHMMSequenceJob();
  const navigate = useNavigate();

  const completedCEJs = cejJobs.filter((j) => j.status === "complete");

  const [sourceId, setSourceId] = useState("");
  const [nStates, setNStates] = useState(4);
  const [pcaDims, setPcaDims] = useState(50);
  const [covType, setCovType] = useState<"diag" | "full">("diag");
  const [nIter, setNIter] = useState(100);
  const [randomSeed, setRandomSeed] = useState(42);
  const [minSeqLen, setMinSeqLen] = useState(10);
  const [error, setError] = useState<string | null>(null);

  // CRNN-only training-mode + tier configuration
  const [trainingMode, setTrainingMode] = useState<TrainingMode>("event_balanced");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [eventCoreProp, setEventCoreProp] = useState<number>(
    DEFAULT_PROPORTIONS.event_core,
  );
  const [nearEventProp, setNearEventProp] = useState<number>(
    DEFAULT_PROPORTIONS.near_event,
  );
  const [backgroundProp, setBackgroundProp] = useState<number>(
    DEFAULT_PROPORTIONS.background,
  );

  const selectedJob = useMemo(
    () => completedCEJs.find((j) => j.id === sourceId) ?? null,
    [completedCEJs, sourceId],
  );

  const isCrnnSource = useMemo(() => {
    if (!selectedJob) return false;
    return continuousEmbeddingSourceKind(selectedJob) === "region_crnn";
  }, [selectedJob]);

  // Reset advanced panel when source kind flips so SurfPerch never
  // sends CRNN-only fields and CRNN never sends stale config.
  useEffect(() => {
    if (!isCrnnSource) {
      setShowAdvanced(false);
    }
  }, [isCrnnSource]);

  const proportionsSum = eventCoreProp + nearEventProp + backgroundProp;
  const proportionsValid = Math.abs(proportionsSum - 1.0) <= 1e-6;

  const canSubmit =
    sourceId !== "" &&
    nStates >= 2 &&
    !createMutation.isPending &&
    (!isCrnnSource || proportionsValid);

  const handleSubmit = () => {
    setError(null);
    if (!canSubmit) return;

    const body: CreateHMMSequenceJobRequest = {
      continuous_embedding_job_id: sourceId,
      n_states: nStates,
      pca_dims: pcaDims,
      covariance_type: covType,
      n_iter: nIter,
      random_seed: randomSeed,
      min_sequence_length_frames: minSeqLen,
    };
    if (isCrnnSource) {
      body.training_mode = trainingMode;
      body.event_balanced_proportions = {
        event_core: eventCoreProp,
        near_event: nearEventProp,
        background: backgroundProp,
      };
    }

    createMutation.mutate(body, {
      onSuccess: (job) =>
        navigate(`/app/sequence-models/hmm-sequence/${job.id}`),
      onError: (err: Error) => setError(err.message),
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>New HMM Sequence Job</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-sm font-medium block mb-1">
              Continuous Embedding Job
            </label>
            <select
              data-testid="hmm-source-select"
              className="w-full border rounded-md px-2 py-1 text-sm"
              value={sourceId}
              onChange={(e) => setSourceId(e.target.value)}
            >
              <option value="">— select a completed job —</option>
              {completedCEJs.map((j) => (
                <option key={j.id} value={j.id}>
                  {j.id.slice(0, 8)}
                  {" · "}
                  {continuousEmbeddingSourceKind(j) === "region_crnn"
                    ? "CRNN"
                    : "SurfPerch"}
                  {continuousEmbeddingSourceKind(j) === "region_crnn"
                    ? j.total_chunks != null
                      ? ` · ${j.total_chunks} chunks`
                      : ""
                    : j.total_windows != null
                      ? ` · ${j.total_windows} windows`
                      : ""}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">States</label>
            <Input
              type="number"
              min={2}
              data-testid="hmm-n-states"
              value={nStates}
              onChange={(e) => setNStates(Number(e.target.value))}
            />
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">PCA Dims</label>
            <Input
              type="number"
              min={1}
              data-testid="hmm-pca-dims"
              value={pcaDims}
              onChange={(e) => setPcaDims(Number(e.target.value))}
            />
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">
              Covariance Type
            </label>
            <select
              data-testid="hmm-cov-type"
              className="w-full border rounded-md px-2 py-1 text-sm"
              value={covType}
              onChange={(e) => setCovType(e.target.value as "diag" | "full")}
            >
              <option value="diag">diag</option>
              <option value="full">full</option>
            </select>
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">
              Max Iterations
            </label>
            <Input
              type="number"
              min={1}
              data-testid="hmm-n-iter"
              value={nIter}
              onChange={(e) => setNIter(Number(e.target.value))}
            />
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">
              Random Seed
            </label>
            <Input
              type="number"
              data-testid="hmm-random-seed"
              value={randomSeed}
              onChange={(e) => setRandomSeed(Number(e.target.value))}
            />
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">
              Min Sequence Length (frames)
            </label>
            <Input
              type="number"
              min={1}
              data-testid="hmm-min-seq-len"
              value={minSeqLen}
              onChange={(e) => setMinSeqLen(Number(e.target.value))}
            />
          </div>
        </div>

        {isCrnnSource && (
          <div
            className="border-t pt-3 space-y-2"
            data-testid="hmm-crnn-training-mode-block"
          >
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">
                Training Mode (CRNN source)
              </label>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAdvanced((v) => !v)}
                data-testid="hmm-advanced-toggle"
              >
                {showAdvanced ? "Hide advanced" : "Show advanced"}
              </Button>
            </div>
            <select
              data-testid="hmm-training-mode"
              className="w-full border rounded-md px-2 py-1 text-sm"
              value={trainingMode}
              onChange={(e) =>
                setTrainingMode(e.target.value as TrainingMode)
              }
            >
              <option value="event_balanced">event_balanced</option>
              <option value="full_region">full_region</option>
              <option value="event_only">event_only</option>
            </select>

            {showAdvanced && (
              <div
                className="grid grid-cols-3 gap-3"
                data-testid="hmm-tier-config-panel"
              >
                <div>
                  <label className="text-sm font-medium block mb-1">
                    event_core
                  </label>
                  <Input
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    data-testid="hmm-tier-event-core"
                    value={eventCoreProp}
                    onChange={(e) => setEventCoreProp(Number(e.target.value))}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium block mb-1">
                    near_event
                  </label>
                  <Input
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    data-testid="hmm-tier-near-event"
                    value={nearEventProp}
                    onChange={(e) => setNearEventProp(Number(e.target.value))}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium block mb-1">
                    background
                  </label>
                  <Input
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    data-testid="hmm-tier-background"
                    value={backgroundProp}
                    onChange={(e) =>
                      setBackgroundProp(Number(e.target.value))
                    }
                  />
                </div>
                <div className="col-span-3 text-xs text-slate-500">
                  Sum: {proportionsSum.toFixed(3)}
                  {!proportionsValid && (
                    <span
                      className="ml-2 text-red-700"
                      data-testid="hmm-proportions-invalid"
                    >
                      must equal 1.0
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {error ? (
          <div className="text-sm text-red-700" data-testid="hmm-create-error">
            {error}
          </div>
        ) : null}

        <Button
          disabled={!canSubmit}
          onClick={handleSubmit}
          data-testid="hmm-create-submit"
        >
          {createMutation.isPending ? "Creating…" : "Create job"}
        </Button>
      </CardContent>
    </Card>
  );
}
