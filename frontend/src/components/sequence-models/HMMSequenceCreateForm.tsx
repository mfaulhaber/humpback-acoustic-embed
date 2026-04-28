import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  useContinuousEmbeddingJobs,
  useCreateHMMSequenceJob,
} from "@/api/sequenceModels";

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

  const canSubmit = sourceId !== "" && nStates >= 2 && !createMutation.isPending;

  const handleSubmit = () => {
    setError(null);
    if (!canSubmit) return;
    createMutation.mutate(
      {
        continuous_embedding_job_id: sourceId,
        n_states: nStates,
        pca_dims: pcaDims,
        covariance_type: covType,
        n_iter: nIter,
        random_seed: randomSeed,
        min_sequence_length_frames: minSeqLen,
      },
      {
        onSuccess: (job) =>
          navigate(`/app/sequence-models/hmm-sequence/${job.id}`),
        onError: (err: Error) => setError(err.message),
      },
    );
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
                  {j.total_windows != null ? ` · ${j.total_windows} windows` : ""}
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
