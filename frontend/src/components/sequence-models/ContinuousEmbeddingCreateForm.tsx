import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useRegionDetectionJobs } from "@/hooks/queries/useCallParsing";
import { useCreateContinuousEmbeddingJob } from "@/api/sequenceModels";

const DEFAULT_MODEL_VERSION = "surfperch-tensorflow2";

export function ContinuousEmbeddingCreateForm() {
  const { data: regionJobs = [] } = useRegionDetectionJobs(0);
  const createMutation = useCreateContinuousEmbeddingJob();

  const [regionJobId, setRegionJobId] = useState<string>("");
  const [hopSeconds, setHopSeconds] = useState<number>(1.0);
  const [padSeconds, setPadSeconds] = useState<number>(10.0);
  const [modelVersion, setModelVersion] = useState<string>(
    DEFAULT_MODEL_VERSION,
  );
  const [error, setError] = useState<string | null>(null);

  const completedRegionJobs = useMemo(
    () => regionJobs.filter((j) => j.status === "complete"),
    [regionJobs],
  );

  const canSubmit =
    regionJobId !== "" &&
    hopSeconds > 0 &&
    padSeconds >= 0 &&
    !createMutation.isPending;

  const handleSubmit = () => {
    setError(null);
    if (!canSubmit) return;
    createMutation.mutate(
      {
        region_detection_job_id: regionJobId,
        model_version: modelVersion,
        hop_seconds: hopSeconds,
        pad_seconds: padSeconds,
      },
      {
        onError: (err: Error) => setError(err.message),
      },
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>New Continuous Embedding Job</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-sm font-medium block mb-1">
              Region Detection Job
            </label>
            <select
              data-testid="cej-region-job-select"
              className="w-full border rounded-md px-2 py-1 text-sm"
              value={regionJobId}
              onChange={(e) => setRegionJobId(e.target.value)}
            >
              <option value="">— select a completed Pass-1 job —</option>
              {completedRegionJobs.map((j) => (
                <option key={j.id} value={j.id}>
                  {j.id.slice(0, 8)}
                  {j.hydrophone_id ? ` · ${j.hydrophone_id}` : ""}
                  {j.region_count != null ? ` · ${j.region_count} regions` : ""}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">
              Model Version
            </label>
            <select
              data-testid="cej-model-version-select"
              className="w-full border rounded-md px-2 py-1 text-sm"
              value={modelVersion}
              onChange={(e) => setModelVersion(e.target.value)}
            >
              <option value={DEFAULT_MODEL_VERSION}>
                surfperch-tensorflow2
              </option>
            </select>
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">
              Hop Seconds
            </label>
            <Input
              type="number"
              step="0.1"
              min="0.1"
              data-testid="cej-hop-seconds"
              value={hopSeconds}
              onChange={(e) => setHopSeconds(Number(e.target.value))}
            />
          </div>
          <div>
            <label className="text-sm font-medium block mb-1">
              Pad Seconds
            </label>
            <Input
              type="number"
              step="0.5"
              min="0"
              data-testid="cej-pad-seconds"
              value={padSeconds}
              onChange={(e) => setPadSeconds(Number(e.target.value))}
            />
          </div>
        </div>

        {error ? (
          <div
            className="text-sm text-red-700"
            data-testid="cej-create-error"
          >
            {error}
          </div>
        ) : null}

        <Button
          disabled={!canSubmit}
          onClick={handleSubmit}
          data-testid="cej-create-submit"
        >
          {createMutation.isPending ? "Creating…" : "Create job"}
        </Button>
      </CardContent>
    </Card>
  );
}
