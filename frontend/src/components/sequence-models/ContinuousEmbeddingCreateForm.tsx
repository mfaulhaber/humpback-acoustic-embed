import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  useSegmentationJobs,
  useRegionDetectionJobs,
  useSegmentationModels,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { useCreateContinuousEmbeddingJob } from "@/api/sequenceModels";
import type {
  ContinuousEmbeddingSourceKind,
  CreateContinuousEmbeddingJobRequest,
} from "@/api/sequenceModels";
import type { EventSegmentationJob, RegionDetectionJob } from "@/api/types";

const DEFAULT_MODEL_VERSION = "surfperch-tensorflow2";
const CRNN_MODEL_VERSION = "crnn-call-parsing-pytorch";

export function ContinuousEmbeddingCreateForm() {
  const { data: segJobs = [] } = useSegmentationJobs(0);
  const { data: regionJobs = [] } = useRegionDetectionJobs();
  const { data: hydrophones = [] } = useHydrophones();
  const { data: segModels = [] } = useSegmentationModels();
  const createMutation = useCreateContinuousEmbeddingJob();

  const [sourceKind, setSourceKind] =
    useState<ContinuousEmbeddingSourceKind>("surfperch");

  // SurfPerch fields
  const [segJobId, setSegJobId] = useState<string>("");
  const [hopSeconds, setHopSeconds] = useState<number>(1.0);
  const [padSeconds, setPadSeconds] = useState<number>(2.0);

  // CRNN fields
  const [regionJobId, setRegionJobId] = useState<string>("");
  const [crnnDisambiguatorSegId, setCrnnDisambiguatorSegId] = useState<string>("");
  const [crnnSegModelId, setCrnnSegModelId] = useState<string>("");
  const [chunkSize, setChunkSize] = useState<number>(0.25);
  const [chunkHop, setChunkHop] = useState<number>(0.25);
  const [projectionKind, setProjectionKind] =
    useState<"identity" | "random" | "pca">("identity");
  const [projectionDim, setProjectionDim] = useState<number>(1024);

  const [error, setError] = useState<string | null>(null);

  const completedSegJobs = useMemo(
    () => segJobs.filter((j) => j.status === "complete"),
    [segJobs],
  );

  const crnnEligibleSegJobs = useMemo(
    () =>
      completedSegJobs.filter(
        (j) => !regionJobId || j.region_detection_job_id === regionJobId,
      ),
    [completedSegJobs, regionJobId],
  );

  const completedRegionJobs = useMemo(
    () => regionJobs.filter((r) => r.status === "complete"),
    [regionJobs],
  );

  const canSubmitSurfPerch =
    sourceKind === "surfperch" &&
    segJobId !== "" &&
    hopSeconds > 0 &&
    padSeconds >= 0 &&
    !createMutation.isPending;

  const canSubmitCrnn =
    sourceKind === "region_crnn" &&
    regionJobId !== "" &&
    crnnDisambiguatorSegId !== "" &&
    crnnSegModelId !== "" &&
    chunkSize > 0 &&
    chunkHop > 0 &&
    projectionDim > 0 &&
    !createMutation.isPending;

  const canSubmit = canSubmitSurfPerch || canSubmitCrnn;

  const handleSubmit = () => {
    setError(null);
    if (!canSubmit) return;
    const body: CreateContinuousEmbeddingJobRequest =
      sourceKind === "surfperch"
        ? {
            event_segmentation_job_id: segJobId,
            model_version: DEFAULT_MODEL_VERSION,
            hop_seconds: hopSeconds,
            pad_seconds: padSeconds,
          }
        : {
            event_segmentation_job_id: crnnDisambiguatorSegId,
            region_detection_job_id: regionJobId,
            crnn_segmentation_model_id: crnnSegModelId,
            model_version: CRNN_MODEL_VERSION,
            chunk_size_seconds: chunkSize,
            chunk_hop_seconds: chunkHop,
            projection_kind: projectionKind,
            projection_dim: projectionDim,
          };
    createMutation.mutate(body, {
      onError: (err: Error) => setError(err.message),
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>New Continuous Embedding Job</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div
          className="flex gap-2 border-b pb-3"
          data-testid="cej-source-toggle"
        >
          <Button
            variant={sourceKind === "surfperch" ? "default" : "outline"}
            size="sm"
            onClick={() => setSourceKind("surfperch")}
            data-testid="cej-source-surfperch"
          >
            Event-padded (SurfPerch 1 s · 5 s window)
          </Button>
          <Button
            variant={sourceKind === "region_crnn" ? "default" : "outline"}
            size="sm"
            onClick={() => setSourceKind("region_crnn")}
            data-testid="cej-source-region-crnn"
          >
            Detection-region (CRNN 250 ms chunks)
          </Button>
        </div>

        {sourceKind === "surfperch" ? (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-sm font-medium block mb-1">
                Segmentation Job
              </label>
              <select
                data-testid="cej-seg-job-select"
                className="w-full border rounded-md px-2 py-1 text-sm"
                value={segJobId}
                onChange={(e) => setSegJobId(e.target.value)}
              >
                <option value="">— select a completed Pass-2 job —</option>
                {completedSegJobs.map((j) => (
                  <option key={j.id} value={j.id}>
                    {segJobLabel(j, regionJobs, hydrophones)} —{" "}
                    {j.event_count ?? 0} events
                  </option>
                ))}
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
        ) : (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-sm font-medium block mb-1">
                Region Detection Job (Pass 1)
              </label>
              <select
                data-testid="cej-region-job-select"
                className="w-full border rounded-md px-2 py-1 text-sm"
                value={regionJobId}
                onChange={(e) => {
                  setRegionJobId(e.target.value);
                  setCrnnDisambiguatorSegId("");
                }}
              >
                <option value="">— select a completed Pass-1 job —</option>
                {completedRegionJobs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {regionJobLabel(r, hydrophones)} — {r.region_count ?? 0}{" "}
                    regions
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-sm font-medium block mb-1">
                Pass 2 Disambiguator
              </label>
              <select
                data-testid="cej-crnn-seg-disambiguator"
                className="w-full border rounded-md px-2 py-1 text-sm"
                value={crnnDisambiguatorSegId}
                onChange={(e) => setCrnnDisambiguatorSegId(e.target.value)}
                disabled={!regionJobId}
              >
                <option value="">
                  {regionJobId
                    ? "— select a Pass-2 job for these regions —"
                    : "— pick a region job first —"}
                </option>
                {crnnEligibleSegJobs.map((j) => (
                  <option key={j.id} value={j.id}>
                    {j.id.slice(0, 8)} — {j.event_count ?? 0} events
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-sm font-medium block mb-1">
                Segmentation Model
              </label>
              <select
                data-testid="cej-crnn-segmentation-model"
                className="w-full border rounded-md px-2 py-1 text-sm"
                value={crnnSegModelId}
                onChange={(e) => setCrnnSegModelId(e.target.value)}
              >
                <option value="">— pick a CRNN checkpoint —</option>
                {segModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-sm font-medium block mb-1">
                Chunk Size (sec)
              </label>
              <Input
                type="number"
                step="0.05"
                min="0.05"
                data-testid="cej-crnn-chunk-size"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
              />
            </div>
            <div>
              <label className="text-sm font-medium block mb-1">
                Chunk Hop (sec)
              </label>
              <Input
                type="number"
                step="0.05"
                min="0.05"
                data-testid="cej-crnn-chunk-hop"
                value={chunkHop}
                onChange={(e) => setChunkHop(Number(e.target.value))}
              />
            </div>
            <div>
              <label className="text-sm font-medium block mb-1">
                Projection
              </label>
              <select
                data-testid="cej-crnn-projection-kind"
                className="w-full border rounded-md px-2 py-1 text-sm"
                value={projectionKind}
                onChange={(e) =>
                  setProjectionKind(
                    e.target.value as "identity" | "random" | "pca",
                  )
                }
              >
                <option value="identity">identity</option>
                <option value="random">random</option>
                <option value="pca">pca</option>
              </select>
            </div>
            <div>
              <label className="text-sm font-medium block mb-1">
                Projection Dim
              </label>
              <Input
                type="number"
                min="1"
                data-testid="cej-crnn-projection-dim"
                value={projectionDim}
                onChange={(e) => setProjectionDim(Number(e.target.value))}
              />
            </div>
          </div>
        )}

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

function formatUtcDate(epoch: number): string {
  const d = new Date(epoch * 1000);
  return d.toISOString().slice(0, 10);
}

function segJobLabel(
  job: EventSegmentationJob,
  regionJobs: {
    id: string;
    hydrophone_id: string | null;
    audio_file_id: string | null;
    start_timestamp: number | null;
    end_timestamp: number | null;
  }[],
  hydrophones: { id: string; name: string }[],
): string {
  const shortId = job.id.slice(0, 8);
  const rj = regionJobs.find((r) => r.id === job.region_detection_job_id);
  const parts: string[] = [];
  if (rj?.hydrophone_id) {
    const h = hydrophones.find((hp) => hp.id === rj.hydrophone_id);
    parts.push(h?.name ?? rj.hydrophone_id);
  }
  if (rj?.start_timestamp != null && rj?.end_timestamp != null) {
    const s = formatUtcDate(rj.start_timestamp);
    const e = formatUtcDate(rj.end_timestamp);
    parts.push(s === e ? s : `${s} – ${e}`);
  }
  parts.push(shortId);
  return parts.join(" - ");
}

function regionJobLabel(
  job: RegionDetectionJob,
  hydrophones: { id: string; name: string }[],
): string {
  const shortId = job.id.slice(0, 8);
  const parts: string[] = [];
  if (job.hydrophone_id) {
    const h = hydrophones.find((hp) => hp.id === job.hydrophone_id);
    parts.push(h?.name ?? job.hydrophone_id);
  }
  if (job.start_timestamp != null && job.end_timestamp != null) {
    const s = formatUtcDate(job.start_timestamp);
    const e = formatUtcDate(job.end_timestamp);
    parts.push(s === e ? s : `${s} – ${e}`);
  }
  parts.push(shortId);
  return parts.join(" - ");
}
