import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  useRegionDetectionJobs,
  useCreateWindowClassificationJob,
} from "@/hooks/queries/useCallParsing";
import { useVocClassifierModels } from "@/hooks/queries/useVocalization";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import type { RegionDetectionJob, VocClassifierModel } from "@/api/types";
import { formatUtcShort } from "@/utils/format";

function regionJobLabel(
  job: RegionDetectionJob,
  hydrophones: Array<{ id: string; name: string }>,
): string {
  const shortId = job.id.slice(0, 8);
  if (job.hydrophone_id) {
    const hName =
      hydrophones.find((hp) => hp.id === job.hydrophone_id)?.name ??
      job.hydrophone_id;
    const dateRange =
      job.start_timestamp != null && job.end_timestamp != null
        ? ` ${formatUtcShort(job.start_timestamp)}–${formatUtcShort(job.end_timestamp)}`
        : "";
    const regions =
      job.region_count != null ? ` · ${job.region_count} regions` : "";
    return `${hName}${dateRange}${regions} — ${shortId}`;
  }
  const regions =
    job.region_count != null ? ` · ${job.region_count} regions` : "";
  return `file${regions} — ${shortId}`;
}

function modelLabel(m: VocClassifierModel): string {
  let info = m.name;
  if (m.vocabulary_snapshot) {
    info += ` (${m.vocabulary_snapshot.length} types)`;
  }
  return info;
}

export function WindowClassifyJobForm() {
  const { data: regionJobs = [] } = useRegionDetectionJobs();
  const { data: hydrophones = [] } = useHydrophones();
  const { data: vocModels = [] } = useVocClassifierModels();
  const createMutation = useCreateWindowClassificationJob();

  const completedRegionJobs = regionJobs.filter(
    (j: RegionDetectionJob) => j.status === "complete",
  );

  const perchEmbeddingModels = vocModels.filter(
    (m) => m.model_family === "sklearn_perch_embedding",
  );

  const [selectedRegionJobId, setSelectedRegionJobId] = useState("");
  const [selectedModelId, setSelectedModelId] = useState("");

  const canSubmit =
    selectedRegionJobId !== "" &&
    selectedModelId !== "" &&
    !createMutation.isPending;

  const handleSubmit = () => {
    if (!canSubmit) return;
    createMutation.mutate(
      {
        region_detection_job_id: selectedRegionJobId,
        vocalization_model_id: selectedModelId,
      },
      undefined,
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>New Window Classification Job</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-sm font-medium">Region Detection Job</label>
            <select
              className="w-full border rounded px-3 py-2 text-sm mt-1"
              value={selectedRegionJobId}
              onChange={(e) => setSelectedRegionJobId(e.target.value)}
            >
              <option value="">Select a completed region job…</option>
              {completedRegionJobs.map((j) => (
                <option key={j.id} value={j.id}>
                  {regionJobLabel(j, hydrophones)}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium">Vocalization Model</label>
            <select
              className="w-full border rounded px-3 py-2 text-sm mt-1"
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
            >
              <option value="">Select a model…</option>
              {perchEmbeddingModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {modelLabel(m)}
                </option>
              ))}
            </select>
          </div>
        </div>

        <Button onClick={handleSubmit} disabled={!canSubmit}>
          {createMutation.isPending ? "Creating…" : "Run Window Classification"}
        </Button>
        {createMutation.isError && (
          <p className="text-sm text-red-600">
            {(createMutation.error as Error).message}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
