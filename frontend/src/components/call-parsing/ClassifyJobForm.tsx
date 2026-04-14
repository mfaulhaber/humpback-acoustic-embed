import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  useSegmentationJobs,
  useEventClassifierModels,
  useCreateClassificationJob,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import type {
  EventSegmentationJob,
  HydrophoneInfo,
  EventClassifierModel,
} from "@/api/types";
import { useRegionDetectionJobs } from "@/hooks/queries/useCallParsing";
import type { RegionDetectionJob } from "@/api/types";

function formatUtcShort(epoch: number): string {
  const d = new Date(epoch * 1000);
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  return `${months[d.getUTCMonth()]} ${d.getUTCDate()}`;
}

function segJobLabel(
  job: EventSegmentationJob,
  regionJobs: RegionDetectionJob[],
  hydrophones: HydrophoneInfo[],
): string {
  const rj = regionJobs.find((r) => r.id === job.region_detection_job_id);
  const shortId = job.id.slice(0, 8);
  if (!rj) return shortId;
  const hName = rj.hydrophone_id
    ? (hydrophones.find((hp) => hp.id === rj.hydrophone_id)?.name ??
        rj.hydrophone_id)
    : "file";
  const dateRange =
    rj.start_timestamp != null && rj.end_timestamp != null
      ? ` · ${formatUtcShort(rj.start_timestamp)}–${formatUtcShort(rj.end_timestamp)}`
      : "";
  const events =
    job.event_count != null ? ` · ${job.event_count} events` : "";
  return `${hName}${dateRange}${events} — ${shortId}`;
}

function modelLabel(m: EventClassifierModel): string {
  let info = m.name;
  if (m.per_class_metrics) {
    try {
      const metrics = JSON.parse(m.per_class_metrics) as Record<
        string,
        Record<string, number>
      >;
      const types = Object.keys(metrics);
      info += ` (${types.length} types)`;
    } catch {
      /* ignore */
    }
  }
  return info;
}

interface ClassifyJobFormProps {
  initialSegmentJobId: string | null;
}

export function ClassifyJobForm({ initialSegmentJobId }: ClassifyJobFormProps) {
  const { data: segJobs = [] } = useSegmentationJobs(3000);
  const { data: regionJobs = [] } = useRegionDetectionJobs();
  const { data: hydrophones = [] } = useHydrophones();
  const { data: models = [] } = useEventClassifierModels();
  const createMutation = useCreateClassificationJob();

  const completedSegJobs = segJobs.filter(
    (j: EventSegmentationJob) => j.status === "complete",
  );

  const [selectedSegJobId, setSelectedSegJobId] = useState(
    initialSegmentJobId ?? "",
  );
  const [selectedModelId, setSelectedModelId] = useState("");

  useEffect(() => {
    if (
      initialSegmentJobId &&
      completedSegJobs.some((j) => j.id === initialSegmentJobId)
    ) {
      setSelectedSegJobId(initialSegmentJobId);
    }
  }, [initialSegmentJobId, completedSegJobs]);

  const canSubmit =
    selectedSegJobId !== "" &&
    selectedModelId !== "" &&
    !createMutation.isPending;

  const handleSubmit = () => {
    if (!canSubmit) return;
    createMutation.mutate(
      {
        event_segmentation_job_id: selectedSegJobId,
        vocalization_model_id: selectedModelId,
      },
      {
        onSuccess: () => {
          setSelectedModelId("");
        },
      },
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>New Classification Job</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-sm font-medium">Segmentation Job</label>
            <select
              className="w-full border rounded px-3 py-2 text-sm mt-1"
              value={selectedSegJobId}
              onChange={(e) => setSelectedSegJobId(e.target.value)}
            >
              <option value="">Select a completed segmentation job…</option>
              {completedSegJobs.map((j) => (
                <option key={j.id} value={j.id}>
                  {segJobLabel(j, regionJobs, hydrophones)}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium">
              Event Classifier Model
            </label>
            <select
              className="w-full border rounded px-3 py-2 text-sm mt-1"
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
            >
              <option value="">Select a model…</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {modelLabel(m)}
                </option>
              ))}
            </select>
          </div>
        </div>

        <Button onClick={handleSubmit} disabled={!canSubmit}>
          {createMutation.isPending ? "Creating…" : "Run Classification"}
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
