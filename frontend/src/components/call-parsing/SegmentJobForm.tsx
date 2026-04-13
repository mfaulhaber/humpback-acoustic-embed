import { useState, useEffect } from "react";
import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useRegionDetectionJobs } from "@/hooks/queries/useCallParsing";
import {
  useSegmentationModels,
  useCreateSegmentationJob,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import type {
  RegionDetectionJob,
  HydrophoneInfo,
  SegmentationModel,
} from "@/api/types";

function formatUtcShort(epoch: number): string {
  const d = new Date(epoch * 1000);
  const months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  return `${months[d.getUTCMonth()]} ${d.getUTCDate()}`;
}

function regionJobLabel(
  job: RegionDetectionJob,
  hydrophones: HydrophoneInfo[],
): string {
  const h = job.hydrophone_id
    ? (hydrophones.find((hp) => hp.id === job.hydrophone_id)?.name ??
      job.hydrophone_id)
    : "file";
  const dateRange =
    job.start_timestamp != null && job.end_timestamp != null
      ? ` · ${formatUtcShort(job.start_timestamp)}–${formatUtcShort(job.end_timestamp)}`
      : "";
  const regions =
    job.region_count != null ? ` · ${job.region_count} regions` : "";
  return `${h}${dateRange}${regions}`;
}

function parseModelEventF1(model: SegmentationModel): string {
  if (!model.config_json) return "";
  try {
    const cfg = JSON.parse(model.config_json) as Record<string, unknown>;
    if (typeof cfg.event_f1_iou_0_3 === "number") {
      return ` (F1: ${cfg.event_f1_iou_0_3.toFixed(2)})`;
    }
  } catch {
    /* ignore */
  }
  return "";
}

interface SegmentJobFormProps {
  initialRegionJobId: string | null;
}

export function SegmentJobForm({ initialRegionJobId }: SegmentJobFormProps) {
  const { data: allJobs = [] } = useRegionDetectionJobs(3000);
  const { data: hydrophones = [] } = useHydrophones();
  const { data: models = [] } = useSegmentationModels();
  const createMutation = useCreateSegmentationJob();

  const completedJobs = allJobs.filter(
    (j: RegionDetectionJob) => j.status === "complete",
  );

  const [selectedRegionJobId, setSelectedRegionJobId] = useState(
    initialRegionJobId ?? "",
  );
  const [selectedModelId, setSelectedModelId] = useState("");

  const [highThreshold, setHighThreshold] = useState(0.5);
  const [lowThreshold, setLowThreshold] = useState(0.3);
  const [minEventSec, setMinEventSec] = useState(0.2);
  const [mergeGapSec, setMergeGapSec] = useState(0.1);

  useEffect(() => {
    if (
      initialRegionJobId &&
      completedJobs.some((j) => j.id === initialRegionJobId)
    ) {
      setSelectedRegionJobId(initialRegionJobId);
    }
  }, [initialRegionJobId, completedJobs]);

  const canSubmit =
    selectedRegionJobId !== "" &&
    selectedModelId !== "" &&
    !createMutation.isPending;

  const handleSubmit = () => {
    if (!canSubmit) return;
    createMutation.mutate(
      {
        region_detection_job_id: selectedRegionJobId,
        segmentation_model_id: selectedModelId,
        config: {
          high_threshold: highThreshold,
          low_threshold: lowThreshold,
          min_event_sec: minEventSec,
          merge_gap_sec: mergeGapSec,
        },
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
        <CardTitle>New Segmentation Job</CardTitle>
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
              {completedJobs.map((j) => (
                <option key={j.id} value={j.id}>
                  {regionJobLabel(j, hydrophones)}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium">Segmentation Model</label>
            <select
              className="w-full border rounded px-3 py-2 text-sm mt-1"
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
            >
              <option value="">Select a model…</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                  {parseModelEventF1(m)}
                </option>
              ))}
            </select>
          </div>
        </div>

        <Collapsible defaultOpen={false}>
          <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium group">
            <ChevronRight className="h-3 w-3 transition-transform group-data-[state=open]:rotate-90" />
            Advanced Settings
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-2 grid grid-cols-4 gap-3">
              <div>
                <label className="text-xs font-medium">High Threshold</label>
                <Input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={highThreshold}
                  onChange={(e) =>
                    setHighThreshold(parseFloat(e.target.value) || 0.5)
                  }
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-xs font-medium">Low Threshold</label>
                <Input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={lowThreshold}
                  onChange={(e) =>
                    setLowThreshold(parseFloat(e.target.value) || 0.3)
                  }
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-xs font-medium">
                  Min Event Duration (s)
                </label>
                <Input
                  type="number"
                  min={0}
                  max={10}
                  step={0.1}
                  value={minEventSec}
                  onChange={(e) =>
                    setMinEventSec(parseFloat(e.target.value) || 0)
                  }
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-xs font-medium">Merge Gap (s)</label>
                <Input
                  type="number"
                  min={0}
                  max={5}
                  step={0.05}
                  value={mergeGapSec}
                  onChange={(e) =>
                    setMergeGapSec(parseFloat(e.target.value) || 0)
                  }
                  className="mt-1"
                />
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>

        <Button onClick={handleSubmit} disabled={!canSubmit}>
          {createMutation.isPending ? "Creating…" : "Start Segmentation"}
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
