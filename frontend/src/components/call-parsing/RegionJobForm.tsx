import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { DateRangePickerUtc } from "@/components/shared/DateRangePickerUtc";
import { useHydrophones, useClassifierModels } from "@/hooks/queries/useClassifier";
import { useModels } from "@/hooks/queries/useAdmin";
import { useCreateRegionJob, resolveModelConfigId } from "@/hooks/queries/useCallParsing";
import type { HydrophoneInfo } from "@/api/types";

export function RegionJobForm() {
  const { data: hydrophones = [] } = useHydrophones();
  const { data: models = [] } = useClassifierModels();
  const { data: modelConfigs = [] } = useModels();
  const createMutation = useCreateRegionJob();

  const [selectedHydrophoneId, setSelectedHydrophoneId] = useState("");
  const [selectedModelId, setSelectedModelId] = useState("");
  const [startEpoch, setStartEpoch] = useState<number | null>(null);
  const [endEpoch, setEndEpoch] = useState<number | null>(null);
  const [highThreshold, setHighThreshold] = useState(0.9);
  const [lowThreshold, setLowThreshold] = useState(0.8);

  const [hopSeconds, setHopSeconds] = useState(1.0);
  const [paddingSec, setPaddingSec] = useState(1.0);
  const [minRegionDuration, setMinRegionDuration] = useState(0.0);
  const [streamChunkSec, setStreamChunkSec] = useState(1800);

  const availableHydrophones = hydrophones.filter(
    (h: HydrophoneInfo) =>
      h.provider_kind === "orcasound_hls" || h.provider_kind === "noaa_gcs",
  );

  const canSubmit =
    selectedHydrophoneId !== "" &&
    selectedModelId !== "" &&
    startEpoch !== null &&
    endEpoch !== null &&
    !createMutation.isPending;

  const handleSubmit = () => {
    if (!canSubmit) return;

    const classifier = models.find((m) => m.id === selectedModelId);
    if (!classifier) return;

    const modelConfigId = resolveModelConfigId(
      classifier.model_version,
      modelConfigs,
    );
    if (!modelConfigId) return;

    createMutation.mutate({
      hydrophone_id: selectedHydrophoneId,
      start_timestamp: startEpoch!,
      end_timestamp: endEpoch!,
      model_config_id: modelConfigId,
      classifier_model_id: selectedModelId,
      config: {
        high_threshold: highThreshold,
        low_threshold: lowThreshold,
        hop_seconds: hopSeconds,
        padding_sec: paddingSec,
        min_region_duration_sec: minRegionDuration,
        stream_chunk_sec: streamChunkSec,
      },
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Region Detection</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-sm font-medium">Hydrophone</label>
            <select
              className="w-full border rounded px-3 py-2 text-sm mt-1"
              value={selectedHydrophoneId}
              onChange={(e) => setSelectedHydrophoneId(e.target.value)}
              data-testid="hydrophone-select"
            >
              <option value="">Select a hydrophone…</option>
              {availableHydrophones.map((h: HydrophoneInfo) => (
                <option key={h.id} value={h.id}>
                  {h.name} — {h.location}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium">Classifier Model</label>
            <select
              className="w-full border rounded px-3 py-2 text-sm mt-1"
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              data-testid="model-select"
            >
              <option value="">Select a model…</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.model_version})
                </option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <label className="text-sm font-medium">Date Range (UTC)</label>
          <DateRangePickerUtc
            value={{ startEpoch, endEpoch }}
            onChange={({ startEpoch: s, endEpoch: e }) => {
              setStartEpoch(s);
              setEndEpoch(e);
            }}
            placeholder="Select date range (UTC)"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-sm font-medium">
              High Threshold: {highThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={highThreshold}
              onChange={(e) => setHighThreshold(parseFloat(e.target.value))}
              className="w-full mt-1"
              data-testid="high-threshold-slider"
            />
          </div>
          <div>
            <label className="text-sm font-medium">
              Low Threshold: {lowThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={lowThreshold}
              onChange={(e) => setLowThreshold(parseFloat(e.target.value))}
              className="w-full mt-1"
              data-testid="low-threshold-slider"
            />
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
                <label className="text-xs font-medium">Hop Size (s)</label>
                <Input
                  type="number"
                  min={0.1}
                  max={10}
                  step={0.1}
                  value={hopSeconds}
                  onChange={(e) =>
                    setHopSeconds(parseFloat(e.target.value) || 1.0)
                  }
                  className="mt-1"
                  data-testid="hop-size-input"
                />
              </div>
              <div>
                <label className="text-xs font-medium">Padding (s)</label>
                <Input
                  type="number"
                  min={0}
                  max={30}
                  step={0.5}
                  value={paddingSec}
                  onChange={(e) =>
                    setPaddingSec(parseFloat(e.target.value) || 0)
                  }
                  className="mt-1"
                  data-testid="padding-input"
                />
              </div>
              <div>
                <label className="text-xs font-medium">
                  Min Region Duration (s)
                </label>
                <Input
                  type="number"
                  min={0}
                  max={60}
                  step={0.5}
                  value={minRegionDuration}
                  onChange={(e) =>
                    setMinRegionDuration(parseFloat(e.target.value) || 0)
                  }
                  className="mt-1"
                  data-testid="min-duration-input"
                />
              </div>
              <div>
                <label className="text-xs font-medium">
                  Stream Chunk (s)
                </label>
                <Input
                  type="number"
                  min={60}
                  max={7200}
                  step={60}
                  value={streamChunkSec}
                  onChange={(e) =>
                    setStreamChunkSec(parseInt(e.target.value) || 1800)
                  }
                  className="mt-1"
                  data-testid="stream-chunk-input"
                />
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>

        <Button
          onClick={handleSubmit}
          disabled={!canSubmit}
          data-testid="start-detection-btn"
        >
          {createMutation.isPending ? "Creating…" : "Start Detection"}
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
