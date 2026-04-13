import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  useSegmentationTrainingDatasets,
  useCreateSegmentationTrainingJob,
} from "@/hooks/queries/useCallParsing";

export function SegmentTrainingForm() {
  const { data: datasets = [] } = useSegmentationTrainingDatasets();
  const createMutation = useCreateSegmentationTrainingJob();

  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);
  const [weightDecay, setWeightDecay] = useState(0.0001);
  const [patience, setPatience] = useState(5);
  const [gradClip, setGradClip] = useState(1.0);
  const [seed, setSeed] = useState(42);

  const canSubmit = selectedDatasetId !== "" && !createMutation.isPending;

  const handleSubmit = () => {
    if (!canSubmit) return;
    createMutation.mutate(
      {
        training_dataset_id: selectedDatasetId,
        config: {
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          weight_decay: weightDecay,
          early_stopping_patience: patience,
          grad_clip: gradClip,
          seed,
        },
      },
      {
        onSuccess: () => {
          setSelectedDatasetId("");
        },
      },
    );
  };

  return (
    <div className="bg-muted/30 rounded-md p-3 space-y-3">
      <div className="flex items-end gap-3">
        <div className="flex-1">
          <label className="text-sm font-medium">Training Dataset</label>
          <select
            className="w-full border rounded px-3 py-2 text-sm mt-1"
            value={selectedDatasetId}
            onChange={(e) => setSelectedDatasetId(e.target.value)}
            disabled={datasets.length === 0}
          >
            {datasets.length === 0 ? (
              <option value="">No datasets available</option>
            ) : (
              <>
                <option value="">Select a dataset…</option>
                {datasets.map((ds) => (
                  <option key={ds.id} value={ds.id}>
                    {ds.name} ({ds.sample_count} samples)
                  </option>
                ))}
              </>
            )}
          </select>
        </div>
        <Button onClick={handleSubmit} disabled={!canSubmit}>
          {createMutation.isPending ? "Creating…" : "Start Training"}
        </Button>
      </div>

      <Collapsible defaultOpen={false}>
        <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium group">
          <ChevronRight className="h-3 w-3 transition-transform group-data-[state=open]:rotate-90" />
          Advanced Settings
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-2 grid grid-cols-4 gap-3">
            <div>
              <label className="text-xs font-medium">Epochs</label>
              <Input
                type="number"
                min={1}
                max={200}
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 30)}
                className="mt-1"
              />
            </div>
            <div>
              <label className="text-xs font-medium">Batch Size</label>
              <Input
                type="number"
                min={1}
                max={256}
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 16)}
                className="mt-1"
              />
            </div>
            <div>
              <label className="text-xs font-medium">Learning Rate</label>
              <Input
                type="number"
                min={0.00001}
                max={1}
                step={0.0001}
                value={learningRate}
                onChange={(e) =>
                  setLearningRate(parseFloat(e.target.value) || 0.001)
                }
                className="mt-1"
              />
            </div>
            <div>
              <label className="text-xs font-medium">Weight Decay</label>
              <Input
                type="number"
                min={0}
                max={1}
                step={0.0001}
                value={weightDecay}
                onChange={(e) =>
                  setWeightDecay(parseFloat(e.target.value) || 0)
                }
                className="mt-1"
              />
            </div>
            <div>
              <label className="text-xs font-medium">Early Stop Patience</label>
              <Input
                type="number"
                min={1}
                max={50}
                value={patience}
                onChange={(e) => setPatience(parseInt(e.target.value) || 5)}
                className="mt-1"
              />
            </div>
            <div>
              <label className="text-xs font-medium">Grad Clip</label>
              <Input
                type="number"
                min={0}
                max={10}
                step={0.1}
                value={gradClip}
                onChange={(e) =>
                  setGradClip(parseFloat(e.target.value) || 1.0)
                }
                className="mt-1"
              />
            </div>
            <div>
              <label className="text-xs font-medium">Seed</label>
              <Input
                type="number"
                min={0}
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
                className="mt-1"
              />
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>

      {createMutation.isError && (
        <p className="text-sm text-red-600">
          {(createMutation.error as Error).message}
        </p>
      )}
    </div>
  );
}
