import { SegmentationJobPicker } from "./SegmentationJobPicker";
import { TrainingDatasetTable } from "./TrainingDatasetTable";
import { SegmentModelTable } from "./SegmentModelTable";

export function SegmentTrainingPage() {
  return (
    <div className="space-y-6">
      <SegmentationJobPicker />
      <TrainingDatasetTable />
      <SegmentModelTable />
    </div>
  );
}
