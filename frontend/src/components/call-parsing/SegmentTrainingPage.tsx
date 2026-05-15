import { SegmentModelTable } from "./SegmentModelTable";
import { SegmentTrainingForm } from "./SegmentTrainingForm";
import { SegmentTrainingJobTable } from "./SegmentTrainingJobTable";

export function SegmentTrainingPage() {
  return (
    <div className="space-y-6">
      <SegmentModelTable />
      <SegmentTrainingForm />
      <SegmentTrainingJobTable />
    </div>
  );
}
