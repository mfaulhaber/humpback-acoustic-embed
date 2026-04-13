import { SegmentModelTable } from "./SegmentModelTable";
import { SegmentTrainingForm } from "./SegmentTrainingForm";
import { SegmentTrainingJobTable } from "./SegmentTrainingJobTable";

export function SegmentTrainingPage() {
  return (
    <div className="space-y-6">
      <SegmentModelTable />

      <div className="space-y-4">
        <SegmentTrainingForm />
        <SegmentTrainingJobTable />
      </div>
    </div>
  );
}
