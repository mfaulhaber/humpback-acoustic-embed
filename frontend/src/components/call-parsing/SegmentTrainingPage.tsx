import { SegmentModelTable } from "./SegmentModelTable";
import { FeedbackTrainingJobTable } from "./FeedbackTrainingJobTable";

export function SegmentTrainingPage() {
  return (
    <div className="space-y-6">
      <SegmentModelTable />
      <FeedbackTrainingJobTable />
    </div>
  );
}
