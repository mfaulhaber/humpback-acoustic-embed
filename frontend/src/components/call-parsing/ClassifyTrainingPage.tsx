import { ClassifyModelTable } from "./ClassifyModelTable";
import { ClassificationJobPicker } from "./ClassificationJobPicker";
import { ClassifyTrainingJobTable } from "./ClassifyTrainingJobTable";

export function ClassifyTrainingPage() {
  return (
    <div className="space-y-6">
      <ClassifyModelTable />
      <ClassificationJobPicker />
      <ClassifyTrainingJobTable />
    </div>
  );
}
