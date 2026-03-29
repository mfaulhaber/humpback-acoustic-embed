import { VocabularyManager } from "./VocabularyManager";
import { VocalizationTrainForm } from "./VocalizationTrainForm";
import { VocalizationModelList } from "./VocalizationModelList";

export function VocalizationTrainingTab() {
  return (
    <div className="space-y-6 max-w-4xl">
      <h2 className="text-lg font-semibold">Vocalization Type Classifier</h2>
      <VocabularyManager />
      <VocalizationTrainForm />
      <VocalizationModelList />
    </div>
  );
}
