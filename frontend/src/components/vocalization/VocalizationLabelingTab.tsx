import { useState } from "react";
import { VocalizationInferenceForm } from "./VocalizationInferenceForm";
import { VocalizationResultsBrowser } from "./VocalizationResultsBrowser";

export function VocalizationLabelingTab() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  return (
    <div className="space-y-6 max-w-5xl">
      <h2 className="text-lg font-semibold">Vocalization Labeling</h2>
      <VocalizationInferenceForm
        selectedJobId={selectedJobId}
        onSelectJob={setSelectedJobId}
      />
      {selectedJobId && <VocalizationResultsBrowser jobId={selectedJobId} />}
    </div>
  );
}
