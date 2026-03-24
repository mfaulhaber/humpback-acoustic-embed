import { useEffect, useState } from "react";
import {
  useCreateVocalizationTrainingJob,
  useVocalizationTrainingJob,
  useTrainingSummary,
} from "@/hooks/queries/useLabeling";
import { GraduationCap, Loader2, XCircle } from "lucide-react";

interface Props {
  onModelReady: (modelId: string) => void;
}

export function VocalizationTrainingPanel({
  onModelReady,
}: Props) {
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const createJob = useCreateVocalizationTrainingJob();
  const trainingJobQuery = useVocalizationTrainingJob(trainingJobId);
  const trainingSummary = useTrainingSummary();

  const status = trainingJobQuery.data?.status;

  // Auto-select model when training completes
  useEffect(() => {
    if (
      status === "complete" &&
      trainingJobQuery.data?.classifier_model_id
    ) {
      onModelReady(trainingJobQuery.data.classifier_model_id);
      setTrainingJobId(null);
    }
  }, [status, trainingJobQuery.data?.classifier_model_id, onModelReady]);

  const labelDistribution = trainingSummary.data?.label_distribution ?? {};
  const distinctLabelCount = Object.keys(labelDistribution).length;
  const labeledRowCount = trainingSummary.data?.labeled_rows ?? 0;
  const labeledJobIds = trainingSummary.data?.labeled_job_ids ?? [];

  const handleTrain = () => {
    if (labeledJobIds.length === 0) return;
    const name = `voc-classifier-${Date.now()}`;
    createJob.mutate(
      {
        name,
        source_detection_job_ids: labeledJobIds,
      },
      {
        onSuccess: (data) => {
          setTrainingJobId(data.id);
        },
      },
    );
  };

  // Training in progress
  if (trainingJobId) {
    if (status === "failed") {
      return (
        <div className="border rounded p-3 bg-white">
          <div className="text-xs text-slate-500 font-medium mb-2">
            Vocalization Classifier
          </div>
          <div className="flex items-center gap-2 text-xs text-red-600 mb-2">
            <XCircle className="h-3.5 w-3.5" />
            Training failed
          </div>
          {trainingJobQuery.data?.error_message && (
            <div className="text-[10px] text-red-500 mb-2 break-words">
              {trainingJobQuery.data.error_message}
            </div>
          )}
          <button
            onClick={() => setTrainingJobId(null)}
            className="text-xs px-2 py-1 bg-slate-100 hover:bg-slate-200 rounded"
          >
            Dismiss
          </button>
        </div>
      );
    }

    return (
      <div className="border rounded p-3 bg-white">
        <div className="text-xs text-slate-500 font-medium mb-2">
          Vocalization Classifier
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-600">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-500" />
          {status === "queued" ? "Queued..." : "Training..."}
        </div>
      </div>
    );
  }

  // Not enough labels
  if (distinctLabelCount < 2) {
    return (
      <div className="border rounded p-3 bg-white">
        <div className="text-xs text-slate-500 font-medium mb-2">
          Vocalization Classifier
        </div>
        <div className="text-xs text-slate-400">
          Label at least 2 distinct vocalization types to enable training.
          <span className="block mt-1 text-slate-500">
            {labeledRowCount === 0
              ? "No rows labeled yet — use the label input below the spectrogram."
              : `${labeledRowCount} row(s) labeled with ${distinctLabelCount} type(s)`}
          </span>
        </div>
      </div>
    );
  }

  // Ready to train
  return (
    <div className="border rounded p-3 bg-white">
      <div className="text-xs text-slate-500 font-medium mb-2">
        Vocalization Classifier
      </div>
      <button
        onClick={handleTrain}
        disabled={createJob.isPending}
        className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500 text-white rounded text-xs hover:bg-emerald-600 disabled:opacity-50"
      >
        <GraduationCap className="h-3.5 w-3.5" />
        Train Classifier
      </button>
      <div className="text-[10px] text-slate-400 mt-2 space-y-0.5">
        <div className="mb-1">
          {labeledRowCount} rows across {labeledJobIds.length} job(s)
        </div>
        {Object.entries(labelDistribution)
          .sort(([, a], [, b]) => b - a)
          .map(([label, count]) => (
            <div key={label} className="flex items-center gap-1.5">
              <span className="truncate max-w-[100px]">{label}</span>
              <span className={count < 2 ? "text-amber-500 font-medium" : "text-slate-500"}>
                {count}
              </span>
              {count < 2 && <span className="text-amber-500">needs 2+</span>}
            </div>
          ))}
      </div>
      {createJob.isError && (
        <div className="text-[10px] text-red-600 mt-1">
          {(createJob.error as Error).message}
        </div>
      )}
    </div>
  );
}
